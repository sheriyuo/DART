import os
import json
import torch
import asyncio
import httpx
import uuid
import time
import random
import subprocess
import argparse

import pandas as pd

from tqdm import tqdm
from typing import List, Union, Optional, Dict, Any, Tuple
from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer


class ToolServerClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        初始化客户端
        :param base_url: 服务地址，例如 "http://localhost:30500"
        :param timeout: 请求超时时间
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        # 使用异步客户端并在整个生命周期内复用连接
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        """关闭客户端连接"""
        await self.client.aclose()

    async def check_health(self) -> Dict:
        """检查服务健康状态"""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def get_observations(
        self, 
        actions: Union[str, List[str]], 
        trajectory_ids: Optional[Union[str, List[str]]] = None,
        extra_fields: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        异步调用工具服务获取观察结果
        
        :param actions: 单个动作字符串 或 动作字符串列表
        :param trajectory_ids: (可选) 对应的轨迹ID，如果不传会自动生成
        :param extra_fields: (可选) 额外的参数字段
        :return: 包含 observations, dones, valids 的字典
        """
        # 1. 规范化输入为列表格式
        if isinstance(actions, str):
            actions = [actions]
        
        # 2. 处理 Trajectory IDs
        if trajectory_ids is None:
            # 如果没传ID，为每个动作生成唯一UUID
            trajectory_ids = [str(uuid.uuid4()) for _ in range(len(actions))]
        elif isinstance(trajectory_ids, str):
            trajectory_ids = [trajectory_ids]

        # 验证长度一致性
        if len(actions) != len(trajectory_ids):
            raise ValueError(f"Actions length ({len(actions)}) must match IDs length ({len(trajectory_ids)})")

        # 3. 构建请求体 (对应服务器端的 ActionRequest 模型)
        payload = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
        }
        
        if extra_fields:
            payload["extra_fields"] = extra_fields

        try:
            # 4. 发送 POST 请求
            response = await self.client.post(
                "/get_observation", 
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as exc:
            print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            print(f"Details: {exc.response.text}")
            raise
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
            raise


class InferenceService:
    def __init__(self, model_tool: str = None, model_reasoning: str = None):
        self.model_tool = model_tool 
        self.model_reasoning = model_reasoning

        # 部署超参数
        self.type_split = 0.75
        self.gpu_utilization = 0.6

        # 初始化工具服务客户端
        SERVER_PORT = 30669  
        SERVER_URL = f"http://localhost:{SERVER_PORT}"
        self.tool_client = ToolServerClient(SERVER_URL)
        self.max_turns = 3  
        self.turn_end_token = "<|im_end|>"
        self.max_obs_length = 512

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_tool)

        self.load_model()
    
    def load_model(self):
        """load the model using VLLM backend"""
        # start a VLLM server using vllm.serve
        vllm_args = [] # 预留接口
        
        host = "0.0.0.0"
        num_models = torch.cuda.device_count()
        first_num = int(num_models * self.type_split)

        ports = random.sample(range(8000, 9000), num_models)
        self.vllm_processes = []
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(torch.cuda.device_count())])).split(",")
        tensor_parallel_size = 1
        gpu_ids_per_model = [gpu_ids[i:i+tensor_parallel_size] for i in range(0, len(gpu_ids), tensor_parallel_size)]
        assert len(gpu_ids) >= num_models * tensor_parallel_size, f"Not enough GPUs available: {len(gpu_ids)} < {num_models * tensor_parallel_size}"
        for i in range(num_models):
            if i < first_num:
                model = self.model_reasoning
            else:
                model = self.model_tool
            cmd = [
                "vllm", "serve", model, "--api-key", "token-abc123",
                "--host", host, "--port", str(ports[i]), 
                "--gpu-memory-utilization", str(self.gpu_utilization),
                "--disable-uvicorn-access-log", "--disable-log-stats", "--disable-log-requests"
            ] + vllm_args
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_per_model[i])
            env["VLLM_LOGGING_LEVEL"] = "ERROR"
            vllm_process = subprocess.Popen(cmd, env=env)
            self.vllm_processes.append(vllm_process)
        self.reasoning_clients = [
            OpenAI(api_key="token-abc123", base_url=f"http://{host}:{ports[i]}/v1") for i in range(first_num)
        ]
        self.tool_clients = [
            OpenAI(api_key="token-abc123", base_url=f"http://{host}:{ports[i]}/v1") for i in range(first_num, num_models)
        ]
        
        # Wait for the service to start (poll the health endpoint)
        max_retries = 60
        retry_interval = 10
        vllm_model_status = [False for _ in range(num_models)]
        for i in range(max_retries):
            for j in range(num_models):
                if vllm_model_status[j]:
                    continue
                try:
                    if j < first_num:
                        response = self.reasoning_clients[j].models.list()
                    else:
                        response = self.tool_clients[j-first_num].models.list()
                    vllm_model_status[j] = True
                    print(f"vLLM instance model-{j} status: {response}")
                except Exception as e:
                    # print(f"vLLM instance model-{j} at {host}:{ports[j]} is not ready yet: {str(e)}")
                    continue
            if all(vllm_model_status):
                print(f"✅ vLLM service started successfully with model")
                return     
            else:
                time.sleep(retry_interval)
        
        # If we get here, the service failed to start
        print("Failed to start one or more vLLM services. Check vLLM logs.")
        for process in self.vllm_processes:
            stderr = process.stderr.read()
            print(f"vLLM stderr: {stderr}")
            process.terminate()
        
        raise RuntimeError("Failed to start vLLM services")
    
    async def send_request(self, client, prompts: List[str], model:str, sampling_params: dict) -> str:
        # Send the request using the client
        sampling_params = sampling_params.copy()
        # Use the async encode method to get tokens
        prompt_lens = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        max_prompt_tokens = max(prompt_lens)
        
        sampling_params['max_tokens'] = min(max(32768 - max_prompt_tokens, 0), sampling_params['max_tokens'])
        # print(f"Sending request to {client.base_url} with sampling params: {sampling_params}")
        
        # Run the API call in an executor to not block the event loop
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.completions.create(
                model=model,
                prompt=prompts,
                echo=False,
                stream=False,
                **sampling_params
            )
        )
        return response

    async def _postprocess_responses(self, outputs: torch.Tensor, active_type: List[str]) -> torch.Tensor:
        """Process responses to stop at python operation or answer operation."""
        active_responses = [outputs.choices[i].text for i in range(len(outputs.choices))]
        active_finish_reasons = [outputs.choices[i].finish_reason for i in range(len(outputs.choices))]
        
        finishes = []
        for i in range(len(active_responses)):
            finish = True
            if active_finish_reasons[i] == "stop" and outputs.choices[i].stop_reason is not None:
                active_responses[i] = active_responses[i] + outputs.choices[i].stop_reason
                if outputs.choices[i].stop_reason == "<search>":
                    active_type[i] = "tool"
                elif outputs.choices[i].stop_reason == "</search>":
                    active_type[i] = "reasoning" 
                finish = False
            finishes.append(finish)
        return active_responses, finishes, active_finish_reasons, active_type
    
    async def post_process_observations(self, next_obs: List[str], dones: List[bool], valid_action: List[bool], finishs: List[bool]):
        """Process observations using the tokenizer with proper async locks"""

        def _obs_to_str(x):
            if x is None:
                return ""
            if isinstance(x, str):
                return x
            if isinstance(x, bytes):
                try:
                    return x.decode("utf-8", "replace")
                except Exception:
                    return str(x)
            # 常见结构：list / dict / tuple
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                try:
                    return "\n".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
                except Exception:
                    return str(x)
            
        safe_obs = []
        for obs, done in zip(next_obs, dones):
            s = "" if done else _obs_to_str(obs)
            safe_obs.append(s)

        is_tructed = False
        next_obs_ids = self.tokenizer(
            safe_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
            padding_side='right',
        )['input_ids'].to(torch.int64)
        if next_obs_ids.shape[1] > self.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.max_obs_length}")
            next_obs_ids = next_obs_ids[:, :self.max_obs_length]
            is_tructed = True
    
        next_obs = self.tokenizer.batch_decode(
            next_obs_ids,
            skip_special_tokens=True,
        )
        if is_tructed:
            next_obs[0] += "</information>\n\n"
        return next_obs

    async def generate_with_tools(self, prompts: List[str], sampling_params: dict) -> Tuple[List[str], List[str]]:
        """
        Dual-Agent Generation:
        - Model 1 (Reasoning): Generates thought process. Stops at <search> or </answer>.
        - Model 2 (Tool): Generates search query. Stops at </search>.
        """
        # 假设 self.clients[0] 是 Model 1 (Reasoning), self.clients[1] 是 Model 2 (Tool)
        # 你也可以在初始化时指定 self.reasoning_client 和 self.tool_client
        reasoning_client = random.choice(self.reasoning_clients)
        tool_client = random.choice(self.tool_clients)
        assert len(prompts) == 1, "Only batch size 1 is supported"

        # 初始化状态, batch 只能为1
        contexts = prompts
        final_responses = ["" for _ in range(len(prompts))]
        traj_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        active_masks = [True for _ in range(len(prompts))] # 标记是否还在对话中
        finish_reasons = [None for _ in range(len(prompts))]
        active_type = ["reasoning" for i in range(len(prompts))] # 默认请求 reasoning model

        # 定义 Stop Tokens
        STOP_REASONING = ["<search>", "</answer>"]
        STOP_TOOL_GEN = ["</search>"]

        for action_step in range(self.max_turns+1):
            active_traj_ids = [traj_ids[i] for i in range(len(traj_ids)) if active_masks[i]]
            active_contexts = [contexts[i] for i in range(len(contexts)) if active_masks[i]]
            if len(active_contexts) == 0:
                break
            
            # 1.2 设置采样参数
            sp_reasoning = sampling_params.copy()
            if action_step != self.max_turns:
                sp_reasoning["stop"] = STOP_REASONING
            else:
                sp_reasoning["stop"] = []
            # 1.3 设置请求参数
            if active_type[0] == "reasoning" or action_step >= self.max_turns:
                client = reasoning_client   
                model = self.model_reasoning  
                sp = sp_reasoning
            else:
                client = tool_client   
                model = self.model_tool 
                sp = sampling_params.copy()
                sp["stop"] = STOP_TOOL_GEN

            outputs = await self.send_request(
                client,
                active_contexts,
                model,
                sp
            )
            active_responses, finishes, active_finish_reasons, active_type = await self._postprocess_responses(outputs, active_type)
            
            if active_responses[0].endswith("</search>"):
                parse_query = "<search>" + active_responses[0]
                tool_responses = await self.tool_client.get_observations(actions=parse_query)
                observations = await self.post_process_observations(tool_responses["observations"], tool_responses["dones"], tool_responses["valids"], finishes)
                dones = tool_responses["dones"]
            elif active_responses[0].endswith("<search>"):
                observations = [""] * len(active_responses)
                dones = [False] * len(active_responses)
            elif active_responses[0].endswith("</answer>"):
                observations = [""] * len(active_responses)
                dones = [True] * len(active_responses)
            else:
                observations = [""] * len(active_responses)
                dones = [False] * len(active_responses)
                print("Unknown stop reason: ", active_responses[0])

            active_idx = 0
            for i in range(len(contexts)):
                if active_masks[i]:
                    contexts[i] += active_responses[active_idx] + observations[active_idx]
                    final_responses[i] += active_responses[active_idx] + observations[active_idx]
                    finish_reasons[i] = active_finish_reasons[active_idx]
                    active_masks[i] = not dones[active_idx]
                    active_idx += 1
        return final_responses, finish_reasons
    
    async def chat_completions_async(self, id, body: Dict[str, Any]) -> Dict[str, Any]:
        """process API request and generate response"""
        # print(f"Received request: {body}")
        
        if "messages" not in body or not body["messages"]:
            raise ValueError("No messages found in the request.")
        if not 'user' in [message["role"] for message in body["messages"]]:
            raise ValueError("No user message found in the request.")
        
        prompt = self.tokenizer.apply_chat_template(body['messages'],
                                                add_generation_prompt=True,
                                                tokenize=False)
        if body.get('n', 1) > 1:
            prompts = [prompt for _ in range(body["n"])]
        else:
            prompts = [prompt]

        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": 8192,
            "top_p": body.get("top_p", 1.0),
            "stop": list(set(body.get("stop", []))),
        }

        # print(f"Sampling params: {sampling_params}")
        all_responses, finish_reasons = await self.generate_with_tools(prompts, sampling_params)
        
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = 0
        for response in all_responses:
            completion_tokens += len(self.tokenizer.encode(response))
        total_tokens = prompt_tokens + completion_tokens
        
        # format the response into OpenAI-compliant format
        return {
            "id": id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "hybird-"+self.model_tool + "-" + self.model_reasoning,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": all_responses[i],
                    },
                    "finish_reason": finish_reasons[i]
                } for i in range(len(all_responses))
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            } 
        }    
    
    async def close(self):
        """Close any resources (like HTTP sessions and processes) when shutting down"""
        # Terminate all VLLM processes
        for process in self.vllm_processes:
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        self.vllm_processes = []
        self.clients = []   
    

async def main(args):
    df = pd.read_parquet("/text2sql/verl-tool/data/searchR1_processed_direct/test.parquet")
    selected = df[df.data_source == args.data_source]
    
    inference = InferenceService(model_tool=args.tool, model_reasoning=args.reasoning)

    sem = asyncio.Semaphore(1024)  # 限制并发数量
    async def run_inference(i, pyload):
        res = {"choices": [], "id": i, "model": "aaa"}
        if pyload.get("n", 1) > 1:
            n = pyload["n"]
            pyload["n"] = 1  # 每次只发一个请求
            for j in range(n):
                async with sem:
                    out = await inference.chat_completions_async(j, pyload)
                    out['choices'][0]['index'] = j
                    res["choices"].append(out['choices'][0])
                    res["model"] = out["model"]
        else:
            async with sem:
                out = await inference.chat_completions_async(i, pyload)
                res = out
        return res

    tasks = [
        asyncio.create_task(run_inference(i, {"n": args.n, "messages": message.tolist()}))
        for i, message in enumerate(selected.prompt.tolist())   
    ]
    results: List[Dict[str, Any]] = [None] * len(tasks)
    pbar = tqdm(total=len(tasks), desc="Inference(whole)", unit="req")
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results[res["id"]] = res
        pbar.update(1)
    pbar.close()

    # 保存结果
    with open(os.path.join(root, args.output_name), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    await inference.close()

if __name__ == "__main__":
    root = "/text2sql/verl-tool/test_outputs/hybird"
    parser = argparse.ArgumentParser("Experiment Runner")
    parser.add_argument("--tool", type=str, default="/text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuery/global_step_120/actor/huggingface")
    parser.add_argument("--reasoning", type=str, default="/text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/global_step_300/actor/huggingface")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--data_source", type=str, default="searchR1_nq")
    parser.add_argument("--output_name", type=str, default="hybird_inference_results.json")
    args = parser.parse_args()
    asyncio.run(main(args))
    