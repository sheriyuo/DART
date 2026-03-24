import argparse
import os
import signal
import subprocess
import sys
import time
import json
import traceback
from pathlib import Path
import asyncio
import pandas as pd
import re
from functools import lru_cache
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from openai import AsyncOpenAI
from transformers import AutoTokenizer

################################################################################
#                         (1) Your eval logic (inlined)
################################################################################

def build_split_input(message, prefix_choices):
    split_inputs = []
    base = list(message) if isinstance(message, list) else list(message.tolist())
    for choice in prefix_choices:
        msg = {
            "role": choice["message"]["role"],
            "content": re.sub(
                r"<answer>.*?</answer>",
                "",
                choice["message"]["content"],
                flags=re.DOTALL | re.IGNORECASE,
            ),
        }
        split_inputs.append(base + [msg])
    return split_inputs


@lru_cache(maxsize=16)
def _get_tokenizer_cached(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


async def _split_inference_async(
    prefix_info: List[Dict],
    messages: List[List],
    model_name: str,
    base_url: str,
    api_key: str = "sk-proj-1234567890",
    temperature: float = 1.0,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    concurrency: int = 32,
    show_progress: bool = True,
):
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    tokenizer = _get_tokenizer_cached(model_name)

    jobs: List[Tuple[int, int, int, str]] = []
    job_idx = 0

    # build list of (global_job_idx, msg_id, choice_idx, prompt_string)
    for msg_id, message in enumerate(messages):
        if not isinstance(message, list):
            message = message.tolist()
        split_inputs = build_split_input(message, prefix_info[msg_id]["choices"])

        for choice_idx, split_input in enumerate(split_inputs):
            prompt = tokenizer.apply_chat_template(
                split_input,
                tokenize=False,
                add_generation_prompt=False
            )
            suffix = "<|im_end|>\n"
            if prompt.endswith(suffix):
                prompt = prompt[: -len(suffix)]
            # prompt = prompt + "<think>"

            jobs.append((job_idx, msg_id, choice_idx, prompt))
            job_idx += 1

    sem = asyncio.Semaphore(concurrency)

    async def run_single_job(global_job_idx: int, msg_id: int, choice_idx: int, prompt: str):
        MAX_TRY = 2
        erro_info = None
        for attempt in range(MAX_TRY + 1):
            try:
                async with sem:
                    rsp = await client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                rsp_dict = rsp.model_dump()
                one_choice = rsp_dict["choices"][0]
                one_choice["index"] = choice_idx
                return global_job_idx, msg_id, ("ok", one_choice)
            except Exception:
                erro_info = traceback.format_exc()
                if attempt < MAX_TRY:
                    # backoff
                    await asyncio.sleep(1.5 ** (attempt + 1))
                else:
                    return global_job_idx, msg_id, ("err", f"[ERROR] {erro_info}")

    tasks = [
        asyncio.create_task(run_single_job(g_idx, msg_id, choice_idx, prompt))
        for (g_idx, msg_id, choice_idx, prompt) in jobs
    ]

    results_per_msg: Dict[int, List[Any]] = {i: [] for i in range(len(messages))}

    if show_progress:
        pbar = tqdm(total=len(tasks), desc="Inference(split)", unit="req")
    else:
        pbar = None

    for coro in asyncio.as_completed(tasks):
        g_idx, msg_id, (status, payload) = await coro
        results_per_msg[msg_id].append(payload)
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    try:
        await client.close()
    except Exception:
        pass

    final_results: List[Dict[str, Any]] = []
    for msg_id in range(len(messages)):
        final_results.append({
            "id": msg_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": results_per_msg[msg_id],
            "usage": None,
        })
    return final_results


def split_inference(
    prefix_info: List[Dict],
    messages: List[List],
    model_name: str,
    base_url: str,
    api_key: str = "sk-proj-1234567890",
    temperature: float = 1.0,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    concurrency: int = 32,
    show_progress: bool = True,
):
    return asyncio.run(
        _split_inference_async(
            prefix_info=prefix_info,
            messages=messages,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            concurrency=concurrency,
            show_progress=show_progress,
        )
    )


async def _inference_async_worker(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    idx: int,
    message: List,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
):
    MAX_TRY = 2
    if type(message) != list:
        message = message.tolist()
    erro_info = None
    for attempt in range(MAX_TRY+1):
        try:
            async with sem:
                rsp = await client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                )
            rsp_dict = rsp.model_dump()
            rsp_dict["id"] = idx
            return rsp_dict
        except Exception:
            erro_info = traceback.format_exc()
            if attempt < MAX_TRY:
                await asyncio.sleep(1.5 ** (attempt+1))

    # failed all tries
    return {
        "id": idx,
        "object": "chat.completion",
        "model": model_name,
        "choices": [erro_info],
        "usage": None
    }


def inference(
    messages: List[List],
    model_name: str,
    base_url: str,
    api_key: str = "sk-proj-1234567890",
    temperature: float = 1.0,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    n: int = 10,
    concurrency: int = 32,
):
    async def run_all():
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        sem = asyncio.Semaphore(concurrency)

        tasks = [
            asyncio.create_task(
                _inference_async_worker(
                    client,
                    sem,
                    idx,
                    m,
                    model_name,
                    temperature,
                    max_tokens,
                    top_p,
                    n,
                )
            )
            for idx, m in enumerate(messages)
        ]

        results: List[Dict[str, Any]] = [None] * len(tasks)
        pbar = tqdm(total=len(tasks), desc="Inference(whole)", unit="req")
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results[res["id"]] = res
            pbar.update(1)
        pbar.close()

        # close client before returning
        try:
            await client.close()
        except Exception:
            pass
        return results

    return asyncio.run(run_all())


################################################################################
#                      (2) helper: launch / kill model server
################################################################################

def launch_backend(
    model_path: str,
    max_turns: int,
    min_turns: int,
    num_models: int,
    tensor_parallel_size: int,
    tool_server_url: str,
    api_host: str,
    api_port: int,
    enable_mtrl: bool,
    action_stop_tokens: str,
) -> subprocess.Popen:
    """
    启动 eval_service/app.py 作为子进程，返回 Popen 对象。
    注意：我们用 mktemp 的逻辑在 Python 里自己做。
    """
    # 写临时文件保存 action_stop_tokens
    import tempfile
    fd, tmp_path = tempfile.mkstemp(prefix="action_stop_tokens_", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(action_stop_tokens)

    cmd = [
        sys.executable,              # python3
        "eval_service/app.py",
        "--host", api_host,
        "--port", str(api_port),
        "--tool_server_url", tool_server_url,
        "--model", model_path,
        "--max_turns", str(max_turns),
        "--min_turns", str(min_turns),
        "--action_stop_tokens", tmp_path,
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--num_models", str(num_models),
        "--enable_mtrl", str(enable_mtrl),
    ]

    print("[runner] launching backend:")
    print(" ", " ".join(cmd))
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid
    )
    return proc


def wait_until_server_ready(proc: subprocess.Popen, timeout_sec: int = 300):
    """
    监控后端启动日志，直到出现 'vLLM service started successfully with mode'
    或进程退出 / 超时。
    """
    target_str = "vLLM service started successfully with mode"
    start_time = time.time()

    print("[runner] Waiting for backend to start...")

    while True:
        # 如果进程挂了，读取剩余日志并报错
        if proc.poll() is not None:
            leftover = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"❌ Backend process exited early.\n"
                f"---- partial log ----\n{leftover}\n----------------------"
            )

        # 尝试读取一行输出（非阻塞）
        line = proc.stdout.readline()
        if line:
            sys.stdout.write(f"[backend] {line}")
            sys.stdout.flush()

            # ✅ 关键检测点
            if target_str in line:
                print("[runner] ✅ Backend is ready (found startup signal)")
                return

        # 超时检查
        if time.time() - start_time > timeout_sec:
            raise TimeoutError(f"❌ Timeout: backend not ready after {timeout_sec}s")

        time.sleep(0.5)


def kill_backend(proc: subprocess.Popen, grace=5):
    if proc.poll() is not None:
        return
    import os, signal, time
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        for _ in range(grace):
            if proc.poll() is not None:
                return
            time.sleep(1)


################################################################################
#                      (3) main experiment pipeline
################################################################################

def main():
    parser = argparse.ArgumentParser("Experiment Runner")

    # backend server params
    parser.add_argument("--model-path", required=True, help="path to HF model")
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=5000)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--min-turns", type=int, default=0)
    parser.add_argument("--num-models", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-mtrl", action="store_true", default=False)
    parser.add_argument("--tool-server-url",type=str, 
                        default="http://0.0.0.0:30669/get_observation")
    parser.add_argument("--action-stop-tokens", default="</search>,</answer>")

    # eval params
    parser.add_argument("--infer-type", choices=["whole", "split"], required=True)
    parser.add_argument("--n-samples", type=int, default=50,
                        help="only used in whole() for n in chat.completions")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=64)

    parser.add_argument("--data-parquet", default="data/searchR1_processed_direct/test.parquet")
    parser.add_argument("--data-source",default="searchR1_hotpotqa")
    parser.add_argument("--prefix-info-path",
                        help="needed when infer-type == split")
    parser.add_argument("--run_name", default="",
                        help="root dir for saving results")
    parser.add_argument("--output-root", default="test_outputs",
                        help="root dir for saving results")

    args = parser.parse_args()

    # 1. 启动后端
    # backend_proc = launch_backend(
    #     model_path=args.model_path,
    #     max_turns=args.max_turns,
    #     min_turns=args.min_turns,
    #     num_models=args.num_models,
    #     tensor_parallel_size=args.tensor_parallel_size,
    #     tool_server_url=args.tool_server_url,
    #     api_host=args.api_host,
    #     api_port=args.api_port,
    #     enable_mtrl=args.enable_mtrl,
    #     action_stop_tokens=args.action_stop_tokens,
    # )

    data_root = "/text2sql/verl-tool"
    # 2. 等后端 ready
    base_url = f"http://{args.api_host}:{args.api_port}"
    # wait_until_server_ready(backend_proc)

    # 3. 读取数据
    df = pd.read_parquet(os.path.join(data_root, args.data_parquet))
    print(f"[runner] filtering data_source == {args.data_source}")
    if args.data_source != "all":
        selected = df[df.data_source == args.data_source]
    else:
        selected = df
    messages = selected.prompt.tolist()

    # 4. 生成输出目录名
    #   - 和你原来的逻辑保持兼容
    model_name_for_save = None
    try:
        model_name_for_save = args.model_path.split("/")[-4]
    except Exception:
        model_name_for_save = args.model_path.split("/")[-1]

    if args.infer_type == "whole":
        save_root = os.path.join(data_root, args.output_root, "whole", model_name_for_save)
    else:
        # split 模式下你之前是 prefix_info_path.split('/')[-2] + tag
        try:
            tag = args.model_path.split("/")[2]
        except Exception:
            tag = args.model_path.split("/")[-1]

        if args.prefix_info_path is None:
            raise ValueError("split mode needs --prefix-info-path")
        prefix_tag = Path(args.prefix_info_path).parts[-2]
        exp_name = prefix_tag + tag
        save_root = os.path.join(data_root, args.output_root, "split", exp_name)

    os.makedirs(save_root, exist_ok=True)

    # 5. 跑推理
    if args.infer_type == "whole":
        results = inference(
            messages=messages,  # debug only
            model_name=os.path.join(data_root, args.model_path),
            base_url=base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            n=args.n_samples,
            concurrency=args.concurrency,
        )
    else:
        prefix_info = json.load(open(os.path.join(data_root, args.prefix_info_path)))
        results = split_inference(
            prefix_info=prefix_info,
            messages=messages,
            model_name=os.path.join(data_root, args.model_path),
            base_url=base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            concurrency=args.concurrency,
        )

    # 6. 保存结果
    out_file = os.path.join(
        save_root,
        f"{args.data_source}_results_{args.n_samples}_{args.run_name}.json"
    )
    with open(out_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[runner] saved results to {out_file}")



if __name__ == "__main__":
    main()
