import re
import torch
import json
import glob, os


import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import Dict, Any
from pprint import pprint
from safetensors.torch import load_file
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_base():
    BASE = "/text2sql/verl-tool/base_model/Qwen2.5-3B"  # 或你自己的 checkpoint

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    new_tokens = ["<search>", "</search>", "<answer>", "</answer>", "<information>", "</information>"]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"成功添加 {num_added} 个普通token")
    print(f"添加后词表大小: {len(tokenizer)}")
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to new vocab size: {len(tokenizer)}")
    model.save_pretrained("/text2sql/verl-tool/base_model/qwen2.5-3B-newtokenizer")
    tokenizer.save_pretrained("/text2sql/verl-tool/base_model/qwen2.5-3B-newtokenizer")


def check_trust_code():
    ckpt_path = "/text2sql/verl-tool/sft_model/Qwen2.5-3B-Instruct-AddTags/v5-20251209-214929/checkpoint-33-merged"
    df = pd.read_parquet("/text2sql/verl-tool/data/searchR1_processed_direct/test.parquet")
    
    # 1. 加载自定义模型：开启 trust_remote_code + flash_attention_2 + bfloat16 + cuda
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,           # 用你本地的 modeling_qwen2.py
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # 用 flash attn v2
        device_map="cuda",                # 直接让 HF 帮你放到 GPU 上
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    print("config.lora_r =", getattr(model.config, "lora_r", None))
    print("attn_implementation =", getattr(model.config, "_attn_implementation", None))

    model.eval()

    prompts = []
    for i in range(4):
        prompt = tokenizer.apply_chat_template(
            df.prompt[i].tolist(),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
   
    # 2. 把输入也搬到同一块 GPU 上
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)

    # 3. 调用 generate（注意要解包 **inputs）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=1.0,
        )
    # model.save_pretrained("/text2sql/verl-tool/base_model/qwen2.5-3B-instruct-lora-custom", safe_serialization=True)
    # tokenizer.save_pretrained("/text2sql/verl-tool/base_model/qwen2.5-3B-instruct-lora-custom")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def test_vllm():
    ckpt_path = "/text2sql/verl-tool/sft_model/Qwen2.5-3B-Instruct-AddTags/v2-20251209-200114/checkpoint-33"
    
    df = pd.read_parquet("/text2sql/verl-tool/data/searchR1_processed_direct/test.parquet")
    # df = json.load(open("sft_data.json"))

    # 最简单的 prompt（你 transformers 测试通过的）
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    llm = LLM(
        model=ckpt_path,
        trust_remote_code=True,
        model_impl="transformers",
        enforce_eager=True, 
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90
    )

    prompts = []
    for i in range(4):
        prompt = tokenizer.apply_chat_template(
            df.prompt[i].tolist(),
            tokenize=False,
            add_generation_prompt=True,
        )
        # prompt = tokenizer.apply_chat_template(
        #     df[i]["messages"][:2],
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        prompts.append(prompt)


    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=1.0,
        top_p=0.9,
    )

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)


def test_rollout():
    ckpt_path = "/text2sql/verl-tool/2lora_ckpts/search_r1_qa_em/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_base_model_qwen2.5-3b-instruct-2lora-grpo-n16-b512-64-t1.0-lr2e-5-LoRA_R8_fixprefill/global_step_80/actor/huggingface"
    all_paths = glob("verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_base_model_qwen2.5-3b-instruct-2lora-grpo-n16-b512-64-t1.0-lr2e-5-LoRA_R8_fixprefill/search_r1_qa_em-step-*.json")
    test_data_paths = [path for path in all_paths if "step-val-" not in path]
    
    def sort_by_step_number(file_path):
        """
        提取文件名中的step数字，作为排序key
        :param file_path: 文件路径
        :return: step对应的整数（用于自然排序）
        """
        # 分割字符串提取step后的数字（适配"step-xxx.json"格式）
        step_part = file_path.split("step-")[-1].split(".json")[0]
        return int(step_part)  # 转为整数实现自然排序
    
    test_data_paths.sort(key=sort_by_step_number)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    device = torch.device("cpu")
    
    def get_lora1_mask(input_ids: torch.Tensor, prefix: str, suffix: str, shift_len: int = 0):
        start_token, end_token = [], []
        last_text_len = 0
        
        for i in range(1, len(input_ids) + 1):
            decoded_cur = tokenizer.decode(input_ids[:i], skip_special_tokens=False)
            cur_len = len(decoded_cur)
            
            p_idx = decoded_cur.rfind(prefix)
            if p_idx != -1:
                p_end = p_idx + len(prefix)
                if last_text_len < p_end <= cur_len:
                    start_token.append(i + shift_len) 

            s_idx = decoded_cur.rfind(suffix)
            if s_idx != -1:
                s_end = s_idx + len(suffix)
                if last_text_len < s_end <= cur_len:
                    print(decoded_cur)
                    end_token.append(i + shift_len)

            # 更新长度，供下一次循环比对
            last_text_len = cur_len
        return start_token, end_token

    # def get_lora1_mask(input_ids: torch.Tensor, prefix: str, suffix: str, shift_len: int = 0):
    #     start_token, end_token = [], []
    #     for i in range(1, len(input_ids)):
    #         decoded_cur = tokenizer.decode(input_ids[:i])
    #         decoded_cur = decoded_cur.rstrip()
    #         if decoded_cur.endswith(prefix):
    #             start_token.append(i + shift_len)
    #         if decoded_cur.endswith(suffix):
    #             end_token.append(i + shift_len)
    #     return start_token, end_token
    
    out = []
    for i in tqdm(range(0, 100, 5)):
        path = test_data_paths[i]
        step_num = path.split('step-')[-1].split('.json')[0]
        print(f"Processing step {step_num}")
        test_data = json.load(open(path, "r"))
        erro_nums, erro_pos = 0, 0
        erro_ids = []
        for idx, record in enumerate(test_data):
            prompt = record["prompt"]
            prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

            response = record["response"]
            response_inputs = tokenizer(response, return_tensors="pt").to(device)

            # concatenate
            input_ids = torch.cat([prompt_inputs["input_ids"], response_inputs["input_ids"]], dim=1)
            attention_mask = torch.cat([prompt_inputs["attention_mask"], response_inputs["attention_mask"]], dim=1)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
            shift_len = prompt_inputs["input_ids"].shape[1]
            start_search, end_search = get_lora1_mask(input_ids=response_inputs["input_ids"][0], prefix="<search>", suffix="</search>", shift_len=shift_len)
            # start_info, end_info = get_lora1_mask(input_ids=response_inputs["input_ids"][0], prefix="<information>", suffix="</information>", shift_len=shift_len)

            if len(start_search) != len(end_search):
                erro_nums += 1
                if record['score'] > 0:
                    erro_pos += 1
                erro_ids.append(idx)
       
        print(f"Step {step_num} Total records with mismatched tags: {erro_nums}/{len(test_data)}")
        print(f"Step {step_num} Positive records with mismatched tags: {erro_pos}, rate: {erro_pos/erro_nums}")
        out.append({
            "step": int(step_num),
            "total_erro_nums": erro_nums,
            "total_records": len(test_data),
            "positive_erro_nums": erro_pos,
            "erro_ids": erro_ids,
        })
    with open("search_r1_qa_em_lora1_tag_erro_stats_newrouter.json", "w") as f:
        json.dump(out, f, indent=4)


def analyze_logits(
    logits: torch.Tensor,
    topk: int = 5,
    confidence_thresholds = (0.5, 0.7, 0.9),
) -> Dict[str, Any]:
    """
    分析 LLM 的 logits 输出，返回若干统计量。
    
    参数
    ----
    logits : torch.Tensor
        形状为 (batch, seq, vocab) 或 (seq, vocab) 的 logits。
    topk : int
        返回 top-k 概率和 index，用于进一步分析。
    confidence_thresholds : tuple[float]
        用于统计 top-1 概率大于某个阈值的比例。
    
    返回
    ----
    stats : Dict[str, Any]
        一个包含多种统计量的字典，方便后续打印或记录到日志中。
    """

    if logits.dim() == 2:
        # (seq, vocab) -> (1, seq, vocab)
        logits = logits.unsqueeze(0)
    if logits.dim() != 3:
        raise ValueError(f"Expected logits dim 2 or 3, got shape {tuple(logits.shape)}")

    bsz, seqlen, vocab_size = logits.shape
    logits_f = logits.float()

    # 概率分布
    probs = torch.softmax(logits_f, dim=-1)

    # ========= 基本 logits 统计 =========
    logits_mean = logits_f.mean().item()
    logits_std = logits_f.std(unbiased=False).item()
    logits_min = logits_f.min().item()
    logits_max = logits_f.max().item()

    # ========= 熵（不确定性） =========
    # 熵越大，分布越平；越小，越确信
    # clamp_min 防止 log(0)
    log_probs = torch.log(probs.clamp_min(1e-9))
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, S)

    entropy_mean = entropy.mean().item()
    entropy_std = entropy.std(unbiased=False).item()
    entropy_min = entropy.min().item()
    entropy_max = entropy.max().item()

    # ========= top-k / top-1 分析 =========
    topk_probs, topk_indices = probs.topk(min(topk, vocab_size), dim=-1)  # (B,S,K)
    top1_probs = topk_probs[..., 0]                                       # (B,S)

    # top-1 与 top-2 的 margin（若 vocab_size==1，margin 设为 1）
    if vocab_size > 1:
        top2_probs = topk_probs[..., 1]
        margin = (top1_probs - top2_probs)  # (B,S)
    else:
        margin = torch.ones_like(top1_probs)

    top1_mean = top1_probs.mean().item()
    top1_std = top1_probs.std(unbiased=False).item()
    top1_min = top1_probs.min().item()
    top1_max = top1_probs.max().item()

    margin_mean = margin.mean().item()
    margin_std = margin.std(unbiased=False).item()
    margin_min = margin.min().item()
    margin_max = margin.max().item()

    # ========= 置信度阈值比例 =========
    confidence_stats = {}
    for thr in confidence_thresholds:
        frac = (top1_probs > thr).float().mean().item()
        confidence_stats[f"p>={thr}"] = frac

    # ========= 汇总结果 =========
    stats: Dict[str, Any] = {
        "shape": {
            "batch_size": bsz,
            "seq_len": seqlen,
            "vocab_size": vocab_size,
        },
        "logits": {
            "mean": logits_mean,
            "std": logits_std,
            "min": logits_min,
            "max": logits_max,
        },
        "entropy": {
            "mean": entropy_mean,
            "std": entropy_std,
            "min": entropy_min,
            "max": entropy_max,
        },
        "top1_prob": {
            "mean": top1_mean,
            "std": top1_std,
            "min": top1_min,
            "max": top1_max,
        },
        "top1_margin": {
            "mean": margin_mean,
            "std": margin_std,
            "min": margin_min,
            "max": margin_max,
        },
        "confidence": confidence_stats,
    }

    return stats


def test_rollout_entropy():
    test_data = json.load(open("verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_base_model_qwen2.5-3b-instruct-2lora-grpo-n16-b512-64-t1.0-lr2e-5-LoRA_R8_fixprefill/search_r1_qa_em-step-80.json", "r"))
    ckpt_path = "/text2sql/verl-tool/2lora_ckpts/search_r1_qa_em/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_base_model_qwen2.5-3b-instruct-2lora-grpo-n16-b512-64-t1.0-lr2e-5-LoRA_R8_fixprefill/global_step_80/actor/huggingface"
    # 1. 加载自定义模型：开启 trust_remote_code + flash_attention_2 + bfloat16 + cuda
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,           # 用你本地的 modeling_qwen2.py
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # 用 flash attn v2
        device_map="cuda",                # 直接让 HF 帮你放到 GPU 上
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    device = torch.device("cuda")

    def get_lora1_mask(input_ids: torch.Tensor, prefix: str, suffix: str, shift_len: int = 0):
        start_token, end_token = [], []
        last_text_len = 0
        
        for i in range(1, len(input_ids) + 1):
            decoded_cur = tokenizer.decode(input_ids[:i], skip_special_tokens=False)
            cur_len = len(decoded_cur)
            
            p_idx = decoded_cur.rfind(prefix)
            if p_idx != -1:
                p_end = p_idx + len(prefix)
                if last_text_len < p_end <= cur_len:
                    start_token.append(i + shift_len) 

            s_idx = decoded_cur.rfind(suffix)
            if s_idx != -1:
                s_end = s_idx + len(suffix)
                if last_text_len < s_end <= cur_len:
                    print(decoded_cur)
                    end_token.append(i + shift_len)

            # 更新长度，供下一次循环比对
            last_text_len = cur_len
        return start_token, end_token
    
    def to_jsonable_stats(stats_dict):
        stats = dict(stats_dict)  # 浅拷贝
        # 去掉包含 tensor 的部分
        stats.pop("topk", None)
        return stats  

    model.eval()
    results = []
    for idx, record in enumerate(tqdm(test_data[:200])):
        prompt = record["prompt"]
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        response = record["response"]
        response_inputs = tokenizer(response, return_tensors="pt").to(device)

        # concatenate
        input_ids = torch.cat([prompt_inputs["input_ids"], response_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_inputs["attention_mask"], response_inputs["attention_mask"]], dim=1)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        shift_len = prompt_inputs["input_ids"].shape[1]
        start_search, end_search = get_lora1_mask(input_ids=response_inputs["input_ids"][0], prefix="<search>", suffix="</search>", shift_len=shift_len)
        start_info, end_info = get_lora1_mask(input_ids=response_inputs["input_ids"][0], prefix="<information>", suffix="</information>", shift_len=shift_len)

        if len(start_search) != len(end_search):
            print("Mismatched <search> tags!")
            print(response)
            print("=="*25)
            continue
        if end_info == [] or start_info == []:
            print(response)
            continue

        _mask = torch.zeros_like(inputs["input_ids"], device=inputs["input_ids"].device, dtype=torch.bool)
        lora1_mask = _mask.clone()

        for s, e in zip(start_search, end_search):
            lora1_mask[:, s:e] = 1

        lora1_mask_wo_info = lora1_mask.clone()
        for s, e in zip(start_info, end_info):
            lora1_mask[:, s:e] = 1

        with torch.no_grad():
            outputs = model(**inputs, lora1_mask=lora1_mask)
            logits_A = outputs.logits[0, end_info[-1]:, :]
            info_logits_A = analyze_logits(logits_A)

            outputs_wo_info = model(**inputs, lora1_mask=lora1_mask_wo_info)
            logits_B = outputs_wo_info.logits[0, end_info[-1]:, :]
            info_logits_B = analyze_logits(logits_B)    
            print("Info process with lora1:")
            pprint(info_logits_A)
            print("Info process with lora2:")
            pprint(info_logits_B)
        
        stats_A_json = to_jsonable_stats(info_logits_A)
        stats_B_json = to_jsonable_stats(info_logits_B)
        sample_result = {
            "idx": idx,
            "record_id": record.get("id", idx),
            "prompt": prompt,
            "response": response,
            "search_spans": list(zip(start_search, end_search)),  # [(s,e), ...]
            "info_spans": list(zip(start_info, end_info)),
            "start_pos_after_info": end_info[-1],
            "stats_with_info": stats_A_json,        # A：information 走 lora1
            "stats_without_info": stats_B_json,     # B：information 不走 lora1
        }

        results.append(sample_result)
    
    with open("logits_analysis_results.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # build_base()
    # test_vllm()
    check_trust_code()
    # test_rollout()
    # test_rollout_entropy()