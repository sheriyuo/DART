import re
import json
import random
import torch

from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer


def load_jsonl(path: str) -> List[Dict]:
    """
    加载 JSONL 文件并返回包含所有行的列表
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def qwen25_chatml_to_messages_strip_info(text: str, build_type: str) -> List[Dict[str, str]]:
    """
    将 Qwen2.5 tokenizer.apply_chat_template 之后的字符串还原为 messages，
    并移除 assistant 的 content 中的 <information>...</information> 段落。
    """

    messages: List[Dict[str, str]] = []

    # ===== 1) 解析标准 ChatML 片段 =====
    pattern = r"<\|im_start\|\>([^\n]+)\n(.*?)<\|im_end\|\>"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))

    for m in matches:
        role = m.group(1).strip()
        content = m.group(2).strip()
        messages.append({"role": role, "content": content})

    # ===== 2) 处理最后可能未闭合的 assistant =====
    if matches:
        last_end = matches[-1].end()
        tail = text[last_end:].strip()
    else:
        tail = text.strip()

    if tail.startswith("<|im_start|>"):
        m2 = re.match(r"<\|im_start\|\>([^\n]+)\n?(.*)", tail, flags=re.DOTALL)
        if m2:
            role = m2.group(1).strip()
            content = m2.group(2).strip()
            messages.append({"role": role, "content": content})

    # ===== 3) 移除 assistant 的 <information> ... </information> =====
    def strip_information_blocks(s: str) -> str:
        """移除所有 <information>xxx</information>，支持跨行、多段"""
        replacement = "<information>Some omitted retrieval results</information>"
        return re.sub(r"<information>.*?</information>", replacement, s, flags=re.DOTALL)

    def strip_observation_blocks(s: str) -> str:
        """移除所有 <observation>xxx</observation>，支持跨行、多段"""
        replacement = "<observation>Some omitted observation results</observation>"
        return re.sub(r"<observation>.*?</observation>", replacement, s, flags=re.DOTALL)

    for msg in messages:
        if msg["role"] == "assistant":
            if random.random() < 0.8:
                if build_type == "search":
                    msg["content"] = strip_information_blocks(msg["content"]).strip()
                elif build_type == "sql":
                    msg["content"] = strip_observation_blocks(msg["content"]).strip()
            else:
                msg["content"] = msg["content"].strip()
    return messages


def get_lora1_mask(input_ids: List[int], start_id: int, end_id: int, shift_len: int = 0):
    """
    极速版：直接查找 Token ID
    """
    # 原逻辑是：input_ids[:i] 解码后以 prefix 结尾，记录 i。
    # 这意味着 input_ids[i-1] 这个 token 就是 prefix。
    # 所以我们只要找到 input_ids 中等于 prefix_id 的索引 idx，记录 idx + 1 即可。
    
    start_token = [i + 1 + shift_len for i, t in enumerate(input_ids) if t == start_id]
    end_token = [i + 1 + shift_len for i, t in enumerate(input_ids) if t == end_id]
            
    return start_token, end_token


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/text2sql/verl-tool/base_model/Qwen2.5-Coder-7B-Instruct-AddTags", trust_remote_code=True)
    steps = range(10, 31)
    sft_data = []
    valid_cnt = 0
    build_type = "sql"

    if build_type == "search":
        START_ID = tokenizer.convert_tokens_to_ids("<search>")
        END_ID = tokenizer.convert_tokens_to_ids("</search>")
    elif build_type == "sql":
        START_ID = tokenizer.convert_tokens_to_ids("<sql>")
        END_ID = tokenizer.convert_tokens_to_ids("</sql>")

    for s in tqdm(steps):
        if build_type == "search":
            data = json.load(open(f"verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_base_model_qwen2.5-3b-instruct-2lora-grpo-n16-b512-64-t1.0-lr2e-5-LoRA_R8_fixprefill/search_r1_qa_em-step-{s}.json"))
        elif build_type == "sql":
            data = load_jsonl(f"verl_step_records/sqlcoder-fsdp-_text2sql_verl-tool_base_model_qwen2.5-coder-7b-instruct-grpo-n5-b256-t0.6-lr1e-6/sqlcoder-step-{s}.jsonl")

        for item in data:
            input_ids = tokenizer(item["response"])["input_ids"]
            start_token, end_token = get_lora1_mask(input_ids, START_ID, END_ID)
            
            if len(start_token) == 0 or len(end_token) == 0:
                print("no search!")
                continue
            if len(start_token) != len(end_token):
                print("start_token != end_token")
                continue
            if item["score"] == 0:
                continue
            
            messages = qwen25_chatml_to_messages_strip_info(item["prompt"]+item["response"], build_type)
            sft_data.append({
                "messages": messages,
                "id":valid_cnt
            })
            valid_cnt += 1
    
    json.dump(sft_data, open("sql_sft_data.json", "w"), ensure_ascii=False, indent=2)
