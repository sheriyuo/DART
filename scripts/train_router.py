import json
from typing import List, Tuple
from transformers import AutoTokenizer


def find_token_bounds_in_full(
    # self,
    tokenizer,
    input_ids: List[int],
    prefix: str,
    suffix: str
) -> Tuple[int, int]:
    start_token, end_token = [], []
    for i in range(1, len(input_ids)):
        decoded_cur = tokenizer.decode(input_ids[:i])
        decoded_cur = decoded_cur.rstrip()
        if decoded_cur.endswith(prefix):
            start_token.append(i)
        if decoded_cur.endswith(suffix):
            end_token.append(i)
    return start_token, end_token


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/text2sql/verl-tool/base_model/qwen2.5-3B-instruct-2lora")

    input_data = json.load(
        open("verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_base_model_qwen2.5-3b-instruct-2lora-grpo-n16-b512-64-t1.0-lr2e-5LoRA_R8/search_r1_qa_em-step-80.json", 
            "r", encoding="utf-8"))
    
    prefix, suffix = "<search>", "</search>"
    for item in input_data:
        text = item["response"]

        tokenizer_output = tokenizer(
            text,
            return_offsets_mapping=True,
        )
        offsets = tokenizer_output["offset_mapping"]
        start_tokens, end_tokens = find_token_bounds_in_full(
            tokenizer,
            tokenizer_output["input_ids"],
            prefix,
            suffix,
        )
        mask_tokens = tokenizer_output["input_ids"]
        for start_token, end_token in zip(start_tokens, end_tokens):
            mask_tokens[start_token:end_token] = [0] * (end_token-start_token)
        mask_str = tokenizer.decode(mask_tokens, skip_special_tokens=True)
        
        print("原文：", text)
        print("掩码后：", mask_str)
        print("="*50)