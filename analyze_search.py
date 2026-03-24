import re
import string
import json 

from tqdm import tqdm
from collections import defaultdict
from pprint import pprint

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def calculate_retrieval_acc(results: list[dict]) -> float:
    total = defaultdict(int)
    correct = defaultdict(int)
    for result in tqdm(results):
        total[result['data_source']] += 1
        if is_retrieval_correct(result["response"], result["ground_truth"]['target']):
            correct[result['data_source']] += 1
    # 计算各个数据集的准确率
    acc = {}
    for ds in total:
        acc[ds] = correct[ds] / total[ds]
    return acc

if __name__ == "__main__":
    single_lora_path = "/text2sql/verl-tool/verl_step_records/search_r1_qa_em-fsdp-agent-_mnt_bn_data-agent_search_r1_base_model_checkpoint-100-2lora-grpo-n16-b256-64-t1.0-lr2e-5-Single-LoraW/search_r1_qa_em-step-val-60.json"
    mutil_lora_path = "verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_sft_model_qwen2.5-3b-instruct-addtags_v12-20251210-014932_checkpoint-100-2lora-grpo-n16-b256-64-t1.0-lr2e-5-cosine_lr/search_r1_qa_em-step-val-100.json"
    full_path = "verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_sft_model_qwen2.5-3b-instruct-addtags_v12-20251210-014932_checkpoint-100-grpo-n16-b256-64-t1.0-lr2e-6-full_train/search_r1_qa_em-step-val-100.json"
    
    with open(single_lora_path, "r") as f:
        single_lora_results = json.load(f)
    with open(mutil_lora_path, "r") as f:
        mutil_lora_results = json.load(f)
    with open(full_path, "r") as f:
        full_results = json.load(f)
    print("Single Lora Retrieval Acc:", calculate_retrieval_acc(single_lora_results))
    print("Mutil Lora Retrieval Acc:", calculate_retrieval_acc(mutil_lora_results))
    print("Full Train Retrieval Acc:", calculate_retrieval_acc(full_results))
