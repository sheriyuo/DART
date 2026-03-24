import sys
import json
import random

import os.path as osp
import pandas as pd

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from verl_tool.workers.reward_manager.search_r1_qa_em import compute_score

# 定义文件路径
json_file_path = "/text2sql/verl-tool/test_outputs/split/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuerysearch_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6debug/searchR1_nq_results_1.json"
parquet_file_path = "/text2sql/verl-tool/data/searchR1_processed_direct/test.parquet"

# 1. 读取 JSON 文件
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# 2. 读取 Parquet 数据
parquet_data = pd.read_parquet(parquet_file_path)

scores = []
for index, row in parquet_data.iterrows():
    if index >= len(json_data):
        print(f"Index {index} exceeds JSON data length {len(json_data)}")
        break
    choice_id = random.randint(0, len(json_data[index]["choices"]) - 1)
    pred = json_data[index]["choices"][choice_id]["text"]
    # pred = json_data[index]["choices"][choice_id]["message"]["content"]
    gt_answer = row.reward_model['ground_truth']  

    score = compute_score(pred, gt_answer, is_print=False)
    print(f"Index: {index}, Prediction: {pred}, Ground Truth: {gt_answer}, Score: {score}")
    scores.append({"source":row.data_source, "score": score})

scores_df = pd.DataFrame(scores)
# 按 source 分组并计算平均分数
grouped_scores = scores_df.groupby('source')['score'].mean().reset_index()
print(grouped_scores)