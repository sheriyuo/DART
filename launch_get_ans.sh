# python3 get_anwer.py --model-path base_model/Qwen2.5-3B --infer-type whole

# python3 get_anwer.py --model-path checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6debug/global_step_370/actor/huggingface --infer-type whole

# python3 get_anwer.py --model-path checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuery/global_step_120/actor/huggingface --infer-type whole

# python3 get_anwer.py --model-path checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/global_step_300/actor/huggingface --infer-type whole

# python3 get_anwer.py --model-path base_model/Qwen2.5-3B-Instruct --infer-type split --prefix-info-path test_outputs/whole/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuery/searchR1_nq_results_100_test.json --max-turns 0

# python3 get_anwer.py --model-path checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/global_step_300/actor/huggingface --infer-type split --prefix-info-path test_outputs/whole/Qwen2.5-3B/searchR1_nq_results_100_test.json --max-turns 0

python3 hybird_eval/hybird_inference.py --tool /text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuery/global_step_120/actor/huggingface --reasoning /text2sql/verl-tool/base_model/Qwen2.5-3B-Instruct --data_source searchR1_nq --output_name searchR1_nq_tools_results.json

python3 hybird_eval/hybird_inference.py --reasoning /text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/global_step_300/actor/huggingface --tool /text2sql/verl-tool/base_model/Qwen2.5-3B-Instruct --data_source searchR1_nq --output_name searchR1_nq_reasoning_results.json

python3 hybird_eval/hybird_inference.py --tool /text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuery/global_step_120/actor/huggingface --reasoning /text2sql/verl-tool/base_model/Qwen2.5-3B-Instruct --data_source searchR1_hotpotqa --output_name searchR1_hotpotqa_tools_results.json

python3 hybird_eval/hybird_inference.py --reasoning /text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/global_step_300/actor/huggingface --tool /text2sql/verl-tool/base_model/Qwen2.5-3B-Instruct --data_source searchR1_hotpotqa --output_name searchR1_hotpotqa_reasoning_results.json

python3 hybird_eval/hybird_inference.py --data_source searchR1_nq --output_name searchR1_nq_hybird_results.json

python3 hybird_eval/hybird_inference.py --data_source searchR1_hotpotqa --output_name searchR1_hotpotqa_hybird_results.json