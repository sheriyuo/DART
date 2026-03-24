CUDA_VISIBLE_DEVICES=0 swift infer \
    --model "/text2sql/verl-tool/sft_model/Qwen2.5-7B-AddTags/v1-20251225-233050/checkpoint-100" \
    --infer_backend pt \
    --stream true \
    --temperature 1 \
    --max_new_tokens 2048