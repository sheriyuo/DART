export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_PORT=12345

MODEL_SINGLE_ABILITY_PATH="/text2sql/verl-tool/base_model/Qwen2.5-7B-AddTags"
MODEL_MULTI_ABILITY_PATH="/text2sql/verl-tool/sft_model/Qwen2.5-7B-AddTags"


swift sft \
    --model "${MODEL_SINGLE_ABILITY_PATH}" \
    --train_type "full" \
    --dataset "sft_data.json" \
    --torch_dtype "bfloat16" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --packing true \
    --split_dataset_ratio 0.001\
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 1 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 6 \
    --response_prefix "" \
    --save_only_model true \
    --output_dir "${MODEL_MULTI_ABILITY_PATH}" \
    --deepspeed "zero3" \
    --use_liger_kernel true \
    --attn_impl "flash_attn" \
    --model_type "qwen2_5"
