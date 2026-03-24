CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  inspect_grpo_grad.py


python3 -m vllm.entrypoints.openai.api_server \
  --model /text2sql/verl-tool/base_model/Qwen2.5-0.5B-2lora \
  --trust-remote-code \
  --model-impl transformers \
  --enforce-eager \
  --dtype auto \
  --max-model-len 8192 \
  --api-key dummy

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
        "model": "/text2sql/verl-tool/base_model/Qwen2.5-0.5B-2lora",
        "messages": [
          {"role": "user", "content": "简单介绍一下你自己，并说明你是LoRA微调过的模型。"}
        ],
        "max_tokens": 128
      }'