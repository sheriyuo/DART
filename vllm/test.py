import json
import pandas as pd

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_parquet_data(tokenizer):
    data = pd.read_parquet("/text2sql/verl-tool/data/searchR1_processed_direct/sub_test.parquet").prompt.tolist()
    prompts = []
    for prompt in data[:100]:
        prompts.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
    return prompts

def load_json_data():
    data = json.load(open("/root/verl-tool/verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_sft_model_qwen2.5-3b-instruct-addtags_v12-20251210-014932_checkpoint-100-2lora-grpo-n16-b512-64-t1.0-lr2e-5-LoRA_R8_newtokenizer/search_r1_qa_em-step-1.json"))
    prompts = []
    for prompt in data[:100]:
        full_text = prompt["prompt"]+prompt["response"]
        if "</information>" in full_text:
            full_text = full_text.split("</information>")[0] + "</information>"
        else:
            continue
        prompts.append(full_text)
    return prompts


model_path = "/text2sql/verl-tool/sft_model/Qwen2.5-3B-Instruct-AddTags/v12-20251210-014932/checkpoint-100-2lora"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

# prompts = load_parquet_data(tokenizer)
prompts = load_json_data()

llm = LLM(
    model=model_path, 
    model_impl="transformers",
    trust_remote_code=True,
    enforce_eager=True)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")