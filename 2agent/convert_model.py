import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modeling_2agent import Qwen2ForCausalLM

def convert_qwen2_to_dual_agent(
    model_name_or_path: str,
    save_directory: str,
    device: str = "cpu"  # 转换建议在 CPU 上做，除非显存很大
):
    print(f"Loading original model from: {model_name_or_path}")
    
    # 1. 加载原始模型
    # 使用 standard transformers 加载，这样我们可以获得标准的 state_dict
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,  # 建议使用 fp16 或 bf16 以节省显存
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # 获取原始 state_dict
    original_sd = original_model.state_dict()
    new_sd = {}
    
    print("Start converting parameters...")
    
    # 定义需要复制的层后缀
    # 注意：这些名字必须和你修改后的 modeling 文件中的定义一致
    target_suffixes = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ]
    
    # 统计计数
    copied_count = 0
    total_keys = 0
    
    for key, weight in original_sd.items():
        new_sd[key] = weight
        total_keys += 1
        
        # 检查这个 key 是否属于我们需要复制的层
        # key 的格式通常是 model.layers.0.self_attn.q_proj.weight
        for suffix in target_suffixes:
            # 匹配 .weight 结尾的线性层
            if f"{suffix}.weight" in key:
                # 构造新的 key，例如 model.layers.0.self_attn.q_proj_b.weight
                # 逻辑：把 "q_proj" 替换为 "q_proj_b"
                
                # 简单替换法 (更稳健的方法是 split)
                new_key = key.replace(suffix, f"{suffix}_b")
                
                # 复制权重 (Clone 一份)
                new_sd[new_key] = weight.clone()
                copied_count += 1
                
            # 如果有 bias (虽然 Qwen2 的 MLP/Attention 大部分没有 bias，但为了稳健性加上)
            if f"{suffix}.bias" in key:
                new_key = key.replace(suffix, f"{suffix}_b")
                new_sd[new_key] = weight.clone()
                copied_count += 1

    print(f"Conversion finished.")
    print(f"Original keys: {total_keys}")
    print(f"Duplicated (Model B) keys: {copied_count}")
    print(f"Total new keys: {len(new_sd)}")

    # 2. 修改 Config
    print("Updating config...")
    config = AutoConfig.from_pretrained(model_name_or_path)
        
    # 标记这是一个双模型结构（可选，方便你自己识别）
    config.architectures = ["Qwen2ForCausalLM"] # 保持不变，还是用原来的类加载，但代码已经被你替换了
    config.is_dual_agent = True 

    # 3. 保存
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    print(f"Saving to {save_directory}...")
    
    # 保存模型权重
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)
        model.load_state_dict(new_sd, strict=True, assign=True)
    
    model.save_pretrained(save_directory)
    
    # 保存 config 和 tokenizer
    config.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    print("Done! You can now load this model with your custom modular_qwen2.py code.")

if __name__ == "__main__":
    # 配置区
    SOURCE_MODEL = "/text2sql/verl-tool/sft_model/Qwen2.5-7B-AddTags/v1-20251225-233050/checkpoint-100"  
    TARGET_DIR = "/text2sql/verl-tool/sft_model/2agent-7B-AddTags/v1-dual-agent"
    
    convert_qwen2_to_dual_agent(SOURCE_MODEL, TARGET_DIR)