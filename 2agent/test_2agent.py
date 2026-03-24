import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer



def test_model():
    model_path = "/text2sql/verl-tool/sft_model/2agent-3B-Instruct-AddTags/v1-dual-agent"
    
    print("Initializing Custom Qwen2Model...")
    # 这里直接实例化你修改过的类
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # 简单推理测试
    model.eval()
    model.to("cuda")
    
    # 构造一个 dummy input
    input_ids = torch.randint(0, 1000, (1, 10)).to("cuda")
    
    # Case 1: 默认 (应该只用 Model A 或 混合)
    # 根据你之前的代码逻辑，如果不传 agent_gate，默认行为取决于具体实现（你之前的代码如果不传好像默认只有A）
    with torch.no_grad():
        out1 = model(input_ids)
    print("Forward pass (default) done.")

    # Case 2: 强制使用 Agent B
    # 构造全 1 的 gate (即全选 index 1 -> Model B)
    # agent_gate shape: (B, T, 2)
    gate_b = torch.zeros((1, 10, 2), device="cuda", dtype=torch.float16) # 假设模型是 fp16
    gate_b[:, :, 1] = 1.0 
    
    with torch.no_grad():
        out2 = model(input_ids, agent_gate=gate_b)
    print("Forward pass (Agent B) done.")
    
    # 验证：由于刚通过 copy 初始化，Model A 和 Model B 权重一样。
    # 所以 out1 (Model A) 和 out2 (Model B) 的输出应该完全一致！
    diff = (out1.logits - out2.logits).abs().max()
    print(f"Difference between Agent A and Agent B output: {diff.item()}")
    
    if diff < 1e-3:
        print("✅ Consistency check passed (Since initialized from copy).")
    else:
        print("⚠️ Warning: Outputs differ. Check if initialization or forward logic is correct.")

if __name__ == "__main__":
    test_model()