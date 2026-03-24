#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config
from modeling_qwen2 import Qwen2Attention, Qwen2MLP


# ==========================
#   核心：转换函数
# ==========================

@torch.no_grad()
def add_lora_to_qwen2(
    src_model_name_or_path: str,
    dst_model_path: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    dtype: str = "auto",
):
    # 1. 读原始 config，并注入 LoRA 超参
    config = AutoConfig.from_pretrained(src_model_name_or_path)
    if not isinstance(config, Qwen2Config):
        raise ValueError(f"Config class must be Qwen2Config, got {type(config)}")
    config.lora_r = int(lora_r)
    config.lora_alpha = int(lora_alpha)
    config.lora_dropout = float(lora_dropout)
    config.lora_init_done = False  # 标记为未初始化（你自己的逻辑可用）
    # 在config中添加auto_map
    config.auto_map = {
        "AutoConfig": "configuration_qwen2.Qwen2Config",
        "AutoModel": "modeling_qwen2.Qwen2Model",
        "AutoModelForCausalLM": "modeling_qwen2.Qwen2ForCausalLM",
        "AutoModelForQuestionAnswering": "modeling_qwen2.Qwen2ForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling_qwen2.Qwen2ForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_qwen2.Qwen2ForTokenClassification"
        }
    config.enter_tokens = [151665]
    config.exit_tokens = [151666]

    # 2. 读原始模型（注意：这里默认单卡 CPU/单 GPU 转换）
    model = AutoModelForCausalLM.from_pretrained(
        src_model_name_or_path,
        config=config,
        torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(src_model_name_or_path)
    # 3. 遍历 decoder layers，替换 MLP 和 Attention
    #    Qwen2ForCausalLM: model.model.layers[i].self_attn / .mlp
    layers = model.model.layers

    for layer_idx, layer in enumerate(layers):
        # ---- Attention ----
        old_attn = layer.self_attn
        new_attn = Qwen2Attention(config, layer_idx=layer_idx)

        # 拷贝原始 Q/K/V/O 权重和 bias
        new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
        new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
        new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
        new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

        if old_attn.q_proj.bias is not None:
            new_attn.q_proj.bias.data.copy_(old_attn.q_proj.bias.data)
        if old_attn.k_proj.bias is not None:
            new_attn.k_proj.bias.data.copy_(old_attn.k_proj.bias.data)
        if old_attn.v_proj.bias is not None:
            new_attn.v_proj.bias.data.copy_(old_attn.v_proj.bias.data)
        # o_proj 对应原本就是 bias=False，所以不用管

        # 替换回 layer
        layer.self_attn = new_attn

        # ---- MLP ----
        old_mlp = layer.mlp
        new_mlp = Qwen2MLP(config)

        new_mlp.gate_proj.weight.data.copy_(old_mlp.gate_proj.weight.data)
        new_mlp.up_proj.weight.data.copy_(old_mlp.up_proj.weight.data)
        new_mlp.down_proj.weight.data.copy_(old_mlp.down_proj.weight.data)

        layer.mlp = new_mlp

    # 4. 保存到目标目录
    model.save_pretrained(dst_model_path)
    config.save_pretrained(dst_model_path)
    tokenizer.save_pretrained(dst_model_path)

    print(f"[OK] LoRA-augmented Qwen2 model saved to: {dst_model_path}")
    print(f"     lora_r={lora_r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str,
                        default="/text2sql/verl-tool/base_model/Qwen2.5-0.5B")
    parser.add_argument("--dst", type=str, 
                        default="/text2sql/verl-tool/base_model/Qwen2.5-0.5B-2lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="auto / float16 / bfloat16 / float32 等")
    args = parser.parse_args()

    add_lora_to_qwen2(
        src_model_name_or_path=args.src,
        dst_model_path=args.dst,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
