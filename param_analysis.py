import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import math
import re

def extract_layer_key(full_key, granularity="block"):
    """
    提取层名粒度。
    granularity:
      - "block": 按 transformer block（如 model.layers.3）
      - "module": 更细粒度（如 model.layers.3.self_attn 或 mlp）
      - "param": 最细粒度（每个 weight/bias 分开）
    """
    parts = full_key.split('.')
    if granularity == "block":
        # 捕捉 model.layers.N 作为层标识
        m = re.match(r"(model\.layers\.\d+)", full_key)
        return m.group(1) if m else parts[0]
    elif granularity == "module":
        m = re.match(r"(model\.layers\.\d+\.[^.]+)", full_key)
        return m.group(1) if m else ".".join(parts[:2])
    elif granularity == "param":
        return full_key
    else:
        raise ValueError("granularity must be block / module / param")


def stable_cosine_angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12):
    # a, b 1D
    a64 = a.detach().double()
    b64 = b.detach().double()

    na = torch.linalg.norm(a64)
    nb = torch.linalg.norm(b64)

    # 处理极端：全零或近零范数
    if na < eps and nb < eps:
        cosv = 1.0  # 两个全零向量：视为同向
    elif na < eps or nb < eps:
        cosv = 0.0  # 只有一个近零：无定义，按 0 处理
    else:
        cosv = torch.dot(a64, b64) / (na * nb)
        cosv = torch.clamp(cosv, -1.0, 1.0).item()

    angle_deg = math.degrees(math.acos(max(min(cosv, 1.0), -1.0)))
    return cosv, angle_deg


def layerwise_cosine_and_normdiff(
    model_path_a, model_path_b, device="cpu", granularity="block"
):
    print(f"Loading model A from {model_path_a}")
    model_a = AutoModelForCausalLM.from_pretrained(
        model_path_a, torch_dtype=torch.float32, device_map=device
    )
    print(f"Loading model B from {model_path_b}")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_path_b, torch_dtype=torch.float32, device_map=device
    )

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    layer_stats = {}

    print(f"Computing cosine and Δnorm ({granularity}-level)...")
    for k in tqdm(sd_a.keys()):
        if k not in sd_b or sd_a[k].shape != sd_b[k].shape:
            continue

        a = sd_a[k].flatten().float().cpu()
        b = sd_b[k].flatten().float().cpu()

        # cosine similarity
        cos_val, angle_deg = stable_cosine_angle(a, b)

        # Δnorm ratio
        delta_norm = torch.norm(b - a)
        base_norm = torch.norm(a)
        rel_norm = (delta_norm / (base_norm + 1e-8)).item()

        layer_name = extract_layer_key(k, granularity)
        if layer_name not in layer_stats:
            layer_stats[layer_name] = []
        layer_stats[layer_name].append((cos_val, angle_deg, rel_norm))

    # 聚合每层平均
    result = {}
    for layer, vals in layer_stats.items():
        cos_mean = sum(v[0] for v in vals) / len(vals)
        angle_mean = sum(v[1] for v in vals) / len(vals)
        rel_mean = sum(v[2] for v in vals) / len(vals)
        result[layer] = dict(cos=cos_mean, angle=angle_mean, rel_norm=rel_mean)

    return result


if __name__ == "__main__":
    model_path_a = "/text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyAns/global_step_300/actor/huggingface"
    model_path_b = "/text2sql/verl-tool/checkpoints/search_r1_qa_em/search_r1_qa_em-fsdp-agent-base_model_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr2e-6onlyQuery/global_step_300/actor/huggingface"
    # result = layerwise_cosine_and_normdiff(model_path_a, model_path_b, device="cuda:5")

    print("\n=== Layerwise cosine similarity ===")
    granularity = "module"
    device = "cuda:5"

    result = layerwise_cosine_and_normdiff(
        model_path_a, model_path_b, device=device, granularity=granularity
    )

    print(f"\n{'Layer':45s} | {'cos':>8s} | {'angle(°)':>8s} | {'Δnorm ratio':>12s}")
    print("-" * 80)
    for k, v in result.items():
        print(f"{k:45s} | {v['cos']:8.4f} | {v['angle']:8.2f} | {v['rel_norm']:12.4e}")

    mean_cos = sum(v["cos"] for v in result.values()) / len(result)
    mean_rel = sum(v["rel_norm"] for v in result.values()) / len(result)
    print("-" * 80)
    print(f"Average cosine: {mean_cos:.6f} | Average Δnorm ratio: {mean_rel:.6e}")
