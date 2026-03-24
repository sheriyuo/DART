import re
import json
import os, random, argparse
import torch
import functools
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl

import numpy as np
from tqdm import tqdm
from pprint import pprint
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==== 你工程内的依赖（保持不变） ====
from verl_tool.workers.reward_manager.search_r1_qa_em import compute_score_old
from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
)

# =========== 配置 ===========
class Cfg:
    model_path = "/mnt/bn/data-agent/search_r1/base_model/Qwen2.5-7B"
    dataset_path = "/mnt/bn/data-agent/search_r1/data/searchR1_processed_direct/test.parquet"
    pred_ans_path = "test_outputs/searchR1_nq_results_100.json"
    ref_ans_path  = "test_outputs/Qwen2.5-7B/searchR1_nq_results_100.json"  # 这版不使用
    data_source = "searchR1_nq"
    seed = 42
    use_flash_attn2 = True
    gradient_checkpointing = True
    bf16 = True
    cpu_offload = True            # 显存吃紧可置 True（会更慢）
    max_prompt_length = 4096
    max_response_length = 4096
    # dataloader：一次只取一个 group（一个样本），N_candidates 在张量第一维
    num_workers = 0
    grad_clip_max_norm = 1.0     
    grad_clip_norm_type = 2.0    # L2 范数
    grad_clip_enable = False

cfg = Cfg()

# =========== 公共工具 ===========
def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)

def init_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def build_input(cfg):
    """
    读取数据 -> 组装 group -> 过滤掉“一个 group 的所有候选要么全错(全 0)，要么全对(全 1)”的样本。
    注意：这里用 choices[i]['message']['content'] 直接打分做快速过滤；
          后续在真正计算 reward 时你依然会基于 token 序列 decode 再打分。
    """
    import pandas as pd
    import orjson
    from tqdm import tqdm

    data = pd.read_parquet(cfg.dataset_path)
    data = data[data.data_source == cfg.data_source].reset_index()

    with open(cfg.pred_ans_path, "rb") as f:
        row_pred_rollout = orjson.loads(f.read())

    pred_rollout, err_ids = [], []
    for i, it in enumerate(row_pred_rollout):
        if len(it.get("choices", [])) > 10:
            it['choices'] = it['choices'][:10]
        if len(it.get("choices", [])) < 2:
            print(f"{i} has less than 2 choices")
            err_ids.append(i)
            continue
        pred_rollout.append(it)
        
    with open(cfg.ref_ans_path, "rb") as f:
        row_ref_ans = orjson.loads(f.read())
    
    ref_ans = []
    for i, it in enumerate(row_ref_ans):
        if i in err_ids:
            continue
        if len(it.get("choices", [])) > 10:
            it['choices'] = it['choices'][:10]
        
        ref_ans.append(it)

    pred_by_id: Dict[int, Dict] = {it.get("id"): it for it in pred_rollout}
    ref_by_id: Dict[int, Dict] = {it.get("id"): it for it in ref_ans}

    grouped_out = []
    kept, dropped_all_zero, dropped_all_one, missing = 0, 0, 0, 0

    for i, row in tqdm(data.iterrows(), total=data.shape[0], desc="build input (filtering)..."):
        # 解析 rid
        if i in err_ids:
            continue
        extra = row.get("extra_info", None)
        rid = int(extra["index"]) if isinstance(extra, dict) and "index" in extra else int(i)

        # 取该 group 的候选与参考
        choices = pred_by_id[i].get('choices', [])
        if not choices:
            missing += 1
            continue

        gt = row.reward_model['ground_truth']    

        scores, f_id = [], -1
        for j, rsp in enumerate(choices): # 
            try:
                msg = rsp.get('message', {})
                content = msg.get('content', '') 
                s = compute_score_old(content, gt, format_score=0.0)
            except Exception:
                # 一旦异常，当作 0（更保守），也可选择 continue 直接丢弃该 group
                s = 0.0
            scores.append(float(s))
            if s > 0:
                f_id = j

        if len(scores) == 0:
            missing += 1
            continue

        all_zero = all(s == 0.0 for s in scores) # NOTE: 需要和reward保持一致
        all_one  = all(s == 1.0 for s in scores)

        if all_zero:
            dropped_all_zero += 1
            continue
        if all_one:
            dropped_all_one += 1
            continue

        # 通过筛选：至少有一个 0 且至少有一个 1
        pred_group = [rsp['message'] for rsp in choices]
        ref_ = ref_by_id[i]['choices'][0]

        grouped_out.append(
            {
                "rid": i,
                "prompt": row.prompt.tolist(),
                "group_rsp": pred_group,
                "ref": ref_,  # 虽然后续流程不一定用 ref，但保持原结构
                "ground_truth": gt,
                "scores": scores
            }
        )
        kept += 1

    print(
        f"[build_input] kept={kept}  dropped_all_zero={dropped_all_zero}  "
        f"dropped_all_one={dropped_all_one}  missing_or_empty={missing}"
    )

    return grouped_out[:100]


def apply_grad_clip(fsdp_model, max_norm: float, norm_type: float = 2.0) -> float:
    """
    对当前 .grad 做裁剪；返回裁剪前的全局 total_norm（float）。
    - 多卡/FSDP: 用 FSDP.clip_grad_norm_（会做必要的通信）
    - 单卡: 用 torch.nn.utils.clip_grad_norm_
    """
    if max_norm is None or max_norm <= 0:
        return 0.0
    try:
        # PyTorch FSDP 的官方接口（优先）
        total_norm = FSDP.clip_grad_norm_(fsdp_model, max_norm, norm_type=norm_type)
        # 有些版本返回的是张量，转 float
        if isinstance(total_norm, torch.Tensor):
            total_norm = float(total_norm.detach().cpu().item())
        else:
            total_norm = float(total_norm)
    except Exception:
        # 回退到普通实现（单机或非 FSDP）
        total_norm = torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_norm, norm_type=norm_type)
        if isinstance(total_norm, torch.Tensor):
            total_norm = float(total_norm.detach().cpu().item())
    return total_norm


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok

def load_model(model_path: str):
    kw = dict(torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32)
    if cfg.use_flash_attn2:
        kw["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model

# =========== 分片 / 片段定位工具 ===========
def _valid_offsets(offsets: List[Tuple[int,int]]):
    out = []
    for i, (s,e) in enumerate(offsets):
        if s == 0 and e == 0: continue
        out.append((i,s,e))
    return out

def _char_to_start_token(valid_offs, char_pos: int) -> int:
    for i,s,e in valid_offs:
        if e > char_pos: return i
    return valid_offs[-1][0] + 1

def _char_to_end_token(valid_offs, char_pos: int) -> int:
    last = valid_offs[0][0]-1
    for i,s,e in valid_offs:
        if s >= char_pos: break
        last = i
    return last

def find_token_bounds_multi(offsets, full, prefix, suffix, include_tags=True):
    valid = _valid_offsets(offsets)
    spans_char = []
    n = len(full)
    if not prefix and not suffix:
        spans_char = [(0,n)]
    elif not prefix:
        pos=0
        while True:
            s_idx = full.find(suffix,pos)
            if s_idx==-1: spans_char.append((0,n)); break
            spans_char.append((0,s_idx+len(suffix))); pos=s_idx+len(suffix)
    elif not suffix:
        pos=0
        while True:
            p_idx=full.find(prefix,pos)
            if p_idx==-1: break
            spans_char.append((p_idx if include_tags else p_idx+len(prefix), n))
            pos=p_idx+len(prefix)
    else:
        pos=0
        while True:
            p_idx=full.find(prefix,pos)
            if p_idx==-1: break
            p_end=p_idx+len(prefix)
            s_idx=full.find(suffix,p_end)
            if s_idx==-1:
                spans_char.append((p_idx if include_tags else p_end, n)); break
            spans_char.append((p_idx if include_tags else p_end, s_idx+(len(suffix) if include_tags else 0)))
            pos=s_idx+len(suffix)
    if not valid: return [(0,-1)]
    spans_tok=[]
    for L,R in spans_char:
        st=_char_to_start_token(valid, L); ed=_char_to_end_token(valid, R)
        if ed>=st: spans_tok.append((st,ed))
    return spans_tok

# =========== Dataset（只做一个 group => [N_candidates, ...] 张量） ===========
class OneGroupDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, samples, sample_num=5):
        self.samples = samples
        self.tok = tokenizer
        self.sample_num = sample_num

    def _get_response_mask(self, response_ids, response_str, offsets, prefix, suffix):
        mask = torch.zeros_like(response_ids["input_ids"])
        for i, rsp in enumerate(response_str):
            spans_tok = find_token_bounds_multi(offsets[i], rsp, prefix, suffix)
            for st, ed in spans_tok:
                mask[i, st:ed+1] = 1
        return mask
    
    def _sample_response(self, group_rsp, scores):
        if self.sample_num >= len(group_rsp):
            return np.asarray(group_rsp).tolist()
        for _ in range(100):
            sample_idx = np.random.choice(a=len(group_rsp), size=self.sample_num, replace=False)
            true_num = np.where(np.array(scores)[sample_idx] == 1)[0].shape[0]
            if true_num > 0 and true_num < self.sample_num:
                return np.asarray(group_rsp)[sample_idx].tolist()
        # 100 次采样都没成功，强制返回一次
        print("[Warning] _sample_response: forced sampling")
        return np.array(group_rsp)[sample_idx].tolist()
    
    def debug_decode_mask(self, input_ids, mask, pad_token_id=None):
        """
        将 mask==0 的 token 替换成 pad，然后 decode 出被 mask 选中的内容。
        Args:
            tokenizer: 你的 tokenizer
            input_ids: Tensor [N, L]
            mask: Tensor [N, L]，0/1
            pad_token_id: 可选，默认取 tokenizer.pad_token_id
        """
        if pad_token_id is None:
            pad_token_id = self.tok.pad_token_id or self.tok.eos_token_id

        # 确保类型匹配
        masked_ids = input_ids.clone()
        masked_ids[mask == 0] = pad_token_id

        # decode 每一行
        for i, ids in enumerate(masked_ids):
            text = self.tok.decode(ids, skip_special_tokens=True)
            print(f"=== sample {i} ===")
            print(text.strip())
        

    def __getitem__(self, idx):
        s = self.samples[idx]
        rid = int(s.get("rid", s.get("id", 0)))
        messages = s["prompt"]

        group_rsp = self._sample_response(s["group_rsp"], s["scores"])

        prompt_rendered = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tok([prompt_rendered], add_special_tokens=False, return_tensors="pt",
                              max_length=cfg.max_prompt_length, truncation=True, padding="max_length")

        response_str = [
            self.tok.apply_chat_template([{"role":"assistant","content": r["content"]}],
                                         add_generation_prompt=False, tokenize=False)
            for r in group_rsp
        ]
        response_ids = self.tok(response_str, add_special_tokens=False, return_tensors="pt",
                                max_length=cfg.max_response_length, truncation=True, padding="max_length",
                                return_offsets_mapping=True)
        offsets = response_ids["offset_mapping"]

        # masks
        search_mask = self._get_response_mask(response_ids, response_str, offsets, "<search>", "</search>")
        answer_mask = self._get_response_mask(response_ids, response_str, offsets, "<answer>", "</answer>")

        # 拼接 prompt + response
        n = len(response_str)
        input_ids = torch.cat([prompt_ids["input_ids"].repeat(n,1), response_ids["input_ids"]], dim=-1).contiguous()
        attention_mask = torch.cat([prompt_ids["attention_mask"].repeat(n,1), response_ids["attention_mask"]], dim=-1).contiguous()

        return {
            "rid": rid,
            "input_ids": input_ids,                       # [N, Lp+Lr]
            "attention_mask": attention_mask,             # [N, Lp+Lr]
            "response_mask_search": search_mask,          # [N, Lr]
            "response_mask_answer": answer_mask,          # [N, Lr]
            "ground_truth": s["ground_truth"],            # dict
            "rsp_str": response_str
        }

    def __len__(self): return len(self.samples)

def grpo_collate_fn(batch):
    # 每次只拿一个 group（batch size = 1），因此直接返回 batch[0]
    return batch[0]

# =========== LogProbs & Loss ===========

def get_logprobs(logits, input_ids):  # pull everything out of the loop except logsumexp
    token_logits = torch.gather(logits, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
    token_log_probs = token_logits - logsumexp_values
    return token_log_probs

def policy_loss_simple(log_prob, advantages, mask):
    # log_prob, advantages, mask: [n_local, T]
    w = (mask > 0).float()
    num = (advantages.detach() * log_prob * w).sum()
    denom_local = w.sum().to(log_prob.dtype)

    denom_local = torch.clamp(denom_local, min=1.0)
    return -(num / denom_local)

# =========== 候选并行切片 ===========
def shard_group_across_ranks(batch: Dict):
    rank, world = get_dist_info()
    N = batch['input_ids'].size(0)
    idx = torch.arange(N)
    local_idx = idx[rank::world]
    if local_idx.numel() == 0:
        return None
    def _slice(t):
        return t.index_select(0, local_idx.to(t.device)) if isinstance(t, torch.Tensor) else t
    local = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            local[k] = _slice(v)
        else:
            local[k] = v
    local['_global_N'] = N
    local['_local_idx'] = local_idx
    return local

def layerwise_grad_cosine_fsdp(
    fsdp_model,
    saved_search_grads_cpu,
    group_regex: str = r"layers\.(\d+)",
    eps: float = 1e-12,
):
    """
    计算每个参数模块的梯度 cos（不在层内平均）。
    每个参数都有一个唯一 key，例如：
        layer_0.attn.q_proj.weight
        layer_0.mlp.fc1.weight
        embeddings.word_embeddings.weight
        lm_head.weight
        others.xxx
    返回 dict: {module_name: cos_value}
    """
    import re

    out_cos = {}
    for name, p in fsdp_model.named_parameters():
        if p.grad is None:
            continue

        g2 = p.grad.detach().float().view(-1)
        g1 = saved_search_grads_cpu[name].to(g2.device).view(-1)

        # 解析层号
        m = re.search(group_regex, name)
        if m:
            layer_k = f"layer_{int(m.group(1))}"
        elif "lm_head" in name:
            layer_k = "lm_head"
        elif ("embed_tokens" in name) or ("wte" in name):
            layer_k = "embeddings"
        else:
            layer_k = "others"

        # 完整模块 key（含层号 + 参数名）
        key = f"{layer_k}.{name.split(layer_k + '.')[-1]}" if layer_k in name else f"{layer_k}.{name}"

        # 计算 cos
        dot = torch.dot(g1, g2)
        n1 = torch.dot(g1, g1)
        n2 = torch.dot(g2, g2)
        denom = (n1.clamp_min(eps).sqrt() * n2.clamp_min(eps).sqrt())
        cos_val = (dot / denom).item() if denom > 0 else 0.0
        out_cos[key] = cos_val

    # 同步所有 rank，确保结果完整
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, out_cos)
        merged = {}
        for d in gathered:
            if d is None:
                continue
            merged.update(d)
        out_cos = merged

    return out_cos


def _noop_allreduce(times=1, device="cuda"):
    if dist.is_initialized():
        z = torch.zeros(1, device=device)
        for _ in range(times):
            dist.all_reduce(z, op=dist.ReduceOp.SUM)


def step_one_sample_candidate(cfg: Cfg, batch: List[Dict], fsdp_model: FSDP, tokenizer, mask_type):
    local_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    autocast_device = f"cuda:{local_rank}"

    out = {}
    # 只取两个回复
    for i, local in enumerate(batch):
        input_ids      = local['input_ids'].to(device)
        attention_mask = local['attention_mask'].to(device)
        if mask_type == 'search':
            mask = local['response_mask_search'].to(device)
        elif mask_type == 'answer':
            mask = local['response_mask_answer'].to(device)
        else:
            raise ValueError("Erro")
        ground_truths  = local['ground_truth']

        def run_forward(input_ids, attention_mask):
            dtype = torch.bfloat16 if cfg.bf16 else torch.float32
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = fsdp_model(input_ids=input_ids, attention_mask=attention_mask)
                logits   = out.logits[:, cfg.max_prompt_length:, :]
                resp_ids = input_ids[:, cfg.max_prompt_length:]
                logprobs = get_logprobs(logits, resp_ids)

                reward = torch.zeros_like(resp_ids, dtype=dtype)
                scores = torch.zeros(input_ids.size(0), dtype=dtype, device=resp_ids.device)
                for i in range(input_ids.size(0)):
                    valid_len = int(attention_mask[i, cfg.max_prompt_length:].sum().item())
                    if valid_len > 0:
                        pred_str = tokenizer.decode(
                            input_ids[i, cfg.max_prompt_length:], skip_special_tokens=True
                        )
                        target = ground_truths[0]['target'] if isinstance(ground_truths, list) else ground_truths['target']
                        score = float(compute_score_old(pred_str, target, format_score=-1, is_print=False))
                        reward[i, valid_len - 1] = score
                        scores[i] = score
            return logprobs, reward, scores

        # Pass 1: 
        fsdp_model.zero_grad(set_to_none=True)
        logprobs_1, reward_1, scores_1 = run_forward(input_ids, attention_mask)
        index = np.zeros(input_ids.shape[0], dtype=int)
        adv_1, _ = compute_grpo_outcome_advantage(reward_1, mask, index)
        loss_1 = policy_loss_simple(logprobs_1, adv_1, mask)
        out[f"loss_{i+1}"] = float(loss_1.detach().cpu().item())
        if local_rank == 0:
            print(f"adv {i+1}:")
            print(adv_1.sum(-1))
        loss_1.backward()

        total_norm_s = 0.0
        if cfg.grad_clip_enable:
            total_norm_s = apply_grad_clip(
                fsdp_model,
                max_norm=cfg.grad_clip_max_norm,
                norm_type=cfg.grad_clip_norm_type
            )

        if i == 0:
            saved_search_grads = {}
            for n, p in fsdp_model.named_parameters():
                if p.grad is not None:
                    saved_search_grads[n] = p.grad.detach().float().cpu().clone()
        elif i == 1:
            layer_cos = layerwise_grad_cosine_fsdp(fsdp_model, saved_search_grads)
        else:
            print("out range")
            break

    layer_cos = layerwise_grad_cosine_fsdp(fsdp_model, saved_search_grads)
    out["cos"] = layer_cos

    return out


# =========== 单 group 双片段梯度对比（核心） ===========
def step_one_group_candidate_parallel(cfg: Cfg, batch: Dict, fsdp_model: FSDP, tokenizer):
    local = batch
    local_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    autocast_device = f"cuda:{local_rank}"

    has_local = local is not None

    if has_local:
        input_ids      = local['input_ids'].to(device)
        attention_mask = local['attention_mask'].to(device)
        mask_search    = local['response_mask_search'].to(device)
        mask_answer    = local['response_mask_answer'].to(device)
        ground_truths  = local['ground_truth']

        def run_forward(input_ids, attention_mask):
            dtype = torch.bfloat16 if cfg.bf16 else torch.float32
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = fsdp_model(input_ids=input_ids, attention_mask=attention_mask)
                logits   = out.logits[:, cfg.max_prompt_length:, :]
                resp_ids = input_ids[:, cfg.max_prompt_length:]
                logprobs = get_logprobs(logits, resp_ids)

                reward = torch.zeros_like(resp_ids, dtype=dtype)
                scores = torch.zeros(input_ids.size(0), dtype=dtype, device=resp_ids.device)
                for i in range(input_ids.size(0)):
                    valid_len = int(attention_mask[i, cfg.max_prompt_length:].sum().item())
                    if valid_len > 0:
                        pred_str = tokenizer.decode(
                            input_ids[i, cfg.max_prompt_length:], skip_special_tokens=True
                        )
                        target = ground_truths[0]['target'] if isinstance(ground_truths, list) else ground_truths['target']
                        score = float(compute_score_old(pred_str, target, is_print=False))
                        reward[i, valid_len - 1] = score
                        scores[i] = score
            return logprobs, reward, scores

        # Pass 1: search
        fsdp_model.zero_grad(set_to_none=True)
        logprobs_1, reward_1, scores_1 = run_forward(input_ids, attention_mask)
        index = np.zeros(input_ids.shape[0], dtype=int)
        adv_search, _ = compute_grpo_outcome_advantage(reward_1, mask_search, index)
        loss_s = policy_loss_simple(logprobs_1, adv_search, mask_search)
        loss_s.backward()

        total_norm_s = 0.0
        if cfg.grad_clip_enable:
            total_norm_s = apply_grad_clip(
                fsdp_model,
                max_norm=cfg.grad_clip_max_norm,
                norm_type=cfg.grad_clip_norm_type
            )

        saved_search_grads = {}
        for n, p in fsdp_model.named_parameters():
            if p.grad is not None:
                saved_search_grads[n] = p.grad.detach().float().cpu().clone()
        # torch.save(saved_search_grads, f"/text2sql/search_grad_{local_rank}_rid={batch['rid']}.bin")

        # Pass 2: answer
        fsdp_model.zero_grad(set_to_none=True)
        logprobs_2, _, _ = run_forward(input_ids, attention_mask)
        adv_answer, _ = compute_grpo_outcome_advantage(reward_1, mask_answer, index)
        loss_a = policy_loss_simple(logprobs_2, adv_answer, mask_answer)
        loss_a.backward()

        total_norm_a = 0.0
        if cfg.grad_clip_enable:
            total_norm_a = apply_grad_clip(
                fsdp_model,
                max_norm=cfg.grad_clip_max_norm,
                norm_type=cfg.grad_clip_norm_type
            )

        layer_cos = layerwise_grad_cosine_fsdp(fsdp_model, saved_search_grads)

        # saved_answer_grads = []
        # for p in fsdp_model.parameters():
        #     if p.grad is not None:
        #         saved_answer_grads.append(p.grad.detach().float().cpu().clone())
        # torch.save(saved_answer_grads, f"/text2sql/answer_grad_{local_rank}_rid={batch['rid']}.bin")

        local_correct = scores_1.sum()
        local_total   = torch.tensor(float(scores_1.numel()), device=scores_1.device)
    else:
        _noop_allreduce(times=2, device=device.type)
        layer_cos = layerwise_grad_cosine_fsdp(fsdp_model, saved_search_grads_cpu=[])
        return {
            "loss_s": 0.0, "loss_a": 0.0,
            "cos": layer_cos, "rollout_acc": 0.0,
            "cos_search": None, "cos_answer": None
        }

    # ---- 跨卡聚合正确率 ----
    if dist.is_initialized():
        correct_global = local_correct.clone()
        total_global   = local_total.clone()
        dist.all_reduce(correct_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_global,   op=dist.ReduceOp.SUM)
    else:
        correct_global = local_correct
        total_global   = local_total

    rollout_acc = (correct_global / torch.clamp(total_global, min=1.0)).item()

    return {
        "loss_s": float(loss_s.detach().cpu().item()),
        "loss_a": float(loss_a.detach().cpu().item()),
        "cos": layer_cos,
        "rollout_acc": rollout_acc,
    }



# =========== FSDP 初始化 ===========
def build_fsdp_model(model: torch.nn.Module) -> FSDP:
    # 1) auto wrap policy（优先 transformer_auto_wrap_policy）
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    wrap_policy = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # NOTE：Qwen2 的 DecoderLayer 类（尽量匹配）
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen2DecoderLayer},
            )
        except Exception:
            pass
    if wrap_policy is None:
        # 回退：按参数量切块（例如 >= 20M 才 wrap）
        wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=20_000_000
        )

    # 2) mixed precision
    mp = MixedPrecision(
        param_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        reduce_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        buffer_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )
    # 3) cpu offload（可选）
    cpu_off = CPUOffload(offload_params=cfg.cpu_offload)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 5) FSDP 封装
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        cpu_offload=cpu_off,
        use_orig_params=True,   # 便于按原始参数维度读取 .grad
        limit_all_gathers=True,
        sync_module_states=True,
    )

    for n, p in fsdp_model.named_parameters():
        if p.device.type != "cuda":
            print(f"[rank{rank}] ⚠️ {n} still on {p.device}")
    return fsdp_model


# =========== 主流程 ===========
def main():
    task = "group"
    set_seed(cfg.seed)
    local_rank, world = init_dist()
    rank = dist.get_rank()

    tokenizer = load_tokenizer(cfg.model_path)
    base_model = load_model(cfg.model_path)
    fsdp_model = build_fsdp_model(base_model).train(True)

    # —— 准备数据（每个迭代仅一个 group；你也可以写一个简单 DataLoader 一次吐 1 个样本）
    samples = build_input(cfg)
    # if task == "group":
    #     samples_num = 10
    # else:
    #     samples_num = 5
    dataset = OneGroupDataset(tokenizer, samples, sample_num=5)
    out = []
    # 这里直接 for idx in range(len(dataset))，每次拿一个 group
    if task == "group":
        for idx in tqdm(range(len(dataset))):
            batch = dataset[idx]  # dict，内部张量 shape: [N_candidates, ...]

            stats = step_one_group_candidate_parallel(cfg, batch, fsdp_model, tokenizer)
            if rank == 0:
                print(f"[group {idx}] search_loss={stats['loss_s']:.4f}  answer_loss={stats['loss_a']:.4f}")
                pprint(stats['cos'])
                out.append({
                    'rid': batch["rid"],
                    'search_loss': stats['loss_s'],
                    'answer_loss': stats['loss_a'],
                    'acc': stats['rollout_acc'],
                    **stats['cos']
                })
    else:
        from copy import deepcopy
        for idx in tqdm(range(len(dataset))):
            # case1 = dataset[4]
            # case2 = deepcopy(case1)
            # batch = [case1, case2]
            batch = [dataset[idx], dataset[idx]]
      
            stats = step_one_sample_candidate(cfg, batch, fsdp_model, tokenizer, "search")
            if rank == 0:
                print(f"[group {idx}] loss_1={stats['loss_1']:.4f}  loss_2={stats['loss_2']:.4f}")
                # pprint(stats['cos'])
                ans = list(stats['cos'].values())
                print(sum(ans)/len(ans))
                out.append({
                    'rid': batch[0]["rid"],
                    'loss_1': stats['loss_1'],
                    'loss_2': stats['loss_2'],
                    'rsp1': batch[0]["rsp_str"],
                    "rsp2": batch[1]["rsp_str"],
                    **stats['cos']
                })
                if sum(ans)/len(ans) < 0.5:
                    with open("grad_case.jsonl", "a") as f:
                        json_str = json.dumps(out[-1])
                        f.write(json_str + "\n")

    if rank == 0:
        with open(f"grad_cos_{task}_search.json", "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()


