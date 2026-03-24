import torch
import json
from typing import List, Optional
from transformers import AutoTokenizer

# ==========================================
# 1. Router 类 (不需要改，保持你的代码)
# ==========================================
class LoRARouter:
    """
    适配 vLLM Eager Mode 的路由器
    实现与训练代码一致的区间检测逻辑 (Interval Detection)，
    解决 Token 粘连 (Merge) 导致的匹配失败问题。
    """
    def __init__(self, tokenizer, enter_strings, exit_strings, initial_mode: int = 2, tail_tokens: int = 128):
        self.tok = tokenizer
        # 预处理：确保是 tuple，方便遍历
        self.enter = tuple(enter_strings) if isinstance(enter_strings, (list, tuple)) else (enter_strings,)
        self.exit = tuple(exit_strings) if isinstance(exit_strings, (list, tuple)) else (exit_strings,)
        
        # 移除空字符串，防止死循环或误判
        self.enter = tuple(s for s in self.enter if s)
        self.exit = tuple(s for s in self.exit if s)

        self.initial_mode = int(initial_mode)
        self.tail_tokens = int(tail_tokens)

        # 状态容器：List[List[int]]
        # index 对应当前 Batch 的 row index
        self.buffers = [] 
        self.mode: Optional[torch.Tensor] = None 

    def _ensure_batch_size(self, batch_size: int, device: torch.device):
        """处理 vLLM 动态 Batch Size 变化"""
        current_size = len(self.buffers)
        
        # 1. 扩展: 新请求加入
        if batch_size > current_size:
            for _ in range(batch_size - current_size):
                self.buffers.append([])
            
            # 重新分配 mode tensor
            new_mode = torch.full((batch_size,), self.initial_mode, dtype=torch.int64, device=device)
            if self.mode is not None:
                # 复制旧状态
                new_mode[:current_size] = self.mode
            self.mode = new_mode
        
        # 2. 截断: 请求结束
        elif batch_size < current_size:
            self.buffers = self.buffers[:batch_size]
            if self.mode is not None:
                self.mode = self.mode[:batch_size]
        
        # 3. 设备同步
        if self.mode is not None and self.mode.device != device:
            self.mode = self.mode.to(device)

    def _find_latest_valid_trigger(self, text: str, patterns, start_bound: int, end_bound: int) -> int:
        """
        在 text 中寻找 patterns 的最后一次出现。
        只有当 pattern 的【结束位置】落在 (start_bound, end_bound] 区间内时，才算有效触发。
        返回该 pattern 的结束位置 (end_pos)，如果没找到或无效，返回 -1。
        """
        max_end_pos = -1
        
        for pattern in patterns:
            # 寻找最后一次出现的位置
            p_idx = text.rfind(pattern)
            if p_idx != -1:
                p_end = p_idx + len(pattern)
                # === 核心逻辑：区间判定 ===
                # 对应训练代码: if last_text_len < p_end <= cur_len:
                if start_bound < p_end <= end_bound:
                    if p_end > max_end_pos:
                        max_end_pos = p_end
                        
        return max_end_pos

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T) - T=1 for decode, T>1 for prefill
        cache_position: (B,) or (B, T)
        """
        B, T = input_ids.shape
        device = input_ids.device
        self._ensure_batch_size(B, device)

        input_ids_list = input_ids.tolist()
        
        # 统一 cache_position 为 (B,)，表示本轮 step 的起始位置
        if cache_position.dim() == 2:
            start_pos = cache_position[:, 0]
        else:
            start_pos = cache_position

        # === 1. 准备阶段：处理重置和构建待解码列表 ===
        prev_buffers_to_decode = [] # 用于计算 last_text_len
        curr_buffers_to_decode = [] # 用于计算 cur_len 和查找 pattern
        
        # 暂存每个 batch 是否 reset 过的标记，用于逻辑判断
        is_reset_mask = [False] * B

        for b in range(B):
            # 如果 start_pos[b] == 0，说明是新请求 (Prefill start)
            if start_pos[b] == 0:
                self.buffers[b] = []
                self.mode[b] = self.initial_mode
                is_reset_mask[b] = True

            # 记录“更新前”的 buffer 用于解码
            # copy 是必须的，因为后面马上要 extend
            prev_buffers_to_decode.append(list(self.buffers[b]))

            # 更新 Buffer
            self.buffers[b].extend(input_ids_list[b])
            
            # 记录“更新后”的 buffer 用于解码
            # 注意：此时尚未 truncate，确保完整性
            curr_buffers_to_decode.append(list(self.buffers[b]))

        # === 2. 批量解码 (Batch Decode) ===
        # 只有在有规则时才解码，节省 CPU
        if self.enter or self.exit:
            # skip_special_tokens=False 是必须的，为了匹配 <search>
            prev_texts = self.tok.batch_decode(prev_buffers_to_decode, skip_special_tokens=False)
            curr_texts = self.tok.batch_decode(curr_buffers_to_decode, skip_special_tokens=False)
            
            # === 3. 状态机流转 ===
            for b in range(B):
                # 这里的逻辑与 get_lora1_mask 完全对应
                prev_len = len(prev_texts[b])
                curr_text = curr_texts[b]
                curr_len = len(curr_text)
                
                # 在 (prev_len, curr_len] 区间内寻找最新的 enter 和 exit
                # 如果是新请求 (reset)，prev_len 应该是 0
                search_start = 0 if is_reset_mask[b] else prev_len

                # 找 enter 的触发位置
                enter_pos = self._find_latest_valid_trigger(curr_text, self.enter, search_start, curr_len)
                # 找 exit 的触发位置
                exit_pos  = self._find_latest_valid_trigger(curr_text, self.exit,  search_start, curr_len)

                # 决策逻辑：谁的位置靠后，谁生效
                if enter_pos != -1 or exit_pos != -1:
                    if enter_pos > exit_pos:
                        self.mode[b] = 1 # Enter Search
                        print(f"Enter Search: {curr_text}")
                    else:
                        self.mode[b] = 2 # Exit Search / Normal
        
        # === 4. 收尾：截断 Buffer ===
        # 在检测完所有逻辑后，再进行 truncate，防止 context 丢失
        for b in range(B):
            if len(self.buffers[b]) > self.tail_tokens:
                self.buffers[b] = self.buffers[b][-self.tail_tokens:]

        return self.mode.clone()

# ==========================================
# 2. 修正后的模拟器
# ==========================================

class MockRequest:
    def __init__(self, req_id, prompt, content, tokenizer):
        self.id = req_id
        full_text = prompt + content
        self.tokens = tokenizer.encode(full_text, add_special_tokens=False)
        self.ptr = 0
        self.finished = False
        self.history_modes = []

    def next_token(self):
        if self.ptr < len(self.tokens):
            val = self.tokens[self.ptr]
            pos = self.ptr
            self.ptr += 1
            # 【修正 1】在这里判断是否是最后一个 token
            if self.ptr >= len(self.tokens):
                self.finished = True
            return val, pos
        else:
            self.finished = True
            return None, None

def simulate_vllm_engine(router, requests_pool, max_parallel=2, device="cpu"):
    print(f"\n🚀 开始模拟 Continuous Batching (Max Parallel={max_parallel})...")
    
    active_slots: List[Optional[MockRequest]] = [None] * max_parallel
    
    step_count = 0
    completed_count = 0
    total_requests = len(requests_pool)
    
    COLORS = ['\033[91m', '\033[92m', '\033[93m', '\033[94m']
    RESET = '\033[0m'

    # 循环直到所有任务完成
    while completed_count < total_requests or any(s is not None for s in active_slots):
        step_count += 1
        current_batch_ids = []
        current_batch_pos = []
        batch_indices = [] # 映射 Tensor index -> Slot index
        
        # 1. 调度与清理
        for i in range(max_parallel):
            # 如果当前 Slot 的任务已完成，清理掉
            if active_slots[i] is not None and active_slots[i].finished:
                active_slots[i] = None 
                completed_count += 1
            
            # 如果有空位且还有待处理任务，装载新任务
            if active_slots[i] is None and len(requests_pool) > 0:
                new_req = requests_pool.pop(0)
                active_slots[i] = new_req
                print(f"Step {step_count}: {COLORS[i%4]}Slot {i} -> 装载新请求 Req {new_req.id}{RESET}")

        # 2. 构建 Batch
        for i in range(max_parallel):
            req = active_slots[i]
            if req is not None and not req.finished:
                tid, pos = req.next_token()
                if tid is not None:
                    current_batch_ids.append([tid])
                    current_batch_pos.append(pos)
                    batch_indices.append(i)
        
        if not current_batch_ids:
            break

        input_tensor = torch.tensor(current_batch_ids, dtype=torch.long, device=device)
        pos_tensor = torch.tensor(current_batch_pos, dtype=torch.long, device=device)

        # 3. Router Step
        modes = router.step(input_tensor, pos_tensor)

        # 4. 打印结果
        print(f"Step {step_count}: ", end="")
        for idx, tensor_row in enumerate(range(len(current_batch_ids))):
            slot_idx = batch_indices[idx]
            mode = modes[tensor_row].item()
            
            # 这里的 tid 需要从 input_tensor 取，而不是重新从 req 取
            t_val = input_tensor[tensor_row][0].item()
            token_char = router.tok.decode([t_val], skip_special_tokens=False)
            token_char = token_char.replace("\n", "\\n")
            
            color = COLORS[slot_idx % 4]
            # MA = Mode 1 (Think), MB = Mode 2 (Normal)
            mode_mark = "MA" if mode==1 else "MB"
            
            # 如果 mode 是 1，用加粗/高亮显示，方便检查
            if mode == 1:
                mode_display = f"\033[1m{mode_mark}\033[0m" # Bold
            else:
                mode_display = mode_mark

            print(f"{color}[S{slot_idx}:{mode_display} '{token_char}']{RESET} ", end="")
        print("")

    print("\n✅ 模拟结束。")

if __name__ == "__main__":
    MODEL_NAME = "/text2sql/verl-tool/base_model/Qwen2.5-3B-Instruct"
    ENTER = ["<search>"]
    EXIT  = ["</search>"]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        router = LoRARouter(tokenizer, ENTER, EXIT, initial_mode=2)
        
        # 【修正 2】数据修改：
        # 1. 在 <think> 前后加空格，避免 tokenizer 粘连，确保 decode 出来是 "... <think> ..."
        # 2. Req 1 长度正常，不会被误判提前结束
        data = json.load(open("verl_step_records/search_r1_qa_em-fsdp-agent-_text2sql_verl-tool_base_model_qwen2.5-3b-instruct-2lora-grpo-n16-b512-64-t1.0-lr2e-5-LoRA_R8_fixprefill/search_r1_qa_em-step-1.json"))
        erro_idx = json.load(open('search_r1_qa_em_lora1_tag_erro_stats.json'))[0]['erro_ids'][2:4]
        data = [d for i, d in enumerate(data) if i in erro_idx]

        pool = [MockRequest(i, d['prompt'], d['response'], tokenizer) for i, d in enumerate(data)]
        
        simulate_vllm_engine(router, pool, max_parallel=2)

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()