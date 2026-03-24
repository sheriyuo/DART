#!/usr/bin/env python3
"""
GPU burner to increase utilization across multiple GPUs without writing any files.

Features:
- Spawns one worker per visible GPU (or a selected subset)
- Heavy GEMM (matrix multiply) workload + optional convolution mix
- Adjustable matrix size, loop iters, dtype (fp16/fp32/bf16)
- Optional duty cycle to throttle (e.g., 0.5 ≈ ~50% avg load)
- Runs until duration elapses or you press Ctrl+C

Usage examples:
  python gpu_burner.py                  # max out all visible GPUs
  python gpu_burner.py --duration 900   # run for 15 minutes
  python gpu_burner.py --duty 0.5       # target ~50% avg via sleep throttling
  python gpu_burner.py --size 4096 --iters 16  # tune workload
  python gpu_burner.py --gpus 0,1,2,3   # use only specific devices

Tip: watch with `nvidia-smi dmon -s pucm` or `nvidia-smi --loop=1`.
"""

import argparse
import os
import signal
import sys
import time
from typing import List, Optional

try:
    import torch
except Exception as e:
    print("This script requires PyTorch with CUDA. Install: pip install torch --index-url https://download.pytorch.org/whl/cu118", file=sys.stderr)
    raise

stop_flag = False

def handle_sigint(signum, frame):
    global stop_flag
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-GPU utilization burner (no artifacts)")
    p.add_argument("--gpus", type=str, default="auto", help="GPU ids as comma list, or 'auto' for all visible")
    p.add_argument("--size", type=int, default=4096, help="Square matmul size N (memory ~ 3*N^2*dtype)")
    p.add_argument("--iters", type=int, default=32, help="Matmul iterations per active cycle")
    p.add_argument("--mix-conv", action="store_true", help="Mix in a small convolution to vary kernels")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Compute dtype")
    p.add_argument("--duty", type=float, default=1.0, help="Duty cycle in (0,1], e.g., 0.4 ≈ ~40% load")
    p.add_argument("--duration", type=int, default=0, help="Run seconds (0 = until interrupted)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (though outputs are discarded)")
    return p.parse_args()


def get_devices(arg: str) -> List[int]:
    total = torch.cuda.device_count()
    if total == 0:
        raise RuntimeError("No CUDA GPUs detected. Check CUDA/PyTorch install.")
    if arg == "auto":
        return list(range(total))
    ids = [int(x) for x in arg.split(',') if x.strip() != '']
    for i in ids:
        if i < 0 or i >= total:
            raise ValueError(f"GPU id {i} out of range (0..{total-1})")
    return ids


def dtype_of(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def worker(gpu: int, N: int, iters: int, use_conv: bool, dt: torch.dtype, duty: float, duration: int, seed: int):
    global stop_flag
    torch.cuda.set_device(gpu)
    # Speed tweaks
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    gen = torch.Generator(device=f"cuda:{gpu}").manual_seed(seed + gpu)

    # Allocate tensors once; reuse to avoid allocator churn
    A = torch.randn((N, N), device=gpu, dtype=dt, generator=gen)
    B = torch.randn((N, N), device=gpu, dtype=dt, generator=gen)
    C = torch.empty((N, N), device=gpu, dtype=dt)

    # Optional small conv workload
    if use_conv:
        conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, device=gpu, dtype=dt)
        x = torch.randn((64, 256, 64, 64), device=gpu, dtype=dt, generator=gen)

    start_time = time.time()
    cycle = 0
    # Active time per cycle and sleep time derived from duty
    active_ms = 900  # compute for ~0.9s
    sleep_ms = max(0, int((1.0 - duty) / duty * active_ms)) if duty < 1.0 else 0

    # Warmup
    for _ in range(5):
        C = A @ B
        if use_conv:
            y = conv(x)
        torch.cuda.synchronize()

    while not stop_flag:
        t0 = time.time()
        # ACTIVE PHASE
        for _ in range(iters):
            C = A @ B
            A = torch.relu_(C)
            if use_conv:
                _ = conv(x)
        torch.cuda.synchronize()
        # Optional throttle
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        cycle += 1

        if duration > 0 and (time.time() - start_time) >= duration:
            break

        # Safety: if loop somehow runs too fast, ensure we don't spin wildly
        elapsed_ms = (time.time() - t0) * 1000
        if duty < 1.0 and elapsed_ms < active_ms:
            time.sleep(max(0.0, (active_ms - elapsed_ms) / 1000.0))

    # Prevent DCE from removing compute (noop readback)
    _ = C[0, 0].item()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Ensure you have a CUDA-capable GPU and correct PyTorch build.")

    devices = get_devices(args.gpus)

    # Bound duty to sane range
    duty = float(max(0.05, min(1.0, args.duty)))
    dt = dtype_of(args.dtype)

    print(f"Using GPUs: {devices}")
    print(f"Workload: N={args.size}, iters/cycle={args.iters}, dtype={args.dtype}, mix_conv={args.mix_conv}, duty={duty}")
    if args.duration > 0:
        print(f"Duration: {args.duration}s\nPress Ctrl+C to stop early.")
    else:
        print("Running until interrupted (Ctrl+C)...")

    # Spawn one process per GPU
    from torch.multiprocessing import Process, set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    procs = []
    for gpu in devices:
        p = Process(target=worker, args=(gpu, args.size, args.iters, args.mix_conv, dt, duty, args.duration, args.seed))
        p.daemon = True
        p.start()
        procs.append(p)

    try:
        # Join with small sleeps so we can react to Ctrl+C
        start = time.time()
        while any(p.is_alive() for p in procs):
            for p in procs:
                p.join(timeout=0.5)
            if args.duration > 0 and (time.time() - start) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        print("\nStopped. ✨")


if __name__ == "__main__":
    main()
