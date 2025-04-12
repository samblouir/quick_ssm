"""
Tests correctness and performance of block_scan_forward_3d against a naive reference.
"""

import torch
import time

from naive_baseline import naive_forward_3d
from triton_forwards import block_scan_forward_3d


def pretty_diff_stats(a, b):
    diff = (a - b).abs()
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "median": float(diff.median()),
        "num>1e-5": int((diff > 1e-5).sum()),
        "shape": tuple(diff.shape),
    }


def test_forwards_scan(
    batch_size: int = 2,
    seq_len: int = 512,
    dim: int = 8,
    block_l: int = 128,
    device_str: str = "cuda",
    atol: float = 1e-5,
    rtol: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> bool:
    print("\n========== Forward Scan Test ==========")
    print(
        f"[Config] B={batch_size}, L={seq_len}, D={dim}, block_l={block_l}, device={device_str}, atol={atol}, rtol={rtol}"
    )

    # Device setup
    if device_str == "cuda" and not torch.cuda.is_available():
        print("!! CUDA requested but unavailable. Switching to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    is_cuda = device_str == "cuda"
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed_all(seed)

    # Input preparation
    dtype = torch.float32
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    a = torch.rand(batch_size, seq_len, dim, device=device, dtype=dtype) * 0.1 + 0.9
    b = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype) * 0.1

    # Triton impl
    print(f"Running Triton (block_scan_forward_3d, BLOCK_L={block_l})...")
    t0 = time.time()
    h_cumprod, h_triton = block_scan_forward_3d(x, a, b, BLOCK_L=block_l)
    if is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    print(f" Triton elapsed: {t1-t0:.4f} sec")

    # Naive reference
    print("Running naive_forward_3d...")
    t2 = time.time()
    h_naive = naive_forward_3d(x, a, b)
    if is_cuda:
        torch.cuda.synchronize()
    t3 = time.time()
    print(f" Naive elapsed: {t3-t2:.4f} sec")

    # Check
    diff_stats = pretty_diff_stats(h_naive, h_triton)
    print(
        f"Max diff {diff_stats['max']:.2e}, Mean {diff_stats['mean']:.2e}, Median {diff_stats['median']:.2e}, Num >1e-5: {diff_stats['num>1e-5']}"
    )

    if verbose:
        print(" h_naive:", h_naive.flatten()[0:8])
        print(" h_triton:", h_triton.flatten()[0:8])
        print(" diff:", (h_naive - h_triton).flatten()[0:8])

    # Allow exact, or within tolerances
    equal = torch.allclose(h_naive, h_triton, rtol=rtol, atol=atol)
    assert (
        equal
    ), f"FAIL: Outputs differ max={diff_stats['max']:.2e} (atol={atol}, rtol={rtol})"
    print("PASS: Triton matches naive baseline within tolerances.")
    return True


def time_performance(name, fn, *args, **kwargs):
    # Warmup
    for _ in range(2):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize() if args[0].is_cuda else None
    t0 = time.time()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize() if args[0].is_cuda else None
    t1 = time.time()
    print(f"{name:35s}: {t1-t0:.4f} sec")
    return out


if __name__ == "__main__":
    # 1. Fast correctness check (using a reasonable BLOCK_L)
    test_forwards_scan(
        batch_size=2, seq_len=512, dim=8, block_l=128, atol=1e-5, rtol=1e-3, seed=42
    )

    # 2. Sanity check edge cases (BLOCK_L >= L, tiny L/D)
    test_forwards_scan(batch_size=1, seq_len=4, dim=1, block_l=8, seed=3)
    test_forwards_scan(batch_size=1, seq_len=1024, dim=1, block_l=1024, seed=4)

    # 3. Perf (LARGE) [Optional: skip on slow GPUs]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        B_perf, L_perf, D_perf = 2, 2**16, 32
        perf_x = torch.randn(B_perf, L_perf, D_perf, device=device_str)
        perf_a = torch.rand(B_perf, L_perf, D_perf, device=device_str) * 0.1 + 0.9
        perf_b = torch.randn(B_perf, L_perf, D_perf, device=device_str) * 0.1
        print("\n-- Performance timings --")
        for blk in [128, 256, 512]:
            time_performance(
                f"Triton BLOCK_L={blk}",
                block_scan_forward_3d,
                perf_x,
                perf_a,
                perf_b,
                blk,
            )
    except Exception as e:
        print("Perf test skipped (device/memory).", e)
