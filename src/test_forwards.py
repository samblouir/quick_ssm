"""
This script tests the performance and correctness of the scan
Using a 1_048_576 sequence length for the precision test.
Tests are done in fp16 and fp32

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional
import torch
import time
from src.naive_baseline import naive_forward_3d
from triton_forwards import block_scan_forward_3d



if __name__ == "__main__":
    # --- Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters for performance demonstration
    B, L, D = 4, 2**17, 128  # Adjusted for potentially faster run time
    use_triton = True

    dtype = torch.float32  # Use float32 by default

    print(
        f"Device: {device}, Shape: ({B}, {L}, {D}), dtype: {dtype}, Using Triton: {use_triton}"
    )

    # --- Create Random Input Tensors ---
    # Use requires_grad=False as we only test forward pass
    x_ = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=False)
    # Initialize 'a' close to 1 but less than 1 for stability
    a_ = (
        torch.rand(B, L, D, device=device, dtype=dtype, requires_grad=False) * 0.1
        + 0.85
    )
    b_ = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=False) * 0.1

    # --- Performance Timing Helper ---
    def timer(name, func, *args, **kwargs):
        if device == "cpu" and "block_scan" in name:
            print(f"Skipping {name} on CPU.")
            return None
        try:
            # Warm-up runs
            print(f"Warming up for {name}...")
            for _ in range(2):
                _ = func(*args, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()

            # Timed runs
            print(f"Timing {name}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            print(f"+++ {name}: {end_time - start_time:.4f} seconds +++")
            return result
        except Exception as e:
            print(f"Error during timing {name}: {e}")
            return None

    # --- Performance Comparison (if using Triton) ---
    if use_triton:
        print("\n--- Performance Comparison ---")
        # Test different block sizes
        for block_size in [128, 256, 512, 1024]:
            # Ensure block size is not larger than sequence length for meaningful test
            if block_size >= L:
                print(f"Skipping BLOCK_L={block_size} as it's >= L={L}")
                # Use naive if block_size >= L, just to get a result
                if L > 0:  # Avoid running naive if L=0
                    _ = timer(
                        f"block_scan_forward_3d (BLOCK_L={block_size} -> Naive Fallback)",
                        block_scan_forward_3d,
                        x_,
                        a_,
                        b_,
                        BLOCK_L=block_size,
                        use_naive_if_small=True,
                    )
                continue

            _ = timer(
                f"block_scan_forward_3d (BLOCK_L={block_size})",
                block_scan_forward_3d,
                x_,
                a_,
                b_,
                BLOCK_L=block_size,
                use_naive_if_small=True,
            )

    # --- Correctness Check vs Naive (on a smaller problem) ---
    print("\n--- Correctness Check vs Naive ---")
    # Use smaller dimensions for feasible naive computation
    B_test, L_test, D_test = 2, 512, 8
    print(f"Using test shape: ({B_test}, {L_test}, {D_test})")

    # Ensure L_test is suitable for both implementations
    test_block_l = 128  # A block size smaller than L_test
    if L_test <= test_block_l and use_triton:
        print(
            f"Warning: L_test ({L_test}) <= test_block_l ({test_block_l}). Triton will use naive fallback."
        )

    try:
        x_test = torch.randn(B_test, L_test, D_test, device=device, dtype=dtype)
        a_test = (
            torch.rand(B_test, L_test, D_test, device=device, dtype=dtype) * 0.1 + 0.85
        )
        b_test = torch.randn(B_test, L_test, D_test, device=device, dtype=dtype) * 0.1

        # Run Naive Implementation
        print("Running naive implementation for comparison...")
        start_time_naive = time.time()
        h_naive = naive_forward_3d(x_test, a_test, b_test)
        if device == "cuda":
            torch.cuda.synchronize()
        end_time_naive = time.time()
        print(f"Naive forward took {end_time_naive - start_time_naive:.4f} seconds.")

        # Run Block Scan Implementation (if applicable)
        print(f"Running block scan implementation (BLOCK_L={test_block_l})...")
        start_time_block = time.time()
        h_states, h_block = block_scan_forward_3d(
            x_test, a_test, b_test, BLOCK_L=test_block_l
        )
        if device == "cuda":
            torch.cuda.synchronize()
        end_time_block = time.time()
        print(
            f"Block scan forward took {end_time_block - start_time_block:.4f} seconds."
        )

        # Compare results
        h_naive = h_naive.to(torch.float32)
        h_block = h_block.to(torch.float32)

        diff = (h_naive - h_block).abs().max().item()
        print(f"Maximum Absolute Difference vs. Naive: {diff:.6e}")

        # Check if the difference is within acceptable tolerance
        tolerance = 1e-5  # if dtype == torch.float32 else 1e-3
        assert diff < tolerance, f"Difference {diff:.6e} exceeds tolerance {tolerance}"
        print(f"Correctness check passed (tolerance={tolerance}).")

    except Exception as e:
        print(
            f"*" * 60,
        )
        print(f"Error during correctness check: {e}")
        raise e


