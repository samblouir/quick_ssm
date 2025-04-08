# test_forwards.py
"""
This script tests the performance and correctness of the forward scan ('block_scan_forward_3d')
against a naive implementation ('naive_forward_3d').
"""

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional

# Assuming these imports exist and are correct
from naive_baseline import naive_forward_3d # Expected signature: naive_forward_3d(x, a, b) -> h
from triton_forwards import block_scan_forward_3d # Expected signature: block_scan_forward_3d(x, a, b, BLOCK_L) -> (out_a, out_x/h)


def test_forwards_scan(
    batch_size: int = 2,
    seq_len: int = 512,  # Default smaller size for feasible naive computation
    dim: int = 8,
    block_l: int = 128, # Block size for Triton implementation
    device_str: str = "cuda",
    atol: float = 1e-5, # Absolute tolerance
    rtol: float = 1e-3, # Relative tolerance (adjust as needed)
    seed: int = 0,      # Seed for reproducibility
    verbose: bool = False # Print outputs if True
) -> bool:
    """
    Verifies the output of the custom 'block_scan_forward_3d' against the
    'naive_forward_3d' implementation.

    This function performs the following steps:
    1. Initializes input tensors (x, a, b) on the specified device.
    2. Computes the hidden state 'h_custom' using 'block_scan_forward_3d'.
    3. Computes the hidden state 'h_naive' using 'naive_forward_3d'.
    4. Compares 'h_custom' and 'h_naive' using torch.allclose.
    5. Asserts that the outputs match within the specified tolerances.

    Args:
        batch_size (int): Batch size (B).
        seq_len (int): Sequence length (L).
        dim (int): Feature dimension (D).
        block_l (int): Block size for the Triton kernel.
        device_str (str): Device ('cuda' or 'cpu').
        atol (float): Absolute tolerance for torch.allclose.
        rtol (float): Relative tolerance for torch.allclose.
        seed (int): Random seed.
        verbose (bool): If True, print intermediate outputs.

    Returns:
        bool: True if the outputs match within tolerance.

    Raises:
        AssertionError: If outputs do not match within tolerances.
        RuntimeError: If CUDA is requested but unavailable.
        ImportError: If naive_baseline.naive_forward_3d is not found or has wrong signature.
        ValueError: If block_l is not suitable for the sequence length.
    """
    print(f"\n--- Running Forward Pass Correctness Check ---")
    print(f"Params: B={batch_size}, L={seq_len}, D={dim}, BLOCK_L={block_l}, Device='{device_str}'")
    print(f"Tolerances: atol={atol}, rtol={rtol}")

    # --- Validate BLOCK_L ---
    # Block scan requires block_l to be less than seq_len for the scan part to be meaningful
    # The kernel might handle block_l >= L, but the test intention is usually block_l < L
    if block_l >= seq_len and device_str == "cuda":
         print(f"Warning: BLOCK_L ({block_l}) >= seq_len ({seq_len}). The Triton kernel might behave like the naive version.")
         # Or raise ValueError("BLOCK_L must be less than seq_len for a meaningful block scan test.")

    # --- Setup Device ---
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # --- Reproducibility ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Tensor Initialization ---
    # Use float32 by default, consistent with typical usage
    dtype = torch.float32
    print(f"Using dtype: {dtype}")

    # No requires_grad needed for forward pass testing
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    # Initialize 'a' carefully for stability in naive recurrence
    a = torch.rand(batch_size, seq_len, dim, device=device, dtype=dtype) * 0.1 + 0.9
    b = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype) * 0.1

    # --- 1. Compute with Custom Triton Implementation ---
    print("Running custom Triton 'block_scan_forward_3d' implementation...")
    if device_str == "cpu":
        print("Skipping Triton execution on CPU.")
        h_custom = None # Cannot run Triton on CPU
    else:
        try:
            start_time_triton = time.time()
            # block_scan_forward_3d returns (cumulative_a, hidden_state_h)
            _, h_custom = block_scan_forward_3d(x, a, b, BLOCK_L=block_l)
            if device == torch.device("cuda"): torch.cuda.synchronize()
            end_time_triton = time.time()
            print(f"  Triton forward took {end_time_triton - start_time_triton:.4f} seconds.")
            if verbose:
                print(f"    h_custom output (sample): {h_custom[0, 0, :4]}") # Print sample
        except Exception as e:
            print(f"Error during Triton forward pass: {e}")
            raise e # Re-raise the exception


    # --- 2. Compute with Naive Implementation ---
    print("Running naive 'naive_forward_3d' implementation...")
    try:
        start_time_naive = time.time()
        # Ensure naive_forward_3d only needs x, a, b
        h_naive = naive_forward_3d(x, a, b)
        if device == torch.device("cuda"): torch.cuda.synchronize() # Sync if naive ran on GPU
        end_time_naive = time.time()
        print(f"  Naive forward took {end_time_naive - start_time_naive:.4f} seconds.")
        if verbose:
            print(f"    h_naive output (sample): {h_naive[0, 0, :4]}") # Print sample
    except TypeError:
         print("\nError: `naive_forward_3d` does not seem to accept 3 arguments (x, a, b) or return 1 value (h).")
         print("Please ensure `naive_baseline.py` defines `naive_forward_3d(x, a, b)` returning `h`.")
         raise ImportError("Incorrect naive_forward_3d signature.")
    except Exception as e:
         print(f"\nError during naive forward pass: {e}")
         raise e

    # --- 3. Compare Outputs ---
    print("Comparing outputs...")

    if h_custom is None:
        print("Skipping comparison because Triton execution was skipped (CPU).")
        # Technically the test didn't fail, but didn't run fully.
        # Depending on requirements, you might return False or raise SkipTest exception.
        return True # Or False, or raise SkipTest

    # Ensure both tensors are on the same device (CPU if naive ran there) and same dtype for comparison
    h_naive_comp = h_naive.to(device=h_custom.device, dtype=h_custom.dtype)

    # Perform comparison using torch.allclose
    outputs_match = torch.allclose(h_custom, h_naive_comp, atol=atol, rtol=rtol)

    print(f"  Outputs match within tolerance: {outputs_match}")

    # Use assertion for clear test failure/success
    if not outputs_match:
        diff = (h_custom - h_naive_comp).abs().max().item()
        assert outputs_match, f"Output mismatch! Max absolute difference: {diff:.6e} (atol={atol}, rtol={rtol})"

    print("\nCustom forward output matches naive output within tolerance! ✔")
    print("--- Forward Pass Correctness Check Complete ---")
    return True


# --- Performance Timing Helper (Optional, kept from original) ---
def time_performance(name, func, device, *args, **kwargs):
    if device == "cpu" and "block_scan" in name:
        print(f"Skipping performance timing for {name} on CPU.")
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

        print(f"+++ Performance {name}: {end_time - start_time:.4f} seconds +++")
        return result
    except Exception as e:
        print(f"Error during performance timing {name}: {e}")
        return None


if __name__ == "__main__":
    # --- Configuration ---
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # --- Correctness Check ---
    try:
        test_forwards_scan(
            batch_size=2,
            seq_len=512,    # Moderate size for check
            dim=8,
            block_l=128,    # Example block size
            device_str=device_str,
            atol=1e-5,
            rtol=1e-3,
            verbose=False
        )
        # Add more calls with different parameters if needed
        # test_forwards_scan(seq_len=1024, block_l=256, device_str=device_str)
        # if device_str == "cuda":
        #     test_forwards_scan(seq_len=3, block_l=4, device_str=device_str) # Test edge case L < BLOCK_L

    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Ensure `naive_baseline.py` and `triton_forwards.py` are accessible.")
    except AssertionError as e:
        print(f"\nCorrectness Check Failed: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during correctness check: {e}")
        # raise e # Optionally re-raise

    # --- Performance Demonstration (Optional, kept from original) ---
    print("\n--- Performance Comparison (Optional) ---")
    # Parameters for performance demonstration (can be larger)
    B_perf, L_perf, D_perf = 4, 2**16, 64 # Example size
    dtype_perf = torch.float32
    use_triton = (device_str == "cuda") # Only run Triton perf on CUDA

    print(f"Performance test setup: Device: {device_str}, Shape: ({B_perf}, {L_perf}, {D_perf}), dtype: {dtype_perf}")

    if use_triton:
        # Create Random Input Tensors for Performance Test
        x_perf = torch.randn(B_perf, L_perf, D_perf, device=device, dtype=dtype_perf)
        a_perf = torch.rand(B_perf, L_perf, D_perf, device=device, dtype=dtype_perf) * 0.1 + 0.9
        b_perf = torch.randn(B_perf, L_perf, D_perf, device=device, dtype=dtype_perf) * 0.1

        # Test different block sizes
        for block_size in [128, 256, 512, 1024]:
            # Ensure block size is reasonable relative to sequence length
            if block_size >= L_perf:
                print(f"Skipping BLOCK_L={block_size} for perf test as it's >= L={L_perf}")
                continue

            _ = time_performance(
                f"block_scan_forward_3d (BLOCK_L={block_size})",
                block_scan_forward_3d,
                device_str, # Pass device string to helper
                x_perf,
                a_perf,
                b_perf,
                BLOCK_L=block_size,
            )
        # Optionally time naive on GPU if desired, but usually much slower
        # _ = time_performance("naive_forward_3d", naive_forward_3d, device_str, x_perf, a_perf, b_perf)

    else:
        print("Skipping performance tests (requires CUDA).")

    print("\n--- Script Finished ---")