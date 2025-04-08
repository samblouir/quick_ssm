# test_backwards.py
import torch
import time
from naive_baseline import naive_full_3d
from triton_backwards import (
    block_scan_backward_3d,
)


def test_backwards_scan(
    batch_size: int = 2,
    seq_len: int = 512,
    dim: int = 4,
    device_str: str = "cuda",
    atol: float = 1e-5,
    rtol: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> bool:
    """
    Verifies the gradients computed by the custom 'block_scan_backward_3d'
    against PyTorch's autograd gradients derived from the 'naive_full_3d' implementation.

    This function performs the following steps:
    1. Initializes input tensors (x, a, b, c) with requires_grad=True.
    2. Computes the forward pass using 'naive_full_3d' to get hidden states (h) and output (y).
    3. Defines a simple scalar loss based on the output 'y'.
    4. Computes reference gradients (dx, da, db, dc) using PyTorch's autograd (`loss.backward()`).
    5. Computes the gradient of the loss with respect to the output 'y' (`grad_y`).
    6. Calls the custom 'block_scan_backward_3d' function with the forward pass tensors
       and `grad_y` to get the custom gradients (dx_custom, da_custom, db_custom, dc_custom).
    7. Compares the autograd gradients and the custom gradients using torch.allclose.
    8. Asserts that the gradients match within the specified tolerances.

    Args:
        batch_size (int): Batch size (B).
        seq_len (int): Sequence length (L).
        dim (int): Feature dimension (D).
        device_str (str): Device ('cuda' or 'cpu').
        atol (float): Absolute tolerance for torch.allclose.
        rtol (float): Relative tolerance for torch.allclose.
        seed (int): Random seed.
        verbose (bool): If True, print gradients.

    Returns:
        bool: True if all gradient comparisons pass.

    Raises:
        AssertionError: If gradients do not match within tolerances.
        RuntimeError: If CUDA is requested but unavailable.
        ImportError: If naive_baseline.naive_full_3d is not found or has wrong signature.
    """
    print(f"\n--- Running Gradient Verification for Backward Pass ---")
    print(f"Params: B={batch_size}, L={seq_len}, D={dim}, Device='{device_str}'")
    print(f"Tolerances: atol={atol}, rtol={rtol}")

    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available."
        )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # --- Reproducibility ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Tensor Initialization ---
    # Use float64 for higher precision in gradient checking if needed, but start with float32
    dtype = torch.float32
    print(f"Using dtype: {dtype}")

    # Requires_grad=True for inputs involved in autograd calculation
    x = torch.randn(
        batch_size, seq_len, dim, device=device, requires_grad=True, dtype=dtype
    )
    # Initialize 'a' carefully for stability in naive recurrence
    a = (
        torch.rand(batch_size, seq_len, dim, device=device, dtype=dtype) * 0.1 + 0.9
    ).requires_grad_(True)
    b = torch.randn(
        batch_size, seq_len, dim, device=device, requires_grad=True, dtype=dtype
    )
    c = torch.randn(
        batch_size, seq_len, dim, device=device, requires_grad=True, dtype=dtype
    )

    # --- 1. Compute Reference Gradients using Autograd ---
    print("Running naive forward and autograd backward...")

    # Perform forward pass with the naive implementation to get h and y
    # Ensure naive_full_3d accepts c and computes y = c*h
    try:
        h_naive, y_naive = naive_full_3d(x, a, b, c)
    except TypeError:
        print(
            "\nError: `naive_full_3d` does not seem to accept 4 arguments (x, a, b, c) or return 2 values (h, y)."
        )
        print(
            "Please ensure `naive_baseline.py` defines `naive_full_3d(x, a, b, c)` returning `(h, y)` where `y=c*h`."
        )
        raise ImportError("Incorrect naive_full_3d signature.")
    except Exception as e:
        print(f"\nError during naive forward pass: {e}")
        raise e

    # Define a simple scalar loss function (e.g., sum of squares or just sum)
    # Using sum() makes grad_y simpler (all ones)
    loss_naive = y_naive.sum()
    # loss_naive = (y_naive * y_naive).sum() # Alternative: sum of squares

    # Compute gradients using PyTorch's autograd
    loss_naive.backward()

    # Store the reference gradients
    dx_naive = x.grad.clone() if x.grad is not None else None
    da_naive = a.grad.clone() if a.grad is not None else None
    db_naive = b.grad.clone() if b.grad is not None else None
    dc_naive = c.grad.clone() if c.grad is not None else None

    if verbose:
        print("  Autograd Gradients (Reference):")
        print(f"    dx_naive: {dx_naive}")
        print(f"    da_naive: {da_naive}")
        print(f"    db_naive: {db_naive}")
        print(f"    dc_naive: {dc_naive}")

    # Ensure gradients are not None
    if any(g is None for g in [dx_naive, da_naive, db_naive, dc_naive]):
        print("Warning: Some autograd gradients are None. Check requires_grad flags.")

    # Remove gradients
    x.grad = None
    a.grad = None
    b.grad = None
    c.grad = None

    # --- 2. Compute Gradients using Custom Triton Backward Pass ---
    print("Running custom Triton backward pass...")

    # Need the gradient of the loss w.r.t. 'y' as input to the custom backward function
    # If loss = y.sum(), then grad_y is all ones.
    grad_y = torch.ones_like(y_naive)
    # If loss = (y*y).sum(), then grad_y is 2*y
    # grad_y = 2 * y_naive.detach() # Detach y_naive value

    # Inputs to the custom backward should be detached values from the forward pass
    # The custom function computes gradients directly, it doesn't track autograd history internally
    dx_custom, da_custom, db_custom, dc_custom = block_scan_backward_3d(
        a.detach(),
        b.detach(),
        c.detach(),
        x.detach(),
        h_naive.detach(),  # Pass the hidden state from the forward pass
        grad_y,  # Pass the gradient of the loss w.r.t. y
    )

    if verbose:
        print("  Custom Triton Gradients:")
        print(f"    dx_custom: {dx_custom}")
        print(f"    da_custom: {da_custom}")
        print(f"    db_custom: {db_custom}")
        print(f"    dc_custom: {dc_custom}")

    # --- 3. Compare Gradients ---
    print("Comparing gradients...")

    results = {}
    all_passed = True

    # Compare dx
    if dx_naive is not None and dx_custom is not None:
        results["dx"] = torch.allclose(dx_custom, dx_naive, atol=atol, rtol=rtol)
        print(f"  Gradient match for dx: {results['dx']}")
        if not results["dx"]:
            all_passed = False
    else:
        print("  Skipping dx comparison (one or both gradients are None).")
        results["dx"] = False
        all_passed = False

    # Compare da
    if da_naive is not None and da_custom is not None:
        results["da"] = torch.allclose(da_custom, da_naive, atol=atol, rtol=rtol)
        print(f"  Gradient match for da: {results['da']}")
        if not results["da"]:
            all_passed = False
    else:
        print("  Skipping da comparison (one or both gradients are None).")
        results["da"] = False
        all_passed = False

    # Compare db
    if db_naive is not None and db_custom is not None:
        results["db"] = torch.allclose(db_custom, db_naive, atol=atol, rtol=rtol)
        print(f"  Gradient match for db: {results['db']}")
        if not results["db"]:
            all_passed = False
    else:
        print("  Skipping db comparison (one or both gradients are None).")
        results["db"] = False
        all_passed = False

    # Compare dc
    if dc_naive is not None and dc_custom is not None:
        results["dc"] = torch.allclose(dc_custom, dc_naive, atol=atol, rtol=rtol)
        print(f"  Gradient match for dc: {results['dc']}")
        if not results["dc"]:
            all_passed = False
    else:
        print("  Skipping dc comparison (one or both gradients are None).")
        results["dc"] = False
        all_passed = False

    # Assertions for clear test failure/success
    if not all_passed:
        # Provide more details on failure
        mismatched = [name for name, passed in results.items() if not passed]
        error_msg = f"Gradient mismatch found for: {', '.join(mismatched)}. "
        # Optionally calculate and show max difference for debugging
        if "dx" in mismatched and dx_naive is not None and dx_custom is not None:
            diff = (dx_custom - dx_naive).abs().max().item()
            error_msg += f"Max dx diff: {diff:.2e}. "
        if "da" in mismatched and da_naive is not None and da_custom is not None:
            diff = (da_custom - da_naive).abs().max().item()
            error_msg += f"Max da diff: {diff:.2e}. "
        if "db" in mismatched and db_naive is not None and db_custom is not None:
            diff = (db_custom - db_naive).abs().max().item()
            error_msg += f"Max db diff: {diff:.2e}. "
        if "dc" in mismatched and dc_naive is not None and dc_custom is not None:
            diff = (dc_custom - dc_naive).abs().max().item()
            error_msg += f"Max dc diff: {diff:.2e}. "

        assert all_passed, error_msg + f"(atol={atol}, rtol={rtol})"

    print(
        "\nAll custom backward gradients match autograd gradients within tolerance! ✔"
    )
    print("--- Backward Gradient Verification Complete ---")
    return True


if __name__ == "__main__":
    try:
        # Example usage: Run the verification
        verification_passed = test_backwards_scan(
            batch_size=2,
            seq_len=32_768,
            dim=8,
            device_str="cuda",
            atol=1e-4,
            rtol=1e-3,
            verbose=False,
        )

    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Ensure `naive_baseline.py` and `triton_backwards.py` are accessible.")
    except AssertionError as e:
        print(f"\nGradient Verification Failed: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
        # raise e
