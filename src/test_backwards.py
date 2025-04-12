# test_backwards.py
import torch
from naive_baseline import naive_full_3d
from triton_backwards import block_scan_backward_3d

def compute_reference_gradients(
    x, a, b, c, loss_func=lambda y: y.sum()
):
    """Compute gradients via autograd using the reference (naive) implementation."""
    h_naive, y_naive = naive_full_3d(x, a, b, c)
    loss = loss_func(y_naive)
    loss.backward()
    dx = x.grad.detach().clone() if x.grad is not None else None
    da = a.grad.detach().clone() if a.grad is not None else None
    db = b.grad.detach().clone() if b.grad is not None else None
    dc = c.grad.detach().clone() if c.grad is not None else None
    # Clean up for next usage
    x.grad, a.grad, b.grad, c.grad = None, None, None, None
    return h_naive, y_naive, (dx, da, db, dc)

def compute_custom_gradients(
    a, b, c, x, h, grad_y
):
    """Run the custom Triton backward pass to get gradients."""
    # All forward values should be detached (no grad tracking)
    dx, da, db, dc = block_scan_backward_3d(
        a.detach(), b.detach(), c.detach(), x.detach(), h.detach(), grad_y
    )
    return dx, da, db, dc

def print_max_error(ref, actual, name):
    if ref is None or actual is None: return
    diff = (ref - actual).abs().max().item()
    print(f"    {name}: max abs diff={diff:.3e}, ref min/max=({ref.min().item():.3e},{ref.max().item():.3e}), actual min/max=({actual.min().item():.3e},{actual.max().item():.3e})")

def test_backwards_scan(
    batch_size=2,
    seq_len=512,
    dim=4,
    device_str="cuda",
    atol=1e-5,
    rtol=1e-3,
    seed=0,
    verbose=False,
):
    """Verifies gradients of the custom 'block_scan_backward_3d' against autograd reference."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but not available!")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dtype = torch.float32

    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    a = (torch.rand(batch_size, seq_len, dim, device=device, dtype=dtype) * 0.1 + 0.9).requires_grad_()
    b = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    c = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

    if verbose:
        print(f"\n[test] Shapes: x {x.shape}, a {a.shape}, b {b.shape}, c {c.shape}, device={device}, dtype={dtype}")

    # ----- Reference Gradients -----
    try:
        h_naive, y_naive, (dx_ref, da_ref, db_ref, dc_ref) = compute_reference_gradients(x, a, b, c)
    except TypeError as e:
        raise ImportError("Check your naive_full_3d signature and return values: " + str(e))

    if verbose:
        print("[ref] dx norm:", dx_ref.norm().item(), "\n      da norm:", da_ref.norm().item(), "\n      db norm:", db_ref.norm().item(), "\n      dc norm:", dc_ref.norm().item())

    # ----- Custom Gradients (Triton) -----
    grad_y = torch.ones_like(y_naive)
    dx_custom, da_custom, db_custom, dc_custom = compute_custom_gradients(a, b, c, x, h_naive, grad_y)

    # ----- Comparison -----
    allclose = lambda a, b: (a is not None and b is not None and torch.allclose(a, b, atol=atol, rtol=rtol))
    all_ok = True
    errors = []

    print("\n[check] Comparing custom and autograd gradients:")
    for name, ref, actual in [
        ("dx", dx_ref, dx_custom),
        ("da", da_ref, da_custom),
        ("db", db_ref, db_custom),
        ("dc", dc_ref, dc_custom),
    ]:
        match = allclose(ref, actual)
        print(f"    {name}: {'OK' if match else 'FAIL'}")
        if verbose or not match:
            print_max_error(ref, actual, name)
        if not match:
            all_ok = False
            errors.append(name)

    if not all_ok:
        msg = "[FAIL] Gradient mismatch for " + ", ".join(errors)
        # Show max diff for each incorrect
        for name, ref, actual in [("dx", dx_ref, dx_custom), ("da", da_ref, da_custom), ("db", db_ref, db_custom), ("dc", dc_ref, dc_custom)]:
            if name in errors: print_max_error(ref, actual, name)
        raise AssertionError(f"{msg} (tolerances: atol={atol}, rtol={rtol})")

    print("\n[SUCCESS] All custom backward gradients match autograd gradients within tolerance.")
    return True

if __name__ == "__main__":
    try:
        test_backwards_scan(
            batch_size=2,
            seq_len=32_768,
            dim=8,
            device_str="cuda",
            atol=1e-4,
            rtol=1e-3,
            verbose=False,
        )
    except ImportError as e:
        print("\nImport Error:", e, "\nCheck that 'naive_baseline.py' and 'triton_backwards.py' are present and correctly implemented.")
    except AssertionError as e:
        print("\n[ERROR] GRADIENT CHECK FAILED:", str(e))
    except Exception as e:
        print("\n[UNEXPECTED ERROR]", type(e).__name__, ":", str(e))

