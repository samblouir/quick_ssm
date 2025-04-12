import torch
import time
from naive_baseline import naive_full_3d
from triton_backwards import block_scan_backward_3d

def compute_reference_gradients(x, a, b, c, loss_func=lambda y: y.sum()):
    """Compute reference gradients via PyTorch Autograd with naive SSM."""
    h, y = naive_full_3d(x, a, b, c)
    loss = loss_func(y)
    loss.backward()
    grads = tuple((v.grad.detach().clone() if v.grad is not None else None) for v in (x, a, b, c))
    for v in (x, a, b, c):
        v.grad = None
    return h.detach(), y.detach(), grads

def compute_custom_gradients(a, b, c, x, h, grad_y):
    """Compute custom gradients using Triton backward kernel."""
    dx, da, db, dc = block_scan_backward_3d(a.detach(), b.detach(), c.detach(), x.detach(), h.detach(), grad_y)
    return dx, da, db, dc

def print_max_error(ref, actual, name):
    if ref is None or actual is None: return
    diff = (ref - actual).abs()
    print(f"    {name}: max|diff|={diff.max().item():.3e}, mean|diff|={diff.mean().item():.3e}, ref min/max=({ref.min().item():.3e},{ref.max().item():.3e}), actual min/max=({actual.min().item():.3e},{actual.max().item():.3e})")

def test_backwards_scan(
    batch_size=2,
    seq_len=512,
    dim=4,
    device_str="cuda",
    atol=1e-5,
    rtol=1e-3,
    seed=0,
    verbose=False,
    triton_warmup_iters=2,
):
    """
    Checks correctness of block_scan_backward_3d (Triton) gradients against autograd (naive SSM).
    """
    print("\n========== Backward Scan Test ==========")
    print(f"[Config] B={batch_size}, L={seq_len}, D={dim}, device={device_str}, atol={atol}, rtol={rtol}")

    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available!")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32, requires_grad=True)
    a = (torch.rand(batch_size, seq_len, dim, device=device, dtype=torch.float32) * 0.1 + 0.9).requires_grad_()
    b = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32, requires_grad=True)
    c = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32, requires_grad=True)

    # Reference
    print("Running PyTorch/naive autograd reference...")
    t0 = time.time()
    try:
        h_naive, y_naive, (dx_ref, da_ref, db_ref, dc_ref) = compute_reference_gradients(x, a, b, c)
    except TypeError as e:
        print("\nError: Check your naive_full_3d signature and return values:", e)
        raise ImportError("Incorrect signature in naive_full_3d.") from e
    t1 = time.time()
    print(f"  Naive/Autograd time: {t1-t0:.3f} s")
    if verbose:
        print("[autograd] dx norm:", dx_ref.norm().item(), "\n      da norm:", da_ref.norm().item(), "\n      db norm:", db_ref.norm().item(), "\n      dc norm:", dc_ref.norm().item())

    # Custom (Triton)
    grad_y = torch.ones_like(y_naive)
    print("Running custom Triton backward (warmup x%d)..." % triton_warmup_iters)
    # Warmup
    for _ in range(triton_warmup_iters):
        _ = block_scan_backward_3d(a.detach(), b.detach(), c.detach(), x.detach(), h_naive.detach(), grad_y)
        if device.type == "cuda":
            torch.cuda.synchronize()

    print("Timing custom Triton backward...")
    t2 = time.time()
    dx_custom, da_custom, db_custom, dc_custom = block_scan_backward_3d(
        a.detach(), b.detach(), c.detach(), x.detach(), h_naive.detach(), grad_y
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.time()
    print(f"  Triton backward time: {t3-t2:.3f} s")

    # Check
    def allclose(a, b): return (a is not None and b is not None and torch.allclose(a, b, atol=atol, rtol=rtol))
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
        raise AssertionError(f"{msg} (atol={atol}, rtol={rtol})")

    print("\n[SUCCESS] All custom backward gradients match autograd gradients within tolerance.")

    return True

if __name__ == "__main__":
    try:
        test_backwards_scan(
            batch_size=2, seq_len=2048, dim=8, device_str="cuda", atol=1e-5, rtol=1e-3, verbose=False
        )
        test_backwards_scan(
            batch_size=1, seq_len=1024, dim=1, device_str="cuda", atol=1e-5, rtol=1e-3, verbose=True
        )
    except ImportError as e:
        print("\nImport Error:", e, "\nCheck that 'naive_baseline.py' and 'triton_backwards.py' are present and correct.")
    except AssertionError as e:
        print("\n[ERROR] GRADIENT CHECK FAILED:", str(e))
    except Exception as e:
        print("\n[UNEXPECTED ERROR]", type(e).__name__, ":", str(e))
