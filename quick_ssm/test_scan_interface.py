import torch

from quick_ssm.scan_interface import scan
from quick_ssm.naive_baseline import naive_full_3d


def _compare(device: str):
    B, L, D = 2, 32, 4
    dtype = torch.float16 if device == "cuda" else torch.float32
    x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)
    a = torch.rand(B, L, D, device=device, dtype=dtype, requires_grad=True) * 0.1 + 0.9
    b = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True) * 0.1
    c = torch.sigmoid(torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True))

    # Reference
    h_ref, y_ref = naive_full_3d(x, a, b, c)
    w = torch.randn_like(y_ref)
    loss_ref = (y_ref * w).sum()
    loss_ref.backward()
    grads_ref = (x.grad.clone(), a.grad.clone(), b.grad.clone(), c.grad.clone())
    for t in (x, a, b, c):
        t.grad = None

    # Under test
    y = scan(x, a, b, c, block_l=16, checkpoint=True, backend="torch" if device == "cpu" else "auto")
    loss = (y * w).sum()
    loss.backward()
    grads = (x.grad, a.grad, b.grad, c.grad)

    for g_ref, g in zip(grads_ref, grads):
        assert torch.allclose(g_ref, g, atol=1e-4, rtol=1e-3)


def test_scan_cpu():
    _compare(device="cpu")


if __name__ == "__main__":
    test_scan_cpu()
    if torch.cuda.is_available():
        _compare(device="cuda")
        print("CUDA check passed.")
