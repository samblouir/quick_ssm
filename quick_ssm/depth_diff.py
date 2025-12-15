"""
Measure drift between stacked SSM layers (Triton scan) and a pure PyTorch reference.

We run 12 layers on random data and report max/mean abs diff of outputs.
Use this to illustrate that fp32 matches tightly, fp16 is close, bf16 drifts more.
"""

import torch

from quick_ssm.layers import SSM
from quick_ssm.test_swiglu_equiv import GatedLinearRNN, _align_weights


def measure(depth: int, compute_dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    B, T, D = 1, 64, 64
    state_mult = 1.0
    eps = 1e-5

    # Build parallel stacks with shared weights per layer
    ssms = [
        SSM(
            hidden_size=D,
            state_size_mult=state_mult,
            compute_dtype=compute_dtype,
            dtype=torch.float32,
            use_residual=True,
            use_norm=True,
            device=device,
        )
        for _ in range(depth)
    ]
    refs = [GatedLinearRNN(state_size=int(D * state_mult), hidden_size=D, use_norm=True, eps=eps).to(device) for _ in range(depth)]

    # tie weights layer-by-layer
    for ssm, ref in zip(ssms, refs):
        _align_weights(ssm, ref)

    x = torch.randn(B, T, D, device=device, dtype=compute_dtype)
    x_ref = x.clone().detach().to(compute_dtype)

    for ssm, ref in zip(ssms, refs):
        x = ssm(x, block_l=64, checkpoint=False, backend="torch")
        x_ref = ref(x_ref)

    diff = (x - x_ref).abs()
    return diff.max().item(), diff.mean().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        mx, mn = measure(depth=12, compute_dtype=dtype, device=device)
        print(f"{dtype}: max_diff={mx:.3e}, mean_diff={mn:.3e}")
