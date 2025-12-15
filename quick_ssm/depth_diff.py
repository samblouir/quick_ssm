"""
Drift measurement after stacking many SSM layers on long sequences.

We compare the Triton SSM output to a CPU reference that computes the
recurrence in closed form (no loops):
    pref = cumprod(f)
    h = pref * cumsum(z / pref)

Defaults: depth=12, seq_len=2**20, hidden=64.
"""

import argparse
import torch

from quick_ssm.layers import SSM


def linear_scan_closed_form(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Closed form with float64 accumulation to reduce underflow
    eps = 1e-12
    f64 = f.double().clamp_min(eps)
    z64 = z.double()
    pref = torch.cumprod(f64, dim=1).clamp_min(eps)
    h = pref * torch.cumsum(z64 / pref, dim=1)
    return h.to(f.dtype)


def cpu_reference(x_cpu, weights, depth: int):
    a_w, a_b, b_w, d_w, c1_w, c2_w, out_w, out_b, rms_w = weights
    h = x_cpu
    for _ in range(depth):
        # RMSNorm
        norm = torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        x_norm = h * (rms_w / norm)

        a = torch.sigmoid(torch.nn.functional.linear(x_norm, a_w, a_b))
        b = torch.nn.functional.linear(x_norm, b_w)
        d = torch.sigmoid(torch.nn.functional.linear(x_norm, d_w))
        c = torch.nn.functional.silu(torch.nn.functional.linear(x_norm, c1_w)) * torch.nn.functional.linear(x_norm, c2_w)

        z = b * d
        h_scan = linear_scan_closed_form(a, z)
        h = torch.nn.functional.linear(c * h_scan, out_w, out_b) + h
    return h


def measure(depth: int, seq_len: int, hidden: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    B = 1
    base = SSM(
        hidden_size=hidden,
        state_size_mult=1.0,
        compute_dtype=dtype,
        dtype=torch.float32,
        use_residual=True,
        use_norm=True,
        device=device,
    )
    with torch.no_grad():
        base.A_proj.bias.fill_(4.0)  # bias f toward ~0.982 to reduce underflow over long seqs

    x = torch.randn(B, seq_len, hidden, device=device, dtype=dtype)

    # Triton path
    y = x
    for _ in range(depth):
        y = base(y, block_l=256, checkpoint=False, backend="auto")
    y_cpu = y.float().cpu()

    # CPU reference with copied weights
    weights = (
        base.A_proj.weight.detach().cpu(),
        base.A_proj.bias.detach().cpu(),
        base.B_proj.weight.detach().cpu(),
        base.D_proj.weight.detach().cpu(),
        base.C_proj_a.weight.detach().cpu(),
        base.C_proj_b.weight.detach().cpu(),
        base.out_proj.weight.detach().cpu(),
        base.out_proj.bias.detach().cpu(),
        base.pre_norm.weight.detach().cpu(),
    )
    x_cpu = x.float().cpu()
    ref = cpu_reference(x_cpu, weights, depth)

    diff = (y_cpu - ref).abs()
    return diff.max().item(), diff.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=2 ** 20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    max_diff, mean_diff = measure(args.depth, args.seq_len, args.hidden, dtype, device)
    print(
        {
            "dtype": args.dtype,
            "depth": args.depth,
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }
    )


if __name__ == "__main__":
    main()
