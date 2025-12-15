"""
Drift measurement after stacking many SSM layers on long sequences.

We compare Triton output to a numerically stable CPU reference.
The naive cumprod/cumsum “closed form” underflows for long L and
changes the recurrence. Here we use a blockwise stable reference.
"""

import argparse
import torch

from quick_ssm.layers import SSM


def linear_scan_blockwise_reference(a: torch.Tensor, z: torch.Tensor, block_l: int = 256) -> torch.Tensor:
    """
    Stable reference for h(t)=a(t)*h(t-1)+z(t) using small blocks and a block scan.
    Works for very long L because it never forms a full-length cumprod.
    """
    if a.shape != z.shape:
        raise ValueError("a and z must have same shape [B,L,D]")
    B, L, D = a.shape
    device = a.device
    a64 = a.double()
    z64 = z.double()

    n_blocks = (L + block_l - 1) // block_l
    L_pad = n_blocks * block_l
    if L_pad != L:
        pad = L_pad - L
        a64 = torch.cat([a64, torch.ones(B, pad, D, device=device, dtype=torch.float64)], dim=1)
        z64 = torch.cat([z64, torch.zeros(B, pad, D, device=device, dtype=torch.float64)], dim=1)

    a_blk = a64.view(B, n_blocks, block_l, D)
    z_blk = z64.view(B, n_blocks, block_l, D)

    a_loc = torch.empty_like(a_blk)
    h_loc = torch.empty_like(z_blk)
    a_run = torch.ones((B, n_blocks, D), device=device, dtype=torch.float64)
    h_run = torch.zeros((B, n_blocks, D), device=device, dtype=torch.float64)
    for i in range(block_l):
        ai = a_blk[:, :, i, :]
        zi = z_blk[:, :, i, :]
        a_run = ai * a_run
        h_run = ai * h_run + zi
        a_loc[:, :, i, :] = a_run
        h_loc[:, :, i, :] = h_run

    a_blk_total = a_run
    h_end0 = h_run

    h_in = torch.empty((B, n_blocks, D), device=device, dtype=torch.float64)
    h_prev = torch.zeros((B, D), device=device, dtype=torch.float64)
    for k in range(n_blocks):
        h_in[:, k, :] = h_prev
        h_prev = a_blk_total[:, k, :] * h_prev + h_end0[:, k, :]

    h_full = a_loc * h_in.unsqueeze(2) + h_loc
    h_full = h_full.view(B, L_pad, D)[:, :L, :]
    return h_full.to(a.dtype)


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
        h_scan = linear_scan_blockwise_reference(a, z, block_l=256)
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
