import torch
import triton
import triton.language as tl
from typing import Tuple

def ensure_contiguous_fp32(*args, device=None):
    outs = []
    for x in args:
        y = x.contiguous()
        if y.dtype != torch.float32:
            y = y.float()
        if device is not None and y.device != device:
            y = y.to(device)
        outs.append(y)
    return tuple(outs)

@triton.jit
def combine_fn(aL, xL, aR, xR):
    new_a = aR * aL
    new_x = aR * xL + xR
    return new_a, new_x

@triton.jit
def ssm_local_forward_kernel(
    x_ptr, a_ptr, b_ptr,
    loc_a_ptr, loc_x_ptr, carry_out_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    n_rows = B * D

    b_idx = row_id // D
    d_idx = row_id % D

    a_run = 1.0
    x_run = 0.0

    t_start = block_id * BLOCK_L
    t_next = t_start + BLOCK_L
    t_end = tl.where(t_next > L, L, t_next)

    for i in tl.static_range(BLOCK_L):
        pos = t_start + i
        if pos < t_end:
            off = b_idx * L * D + pos * D + d_idx
            a_i = tl.load(a_ptr + off)
            b_i = tl.load(b_ptr + off)
            x_i = tl.load(x_ptr + off)

            a_new = a_i * a_run
            x_new = a_i * x_run + b_i * x_i

            tl.store(loc_a_ptr + off, a_new)
            tl.store(loc_x_ptr + off, x_new)

            a_run = a_new
            x_run = x_new

    n_blocks = (L + BLOCK_L - 1) // BLOCK_L
    c_off = row_id * n_blocks + block_id
    tl.store(carry_out_ptr + c_off, a_run)
    tl.store(carry_out_ptr + n_rows * n_blocks + c_off, x_run)

@triton.jit
def ssm_forward_carry_scan_kernel(
    carry_io,
    n_rows: tl.constexpr,
    n_blocks: tl.constexpr,
):
    row_id = tl.program_id(0)
    ks = tl.arange(0, n_blocks)

    a_in = tl.load(carry_io + row_id * n_blocks + ks)
    x_in = tl.load(carry_io + n_rows * n_blocks + row_id * n_blocks + ks)

    a_out, x_out = tl.associative_scan((a_in, x_in), 0, combine_fn, reverse=False)
    tl.store(carry_io + row_id * n_blocks + ks, a_out)
    tl.store(carry_io + n_rows * n_blocks + row_id * n_blocks + ks, x_out)

@triton.jit
def ssm_fwd_apply_kernel(
    loc_a_ptr, loc_x_ptr, carry_io_ptr,
    out_a_ptr, out_x_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L

    b_idx = row_id // D
    d_idx = row_id % D

    c_off = row_id * n_blocks + (block_id - 1)
    if block_id == 0:
        a_pref = 1.0
        x_pref = 0.0
    else:
        a_pref = tl.load(carry_io_ptr + c_off)
        x_pref = tl.load(carry_io_ptr + n_rows * n_blocks + c_off)

    t_start = block_id * BLOCK_L
    t_next = t_start + BLOCK_L
    t_end = tl.where(t_next > L, L, t_next)

    for i in tl.static_range(BLOCK_L):
        pos = t_start + i
        if pos < t_end:
            off = b_idx * L * D + pos * D + d_idx
            a_loc = tl.load(loc_a_ptr + off)
            x_loc = tl.load(loc_x_ptr + off)

            a_out, x_out = combine_fn(a_pref, x_pref, a_loc, x_loc)
            tl.store(out_a_ptr + off, a_out)
            tl.store(out_x_ptr + off, x_out)

def block_scan_forward_3d(x, a, b, BLOCK_L=256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward block-scan to compute h(t) in parallel blocks.
    Returns:
        out_a, out_x of shape [B, L, D].
        (out_x is the SSM output, out_a the cumulative product.)
    """
    for t in (x, a, b):
        assert t.dim() == 3, "All inputs should be [B, L, D]."
        assert t.device.type == "cuda", "Inputs must be on CUDA device."
    B, L, D = x.shape
    x, a, b = ensure_contiguous_fp32(x, a, b, device=x.device)
    loc_a = torch.empty_like(a)
    loc_x = torch.empty_like(a)
    out_a = torch.empty_like(a)
    out_x = torch.empty_like(x)

    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L

    carry_out = torch.empty((2, n_rows, n_blocks), dtype=x.dtype, device=x.device)
    carry_flat = carry_out.reshape(-1)

    grid1 = (n_rows, n_blocks)
    ssm_local_forward_kernel[grid1](x, a, b, loc_a, loc_x, carry_flat, B, L, D, BLOCK_L)
    ssm_forward_carry_scan_kernel[(n_rows,)](carry_flat, n_rows, n_blocks)
    ssm_fwd_apply_kernel[grid1](
        loc_a, loc_x, carry_flat, out_a, out_x, B, L, D, BLOCK_L
    )
    return out_a, out_x
