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
def aggregator_combine(aL, xL, aR, xR):
    new_a = aR * aL
    new_x = aR * xL + xR
    return new_a, new_x

@triton.jit
def ssm_local_backward_kernel(
    a_ptr, c_ptr, dY_ptr,
    loc_a_ptr, loc_x_ptr, carry_out_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    b_idx = row_id // D
    d_idx = row_id % D
    n_rows = B * D

    a_run = 1.0
    x_run = 0.0

    t_start = block_id * BLOCK_L
    t_next = t_start + BLOCK_L
    t_end = tl.where(t_next > L, L, t_next)

    for i in tl.static_range(BLOCK_L):
        pos = (t_end - 1) - i
        if pos >= t_start:
            off = b_idx * L * D + pos * D + d_idx

            # a(t+1)
            at1_off = b_idx * L * D + (pos + 1) * D + d_idx
            mask_a_next = (pos + 1) < L
            a_next = tl.load(a_ptr + at1_off, mask=mask_a_next, other=1.0)

            c_i = tl.load(c_ptr + off)
            gY_i = tl.load(dY_ptr + off)
            # Update aggregator
            a_run_new = a_next * a_run
            x_run_new = a_next * x_run + c_i * gY_i

            tl.store(loc_a_ptr + off, a_run_new)
            tl.store(loc_x_ptr + off, x_run_new)

            a_run = a_run_new
            x_run = x_run_new

    n_blocks = (L + BLOCK_L - 1) // BLOCK_L
    c_off = row_id * n_blocks + block_id
    tl.store(carry_out_ptr + c_off, a_run)
    tl.store(carry_out_ptr + n_rows * n_blocks + c_off, x_run)

@triton.jit
def ssm_backward_carry_scan_kernel(
    carry_io,
    n_rows: tl.constexpr,
    n_blocks: tl.constexpr,
):
    row_id = tl.program_id(0)
    ks = tl.arange(0, n_blocks)

    a_in = tl.load(carry_io + row_id * n_blocks + ks)
    x_in = tl.load(carry_io + (n_rows * n_blocks) + row_id * n_blocks + ks)

    a_out, x_out = tl.associative_scan((a_in, x_in), 0, aggregator_combine, reverse=True)
    tl.store(carry_io + row_id * n_blocks + ks, a_out)
    tl.store(carry_io + (n_rows * n_blocks) + row_id * n_blocks + ks, x_out)

@triton.jit
def ssm_bwd_apply_kernel(
    loc_a_ptr, loc_x_ptr, carry_io_ptr,
    dx_ptr, da_ptr, db_ptr, dc_ptr,
    x_ptr, a_ptr, b_ptr, h_ptr,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    b_idx = row_id // D
    d_idx = row_id % D
    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L

    if block_id + 1 < n_blocks:
        c_off = row_id * n_blocks + (block_id + 1)
        a_pref = tl.load(carry_io_ptr + c_off)
        x_pref = tl.load(carry_io_ptr + n_rows * n_blocks + c_off)
    else:
        a_pref = 1.0
        x_pref = 0.0

    t_start = block_id * BLOCK_L
    t_next = t_start + BLOCK_L
    t_end = tl.where(t_next > L, L, t_next)

    for i in tl.static_range(BLOCK_L):
        pos = (t_end - 1) - i
        if pos >= t_start:
            off = b_idx * L * D + pos * D + d_idx
            a_loc = tl.load(loc_a_ptr + off)
            x_loc = tl.load(loc_x_ptr + off)

            a_out, x_out = aggregator_combine(a_pref, x_pref, a_loc, x_loc)

            dH = x_out

            # dC: directly h(t)
            h_val = tl.load(h_ptr + off)
            tl.store(dc_ptr + off, h_val)

            # dX: b(t) * dH
            b_val = tl.load(b_ptr + off)
            dx_val = b_val * dH
            tl.store(dx_ptr + off, dx_val)

            # dA: h(t-1) * dH
            if pos == 0:
                h_prev = 0.0
            else:
                h_prev = tl.load(h_ptr + b_idx * L * D + (pos - 1) * D + d_idx)
            da_val = h_prev * dH
            tl.store(da_ptr + off, da_val)

            # dB: x(t) * dH
            x_val = tl.load(x_ptr + off)
            db_val = x_val * dH
            tl.store(db_ptr + off, db_val)

def block_scan_backward_3d(a, b, c, x, h, grad_y, BLOCK_L=256) -> Tuple[torch.Tensor, ...]:
    """
    Perform block-scan backward computation for 1D SSM, batch and dim parallel.
    Args:
        a, b, c, x, h, grad_y: [B, L, D] tensors.
        BLOCK_L: time block size (default 256)
    Returns:
        dx, da, db, dc: gradients on [B, L, D]
    """
    for t in (a, b, c, x, h, grad_y):
        assert t.dim() == 3, "All inputs should be [B, L, D]."
        assert t.device.type == "cuda", "Inputs must be on CUDA device."
    B, L, D = x.shape
    a, b, c, x, h, grad_y = ensure_contiguous_fp32(a, b, c, x, h, grad_y, device=x.device)
    loc_a = torch.empty_like(a)
    loc_x = torch.empty_like(a)
    dx = torch.empty_like(x)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dc = torch.empty_like(c)

    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L
    carry_out = torch.empty((2, n_rows, n_blocks), dtype=a.dtype, device=a.device)
    carry_flat = carry_out.reshape(-1)

    grid1 = (n_rows, n_blocks)
    ssm_local_backward_kernel[grid1](a, c, grad_y, loc_a, loc_x, carry_flat, B, L, D, BLOCK_L)
    ssm_backward_carry_scan_kernel[(n_rows,)](carry_flat, n_rows, n_blocks)
    ssm_bwd_apply_kernel[grid1](
        loc_a, loc_x, carry_flat, dx, da, db, dc, x, a, b, h, B, L, D, BLOCK_L
    )
    return dx, da, db, dc

