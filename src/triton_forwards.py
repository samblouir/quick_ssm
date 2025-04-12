import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def combine_fn(aL, xL, aR, xR):
    """
    Combine two forward aggregator tuples (aL, xL) and (aR, xR).

    Represents:
        new_a = aR * aL
        new_x = aR * xL + xR

    Returns
    -------
    (float, float)
        The combined aggregator tuple (a_out, x_out).
    """
    a_out = aR * aL
    x_out = aR * xL + xR
    return a_out, x_out


@triton.jit
def ssm_local_forward_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    loc_a_ptr,
    loc_x_ptr,
    carry_out_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Local forward pass within a single block.

    Each thread processes a single [batch, dim] row over the block range in time.
    We accumulate aggregator (a_run, x_run) for the range, store partial results.

    Parameters
    ----------
    x_ptr, a_ptr, b_ptr : pointers
        Inputs x, a, b in flattened shape [B*L*D].
    loc_a_ptr, loc_x_ptr : pointers
        Outputs storing the local aggregator for each timestep in this block.
    carry_out_ptr : pointer
        Output storing the final aggregator for this block (needed for the scan).
    B, L, D : int
        Batch size, sequence length, dimension.
    BLOCK_L : int
        Block size along the time dimension.
    """
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    n_rows = B * D

    b_idx = row_id // D
    d_idx = row_id % D

    a_run = 1.0
    x_run = 0.0

    t_start = block_id * BLOCK_L
    for i in tl.static_range(BLOCK_L):
        pos = t_start + i
        if pos < L:
            off = b_idx * L * D + pos * D + d_idx
            a_i = tl.load(a_ptr + off)
            b_i = tl.load(b_ptr + off)
            x_i = tl.load(x_ptr + off)

            # Accumulate aggregator
            a_new = a_i * a_run
            x_new = a_i * x_run + b_i * x_i

            # Store per-timestep aggregator
            tl.store(loc_a_ptr + off, a_new)
            tl.store(loc_x_ptr + off, x_new)

            a_run = a_new
            x_run = x_new

    n_blocks = (L + BLOCK_L - 1) // BLOCK_L
    c_off = row_id * n_blocks + block_id

    # Store final aggregator for this block
    tl.store(carry_out_ptr + c_off, a_run)
    tl.store(carry_out_ptr + n_rows * n_blocks + c_off, x_run)


@triton.jit
def ssm_forward_carry_scan_kernel(
    carry_io,
    n_rows,
    n_blocks: tl.constexpr,
):
    """
    Performs an associative scan across blocks (forward direction).

    carry_io is shaped [2 * n_rows * n_blocks], storing aggregator (a, x) for each block.
    The combine_fn(...) function merges partial blocks.

    Parameters
    ----------
    carry_io : pointer
        Flattened aggregator array of shape [2 * n_rows * n_blocks].
    n_rows : int
        B*D.
    n_blocks : int
        Number of blocks along the sequence.
    """
    row_id = tl.program_id(0)
    ks = tl.arange(0, n_blocks)

    # Load the aggregator arrays for this row
    a_in = tl.load(carry_io + row_id * n_blocks + ks)
    x_in = tl.load(carry_io + n_rows * n_blocks + row_id * n_blocks + ks)

    # Perform a forward associative scan
    a_out, x_out = tl.associative_scan(
        (a_in, x_in),  # aggregator tuples
        axis=0,
        combine_fn=combine_fn,
        reverse=False
    )

    # Store results back
    tl.store(carry_io + row_id * n_blocks + ks, a_out)
    tl.store(carry_io + n_rows * n_blocks + row_id * n_blocks + ks, x_out)


@triton.jit
def ssm_fwd_apply_kernel(
    loc_a_ptr,
    loc_x_ptr,
    carry_io_ptr,
    out_a_ptr,
    out_x_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Merges each block's local aggregator with the scanned carry aggregator
    to produce final aggregator values for each timestep in that block.

    The final aggregator for time t is combine_fn( carry_pref, local(t) ).

    Parameters
    ----------
    loc_a_ptr, loc_x_ptr : pointers
        Local aggregator from the local forward pass (per time step).
    carry_io_ptr : pointer
        Flattened aggregator array from the block-scan carry pass.
    out_a_ptr, out_x_ptr : pointers
        Final aggregator outputs (per time step) of shape [B*L*D].
    B, L, D : int
        Batch size, sequence length, dimension.
    BLOCK_L : int
        Block size along the time dimension.
    """
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L

    b_idx = row_id // D
    d_idx = row_id % D

    # Load the carry aggregator from the "previous" block
    c_off = row_id * n_blocks + (block_id - 1)
    if block_id == 0:
        a_pref = 1.0
        x_pref = 0.0
    else:
        a_pref = tl.load(carry_io_ptr + c_off)
        x_pref = tl.load(carry_io_ptr + n_rows * n_blocks + c_off)

    # Merge local aggregator with carry aggregator
    t_start = block_id * BLOCK_L
    for i in tl.static_range(BLOCK_L):
        pos = t_start + i
        if pos < L:
            off = b_idx * L * D + pos * D + d_idx
            a_loc = tl.load(loc_a_ptr + off)
            x_loc = tl.load(loc_x_ptr + off)

            # combine_fn( (a_pref,x_pref), (a_loc,x_loc) )
            a_out, x_out = combine_fn(a_pref, x_pref, a_loc, x_loc)
            tl.store(out_a_ptr + off, a_out)
            tl.store(out_x_ptr + off, x_out)


def block_scan_forward_3d(x, a, b, BLOCK_L=256, use_naive_if_small=False):
    """
    Forward block-scan to compute h(t) in parallel blocks.

    Each block is processed locally, then we do an associative scan
    on the aggregator for each block, and finally apply that carry
    to each block's local aggregator.

    Parameters
    ----------
    x : torch.Tensor
        Shape [B, L, D].
    a : torch.Tensor
        Shape [B, L, D].
    b : torch.Tensor
        Shape [B, L, D].
    BLOCK_L : int, optional
        Block size in the time dimension.
    use_naive_if_small : bool, optional
        NOT YET IMPLEMENTED
        If True, use naive implementation for short sequences.
        Default is False.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        out_a, out_x of shape [B, L, D].
        Here out_x corresponds to the final h(t) values,
        and out_a is the cumulative product aggregator.
    """
    B, L, D = x.shape

    # Buffers for storing local aggregator
    loc_a = torch.empty_like(a)
    loc_x = torch.empty_like(a)

    # Buffers for final aggregator
    out_a = torch.empty_like(a)
    out_x = torch.empty_like(x)

    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L

    # Flatten the aggregator carry buffer: shape [2, n_rows, n_blocks]
    carry_out = torch.empty((2, n_rows, n_blocks), dtype=x.dtype, device=x.device)
    carry_flat = carry_out.view(-1)

    # 1) Local forward pass for each block
    grid1 = (n_rows, n_blocks)
    ssm_local_forward_kernel[grid1](
        x, a, b, loc_a, loc_x, carry_flat,
        B=B, L=L, D=D, BLOCK_L=BLOCK_L
    )

    # 2) Carry aggregator scan across blocks
    grid2 = (n_rows,)
    ssm_forward_carry_scan_kernel[grid2](carry_flat, n_rows, n_blocks)

    # 3) Apply the scanned carry aggregator to each block
    grid3 = (n_rows, n_blocks)
    ssm_fwd_apply_kernel[grid3](
        loc_a, loc_x, carry_flat, out_a, out_x,
        B=B, L=L, D=D, BLOCK_L=BLOCK_L
    )
    return out_a, out_x
