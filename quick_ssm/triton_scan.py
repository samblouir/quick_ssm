import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
# from naive import ssm_naive_forward_3d, ssm_naive_full_3d

'''
Implements the forward and backward passes of a State Space Model (SSM)
using a block parallel scan algorithm accelerated with Triton.

The core recurrence is: h(t) = a(t) * h(t-1) + b(t) * x(t)
The final output is y(t) = c(t) * h(t)

Algorithm: Block Scan (Parallel Prefix Sum)
-------------------------------------------
Long sequences (length L) are divided into blocks of size `BLOCK_L`.

1.  Local Scan (Parallel): Within each block, compute the recurrence
    independently. Store the
    intermediate hidden states (`loc_x`) and the final state (`carry_out`)
    for each block. This step is parallel across blocks and batch/dimension.
2.  Carry Scan (Parallel Prefix Sum): Compute the true carry-in state for
    each block by performing a parallel scan (prefix sum) on the `carry_out`
    values from the local scan. This uses the associative property of the
    SSM recurrence.
3.  Final Application (Parallel): Update the intermediate hidden states
    (`loc_x`) within each block by applying the corresponding carry computed
    in the previous step. This yields the final correct hidden states `h(t)`.
    This step is also parallel across blocks and batch/dimension.

TODO: Finish adding gradient checkpointing support

'''



@triton.jit
def aggregator_combine(aL, xL, aR, xR):
    """
    Associative combine operation for the forward scan.
    Represents combining two sequential SSM steps:
    h2 = aR * h1 + xR
    h1 = aL * h0 + xL
    ------------------
    h2 = aR * (aL * h0 + xL) + xR
       = (aR * aL) * h0 + (aR * xL + xR)
    Returns (a_out, x_out) corresponding to (aR * aL, aR * xL + xR)
    """
    a_out = aR * aL
    x_out = aR * xL + xR
    return a_out, x_out


@triton.jit
def aggregator_combine(aR, xR, aL, xL):
    """
    Associative combine operation for the backward scan (reversed time).
    Combines dH(t) based on dH(t+1). Corresponds to the structure:
    dH(t) = a(t+1)*dH(t+1) + local_grad(t)
    Represents combining two steps in the *reversed* direction.
    If the combined state represents `(a_combined, x_combined)` such that
    `output = a_combined * input + x_combined`, this function computes
    the parameters for the combined operation `L -> R`.
    Let operation L be dH(t+1) = aL * dH(t+2) + xL (corresponds to time step t+1)
    Let operation R be dH(t)   = aR * dH(t+1) + xR (corresponds to time step t)
    Then dH(t) = aR * (aL * dH(t+2) + xL) + xR
              = (aR * aL) * dH(t+2) + (aR * xL + xR)
    The kernel `ssm_backward_carry_scan_kernel` uses reverse=True, meaning it combines
    elements from right to left (e.g., block k+1 combined into block k).
    The `tl.associative_scan` with reverse=True effectively computes R composed with L.
    So, if the function receives `(aR, xR)` from the right (future) and `(aL, xL)` from the
    left (current block's contribution), the combination should represent applying R then L.
    The function as written `aggregator_combine(aR, xR, aL, xL)` computes:
    a_out = aL * aR  (Apply R then L for the 'a' part)
    x_out = aL * xR + xL (Apply R then L for the 'x' part)
    This matches the composition `R -> L`.
    """
    a_out = aL * aR
    x_out = aL * xR + xL
    return a_out, x_out


@triton.jit
def ssm_local_forward_kernel(
    # --- Input/Output Pointers ---
    x_ptr,         # Input sequence X: shape (B, L, D)
    a_ptr,         # State transition A: shape (B, L, D)
    b_ptr,         # Input transition B: shape (B, L, D)
    loc_a_ptr,     # Output: Local transition accumulation (cumulative a within block)
    loc_x_ptr,     # Output: Local state H computed assuming h(-1)=0 within block
    carry_out_ptr, # Output: Carry (final A, final H) for each block
    # --- Tensor Shapes ---
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # --- Algorithm Parameters ---
    BLOCK_L: tl.constexpr,
):
    """
    Triton kernel for the local forward scan within each block.
    Computes h(t) = a(t) * h(t-1) + b(t) * x(t) independently per block,
    assuming h(-1) = 0 at the start of the block. Stores the intermediate
    local state `loc_x` and the block's overall transformation (`carry_out`).

    Grid: (B * D, num_blocks)
        - Program 0: Indexes batch and dimension (row_id = b * D + d)
        - Program 1: Indexes block (block_id)
    """
    # Identify the specific sequence (batch B, dimension D) and block this thread works on
    row_id = tl.program_id(0)    # Index for B and D combined (0 to B*D - 1)
    block_id = tl.program_id(1)  # Index for the block (0 to num_blocks - 1)
    n_rows = B * D               # Total number of independent sequences (B*D)

    # Deconstruct row_id to get batch and dimension indices
    b_ = row_id // D
    d_ = row_id % D

    # Initialize running state for the forward recurrence within the block
    # (a_run, x_run) represents the transform h(t) = a_run * h(start-1) + x_run
    a_run = 1.0 # Represents cumulative product of 'a's
    x_run = 0.0 # Represents the accumulated state assuming h(start-1)=0

    # Iterate through the time steps within the assigned block
    t_start = block_id * BLOCK_L
    for i in tl.static_range(BLOCK_L): # Use static_range for potential unrolling
        pos = t_start + i
        if pos < L: # Boundary check for the last block
            # Calculate memory offset for the current element (B, L, D) -> flat
            off = b_ * L * D + pos * D + d_

            # Load input values for this time step
            a_i = tl.load(a_ptr + off)
            b_i = tl.load(b_ptr + off)
            x_i = tl.load(x_ptr + off)

            # Apply the SSM recurrence relation update
            # h(i) = a_i * h(i-1) + b_i * x_i
            # We track the transformation (a_run, x_run) such that h(i) = a_run * h(start-1) + x_run
            # Combine (a_i, b_i*x_i) with previous (a_run, x_run) using aggregator_combine
            # Let L = (a_run, x_run), R = (a_i, b_i*x_i)
            # a_new = a_i * a_run
            # x_new = a_i * x_run + b_i * x_i
            a_new, x_new = aggregator_combine(a_run, x_run, a_i, b_i * x_i)

            # Store the intermediate local results
            # loc_a stores the cumulative 'a' product up to time 'pos' within the block
            # loc_x stores the state 'h(pos)' computed assuming h(start-1)=0
            tl.store(loc_a_ptr + off, a_new)
            tl.store(loc_x_ptr + off, x_new)

            # Update the running state for the next iteration
            a_run = a_new
            x_run = x_new

    # After processing the block, store the final (a_run, x_run) as the carry-out
    # This represents the transformation applied by the entire block
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L
    # Offset for storing carries: layout is [A_carries | X_carries] flattened
    # A carries: index `row_id * n_blocks + block_id`
    # X carries: index `n_rows * n_blocks + row_id * n_blocks + block_id`
    carry_a_off = row_id * n_blocks + block_id
    carry_x_off = n_rows * n_blocks + carry_a_off

    tl.store(carry_out_ptr + carry_a_off, a_run)
    tl.store(carry_out_ptr + carry_x_off, x_run)


@triton.jit
def ssm_forward_carry_scan_kernel(
    carry_io,      # Input/Output: Carries (A, X), shape (2, B*D, num_blocks), flattened
    n_rows: tl.constexpr,       # B * D
    n_blocks: tl.constexpr,     # Number of blocks
):
    """
    Triton kernel for the parallel scan (prefix sum) on the forward carries.
    Modifies `carry_io` in-place using `tl.associative_scan`. Each element `k`
    in the output represents the combined transformation from block 0 to block `k`.

    Grid: (B * D,)
        - Program 0: Indexes batch and dimension (row_id)
    """
    row_id = tl.program_id(0) # Index for B and D combined (0 to B*D - 1)

    # --- Load carries for the current sequence (row_id) ---
    # Create offsets for all blocks for this row
    ks = tl.arange(0, n_blocks) # [0, 1, ..., n_blocks-1]
    # Offsets for A carries: row_id * n_blocks + block indices
    a_carry_offset = row_id * n_blocks + ks
    # Offsets for X carries: starts after all A carries
    x_carry_offset = n_rows * n_blocks + a_carry_offset

    a_in = tl.load(carry_io + a_carry_offset, mask = ks < n_blocks) # Load A carries for this row
    x_in = tl.load(carry_io + x_carry_offset, mask = ks < n_blocks) # Load X carries for this row

    # --- Perform associative scan ---
    # `aggregator_combine` defines how to combine transforms from adjacent blocks
    # The scan computes the cumulative transform up to each block
    a_out, x_out = tl.associative_scan(
        (a_in, x_in), axis=0, combine_fn=aggregator_combine, reverse=False
    )

    # --- Store results back ---
    # `carry_io` now contains the scanned carries (cumulative transformations)
    tl.store(carry_io + a_carry_offset, a_out, mask = ks < n_blocks)
    tl.store(carry_io + x_carry_offset, x_out, mask = ks < n_blocks)


@triton.jit
def ssm_fwd_apply_kernel(
    # --- Input Pointers ---
    loc_a_ptr,     # Local 'a' accumulation from local scan (A_loc)
    loc_x_ptr,     # Local 'h' state from local scan (H_loc, assumes h_start=0)
    carry_io_ptr,  # Scanned forward carries (A_scan, H_scan) from carry scan kernel
    # --- Output Pointers ---
    out_a_ptr,     # Final 'a' accumulation (optional, for debugging/testing)
    out_x_ptr,     # Final state 'h' output (H_final)
    # --- Tensor Shapes ---
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # --- Algorithm Parameters ---
    BLOCK_L: tl.constexpr,
):
    """
    Triton kernel to apply the scanned forward carries to the local results.
    Computes the final h(t) using the formula:
    H_final(t) = Apply(Carry_scan[block-1], Local_result(t))
    where Carry_scan[block-1] = (A_prefix, H_prefix) is the cumulative transform up to block-1
    and Local_result(t) = (A_loc(t), H_loc(t)) is the transform from block start to time t.
    The combination is: H_final(t) = A_loc(t) * H_prefix + H_loc(t)

    Grid: (B * D, num_blocks)
        - Program 0: Indexes batch and dimension (row_id)
        - Program 1: Indexes block (block_id)
    """
    # Identify the specific sequence (batch B, dimension D) and block
    row_id = tl.program_id(0)    # Index for B and D combined
    block_id = tl.program_id(1)  # Index for the block
    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L

    # Deconstruct row_id to get batch and dimension indices
    b_ = row_id // D
    d_ = row_id % D

    # --- Determine the carry-in prefix from the *previous* block ---
    # The scanned carry at index `k` represents the cumulative transform up to block `k`.
    # We need the prefix *up to* the end of block `k-1`.
    # Offsets for A prefix carry: row_id * n_blocks + (block_id - 1)
    carry_a_off_prefix = row_id * n_blocks + (block_id - 1)
    # Offsets for X prefix carry: n_rows * n_blocks + A_prefix_offset
    carry_x_off_prefix = n_rows * n_blocks + carry_a_off_prefix

    if block_id == 0:
        # First block has no prefix, equivalent to identity transform (A=1, H=0)
        a_pref = 1.0
        x_pref = 0.0
    else:
        # Load the scanned carry from the end of the previous block
        a_pref = tl.load(carry_io_ptr + carry_a_off_prefix)
        x_pref = tl.load(carry_io_ptr + carry_x_off_prefix)

    # Iterate through the time steps within the assigned block
    t_start = block_id * BLOCK_L
    for i in tl.static_range(BLOCK_L):
        pos = t_start + i
        if pos < L: # Boundary check
            # Calculate memory offset
            off = b_ * L * D + pos * D + d_

            # Load the locally computed results for this time step
            a_loc = tl.load(loc_a_ptr + off) # Local cumulative A within block to time pos
            x_loc = tl.load(loc_x_ptr + off) # Local H at time pos (assuming h_start=0)

            # Combine the prefix carry (from block 0 to block_id-1) with the local result
            # Prefix transform: h_prev = a_pref * h(-1) + x_pref
            # Local transform:  h_loc = a_loc * h(start-1) + x_loc
            # We want h_final = combined_a * h(-1) + combined_x
            # Where h(start-1) = h_prev
            # h_final = a_loc * h_prev + x_loc
            #         = a_loc * (a_pref * h(-1) + x_pref) + x_loc
            #         = (a_loc * a_pref) * h(-1) + (a_loc * x_pref + x_loc)
            # The aggregator_combine(aL, xL, aR, xR) calculates (aR*aL, aR*xL + xR)
            # We need combine(prefix, local) => L=prefix=(a_pref, x_pref), R=local=(a_loc, x_loc)
            a_out, x_out = aggregator_combine(a_pref, x_pref, a_loc, x_loc)

            # Store the final results
            tl.store(out_a_ptr + off, a_out) # Store final cumulative A (optional)
            tl.store(out_x_ptr + off, x_out) # Store final state H


def block_scan_forward_3d(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, BLOCK_L: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs the block scan forward pass for the SSM: h(t) = a(t)h(t-1) + b(t)x(t).

    Args:
        x: Input tensor of shape (B, L, D).
        a: State transition tensor of shape (B, L, D).
        b: Input transition tensor of shape (B, L, D).
        BLOCK_L: Block size for the scan algorithm.

    Returns:
        A tuple (out_a, out_x):
        - out_a: Final accumulated 'a' values (cumulative product), shape (B, L, D). (Mainly for debugging)
        - out_x: Final hidden state 'h', shape (B, L, D).
    """
    B, L, D = x.shape
    assert a.shape == (B, L, D)
    assert b.shape == (B, L, D)

    # Allocate memory for intermediate and final results
    loc_a = torch.empty_like(a) # Stores local cumulative A within blocks
    loc_x = torch.empty_like(a) # Stores local H within blocks (h_start=0)
    out_a = torch.empty_like(a) # Final cumulative A
    out_x = torch.empty_like(x) # Final H

    n_rows = B * D # Number of independent sequences to process
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L # Number of blocks

    # Allocate memory for carries (A and X components)
    # Shape: (2, n_rows, n_blocks) -> Flattened for Triton
    carry_out = torch.empty((2, n_rows, n_blocks), dtype=x.dtype, device=x.device)
    carry_flat = carry_out.view(-1) # Flatten to shape (2 * n_rows * n_blocks)

    # --- Kernel 1: Local Scan ---
    # Computes local h within blocks & the block transform. Parallel over B, D, n_blocks.
    grid1 = (n_rows, n_blocks)
    ssm_local_forward_kernel[grid1](
        x, a, b, loc_a, loc_x, carry_flat,
        B=B, L=L, D=D, BLOCK_L=BLOCK_L
    )

    # --- Kernel 2: Carry Scan ---
    # Performs parallel prefix sum on block transforms. Parallel over B, D.
    grid2 = (n_rows,)
    # Note: n_blocks needs to be compile-time constant for tl.associative_scan's internal blocking.
    # Workaround if n_blocks is too large/dynamic: pad L or use a segmented scan.
    ssm_forward_carry_scan_kernel[grid2](carry_flat, n_rows=n_rows, n_blocks=n_blocks)

    # --- Kernel 3: Apply Carry ---
    # Applies the scanned carries to local results. Parallel over B, D, n_blocks.
    grid3 = (n_rows, n_blocks)
    ssm_fwd_apply_kernel[grid3](
        loc_a, loc_x, carry_flat, out_a, out_x,
        B=B, L=L, D=D, BLOCK_L=BLOCK_L
    )

    # out_x contains the final hidden states h(t)
    return out_a, out_x


# --- Backward Pass Kernels and Functions ---

@triton.jit
def ssm_local_backward_kernel(
    # --- Input Pointers ---
    a_ptr,         # State transition A: shape (B, L, D)
    c_ptr,         # Output projection C: shape (B, L, D)
    dY_ptr,        # Gradient dL/dY: shape (B, L, D)
    # --- Output Pointers ---
    loc_a_ptr,     # Output: Local backward transition accumulation (A')
    loc_x_ptr,     # Output: Local dH computed assuming dH(end+1)=0 within block
    carry_out_ptr, # Output: Carry (final A', final dH') for each block backward
    # --- Tensor Shapes ---
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # --- Algorithm Parameters ---
    BLOCK_L: tl.constexpr,
):
    """
    Triton kernel for the local backward scan within each block.
    Computes the backward recurrence dH(t) = a(t+1)*dH(t+1) + c(t)*dY(t)
    independently per block, assuming dH(end+1) = 0 at the end of the block
    (where end is the last time step in the block).
    Processes time in reverse order within the block. Stores the intermediate
    local gradient `loc_x` (local dH) and the block's overall backward
    transformation (`carry_out`).

    Grid: (B * D, num_blocks)
        - Program 0: Indexes batch and dimension (row_id)
        - Program 1: Indexes block (block_id)
    """
    # Identify the specific sequence (batch B, dimension D) and block
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    n_rows = B * D

    b_ = row_id // D
    d_ = row_id % D

    # Initialize running state for the backward recurrence within the block (processed in reverse)
    # (a_run, x_run) represents the transform dH(t) = a_run * dH(end+1) + x_run
    # where dH(end+1) is the gradient flowing *into* the end of the block.
    a_run = 1.0 # Represents cumulative product of 'a's (from right-to-left)
    x_run = 0.0 # Represents the accumulated gradient dH assuming dH(end+1)=0

    # Iterate through the time steps *in reverse* within the assigned block
    t_start = block_id * BLOCK_L
    t_end = tl.minimum(t_start + BLOCK_L, L) # Exclusive end index

    # Loop from (t_end - 1) down to t_start
    for i in tl.static_range(BLOCK_L):
        pos = (t_end - 1) - i # Reverse index: t_end-1, t_end-2, ..., t_start
        if pos >= t_start: # Process only valid indices within the block
            # Calculate memory offset for the current element
            off = b_ * L * D + pos * D + d_

            # Need a(t+1) for the recurrence dH(t) = a(t+1)*dH(t+1) + c(t)*dY(t)
            # Load a(t+1) with boundary check
            a_next_off = b_ * L * D + (pos + 1) * D + d_
            mask_a_next = (pos + 1) < L
            # If pos+1 is out of bounds (i.e., pos = L-1), a(t+1) doesn't exist, conceptually 0.
            a_next = tl.load(a_ptr + a_next_off, mask=mask_a_next, other=0.0)

            # Load other inputs for this time step
            c_i = tl.load(c_ptr + off)
            dY_i = tl.load(dY_ptr + off) # Gradient from upstream dL/dY

            # Apply the backward SSM recurrence relation update
            # dH(i) = a_{i+1} * dH(i+1) + c_i * dY_i
            # We track the transform (a_run, x_run) s.t. dH(i) = a_run * dH(end+1) + x_run
            # Combine (a_next, c_i*dY_i) with previous (a_run, x_run) using reverse aggregator
            # Let L = (a_next, c_i*dY_i), R = (a_run, x_run) [R is from "future"]
            # aggregator_combine(aR, xR, aL, xL) computes aL*aR, aL*xR + xL
            # This represents applying the future transform R, then the current step L.
            a_new, x_new = aggregator_combine(a_run, x_run, a_next, c_i * dY_i)

            # Store the intermediate local results for the backward pass
            tl.store(loc_a_ptr + off, a_new) # Stores cumulative A' for backward to time pos
            tl.store(loc_x_ptr + off, x_new) # Stores local dH at time pos assuming dH(end+1)=0

            # Update the running state for the next iteration (previous time step)
            a_run = a_new
            x_run = x_new

    # After processing the block, store the final (a_run, x_run) as the backward carry-out
    # This represents the backward transformation applied by the entire block.
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L
    # Offset for storing carries (same layout as forward)
    carry_a_off = row_id * n_blocks + block_id
    carry_x_off = n_rows * n_blocks + carry_a_off

    tl.store(carry_out_ptr + carry_a_off, a_run)
    tl.store(carry_out_ptr + carry_x_off, x_run)


@triton.jit
def ssm_backward_carry_scan_kernel(
    carry_io,      # Input/Output: Backward carries (A', dH'), flattened
    n_rows: tl.constexpr,       # B * D
    n_blocks: tl.constexpr,     # Number of blocks
):
    """
    Triton kernel for the parallel scan (prefix sum) on the backward carries.
    Uses `tl.associative_scan` with `reverse=True`. Modifies `carry_io` in-place.
    Each element `k` in the output represents the combined backward transformation
    from block `k` to the end of the sequence (block `n_blocks-1`).

    Grid: (B * D,)
        - Program 0: Indexes batch and dimension (row_id)
    """
    row_id = tl.program_id(0)

    # --- Load backward carries for the current sequence (row_id) ---
    ks = tl.arange(0, n_blocks)
    a_carry_offset = row_id * n_blocks + ks
    x_carry_offset = n_rows * n_blocks + a_carry_offset

    a_in = tl.load(carry_io + a_carry_offset, mask = ks < n_blocks)
    x_in = tl.load(carry_io + x_carry_offset, mask = ks < n_blocks)

    # --- Perform associative scan in reverse ---
    # `aggregator_combine` defines how to combine backward transforms
    # The scan computes the cumulative transform from each block to the end
    a_out, x_out = tl.associative_scan(
        (a_in, x_in), axis=0, combine_fn=aggregator_combine, reverse=True
    )

    # --- Store results back ---
    # `carry_io` now contains the scanned backward carries (cumulative backward transformations)
    tl.store(carry_io + a_carry_offset, a_out, mask = ks < n_blocks)
    tl.store(carry_io + x_carry_offset, x_out, mask = ks < n_blocks)


@triton.jit
def ssm_bwd_apply_kernel(
    # --- Input Pointers ---
    loc_a_ptr,     # Local backward 'a' accumulation from local bwd scan (A'_loc)
    loc_x_ptr,     # Local backward 'dH' state from local bwd scan (dH_loc)
    carry_io_ptr,  # Scanned backward carries (A'_scan, dH'_scan) from bwd carry scan
    x_ptr,         # Original input X
    a_ptr,         # Original state transition A
    b_ptr,         # Original input transition B
    c_ptr,         # Original output projection C (needed for dY term if not pre-multiplied)
    h_ptr,         # Forward hidden state H (recomputed or from ctx)
    dY_ptr,        # Gradient dL/dY (needed for dc calculation)
    # --- Output Pointers ---
    dx_ptr,        # Gradient dL/dX
    da_ptr,        # Gradient dL/dA
    db_ptr,        # Gradient dL/dB
    dc_ptr,        # Gradient dL/dC
    # --- Tensor Shapes ---
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # --- Algorithm Parameters ---
    BLOCK_L: tl.constexpr,
):
    """
    Triton kernel to apply the scanned backward carries and compute final gradients.
    Processes time in reverse order within the block (consistent with backward pass).

    Computes final dH(t) using the scanned carry from block t+1 onwards.
    Then calculates gradients based on this final dH(t) = dL/dh(t):
    dL/dc(t) = dL/dy(t) * dy(t)/dc(t) = dY(t) * h(t)
    dL/dx(t) = dL/dh(t) * dh(t)/dx(t) = dH(t) * b(t)
    dL/da(t) = dL/dh(t) * dh(t)/da(t) = dH(t) * h(t-1)
    dL/db(t) = dL/dh(t) * dh(t)/db(t) = dH(t) * x(t)

    Grid: (B * D, num_blocks)
        - Program 0: Indexes batch and dimension (row_id)
        - Program 1: Indexes block (block_id)
    """
    # Identify the specific sequence (batch B, dimension D) and block
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    n_rows = B * D
    b_ = row_id // D
    d_ = row_id % D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L

    # --- Determine the backward carry-in prefix from the *next* block (future in time) ---
    # The scanned backward carry at index `k` represents the cumulative transform from block `k` to the end.
    # We need the prefix *starting from* block `k+1`.
    # Offset for A' prefix carry: row_id * n_blocks + (block_id + 1)
    carry_a_off_prefix = row_id * n_blocks + (block_id + 1)
    # Offset for dH' prefix carry: n_rows * n_blocks + A'_prefix_offset
    carry_x_off_prefix = n_rows * n_blocks + carry_a_off_prefix

    if block_id == (n_blocks - 1):
        # Last block (first in backward time) has no prefix from future blocks.
        # Gradient flowing in from beyond sequence end is 0. (A'=1, dH'=0 identity transform)
        a_pref = 1.0 # Identity for A' multiplier
        x_pref = 0.0 # Zero for dH' additive term
    else:
        # Load the scanned backward carry from the start of the next block (block_id + 1)
        a_pref = tl.load(carry_io_ptr + carry_a_off_prefix)
        x_pref = tl.load(carry_io_ptr + carry_x_off_prefix)

    # Iterate through the time steps *in reverse* within the assigned block
    t_start = block_id * BLOCK_L
    t_end = tl.minimum(t_start + BLOCK_L, L)

    for i in tl.static_range(BLOCK_L):
        pos = (t_end - 1) - i # Reverse index: t_end-1, ..., t_start
        if pos >= t_start: # Boundary check
            # Calculate memory offset
            off = b_ * L * D + pos * D + d_

            # Load the locally computed backward results for this time step
            a_loc = tl.load(loc_a_ptr + off) # Local cumulative A' backward to time pos
            x_loc = tl.load(loc_x_ptr + off) # Local dH backward at time pos (assuming dH_end=0)

            # Combine the prefix carry (from future blocks k+1 to end) with the local result
            # Prefix transform (future): dH_future = a_pref * dH(end+1) + x_pref
            # Local transform (current): dH_loc    = a_loc * dH(block_end+1) + x_loc
            # We want final dH(t) = combined_a' * dH(end+1) + combined_dh'
            # Where dH(block_end+1) = dH_future
            # final dH(t) = a_loc * dH_future + x_loc
            #             = a_loc * (a_pref * dH(end+1) + x_pref) + x_loc
            #             = (a_loc * a_pref) * dH(end+1) + (a_loc * x_pref + x_loc)
            # Using aggregator_combine(aR, xR, aL, xL) -> (aL*aR, aL*xR + xL)
            # We need combine(prefix, local) => R=prefix=(a_pref, x_pref), L=local=(a_loc, x_loc)
            _, dH = aggregator_combine(a_pref, x_pref, a_loc, x_loc)
            # The second output `dH` is the final dL/dh(t)

            # --- Compute Gradients based on final dH(t) ---

            # Load required values for gradient calculation
            h_val = tl.load(h_ptr + off)
            x_val = tl.load(x_ptr + off)
            b_val = tl.load(b_ptr + off)
            dY_val = tl.load(dY_ptr + off) # Load dL/dY

            # dL/dc(t) = dL/dy(t) * dy(t)/dc(t) = dY(t) * h(t)
            dc_ = dY_val * h_val
            tl.store(dc_ptr + off, dc_)

            # dL/dx(t) = dL/dh(t) * dh(t)/dx(t) = dH(t) * b(t)
            dx_ = dH * b_val
            tl.store(dx_ptr + off, dx_)

            # dL/da(t) = dL/dh(t) * dh(t)/da(t) = dH(t) * h(t-1)
            h_prev = 0.0 # Default for t=0
            if pos > 0:
                # Load h(t-1)
                h_prev_off = b_ * L * D + (pos - 1) * D + d_
                h_prev = tl.load(h_ptr + h_prev_off)
            da_ = dH * h_prev
            tl.store(da_ptr + off, da_)

            # dL/db(t) = dL/dh(t) * dh(t)/db(t) = dH(t) * x(t)
            db_ = dH * x_val
            tl.store(db_ptr + off, db_)


def block_scan_backward_3d(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, x: torch.Tensor, h: torch.Tensor, grad_y: torch.Tensor, BLOCK_L: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs the block scan backward pass for the SSM.

    Computes gradients dL/dx, dL/da, dL/db, dL/dc given dL/dy (grad_y).
    Uses the block scan algorithm with three Triton kernels for efficiency.
    The backward recurrence is: dH(t) = a(t+1)*dH(t+1) + c(t)*grad_y(t), where dH is dL/dh.

    Args:
        a: State transition tensor, shape (B, L, D).
        b: Input transition tensor, shape (B, L, D).
        c: Output projection tensor, shape (B, L, D).
        x: Original input tensor, shape (B, L, D).
        h: Forward hidden states, shape (B, L, D) (computed in fwd pass or recomputed).
        grad_y: Gradient of the loss w.r.t. output y (dL/dy), shape (B, L, D).
        BLOCK_L: Block size for the scan algorithm.

    Returns:
        A tuple (dx, da, db, dc):
        - dx: Gradient dL/dx, shape (B, L, D).
        - da: Gradient dL/da, shape (B, L, D).
        - db: Gradient dL/db, shape (B, L, D).
        - dc: Gradient dL/dc, shape (B, L, D).
    """
    B, L, D = x.shape
    assert a.shape == (B, L, D)
    assert b.shape == (B, L, D)
    assert c.shape == (B, L, D)
    assert h.shape == (B, L, D)
    assert grad_y.shape == (B, L, D)

    # Allocate memory for intermediate results and final gradients
    loc_a_bwd = torch.empty_like(a) # Stores local cumulative A' backward within blocks
    loc_x_bwd = torch.empty_like(a) # Stores local dH backward within blocks (assuming dH_end=0)
    dx = torch.empty_like(x)        # Gradient dL/dx
    da = torch.empty_like(a)        # Gradient dL/da
    db = torch.empty_like(b)        # Gradient dL/db
    dc = torch.empty_like(c)        # Gradient dL/dc

    n_rows = B * D                  # Number of independent sequences
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L # Number of blocks

    # Allocate memory for backward carries (A' and dH' components)
    # Shape: (2, n_rows, n_blocks) -> Flattened for Triton
    carry_out_bwd = torch.empty((2, n_rows, n_blocks), dtype=a.dtype, device=a.device)
    carry_flat_bwd = carry_out_bwd.view(-1) # Flatten

    # --- Kernel 1: Local Backward Scan ---
    # Computes local dH within blocks & the block backward transform. Parallel over B, D, n_blocks.
    grid1 = (n_rows, n_blocks)
    ssm_local_backward_kernel[grid1](
        a, c, grad_y,               # Inputs for recurrence
        loc_a_bwd, loc_x_bwd,       # Outputs: local results
        carry_flat_bwd,             # Output: block transforms (carries)
        B=B, L=L, D=D, BLOCK_L=BLOCK_L
    )

    # --- Kernel 2: Backward Carry Scan ---
    # Performs parallel prefix sum (reversed) on backward block transforms. Parallel over B, D.
    grid2 = (n_rows,)
    ssm_backward_carry_scan_kernel[grid2](
        carry_flat_bwd,             # Input/Output: carries to be scanned
        n_rows=n_rows, n_blocks=n_blocks
    )

    # --- Kernel 3: Apply Backward Carry & Compute Gradients ---
    # Applies scanned carries to local dH, computes final gradients. Parallel over B, D, n_blocks.
    grid3 = (n_rows, n_blocks)
    ssm_bwd_apply_kernel[grid3](
        loc_a_bwd, loc_x_bwd,       # Inputs: local results
        carry_flat_bwd,             # Input: scanned carries
        x, a, b, c, h, grad_y,      # Inputs: original tensors needed for grad computation
        dx, da, db, dc,             # Outputs: final gradients
        B=B, L=L, D=D, BLOCK_L=BLOCK_L
    )

    # Return the computed gradients
    return dx, da, db, dc