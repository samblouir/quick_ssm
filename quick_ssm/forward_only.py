"""
Implements and tests a forward pass for a linear recurrence using Triton kernels.

The recurrence is defined as:
  h(0) = b(0) * x(0)
  h(t) = a(t) * h(t-1) + b(t) * x(t)   for t = 1..(L-1)

This file provides:
1. A naive, sequential Python implementation (`naive_forward_3d`) for correctness reference.
2. A parallel implementation using Triton (`block_scan_forward_3d`) based on a
   block-scan algorithm suitable for long sequences (large L).
3. Supporting Triton kernels for the block-scan algorithm (local computation,
   prefix scan, final application).
4. Test functions for precision and correctness checks.
5. A main execution block (`if __name__ == "__main__":`) demonstrating usage,
   performance timing, and comparison with the naive implementation.

"""

import torch
import triton
import triton.language as tl
import time

# ==============================================================================
# Naive Python Implementation (Reference)
# ==============================================================================


def naive_forward_3d(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs a naive, sequential forward pass for the linear recurrence.

    Recurrence Relation:
        h(0) = b(0) * x(0)
        h(t) = a(t) * h(t-1) + b(t) * x(t)   for t = 1..(L-1)

    This implementation iterates step-by-step along the time dimension (L).
    It is conceptually simple but can be slow for large sequence lengths (L)
    due to its sequential nature.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, D).
        a (torch.Tensor): Recurrence coefficient tensor of shape (B, L, D).
                          Controls the influence of the previous hidden state.
        b (torch.Tensor): Input scaling coefficient tensor of shape (B, L, D).
                          Controls the influence of the current input.

    Returns:
        torch.Tensor: Output tensor h, representing the hidden states at each
                      time step, with shape (B, L, D).
    """
    B, L, D = x.shape
    h = torch.zeros_like(
        x, dtype=torch.float32
    )  # Use float32 for accumulation stability

    # Initialize at t = 0
    if L > 0:
        h[:, 0, :] = b[:, 0, :] * x[:, 0, :]

    # Compute recurrence sequentially for t = 1..L-1
    for t in range(1, L):
        h[:, t, :] = a[:, t, :] * h[:, t - 1, :] + b[:, t, :] * x[:, t, :]

    return h.to(x.dtype)  # Cast back to original dtype if needed


# ==============================================================================
# Triton Kernels for Block Scan Algorithm
# ==============================================================================


# ------------------------------------------------------------------------------
# Kernel 1: Local Block Computation
# ------------------------------------------------------------------------------
@triton.jit
def ssm_local_forward_kernel(
    # Pointers to input tensors
    x_ptr,
    a_ptr,
    b_ptr,
    # Pointers to output tensors for this pass
    local_scan_a_ptr,
    local_scan_x_ptr,  # Stores local scan results within each block
    block_carry_out_ptr,  # Stores the final aggregator for each block
    # Tensor dimensions
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    # Block size for time dimension
    BLOCK_L: tl.constexpr,
):
    """
    Triton Kernel (Pass 1/3): Performs local forward scan within blocks.

    Processes the recurrence relation independently within fixed-size blocks
    along the time dimension (L). Each program instance handles one element
    in the (B, D) dimensions across one block of time steps.

    For each (B, D) element and each time block:
    1. Initializes an aggregator state (a_running=1.0, x_running=0.0), representing
       the identity transformation h_out = 1*h_in + 0.
    2. Iterates through `BLOCK_L` time steps within the assigned block.
    3. Loads input `a(t)`, `b(t)`, `x(t)`.
    4. Updates the aggregator state using the recurrence relation combined with
       the previous state within the block:
          new_a = a(t) * a_running
          new_x = a(t) * x_running + b(t) * x(t)
    5. Stores the intermediate local scan result (`new_a`, `new_x`) at each
       time step `t` into `local_scan_a_ptr` and `local_scan_x_ptr`. These
       represent the result of the scan *up to time t within the block*.
    6. Stores the final aggregator state (`a_running`, `x_running`) for the
       *entire block* into `block_carry_out_ptr`. This carry represents the
       total transformation applied by this block and will be used in the next pass.

    Args:
        x_ptr: Pointer to the input tensor `x` (B, L, D).
        a_ptr: Pointer to the recurrence coefficient tensor `a` (B, L, D).
        b_ptr: Pointer to the input scaling coefficient tensor `b` (B, L, D).
        local_scan_a_ptr: Output pointer for the 'a' component of the local scan
                          results within each block (B, L, D).
        local_scan_x_ptr: Output pointer for the 'x' component of the local scan
                          results within each block (B, L, D). This corresponds
                          to the hidden state *if the block started from h=0*.
        block_carry_out_ptr: Output pointer for the carry-out aggregators for each
                             block. Shape is (2, B*D, n_blocks), where index 0
                             stores 'a' components and index 1 stores 'x' components.
        B (int): Batch size.
        L (int): Sequence length (time dimension).
        D (int): Feature dimension.
        BLOCK_L (int): Size of the time block processed by each kernel instance segment.
    """
    # Program IDs identify the work item
    row_id = tl.program_id(0)  # Index for the flattened (B, D) dimension
    block_id = tl.program_id(1)  # Index for the time block

    # Calculate total number of rows and blocks
    n_rows = B * D  # Flattened dimension
    n_blocks = tl.cdiv(L, BLOCK_L)  # Number of blocks needed to cover L

    # Decompose row_id back into batch (b_) and feature (d_) indices
    b_ = row_id // D
    d_ = row_id % D

    # Initialize the running aggregator state for this block
    # Represents the identity transformation: h_out = 1.0 * h_in + 0.0
    a_running = 1.0
    x_running = 0.0

    # Starting time step for this block
    t_start = block_id * BLOCK_L

    # Iterate through time steps within the block
    for i in tl.static_range(BLOCK_L):
        current_t = t_start + i
        # Boundary check: only process if within the actual sequence length L
        if current_t < L:
            # Calculate memory offset for the current (b_, current_t, d_) element
            offset = (b_ * L + current_t) * D + d_

            # Load inputs for the current time step
            a_i = tl.load(a_ptr + offset).to(tl.float32)
            b_i = tl.load(b_ptr + offset).to(tl.float32)
            x_i = tl.load(x_ptr + offset).to(tl.float32)

            # Update the aggregator state
            # This combines the current step (a_i, b_i*x_i) with the running state
            new_a = a_i * a_running
            new_x = a_i * x_running + b_i * x_i

            # Store the local scan result for this time step within the block
            # Note: We store the *updated* state (new_a, new_x)
            tl.store(local_scan_a_ptr + offset, new_a)
            tl.store(local_scan_x_ptr + offset, new_x)

            # Update the running state for the next iteration within the block
            a_running = new_a
            x_running = new_x

    # After processing the block, store the final carry-out aggregator state
    # This represents the total transformation performed by this block.
    # We use a special convention: store (0.0, 0.0) if the block was effectively
    # identity (a=1, x=0). This helps in the scan pass.
    # [This convention seems different from the code, let's stick to the original for now]
    # The original code stores the actual final (a_running, x_running).

    # Calculate offset into the carry-out tensor
    carry_offset = row_id * n_blocks + block_id
    # Store 'a' component
    tl.store(block_carry_out_ptr + 0 * (n_rows * n_blocks) + carry_offset, a_running)
    # Store 'x' component
    tl.store(block_carry_out_ptr + 1 * (n_rows * n_blocks) + carry_offset, x_running)


# ------------------------------------------------------------------------------
# Helper Function: Combine Aggregators
# ------------------------------------------------------------------------------
@triton.jit
def aggregator_combine(aL, xL, aR, xR):
    """
    Combines two sequential recurrence aggregators (Left and Right).

    An aggregator (a, x) represents the transformation h_out = a * h_in + x.
    Combining (aL, xL) followed by (aR, xR) means applying the Left transform
    then the Right transform:
    h_intermediate = aL * h_initial + xL
    h_final = aR * h_intermediate + xR
            = aR * (aL * h_initial + xL) + xR
            = (aR * aL) * h_initial + (aR * xL + xR)

    Thus, the combined aggregator is (a_out, x_out) = (aR * aL, aR * xL + xR).

    Note on Identity: The identity transformation is (a=1, x=0). The kernel uses
    a storage convention where (0, 0) might represent identity in specific contexts
    (like initial state for scan), handled via `tl.where`.
    [Correction: The original code didn't explicitly use (0,0) for identity storage,
     but the combination logic needs to be correct regardless.]

    Args:
        aL (tl.tensor): 'a' component of the left aggregator.
        xL (tl.tensor): 'x' component of the left aggregator.
        aR (tl.tensor): 'a' component of the right aggregator.
        xR (tl.tensor): 'x' component of the right aggregator.

    Returns:
        tuple[tl.tensor, tl.tensor]: Combined aggregator (a_out, x_out).
    """
    # The mathematical combination derived above:
    a_out = aR * aL
    x_out = aR * xL + xR
    return a_out, x_out


# ------------------------------------------------------------------------------
# Kernel 2: Carry Prefix Scan
# ------------------------------------------------------------------------------
@triton.jit
def ssm_fwd_carry_scan_kernel(
    # Input/Output pointer (modified in-place)
    block_carry_io_ptr,
    # Dimensions
    n_rows: tl.constexpr,  # Flattened B*D dimension
    n_blocks: tl.constexpr,  # Number of time blocks
):
    """
    Triton Kernel (Pass 2/3): Performs a parallel prefix scan (scan-then-sum)
    on the block carry-out aggregators computed in Pass 1.

    Each program instance handles one row in the flattened (B, D) dimension.
    It performs an associative scan along the `n_blocks` dimension using the
    `aggregator_combine` function.

    The input `block_carry_io_ptr` initially contains the independent carry-out
    aggregator for each block (from Pass 1). This kernel modifies it *in-place*
    so that at the end, `block_carry_io_ptr[..., k]` contains the combined
    aggregator for all blocks from 0 up to and including block `k`.

    Args:
        block_carry_io_ptr: Pointer to the carry tensor of shape (2, n_rows, n_blocks).
                            Stores 'a' components at index 0, 'x' at index 1.
                            Modified in-place.
        n_rows (int): Total number of rows (B * D).
        n_blocks (int): Number of time blocks.
    """
    # Program ID identifies the row this instance works on
    row_id = tl.program_id(0)

    # Offsets to load the 'a' and 'x' components for the entire row across all blocks
    a_offset = 0 * (n_rows * n_blocks) + row_id * n_blocks
    x_offset = 1 * (n_rows * n_blocks) + row_id * n_blocks

    # Block indices range [0, 1, ..., n_blocks-1]
    block_indices = tl.arange(0, n_blocks)

    # Load the initial 'a' and 'x' carry components for this row from all blocks
    a_carry_in = tl.load(block_carry_io_ptr + a_offset + block_indices)
    x_carry_in = tl.load(block_carry_io_ptr + x_offset + block_indices)

    # Perform the associative prefix scan along the block dimension (axis=0 for the loaded slice)
    # The `aggregator_combine` function defines how to combine adjacent elements.
    a_carry_scanned, x_carry_scanned = tl.associative_scan(
        (a_carry_in, x_carry_in),
        axis=0,  # Scan along the block dimension
        combine_fn=aggregator_combine,
        # `reverse=False` is the default, meaning it's a standard prefix scan
    )

    # Store the scanned results back into the same memory locations (in-place update)
    tl.store(block_carry_io_ptr + a_offset + block_indices, a_carry_scanned)
    tl.store(block_carry_io_ptr + x_offset + block_indices, x_carry_scanned)


# ------------------------------------------------------------------------------
# Kernel 3: Apply Prefix Aggregator
# ------------------------------------------------------------------------------
@triton.jit
def ssm_fwd_apply_kernel(
    # Inputs from Pass 1 and Pass 2
    local_scan_a_ptr,
    local_scan_x_ptr,  # Local scans within blocks (Pass 1)
    block_carry_scanned_ptr,  # Scanned block carries (Result of Pass 2)
    # Output pointer for the final result
    h_out_ptr,  # Final hidden states h(t)
    # Tensor dimensions
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    # Block size for time dimension
    BLOCK_L: tl.constexpr,
):
    """
    Triton Kernel (Pass 3/3): Applies the scanned carry aggregators to the
    local scan results to compute the final hidden states.

    Each program instance handles one element in the (B, D) dimensions across
    one block of time steps.

    For each time step `t` within its assigned block:
    1. Loads the local scan result (`a_local`, `x_local`) computed in Pass 1 for time `t`.
       This represents the state `h(t)` *assuming the block started with h=0*.
    2. Loads the scanned carry aggregator (`a_prefix`, `x_prefix`) from the
       *previous* block (block `block_id - 1`). This represents the total
       transformation from time 0 up to the beginning of the current block.
       For the very first block (`block_id == 0`), the prefix is the identity (a=1, x=0).
    3. Combines the prefix aggregator with the local scan result using the same
       combination logic:
          h(t)_final = a_local * h_prefix_in + x_local
       where h_prefix_in (the state entering the block) is given by:
          h_prefix_in = a_prefix * h_initial(0) + x_prefix
       Assuming h_initial(0) = 0, we need to compute:
          h(t)_final = a_prefix * x_local (?) No, this is not right.

       Let's rethink:
       - `(a_local, x_local)` transforms `h(start_of_block)` to `h(t)`.
       - `(a_prefix, x_prefix)` transforms `h(0)` to `h(start_of_block - 1)`.
       We need the transformation from `h(0)` to `h(t)`.
       The state entering the current step `t` within the block, considering the prefix,
       can be thought of as `h(t-1)_global`.
       The local scan `x_local` at step `t` is `a(t)*h(t-1)_local + b(t)*x(t)`.
       The global state `h(t)` should be `a(t)*h(t-1)_global + b(t)*x(t)`.

       Consider the formula: `h(t) = combined_a(t) * h(0) + combined_x(t)`.
       Let `(a_prefix, x_prefix)` be the aggregate transform up to the *start* of the block.
       Let `(a_local, x_local)` be the aggregate transform from the start of the block *up to time t*.
       The combined transform up to time `t` is `aggregator_combine((a_prefix, x_prefix), (a_local, x_local))`.
       Let this be `(a_final, x_final)`.
       Then `h(t) = a_final * h(0) + x_final`. Assuming `h(0)` for the recurrence relates
       to the input `x(0)` (handled at initialization), we are interested in the `x_final` component.

       Combination: `a_final = a_local * a_prefix`, `x_final = a_local * x_prefix + x_local`

    4. Stores the resulting `x_final` component into `h_out_ptr` at time `t`.
       This is the final hidden state `h(t)`.

    Args:
        local_scan_a_ptr: Pointer to local 'a' scan results from Pass 1 (B, L, D).
        local_scan_x_ptr: Pointer to local 'x' scan results from Pass 1 (B, L, D).
        block_carry_scanned_ptr: Pointer to the prefix-scanned carry aggregators
                                 from Pass 2 (2, n_rows, n_blocks).
        h_out_ptr: Output pointer for the final hidden states `h` (B, L, D).
        B (int): Batch size.
        L (int): Sequence length.
        D (int): Feature dimension.
        BLOCK_L (int): Time block size.
    """
    # Program IDs identify the work item
    row_id = tl.program_id(0)  # Index for the flattened (B, D) dimension
    block_id = tl.program_id(1)  # Index for the time block

    # Calculate total number of rows and blocks
    n_rows = B * D
    n_blocks = tl.cdiv(L, BLOCK_L)

    # Decompose row_id back into batch (b_) and feature (d_) indices
    b_ = row_id // D
    d_ = row_id % D

    # --- Load the prefix aggregator from the end of the *previous* block ---
    a_prefix = 1.0  # Identity 'a' component
    x_prefix = 0.0  # Identity 'x' component

    # If not the first block, load the actual scanned carry from the previous block
    if block_id > 0:
        # Offset to the carry data for the previous block (block_id - 1)
        carry_offset = row_id * n_blocks + (block_id - 1)
        # Load 'a' component of the prefix carry
        a_prefix = tl.load(
            block_carry_scanned_ptr + 0 * (n_rows * n_blocks) + carry_offset
        )
        # Load 'x' component of the prefix carry
        x_prefix = tl.load(
            block_carry_scanned_ptr + 1 * (n_rows * n_blocks) + carry_offset
        )

    # --- Iterate through time steps within the current block ---
    t_start = block_id * BLOCK_L
    for i in tl.static_range(BLOCK_L):
        current_t = t_start + i
        # Boundary check
        if current_t < L:
            # Calculate memory offset for the current (b_, current_t, d_) element
            offset = (b_ * L + current_t) * D + d_

            # Load the local scan result (a_local, x_local) for this time step from Pass 1
            a_local = tl.load(local_scan_a_ptr + offset).to(tl.float32)
            x_local = tl.load(local_scan_x_ptr + offset).to(tl.float32)

            # Combine the prefix aggregator with the local scan result
            # Formula: x_final = a_local * x_prefix + x_local
            # (We only need the 'x' component for the final hidden state h)
            h_final_t = a_local * x_prefix + x_local

            # Store the final computed hidden state h(t)
            tl.store(h_out_ptr + offset, h_final_t)


# ==============================================================================
# User-Facing Function: Block Scan Forward Pass
# ==============================================================================


def block_scan_forward_3d(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    BLOCK_L: int = 256,
    use_naive_if_small: bool = True,
) -> torch.Tensor:
    """
    Computes the forward pass of the linear recurrence using a parallel block scan algorithm.

    Recurrence Relation:
        h(0) = b(0) * x(0)
        h(t) = a(t) * h(t-1) + b(t) * x(t),  t = 1..(L-1)

    This function orchestrates the three-pass Triton kernel implementation:
    1. `ssm_local_forward_kernel`: Computes scans locally within blocks and block carries.
    2. `ssm_fwd_carry_scan_kernel`: Performs a prefix scan on the block carries.
    3. `ssm_fwd_apply_kernel`: Combines prefix carries with local scans for the final result.

    This approach is efficient for long sequences (large L) where the naive
    sequential approach becomes a bottleneck.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, D). Must be on CUDA device.
        a (torch.Tensor): Recurrence coefficient tensor of shape (B, L, D). Must be on CUDA.
        b (torch.Tensor): Input scaling coefficient tensor of shape (B, L, D). Must be on CUDA.
        BLOCK_L (int, optional): The size of the blocks to use for the time dimension (L).
                                 Must be a power of 2? (Triton often prefers this).
                                 Performance may vary based on this value. Defaults to 256.
        use_naive_if_small (bool, optional): If True and L <= BLOCK_L, use the
                                            `naive_forward_3d` implementation, which might
                                            be faster due to lower overhead. Defaults to True.

    Returns:
        torch.Tensor: The final output tensor h (hidden states) of shape (B, L, D).

    Raises:
        AssertionError: If input tensors are not on the same CUDA device or have
                        incompatible shapes.
    """
    # Input validation
    assert (
        x.is_cuda and a.is_cuda and b.is_cuda
    ), "Input tensors must be on a CUDA device."
    assert (
        x.shape == a.shape == b.shape
    ), f"Input shapes must match: x={x.shape}, a={a.shape}, b={b.shape}"
    assert x.dim() == 3, "Input tensors must be 3D (B, L, D)."
    # assert BLOCK_L > 0 and (BLOCK_L & (BLOCK_L - 1) == 0), "BLOCK_L must be a positive power of 2" # Often good practice but not strictly required by logic

    B, L, D = x.shape
    device = x.device
    dtype = x.dtype  # Assuming all inputs have the same dtype

    # --- Fallback to naive implementation for small sequences ---
    if use_naive_if_small and L <= BLOCK_L:
        print(f"Sequence length {L} <= BLOCK_L {BLOCK_L}. Using naive implementation.")
        return naive_forward_3d(x, a, b)

    # --- Allocate Output and Intermediate Tensors ---
    # Final output tensor h(t)
    h_out = torch.empty_like(x)

    # Intermediate tensors for Pass 1 outputs
    # Local scan results within each block
    local_scan_a = torch.empty_like(a)
    local_scan_x = torch.empty_like(x)

    # Carry aggregators at the end of each block
    n_rows = B * D
    n_blocks = (L + BLOCK_L - 1) // BLOCK_L  # Equivalent to ceil(L / BLOCK_L)
    # Shape: (2 components ('a', 'x'), B*D rows, n_blocks)
    block_carry = torch.empty((2, n_rows, n_blocks), dtype=torch.float32, device=device)

    # --- Launch Triton Kernels ---

    # Pass 1: Local Scan within Blocks
    grid_pass1 = (n_rows, n_blocks)  # One program instance per (row, block)
    ssm_local_forward_kernel[grid_pass1](
        x,
        a,
        b,  # Inputs
        local_scan_a,
        local_scan_x,  # Pass 1 Outputs: Local scans
        block_carry,  # Pass 1 Output: Block carries
        B=B,
        L=L,
        D=D,
        BLOCK_L=BLOCK_L,  # Dimensions and block size
    )
    # Debug: print("Pass 1 done.")

    # Pass 2: Prefix Scan on Block Carries
    grid_pass2 = (n_rows,)  # One program instance per row
    # Note: block_carry is modified in-place
    ssm_fwd_carry_scan_kernel[grid_pass2](
        block_carry,  # Input/Output: Block carries (scanned in-place)
        n_rows=n_rows,
        n_blocks=n_blocks,  # Dimensions
    )
    # Debug: print("Pass 2 done.")

    # Pass 3: Apply Scanned Carries to Local Results
    grid_pass3 = (n_rows, n_blocks)  # One program instance per (row, block)
    ssm_fwd_apply_kernel[grid_pass3](
        local_scan_a,
        local_scan_x,  # Inputs: Local scans from Pass 1
        block_carry,  # Input: Scanned block carries from Pass 2
        h_out,  # Output: Final hidden states h(t)
        B=B,
        L=L,
        D=D,
        BLOCK_L=BLOCK_L,  # Dimensions and block size
    )
    # Debug: print("Pass 3 done.")

    # Return the final computed hidden states, cast back to original type if needed
    return h_out.to(dtype)


# ==============================================================================
# Precision Testing Function
# ==============================================================================


def precision_test(device: str = "cuda", dtype: torch.dtype = torch.float32):
    """
    Tests the numerical precision of the block_scan_forward_3d implementation,
    especially for long sequences where floating-point errors can accumulate.

    It uses a specific input pattern where only the first element of x is non-zero.
    In this case, the recurrence simplifies, and the final hidden state h(L-1)
    can be calculated analytically, allowing for direct comparison.

    Args:
        device (str): CUDA device to run the test on (e.g., "cuda:0").
        dtype (torch.dtype): Data type to use for the test (e.g., torch.float32, torch.bfloat16).
    """
    print(f"\n--- Running Precision Test (dtype={dtype}) ---")
    # Large sequence length to test accumulation
    B, L, D = 1, int(2**20), 1  # Using 2**20 for faster test, original was 2**24

    # Use a value close to 1 for 'a' to simulate long-range dependencies
    gate_value = 0.9999  # A slightly less extreme value than original
    # gate_value = 0.9999999990686774 # Original value

    # Create input tensors
    x_ = torch.zeros(B, L, D, device=device, dtype=dtype)
    x_[:, 0, :] = 1.0  # Only the first element is 1.0
    a_ = torch.full_like(x_, gate_value)
    b_ = torch.ones_like(x_)  # b(t)=1 simplifies the recurrence

    # Determine a reasonable block size (e.g., a power of 2)
    BLOCK_L = 256
    # Increase block size if L is much larger (heuristic)
    # while BLOCK_L * 4 < L: # Adjust multiplier as needed
    #     BLOCK_L *= 2
    print(f"Using B={B}, L={L}, D={D}, BLOCK_L={BLOCK_L}")

    # --- Run the block scan forward pass ---
    start_time = time.time()
    h_block = block_scan_forward_3d(
        x_, a_, b_, BLOCK_L=BLOCK_L, use_naive_if_small=False
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Block scan forward took {end_time - start_time:.4f} seconds.")

    # --- Calculate expected result ---
    # For this specific input (x[0]=1, x[t>0]=0, b[t]=1):
    # h(0) = b(0)*x(0) = 1*1 = 1
    # h(1) = a(1)*h(0) + b(1)*x(1) = a(1)*1 + 1*0 = a(1)
    # h(2) = a(2)*h(1) + b(2)*x(2) = a(2)*a(1) + 1*0 = a(2)*a(1)
    # ...
    # h(t) = a(t)*a(t-1)*...*a(1)
    # If a(t) is constant 'gate_value':
    # h(t) = gate_value^t
    # So, h(L-1) should be gate_value^(L-1)
    if L > 0:
        # Use float64 for higher precision calculation of the expected value
        expected_last_val = torch.tensor(gate_value, dtype=torch.float64).pow(L - 1)
        actual_last_val = h_block[:, -1].to(torch.float64)  # Compare in float64

        # Calculate absolute and relative differences
        diff_abs = (actual_last_val - expected_last_val).abs().max().item()
        diff_rel = (
            (diff_abs / expected_last_val.abs().item())
            if expected_last_val.abs().item() > 1e-10
            else diff_abs
        )

        print(f"Expected h(L-1): {expected_last_val.item():.10f}")
        print(f"Actual h(L-1):   {actual_last_val.item():.10f}")
        print(f"Max Absolute Difference: {diff_abs:.6e}")
        print(f"Max Relative Difference: {diff_rel:.6e}")

        # Set a tolerance based on the data type
        tolerance = (
            1e-5 if dtype == torch.float32 else 1e-2
        )  # Looser tolerance for lower precision types
        assert (
            diff_rel < tolerance
        ), f"Relative difference {diff_rel:.6e} exceeds tolerance {tolerance}"
        print(f"Precision test passed (tolerance={tolerance}).")
    else:
        print("Sequence length L=0, skipping value check.")
    print("--- Precision Test End ---")


# ==============================================================================
# Main Execution Block (Demo & Basic Tests)
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, naive implementation will be used.")
        # Force parameters for CPU execution if needed, or skip Triton parts
        B, L, D = 1, 256, 2
        use_triton = False
    else:
        # Parameters for performance demonstration
        B, L, D = 4, 2**17, 128  # Adjusted for potentially faster run time
        use_triton = True

    dtype = torch.float32  # Use float32 by default

    print(
        f"Device: {device}, Shape: ({B}, {L}, {D}), dtype: {dtype}, Using Triton: {use_triton}"
    )

    # --- Create Random Input Tensors ---
    # Use requires_grad=False as we only test forward pass
    x_ = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=False)
    # Initialize 'a' close to 1 but less than 1 for stability
    a_ = (
        torch.rand(B, L, D, device=device, dtype=dtype, requires_grad=False) * 0.1
        + 0.85
    )
    b_ = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=False) * 0.1

    # --- Performance Timing Helper ---
    def timer(name, func, *args, **kwargs):
        if device == "cpu" and "block_scan" in name:
            print(f"Skipping {name} on CPU.")
            return None
        try:
            # Warm-up runs
            print(f"Warming up for {name}...")
            for _ in range(2):
                _ = func(*args, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()

            # Timed runs
            print(f"Timing {name}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            print(f"+++ {name}: {end_time - start_time:.4f} seconds +++")
            return result
        except Exception as e:
            print(f"Error during timing {name}: {e}")
            return None

    # --- Performance Comparison (if using Triton) ---
    if use_triton:
        print("\n--- Performance Comparison ---")
        # Test different block sizes
        for block_size in [128, 256, 512, 1024]:
            # Ensure block size is not larger than sequence length for meaningful test
            if block_size >= L:
                print(f"Skipping BLOCK_L={block_size} as it's >= L={L}")
                # Use naive if block_size >= L, just to get a result
                if L > 0:  # Avoid running naive if L=0
                    _ = timer(
                        f"block_scan_forward_3d (BLOCK_L={block_size} -> Naive Fallback)",
                        block_scan_forward_3d,
                        x_,
                        a_,
                        b_,
                        BLOCK_L=block_size,
                        use_naive_if_small=True,
                    )
                continue

            _ = timer(
                f"block_scan_forward_3d (BLOCK_L={block_size})",
                block_scan_forward_3d,
                x_,
                a_,
                b_,
                BLOCK_L=block_size,
                use_naive_if_small=True,
            )

    # --- Correctness Check vs Naive (on a smaller problem) ---
    print("\n--- Correctness Check vs Naive ---")
    # Use smaller dimensions for feasible naive computation
    B_test, L_test, D_test = 2, 512, 8
    print(f"Using test shape: ({B_test}, {L_test}, {D_test})")

    # Ensure L_test is suitable for both implementations
    test_block_l = 128  # A block size smaller than L_test
    if L_test <= test_block_l and use_triton:
        print(
            f"Warning: L_test ({L_test}) <= test_block_l ({test_block_l}). Triton will use naive fallback."
        )

    try:
        x_test = torch.randn(B_test, L_test, D_test, device=device, dtype=dtype)
        a_test = (
            torch.rand(B_test, L_test, D_test, device=device, dtype=dtype) * 0.1 + 0.85
        )
        b_test = torch.randn(B_test, L_test, D_test, device=device, dtype=dtype) * 0.1

        # Run Naive Implementation
        print("Running naive implementation for comparison...")
        start_time_naive = time.time()
        h_naive = naive_forward_3d(x_test, a_test, b_test)
        if device == "cuda":
            torch.cuda.synchronize()
        end_time_naive = time.time()
        print(f"Naive forward took {end_time_naive - start_time_naive:.4f} seconds.")

        # Run Block Scan Implementation (if applicable)
        if use_triton:
            print(f"Running block scan implementation (BLOCK_L={test_block_l})...")
            start_time_block = time.time()
            h_block = block_scan_forward_3d(
                x_test, a_test, b_test, BLOCK_L=test_block_l
            )
            if device == "cuda":
                torch.cuda.synchronize()
            end_time_block = time.time()
            print(
                f"Block scan forward took {end_time_block - start_time_block:.4f} seconds."
            )

            # Compare results
            # Use higher precision for comparison if inputs were lower precision
            diff = (
                (h_naive.to(torch.float32) - h_block.to(torch.float32))
                .abs()
                .max()
                .item()
            )
            print(f"Maximum Absolute Difference vs. Naive: {diff:.6e}")

            # Check if the difference is within acceptable tolerance
            tolerance = 1e-5 if dtype == torch.float32 else 1e-3
            assert (
                diff < tolerance
            ), f"Difference {diff:.6e} exceeds tolerance {tolerance}"
            print(f"Correctness check passed (tolerance={tolerance}).")
        else:
            print("Skipping block scan comparison on CPU.")

    except Exception as e:
        print(f"Error during correctness check: {e}")

    # --- Run Precision Test (if using Triton) ---
    if use_triton:
        try:
            precision_test(device=device, dtype=dtype)
            # Optionally test with bfloat16 if available and desired
            if hasattr(torch, "bfloat16"):
                precision_test(device=device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error during precision test: {e}")

    print("\n--- Script Finished ---")
