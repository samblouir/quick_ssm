
###############################################################################
#                   Triton-Based Block-Scan Implementation                    #
###############################################################################

#
# 1) aggregator_combine(...) and aggregator_combine_rev(...) define how
#    partial aggregates are merged in forward/backward scans.
#
#    For the forward aggregator, (a, x) represents:
#      a(t) = product of all a up to time t
#      x(t) = resulting partial sum for h(t)
#    So the combination is a standard "prefix" operation for SSM recurrences:
#      (aL, xL) combined with (aR, xR) => (aR * aL, aR * xL + xR).
#
# 2) The local_*_kernel(...) Triton kernels process a single block of the sequence
#    to compute partial (a, x) aggregator values.  This is done for each block in parallel.
#
# 3) The carry_*_scan_kernel(...) then merges these partial aggregates block-by-block
#    using an associative scan across blocks, giving a final aggregator for each block.
#
# 4) The *_apply_kernel(...) merges the carry aggregator back into each block
#    so that each element h(t) can be obtained correctly.
#


@triton.jit
def aggregator_combine(aL, xL, aR, xR):
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
def aggregator_combine_rev(aR, xR, aL, xL):
    """
    Combine two backward aggregator tuples (aR, xR) and (aL, xL).

    This is the reversed version for backward pass.

    Represents:
        new_a = aL * aR
        new_x = aL * xR + xL

    Returns
    -------
    (float, float)
        The combined aggregator tuple (a_out, x_out).
    """
    a_out = aL * aR
    x_out = aL * xR + xL
    return a_out, x_out

