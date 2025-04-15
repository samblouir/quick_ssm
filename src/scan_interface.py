'''
This contains an interface for Torch to use the Triton kernel for the SSM scan.

Please see ../example.py for an example of how to use this.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional
from quick_ssm.src.triton_backwards import block_scan_backward_3d
from quick_ssm.src.triton_forwards import block_scan_forward_3d


class ScanFunctionCheckpoint(torch.autograd.Function):
    """
    A custom function that does y = c*h.
    """

    @staticmethod
    def forward(ctx, x, a, b, c, BLOCK_L=256, checkpoint=True):
        """
        x,a,b,c => shape (B,L,D).
        returns y => shape (B,L,D).
        - If checkpoint=False, we store h in ctx for direct use in backward.
        - If checkpoint=True, we do NOT store h; we re-run forward in backward.
        """
        ctx.checkpoint = checkpoint
        ctx.BLOCK_L = BLOCK_L

        ctx.save_for_backward(x, a, b, c)

        out_a, out_x = block_scan_forward_3d(x, a, b, BLOCK_L=BLOCK_L)
        h = out_x
        y = c * h

        if not checkpoint:
            ctx.save_for_backward(x, a, b, c, h)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output => shape (B,L,D)
        aggregator => dH(t)= a(t+1)*dH(t+1)+ c(t)* grad_output[t]
        Returns dx, da, db, dc
        """
        saved = ctx.saved_tensors
        BLOCK_L = ctx.BLOCK_L
        checkpoint = ctx.checkpoint

        if checkpoint:
            raise NotImplementedError(
				"Checkpointing is not yet implemented."
			)
            x, a, b, c = saved
            h = block_scan_forward_recompute(x, a, b, BLOCK_L=BLOCK_L)
        else:
            x, a, b, c, h = saved

        dx, da, db, dc = block_scan_backward_3d(
            a, b, c, x, h, grad_output, BLOCK_L=BLOCK_L
        )

        return dx, da, db, dc, None, None

# TODO: torch.compile may be able to be removed now
@torch.compile(disable=True)
def scan(a, x, b, c, BLOCK_L=512, checkpoint=True):
    """
    A user-level function that calls:
        y= ssm_scan(x,a,b,c, checkpoint=True)
    for gradient checkpointing if desired.
    """
    return ScanFunctionCheckpoint.apply(x, a, b, c, BLOCK_L, checkpoint)

if __name__ == "__main__":
	# Example usage
	B, L, D = 2, 5, 3
	x = torch.randn(B, L, D, requires_grad=True)
	a = torch.randn(B, L, D, requires_grad=True)
	b = torch.randn(B, L, D, requires_grad=True)
	c = torch.randn(B, L, D, requires_grad=True)

	y = scan(x, a, b, c)
	print("Output y:", y)
     