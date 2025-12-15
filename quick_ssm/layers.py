
import torch
import torch.nn as nn
from typing import Optional

from quick_ssm.scan_interface import scan


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, dtype=torch.float32, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * (self.weight / norm)


class SSM(nn.Module):
    """
    Compact SSM layer that wraps the Triton scan.

    Notes on stability:
        * Start training in fp32 (compute_dtype) for the first few thousand steps
          if you see early NaNs; switch to fp16/bf16 later for speed.
        * Set `checkpoint=True` (default) to save VRAM at the cost of extra compute.
        * Use `tile_b`/`tile_d` to further bound activation memory when B or D is big.
    """

    def __init__(
        self,
        hidden_size: int,
        state_size_mult: float = 1.0,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size_dim = int(hidden_size * state_size_mult)
        self.compute_dtype = compute_dtype

        # Projections
        self.A_proj = nn.Linear(hidden_size, self.state_size_dim, bias=True, device=device, dtype=dtype)
        self.B_proj = nn.Linear(hidden_size, self.state_size_dim, bias=False, device=device, dtype=dtype)
        self.C_proj = nn.Linear(hidden_size, self.state_size_dim, bias=False, device=device, dtype=dtype)
        self.D_proj = nn.Linear(hidden_size, self.state_size_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.state_size_dim, hidden_size, bias=True, device=device, dtype=dtype)

        self.pre_norm = RMSNorm(hidden_size, eps=eps, dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        block_l: int = 256,
        checkpoint: bool = True,
        tile_b: Optional[int] = None,
        tile_d: Optional[int] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        if (L & (L - 1)) != 0:
            raise ValueError("Sequence length L must currently be a power of 2.")

        residual = x
        x_norm = self.pre_norm(x)

        a = torch.sigmoid(self.A_proj(x_norm))              # [B, L, state_dim]
        b = self.B_proj(x_norm)                             # input transform
        c = torch.sigmoid(self.C_proj(x_norm))              # output gate
        d = torch.sigmoid(self.D_proj(x_norm))              # input gate

        # Cast to compute dtype for the kernel; outputs will be cast back later
        a = a.to(self.compute_dtype)
        b = b.to(self.compute_dtype)
        c = c.to(self.compute_dtype)
        d = d.to(self.compute_dtype)

        h = scan(
            x=b,  # treat projected input as x in the recurrence
            a=a,
            b=d,
            c=c,
            block_l=block_l,
            checkpoint=checkpoint,
            tile_b=tile_b,
            tile_d=tile_d,
            out_dtype=self.compute_dtype,
        )

        y = self.out_proj(h.to(self.out_proj.weight.dtype))
        return y + residual


__all__ = ["SSM", "RMSNorm"]

