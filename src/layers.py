
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dtype=torch.float32, device=None):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * (self.weight / norm)
    

class SSM(nn.Module):
    """
    A recurrent SSM
    """

    def __init__(self, **kwargs):
        
        """
        Initializes the SSM layer components.

        Args:
            hidden_size: Dimension of the input and output features.
            state_size_mult: Multiplier to determine the internal state size relative
                             to hidden_size.
            eps: Epsilon value for RMSNorm layers.
            dtype: Data type for layers.
            device: Device for layers.
            **kwargs: Additional arguments passed to LinearProjection layers.
        """
        
        super().__init__()
        hidden_size = kwargs["hidden_size"]
        state_size_mult = kwargs.get("state_size_mult", 1.0)

        state_size_dim = int(hidden_size * state_size_mult)

        A_proj_kwargs = {
            **kwargs,
            "projection_layers_use_bias": True,
        }
        self.A_proj = nn.LinearProjection(hidden_size, state_size_dim, **A_proj_kwargs)
        self.x_proj = nn.LinearProjection(hidden_size, state_size_dim, **kwargs)
        self.sidegate_proj = nn.LinearProjection(hidden_size, state_size_dim, **kwargs)
        self.output_proj = nn.LinearProjection(hidden_size, state_size_dim, **kwargs)

        self.compute_dtype = kwargs.get("compute_dtype", torch.float32)

        self.wo = nn.LinearProjection(state_size_dim, hidden_size, **kwargs)

        self.pre_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=kwargs.get("eps", 1e-5),
            dtype=kwargs.get("dtype", torch.float32),
            device=kwargs.get("device", None),
        )

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        
        
        This casts to float16 for performance gains
        Switch to fp32 if you encounter NaN issues
        Empirically speaking, once the model is past the initial few thousand training steps,
        outliers in the scan virtually entirely dissappear.
        But you may need to start out with fp32
        
        """
        B, L, D = x.shape
        # B is the batch size
        # L is the sequence length
        # D is the hidden dimension
        # assert L is a power of 2
        assert(L & (L - 1)) == 0, "L must be a power of 2"

        residual = x
        x = self.pre_norm(x)

        A_elements = self.A_proj(x)

        x_elements = self.x_proj(x)
        sidegate_elements = torch.sigmoid(self.output_proj(x))
        gate = torch.sigmoid(self.sidegate_proj(x))

        A_elements = F.sigmoid(A_elements)

        A_elements = A_elements.to(self.compute_dtype)
        x_elements = x_elements.to(self.compute_dtype)
        sidegate_elements = sidegate_elements.to(self.compute_dtype)
        gate = gate.to(self.compute_dtype)
        


        #   a = State transition vector
        #   x = Input sequence
        #   b = Input gate
        #   c = Sidegate

        # The recurrence is h(t) = A(t) * h(t-1) + b(t) * u(t)
        # The output is y(t) = gate(t) * h(t)

        h = scan(A_elements, x_elements, sidegate_elements, gate)

        h = self.wo(h)

        return h + residual


