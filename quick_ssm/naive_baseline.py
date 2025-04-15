import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional


# Naive PyTorch Scan Implementations (For Reference & Testing)


def naive_forward_3d(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Computes the forward recurrence h(t) = a(t)*h(t-1) + b(t)*x(t) naively.

    Args:
        x: Input tensor of shape (B, L, D).
        a: State transition factor tensor of shape (B, L, D).
        b: Input scaling factor tensor of shape (B, L, D).

    Returns:
        Hidden state tensor h of shape (B, L, D).
    """
    B, L, D = x.shape
    h_out = torch.zeros_like(x)
    for bb in range(B):
        h_prev = torch.zeros(D, device=x.device, dtype=x.dtype)
        for t in range(L):
            h_curr = a[bb, t] * h_prev + b[bb, t] * x[bb, t]
            h_out[bb, t] = h_curr
            h_prev = h_curr  # Update previous hidden state
    return h_out


def naive_full_3d(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the full forward pass naively:
    h(t) = a(t)*h(t-1) + b(t)*x(t)
    y(t) = c(t)*h(t)

    Args:
        x: Input tensor of shape (B, L, D).
        a: State transition factor tensor of shape (B, L, D).
        b: Input scaling factor tensor of shape (B, L, D).
        c: Output scaling factor tensor of shape (B, L, D).

    Returns:
        A tuple (h, y):
        h: Hidden state tensor of shape (B, L, D).
        y: Output tensor of shape (B, L, D).
    """
    h = naive_forward_3d(x, a, b)
    y = c * h
    return h, y


def naive_backward_3d(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    grad_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the backward pass naively for the loss L = sum(y * grad_y).

    Calculates gradients dL/dx, dL/da, dL/db, dL/dc.

    Args:
        x: Original input tensor of shape (B, L, D).
        a: Original state transition factor tensor of shape (B, L, D).
        b: Original input scaling factor tensor of shape (B, L, D).
        c: Original output scaling factor tensor of shape (B, L, D).
        grad_y: Gradient of the loss with respect to the output y, shape (B, L, D).

    Returns:
        A tuple (dx, da, db, dc):
        dx: Gradient w.r.t. x, shape (B, L, D).
        da: Gradient w.r.t. a, shape (B, L, D).
        db: Gradient w.r.t. b, shape (B, L, D).
        dc: Gradient w.r.t. c, shape (B, L, D).
    """
    B, L, D = x.shape

    # Recompute forward hidden states needed for backward
    h_out = naive_forward_3d(x, a, b)

    # Initialize gradients
    dh_next = torch.zeros(
        (B, D), device=x.device, dtype=x.dtype
    )  # dL/dh(t) flowing from t+1
    dx = torch.zeros_like(x)
    da = torch.zeros_like(a)
    db = torch.zeros_like(b)
    dc = torch.zeros_like(c)

    # Backward pass iteration
    for t in reversed(range(L)):
        # Gradient from the output projection: dL/dh(t) += dL/dy(t) * dy(t)/dh(t)
        dh_local = grad_y[:, t, :] * c[:, t, :]

        # Total gradient for h(t): Contribution from local output + contribution from next step
        dh_total = dh_local + dh_next  # Shape (B, D)

        # Gradient w.r.t. c: dL/dc(t) = dL/dy(t) * dy(t)/dc(t) = grad_y(t) * h(t)
        dc[:, t, :] = grad_y[:, t, :] * h_out[:, t, :]

        # Gradient w.r.t. b: dL/db(t) = dL/dh(t) * dh(t)/db(t) = dh_total * x(t)
        db[:, t, :] = dh_total * x[:, t, :]

        # Get h(t-1) for gradient w.r.t. a
        h_prev = (
            h_out[:, t - 1, :]
            if t > 0
            else torch.zeros((B, D), device=x.device, dtype=x.dtype)
        )

        # Gradient w.r.t. a: dL/da(t) = dL/dh(t) * dh(t)/da(t) = dh_total * h(t-1)
        da[:, t, :] = dh_total * h_prev

        # Gradient w.r.t. x: dL/dx(t) = dL/dh(t) * dh(t)/dx(t) = dh_total * b(t)
        dx[:, t, :] = dh_total * b[:, t, :]

        # Propagate gradient to h(t-1): dL/dh(t-1) = dL/dh(t) * dh(t)/dh(t-1) = dh_total * a(t)
        dh_next = dh_total * a[:, t, :]

    return dx, da, db, dc
