"""
High-level scan interface exposed to PyTorch.

Goals of this rewrite
---------------------
* Fix the previous autograd bug where `checkpoint=True` raised
  `NotImplementedError` during backward.
* Add optional tiling/chunking across batch and feature dimensions so that
  very large models can be trained in low‑VRAM settings (at the cost of
  extra kernel launches and recomputation).
* Provide an automatic fallback to a pure-Torch implementation when CUDA/Triton
  is unavailable so the code paths remain testable on CPU.

Notation (shapes always [B, L, D]):
    x(t): input signal
    a(t): state transition (0..1 recommended)
    b(t): input gate
    c(t): output gate
    h(t) = a(t) * h(t-1) + b(t) * x(t)
    y(t) = c(t) * h(t)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from quick_ssm.naive_baseline import naive_forward_3d
from quick_ssm.triton_backwards import block_scan_backward_3d
from quick_ssm.triton_forwards import block_scan_forward_3d

Tensor = torch.Tensor


def _chunk_ranges(total: int, tile: Optional[int]) -> List[Tuple[int, int]]:
    """Return half‑open index ranges that cover ``[0, total)``.

    Args:
        total: length to cover.
        tile:  tile size or ``None`` for a single range.
    """

    if tile is None or tile <= 0 or tile >= total:
        return [(0, total)]
    return [(start, min(start + tile, total)) for start in range(0, total, tile)]


class _TritonScanFunction(torch.autograd.Function):
    """Memory-aware wrapper around the Triton block-scan kernels.

    Tiling is implemented in Python (not inside the kernel) so we can keep the
    scratch buffers proportional to ``tile_b * L * tile_d`` instead of the full
    ``B * L * D`` when desired. Backward either reuses the saved hidden states
    (checkpoint=False) or recomputes them per tile (checkpoint=True).
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        block_l: int,
        checkpoint: bool,
        tile_b: Optional[int],
        tile_d: Optional[int],
        out_dtype: torch.dtype,
    ) -> Tensor:
        if x.shape != a.shape or x.shape != b.shape or x.shape != c.shape:
            raise ValueError("x, a, b, c must share the same shape [B, L, D].")
        if x.device.type != "cuda":
            raise RuntimeError("Triton backend requires CUDA tensors.")

        ctx.block_l = int(block_l)
        ctx.checkpoint = bool(checkpoint)
        ctx.tile_b = tile_b
        ctx.tile_d = tile_d
        ctx.out_dtype = out_dtype

        ctx.save_for_backward(x, a, b, c)

        # Preallocate outputs; h_saved is only kept when checkpoint is disabled.
        y = torch.empty_like(x, dtype=out_dtype)
        h_saved = torch.empty_like(x, dtype=torch.float32) if not checkpoint else None

        b_tiles = _chunk_ranges(x.shape[0], tile_b)
        d_tiles = _chunk_ranges(x.shape[2], tile_d)

        for b0, b1 in b_tiles:
            for d0, d1 in d_tiles:
                xs = x[b0:b1, :, d0:d1].contiguous()
                as_ = a[b0:b1, :, d0:d1].contiguous()
                bs = b[b0:b1, :, d0:d1].contiguous()
                cs = c[b0:b1, :, d0:d1].contiguous()

                # Compute block scan for this tile
                _, h_chunk = block_scan_forward_3d(xs, as_, bs, BLOCK_L=ctx.block_l)

                if h_saved is not None:
                    h_saved[b0:b1, :, d0:d1] = h_chunk

                y_chunk = cs * h_chunk
                if y_chunk.dtype != out_dtype:
                    y_chunk = y_chunk.to(out_dtype)
                y[b0:b1, :, d0:d1] = y_chunk

        ctx.h_saved = h_saved
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, a, b, c = ctx.saved_tensors
        block_l = ctx.block_l

        dx = torch.zeros_like(x, dtype=torch.float32)
        da = torch.zeros_like(a, dtype=torch.float32)
        db = torch.zeros_like(b, dtype=torch.float32)
        dc = torch.zeros_like(c, dtype=torch.float32)

        b_tiles = _chunk_ranges(x.shape[0], ctx.tile_b)
        d_tiles = _chunk_ranges(x.shape[2], ctx.tile_d)

        for b0, b1 in b_tiles:
            for d0, d1 in d_tiles:
                xs = x[b0:b1, :, d0:d1].contiguous()
                as_ = a[b0:b1, :, d0:d1].contiguous()
                bs = b[b0:b1, :, d0:d1].contiguous()
                cs = c[b0:b1, :, d0:d1].contiguous()
                g_chunk = grad_output[b0:b1, :, d0:d1].contiguous()

                if ctx.checkpoint or ctx.h_saved is None:
                    _, h_chunk = block_scan_forward_3d(xs, as_, bs, BLOCK_L=block_l)
                else:
                    h_chunk = ctx.h_saved[b0:b1, :, d0:d1]

                dx_s, da_s, db_s, dc_s = block_scan_backward_3d(
                    as_, bs, cs, xs, h_chunk, g_chunk, BLOCK_L=block_l
                )

                dx[b0:b1, :, d0:d1] = dx_s
                da[b0:b1, :, d0:d1] = da_s
                db[b0:b1, :, d0:d1] = db_s
                dc[b0:b1, :, d0:d1] = dc_s

        # Cast gradients back to the dtypes of their respective inputs
        return (
            dx.to(x.dtype),
            da.to(a.dtype),
            db.to(b.dtype),
            dc.to(c.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def _torch_scan(x: Tensor, a: Tensor, b: Tensor, c: Tensor, out_dtype: torch.dtype) -> Tensor:
    """CPU / fallback implementation using the naive PyTorch recurrence."""

    h = naive_forward_3d(x, a, b)
    y = c * h
    return y.to(out_dtype)


def scan(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    *,
    block_l: int = 256,
    checkpoint: bool = True,
    tile_b: Optional[int] = None,
    tile_d: Optional[int] = None,
    backend: str = "auto",
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Run the SSM scan with optional tiling/chunking.

    Args:
        x, a, b, c: tensors of shape ``[B, L, D]``.
        block_l: block length for the Triton scan kernels.
        checkpoint: when ``True`` (default) we recompute hidden states in
            backward instead of storing them, trading compute for lower VRAM.
        tile_b: optional batch tile to reduce scratch buffers; set to a small
            integer (e.g. 1 or 2) when B is large and memory is tight.
        tile_d: optional feature tile; set to values like 256/512/1024 to trade
            more kernel launches for reduced peak memory.
        backend: "auto" (default), "triton", or "torch".
        out_dtype: dtype of the returned tensor. Defaults to ``x.dtype``.

    Returns:
        Tensor y with shape ``[B, L, D]`` and dtype ``out_dtype``.
    """

    out_dtype = out_dtype or x.dtype
    chosen_backend = backend
    if backend == "auto":
        chosen_backend = "triton" if x.device.type == "cuda" else "torch"

    if chosen_backend == "triton":
        return _TritonScanFunction.apply(
            x, a, b, c, block_l, checkpoint, tile_b, tile_d, out_dtype
        )

    if chosen_backend == "torch":
        return _torch_scan(x, a, b, c, out_dtype)

    raise ValueError(f"Unsupported backend '{backend}'. Use 'auto', 'triton', or 'torch'.")


__all__ = ["scan"]
