# SSM Scan — Low-VRAM Tiling with Triton Block Scan

## When to use this skill
- Training or finetuning SSM/RNN models using `quick_ssm.scan` where GPU VRAM is tight.
- Debugging NaNs/instability during early training caused by large activations.
- Running correctness checks on CPU when CUDA/Triton is unavailable.

## Inputs / outputs
- Inputs: PyTorch tensors `x, a, b, c` shaped `[B, L, D]`; `L` must be a power of 2 for the Triton path.
- Key knobs: `block_l` (time block length), `checkpoint` (recompute h in backward), `tile_b`, `tile_d`, `backend` (`auto|triton|torch`), `out_dtype`.
- Output: `y` shaped `[B, L, D]` with dtype `out_dtype`.

## Standard workflow
1) Start stable: set `checkpoint=True`, `backend="auto"`, `block_l=256`, `compute_dtype=torch.float32` for the first few thousand steps if seeing NaNs; later switch compute dtype to fp16/bf16 for speed.  
2) Choose chunking:\n   - If OOM on batch: set `tile_b=1` or a small divisor of `B`.\n   - If OOM on feature dim: set `tile_d` to 256/512/1024 depending on GPU cache size.\n   - Keep `block_l` at 256–512; smaller reduces scratch but increases launches.\n3) Call `scan(x, a, b, c, block_l=..., checkpoint=..., tile_b=..., tile_d=...)`.\n4) Backward automatically recomputes per tile when `checkpoint=True`; no extra code needed.\n5) For CPU-only validation, set `backend="torch"`; this runs the naive recurrence for correctness checks.\n6) Monitor memory with `torch.cuda.max_memory_allocated()` before/after to tune tiles.\n7) After convergence is stable, optionally disable checkpointing for speed if VRAM allows.

## Pitfalls and fixes
- Symptom: `ValueError: Sequence length L must currently be a power of 2.` → pad the sequence or crop to nearest power of two before calling `scan`.\n- Symptom: NaNs early in training → use `compute_dtype=torch.float32` (or keep parameters in fp32 and cast outputs), reduce LR, or clamp `a` into `[0,1]` via sigmoid as done in the layer.\n- Symptom: OOM despite tiling → increase `tile_b` granularity (down to 1), reduce `tile_d`, and/or lower `block_l`.\n- Symptom: Very slow when `tile_d` is tiny → increase `tile_d` to reduce kernel launch overhead; prefer batch tiling first if possible.\n- Symptom: Gradients mismatch in tests → ensure inputs are contiguous and all tensors on the same device; for CPU tests use `backend=\"torch\"`.\n- Mixed dtypes across inputs can lead to silent up/downcasts; explicitly set `out_dtype` if you need a specific output dtype.

## Validation checklist
- On a small CPU case, compare to naive: `torch.allclose` of outputs and grads (see `quick_ssm/test_scan_interface.py`).\n- Ensure `torch.cuda.is_available()` path passes a forward/backward run without OOM when tiles are set.\n- Confirm `checkpoint=True` actually reduces peak memory via `torch.cuda.reset_peak_memory_stats()` before the call.

## Snippets / templates
```python
# Low-VRAM forward/backward with tiling and checkpointing
y = scan(
    x, a, b, c,
    block_l=256,
    checkpoint=True,   # recompute h in backward
    tile_b=1,          # chunk batch
    tile_d=512,        # chunk feature dim
    out_dtype=torch.float16,  # output dtype
)
```
```python
# CPU correctness check
y = scan(x, a, b, c, backend="torch", checkpoint=False)
```

## Changelog
- 2025-12-15: Added initial tiling/checkpointing playbook and validation steps.
