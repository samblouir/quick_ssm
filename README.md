# Quick SSM

Quick SSM provides blocked associative scan operations implemented with Triton kernels.
A Torch layer and interface are included.
Please see example_layer.py for a simple example of how to use the layer, and example_interface.py

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/samblouir/quick_ssm
```

Or clone the repository and install it locally:

```bash
git clone https://github.com/samblouir/quick_ssm
cd quick_ssm
pip install -e .
```

## Usage

```python
import torch
from quick_ssm import scan_interface

if __name__ == "__main__":
	# Example usage
	B, L, D = 2, 5, 3
	x = torch.randn(B, L, D, requires_grad=True)
	a = torch.randn(B, L, D, requires_grad=True)
	b = torch.randn(B, L, D, requires_grad=True)
	c = torch.randn(B, L, D, requires_grad=True)

	y = scan(x, a, b, c)
	print("Output y:", y)
```

## 

This codebase does the following scan, based on [Gated State Spaces](https://arxiv.org/abs/2206.13947), [Mamba](https://arxiv.org/abs/2312.00752), and is inspired by code found in Sasha Rush' [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html).

* `h(t) = a(t) * h(t-1) + b(t) * x(t)`
* `y(t) = c(t) * h(t)`

Where:
* `x(t)` is the input at time `t`.
* `h(t)` is the hidden state at time `t`.
* `a(t)`, `b(t)`, `c(t)` are learned time-varying parameters (gates/transitions).
* `y(t)` is the output at time `t`.

## Features

* **Triton Acceleration:** Uses custom Triton kernels for efficient parallel execution of the scan operation on GPUs.
* **Block Scan Algorithm:** Implements the parallel prefix sum (scan) algorithm over blocks for handling long sequences.
* **Autograd Interface:** Provides `quick_ssm.scan_interface.scan`, a `torch.autograd.Function`, for seamless backpropagation.
* **Gradient Checkpointing:** Supports optional gradient checkpointing within the `scan` interface to save memory.
* **Naive Baseline:** Includes naive PyTorch implementations (`naive_baseline.py`) for testing and reference.
