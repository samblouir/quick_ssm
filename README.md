# Quick SSM: Efficient Triton-based Scan for RNNs/SSMs

Quick SSM provides an optimized implementation of the associative scan operation using [Triton](https://github.com/openai/triton) kernels for high performance.
It includes a convenient PyTorch interface for the scan function for easy integration into existing SSM codebases, asa well as a layer to easily add an SSM layer to your model.


This implementation is inspired by code and techniques found in:
* [accelerated-scan](https://github.com/proger/accelerated-scan) by Volodymyr Kyrylov
* [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html) by Sasha Rush

Papers:
* [Gated State Spaces](https://arxiv.org/abs/2206.13947)
* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)


<div align="center">
  <a href="https://github.com/samblouir/quick_ssm/blob/main/birdie_emnlp_2024_poster.jpg?raw=true"><img src="https://github.com/samblouir/quick_ssm/blob/main/quickssm.png?raw=true" alt="quick ssm logo of a rocket" width="200" /></a>
</div>

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/samblouir/quick_ssm
```

Alternatively, install it in editable mode for development:

```bash
git clone https://github.com/samblouir/quick_ssm
cd quick_ssm
pip install -e .
```

*Note: Requires a compatible PyTorch installation and Triton (which typically requires an NVIDIA GPU and Linux).*

## Usage

Here's a basic example using the `scan_interface`:

```python
import torch
from quick_ssm.scan_interface import scan

device = 'cuda'
dtype = torch.float16 # fp16 stays finite later in training after the intermediates calm down, I recommend using fp32 for the first few thousand training steps. Your mileage will vary.

# Example dimensions (Batch, Sequence Length, Hidden Dimension)
# Note: Sequence length L must be a power of 2
B =	4
L = 2048
D = 16

x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)
a = torch.rand(B, L, D, device=device, dtype=dtype, requires_grad=True)
b = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)
c = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)

# Note: h(t) is currently materialized.
# h(t) = a(t) * h(t-1) + b(t) * x(t)
# y(t) = c(t) * h(t)
# `checkpoint=False` currently (checkpointing is WIP)
y = scan(x, a, b, c, checkpoint=False)
```

### Layer Example

```python
import torch
import torch.nn as nn
from quick_ssm.layers import SSM

# Basic Torch Model
class AnyTorchModel(nn.Module):
	def __init__(self, hidden_size):
		super(AnyTorchModel, self).__init__()
		self.ssm = SSM(
			hidden_size=hidden_size,
			state_size_mult=(hidden_size * 4),
			dtype=torch.float32, # Parameter dtype
			# (FOR NLP: USE FP32 FOR THE FIRST FEW THOUSAND PRE-TRAINING STEPS TO AVOID NaN with FP16.)
			compute_dtype=torch.float16, # Computation dtype 
		)
			

	def forward(self, x):
		return self.ssm(x)
```




## Key Features

* **High-Performance Kernels:** Utilizes Triton kernels for efficient parallel execution of the associative scan operation.
* **Block Scan Algorithm:** Implements the parallel prefix sum (scan) algorithm over blocks, enabling efficient processing of long sequences.
* **PyTorch Integration:** Offers `quick_ssm.scan_interface.scan`, a `torch.autograd.Function`, allowing seamless use inside PyTorch models.
* **SSM Layer:** Provides a simple `quick_ssm.layers.SSM` layer for easy integration into existing models.

## Core Concept: SSM Scan

This library efficiently computes the following core SSM recurrence relation:

1.  **Hidden State Update:** `h(t) = a(t) * h(t-1) + b(t) * x(t)`
2.  **Output Calculation:** `y(t) = c(t) * h(t)`

Where:
* `x(t)`: Input sequence tensor at time `t`.
* `h(t)`: Hidden state tensor at time `t`.
* `a(t)`: State transition factor
* `b(t)`: Input gate/projection factor.
* `c(t)`: Output gate/projection factor (sometimes called a side gate).
* `y(t)`: Output sequence tensor at time `t`.

All tensors are of shape `(B, L, D)`, where:
* `B`: Batch size
* `L`: Sequence length (must be a power of 2)
* `D`: Hidden dimension



## Repository Structure

* `example_interface.py`: Minimal `scan` function usage example.
* `example_layer.py`: Minimal `SSM` layer example.
* `src/`: Contains the core library code.
    * `triton_scan.py`: Triton kernels for the forward and backward scan passes.
    * `scan_interface.py`: The main `torch.autograd.Function` interface (`scan`).
    * `layers.py`: An example `nn.Module` (`SSM`) demonstrating usage.
    * `naive_baseline.py`: Pure PyTorch implementations of the scan for testing.
    * `test_forwards.py`: Correctness tests for the forward pass.
    * `test_backwards.py`: Correctness tests for the backward pass (gradient calculation).


## TODO / Future Work
* Add tensor-parallel support for the scan.
* Add automatic padding to support non-power-of-2 sequence lengths.
* Verify torch.compile compatibility with distributed training.
* Complete Gradient Checkpointing support to reduce VRAM usage during training.
* Explore additional VRAM optimization strategies.
* Implement a fast inference/generation mode (e.g., for autoregressive sampling).
* Investigate support for related SSM variants like those used in [Hawk](https://arxiv.org/abs/2402.19427).

## Not currently planned, but possible
* Pipeline scan, splitting the sequence time-wise across devices.

## Contributing

Contributions are welcome!
Please feel free to open an issue to report bugs or suggest features.

## License

Apache 2.0
