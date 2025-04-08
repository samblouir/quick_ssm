# Quick SSM: Efficient Triton-based SSM Scan Operations

Quick SSM provides an optimized implementation of the associative scan operation using [Triton](https://github.com/openai/triton) kernels for high performance.
It includes a convenient PyTorch interface for the scan function for easy integration into existing SSM codebases, asa well as a layer to easily add an SSM layer to your model.


This implementation is inspired by code and techniques found in:
* [accelerated-scan](https://github.com/proger/accelerated-scan) by Volodymyr Kyrylov
* [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html) by Sasha Rush
* [Gated State Spaces Layer](https://arxiv.org/abs/2206.13947) (Note: The gating structure here is a simplified version)
* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

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
* `a(t)`: State transition factor (often related to decay).
* `b(t)`: Input gate/projection factor.
* `c(t)`: Output gate/projection factor (sometimes called a side gate).
* `y(t)`: Output sequence tensor at time `t`.

The tensors `a`, `b`, and `c` are typically learned parameters or dynamically computed within a larger neural network layer.



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

* Complete Gradient Checkpointing support to reduce VRAM usage during training.
* Explore additional VRAM optimization strategies.
* Implement a fast inference/generation mode (e.g., for autoregressive sampling).
* Investigate support for related SSM variants like those used in [Hawk](https://arxiv.org/abs/2402.19427).

## Contributing

Contributions are welcome!
Please feel free to open an issue to report bugs or suggest features.

## License

Apache 2.0
