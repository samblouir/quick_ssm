"""
Minimal usage example for the scan interface.
"""

import torch

from quick_ssm.scan_interface import scan


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    B, L, D = 2, 1024, 16
    x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)
    a = torch.rand(B, L, D, device=device, dtype=dtype) * 0.1 + 0.9
    b = torch.randn(B, L, D, device=device, dtype=dtype) * 0.1
    c = torch.sigmoid(torch.randn(B, L, D, device=device, dtype=dtype))

    y = scan(x, a, b, c, block_l=256, checkpoint=True, tile_b=1, tile_d=512)
    loss = y.sum()
    loss.backward()
    print("y shape", y.shape, "loss", loss.item())


if __name__ == "__main__":
    main()
