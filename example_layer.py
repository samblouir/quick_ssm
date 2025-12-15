import torch
import torch.nn as nn

from quick_ssm.layers import SSM


class Model(nn.Module):
    """Minimal stack of SSM layers."""

    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 512,
        state_size_mult: float = 4.0,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        device=None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SSM(
                    hidden_size=hidden_size,
                    state_size_mult=state_size_mult,
                    eps=eps,
                    dtype=dtype,
                    device=device,
                    compute_dtype=torch.float16,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = Model(num_layers=2, hidden_size=128, state_size_mult=4.0)
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.randn(2, 1024, 128, device=device)
    out = model(data)
    print("Output shape:", out.shape)
