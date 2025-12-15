import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_ssm.layers import SSM, RMSNorm


class GatedLinearRNN(nn.Module):
    """
    Reference PyTorch implementation matching the SSM layer mapping:
        f = sigmoid(W_f)
        z = W_z * sigmoid(W_z_gate)
        h = scan(f, z)
        out_gate = swiglu(W_out_gate_a, W_out_gate_b)
        y = W_out(out_gate * h)
    """

    def __init__(self, state_size: int, hidden_size: int, batch_first: bool = True, use_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.use_norm = use_norm
        self.W_f = nn.Linear(hidden_size, state_size)
        self.W_z_gate = nn.Linear(hidden_size, state_size, bias=False)
        self.W_z = nn.Linear(hidden_size, state_size, bias=False)
        self.W_out_gate_a = nn.Linear(hidden_size, state_size, bias=False)
        self.W_out_gate_b = nn.Linear(hidden_size, state_size, bias=False)
        self.W_out = nn.Linear(state_size, hidden_size)
        self.rms = RMSNorm(hidden_size, eps=eps) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected [B, T, D]")
        B, T, D = x.shape
        residual = x
        x = self.rms(x)
        out_gate = F.silu(self.W_out_gate_a(x)) * self.W_out_gate_b(x)
        f = torch.sigmoid(self.W_f(x))
        z = self.W_z(x) * torch.sigmoid(self.W_z_gate(x))
        # Mimic SSM quantization: cast to bf16 before scan and keep accumulation in bf16
        f_q = f.to(torch.bfloat16)
        z_q = z.to(torch.bfloat16)
        h = self.linear_scan(f_q, z_q)
        y = self.W_out((out_gate.to(torch.bfloat16) * h).to(torch.float32))
        return y + residual

    @staticmethod
    def linear_scan(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h_t = f_t * h_{t-1} + z_t
        B, T, S = f.shape
        h_prev = torch.zeros(B, S, device=f.device, dtype=f.dtype)
        hs = []
        for t in range(T):
            h_prev = f[:, t] * h_prev + z[:, t]
            hs.append(h_prev)
        return torch.stack(hs, dim=1)


def _align_weights(ssm: SSM, ref: GatedLinearRNN):
    # Map SSM params to reference model
    with torch.no_grad():
        ref.W_f.weight.copy_(ssm.A_proj.weight)
        ref.W_f.bias.copy_(ssm.A_proj.bias)
        ref.W_z.weight.copy_(ssm.B_proj.weight)
        ref.W_z_gate.weight.copy_(ssm.D_proj.weight)
        ref.W_out_gate_a.weight.copy_(ssm.C_proj_a.weight)
        ref.W_out_gate_b.weight.copy_(ssm.C_proj_b.weight)
        ref.W_out.weight.copy_(ssm.out_proj.weight)
        ref.W_out.bias.copy_(ssm.out_proj.bias)


def test_swiglu_equivalence():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D = 2, 16, 8
    state_mult = 1.0
    eps = 1e-5
    ATOL = 1e-2
    RTOL = 1e-2

    ssm = SSM(
        hidden_size=D,
        state_size_mult=state_mult,
        compute_dtype=torch.bfloat16,
        dtype=torch.float32,
        use_residual=True,
        use_norm=True,
        device=device,
    )
    ref = GatedLinearRNN(state_size=int(D * state_mult), hidden_size=D, use_norm=True, eps=eps).to(device)
    _align_weights(ssm, ref)

    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    y_ssm = ssm(x, block_l=64, checkpoint=False, tile_b=None, tile_d=None, backend="torch")
    y_ref = ref(x_ref)

    diff = (y_ssm - y_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert torch.allclose(y_ssm, y_ref, atol=ATOL, rtol=RTOL), f"forward mismatch max_diff={max_diff:.3e} mean_diff={mean_diff:.3e}"

    # Backward match
    g = torch.randn_like(y_ssm)
    y_ssm.backward(g)
    y_ref.backward(g)

    for name, p_ssm, p_ref in [
        ("x", x.grad, x_ref.grad),
        ("A", ssm.A_proj.weight.grad, ref.W_f.weight.grad),
        ("B", ssm.B_proj.weight.grad, ref.W_z.weight.grad),
        ("D", ssm.D_proj.weight.grad, ref.W_z_gate.weight.grad),
        ("C_a", ssm.C_proj_a.weight.grad, ref.W_out_gate_a.weight.grad),
        ("C_b", ssm.C_proj_b.weight.grad, ref.W_out_gate_b.weight.grad),
        ("out", ssm.out_proj.weight.grad, ref.W_out.weight.grad),
    ]:
        assert torch.allclose(p_ssm, p_ref, atol=ATOL, rtol=RTOL), f"Gradient mismatch {name}"


if __name__ == "__main__":
    test_swiglu_equivalence()
    print("SwiGLU equivalence test passed.")
