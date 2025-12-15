"""
Minimal copying task to sanity‑check training with quick_ssm.

Task: given a random string of a‑zA‑Z0‑9, predict the exact same sequence
at every position (not next‑token). This forces the model to store and
recall information through the SSM state.
"""

import argparse
import string
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_ssm.layers import SSM


VOCAB = string.ascii_letters + string.digits  # 62 chars
VOCAB_SIZE = len(VOCAB)


def sample_batch(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Return integer tokens shaped [B, L]."""
    return torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device, dtype=torch.long)


class CopyModel(nn.Module):
    def __init__(self, hidden: int, device: torch.device, compute_dtype: torch.dtype):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, hidden, device=device)
        self.ssm = SSM(
            hidden_size=hidden,
            state_size_mult=4.0,
            compute_dtype=compute_dtype,
            dtype=torch.float32,
            device=device,
        )
        self.head = nn.Linear(hidden, VOCAB_SIZE, device=device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)  # [B, L, H]
        h = self.ssm(x, block_l=128, checkpoint=True, tile_b=1, tile_d=512)
        return self.head(h)  # [B, L, V]


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = CopyModel(hidden=args.hidden, device=device, compute_dtype=compute_dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    for step in range(1, args.steps + 1):
        tokens = sample_batch(args.batch_size, args.seq_len, device)
        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tokens.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % args.log_interval == 0 or step == 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == tokens).float().mean().item()
            print(f"step {step:04d} | loss {loss.item():.4f} | token-acc {acc:.3f}")

    # Final qualitative sample
    with torch.no_grad():
        tokens = sample_batch(1, args.seq_len, device)
        logits = model(tokens)
        preds = logits.argmax(dim=-1)
        source = "".join(VOCAB[i] for i in tokens[0].tolist())
        copied = "".join(VOCAB[i] for i in preds[0].tolist())
        print("\nSource :", source)
        print("Pred   :", copied)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copying toy task with quick_ssm")
    p.add_argument("--steps", type=int, default=200, help="Training steps")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--seq-len", type=int, default=64, help="Sequence length (power of 2 recommended)")
    p.add_argument("--hidden", type=int, default=256, help="Model hidden size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--log-interval", type=int, default=20, help="Steps between logs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (args.seq_len & (args.seq_len - 1)) != 0:
        raise ValueError("seq_len must be a power of 2 for the current Triton kernels.")
    train(args)
