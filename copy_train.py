import argparse
import string
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_ssm.layers import SSM


VOCAB = string.ascii_letters + string.digits  # 62 chars
VOCAB_SIZE = len(VOCAB)


def sample_batch(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
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
            use_residual=True,
            use_norm=True,
            device=device,
        )
        self.head = nn.Linear(hidden, VOCAB_SIZE, device=device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        h = self.ssm(x, block_l=128, checkpoint=True, tile_b=1, tile_d=512, scan_out_dtype=torch.float32)
        return self.head(h)


def build_teacher_batch(batch_size: int, seq_len: int, device: torch.device):
    """
    Build inputs/labels for teacher-forced copying:
      inputs  = [src, src]
      labels  = [-100, src]  (mask loss on first half)
    """
    src = sample_batch(batch_size, seq_len, device)
    inp = torch.cat([src, src], dim=1)
    labels = torch.cat([torch.full_like(src, -100), src], dim=1)
    return inp, labels


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = CopyModel(hidden=args.hidden, device=device, compute_dtype=compute_dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    for step in range(1, args.steps + 1):
        if args.teacher_forcing:
            tokens, labels = build_teacher_batch(args.batch_size, args.seq_len, device)
        else:
            tokens = sample_batch(args.batch_size, args.seq_len, device)
            labels = tokens

        logits = model(tokens)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            labels.view(-1),
            ignore_index=-100,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % args.log_interval == 0 or step == 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                if args.teacher_forcing:
                    mask = labels != -100
                    acc = (preds[mask] == labels[mask]).float().mean().item()
                else:
                    acc = (preds == labels).float().mean().item()
            print(f"step {step:04d} | loss {loss.item():.6f} | token-acc {acc:.4f}")

    # Final qualitative sample
    with torch.no_grad():
        if args.teacher_forcing:
            tokens, labels = build_teacher_batch(1, args.seq_len, device)
            src = tokens[0, : args.seq_len]
            logits = model(tokens)
            preds = logits.argmax(dim=-1)[0, args.seq_len :]
        else:
            tokens = sample_batch(1, args.seq_len, device)
            logits = model(tokens)
            preds = logits.argmax(dim=-1)[0]
            src = tokens[0]
        source = "".join(VOCAB[i] for i in src.tolist()[:80])
        copied = "".join(VOCAB[i] for i in preds.tolist()[:80])
        print("\nSource :", source + ("..." if len(src) > 80 else ""))
        print("Pred   :", copied + ("..." if len(preds) > 80 else ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copying toy task with quick_ssm")
    p.add_argument("--steps", type=int, default=200, help="Training steps")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    p.add_argument("--hidden", type=int, default=256, help="Model hidden size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--log-interval", type=int, default=20, help="Steps between logs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--teacher-forcing", action="store_true", help="Use inputs [src, src], loss only on second half")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
