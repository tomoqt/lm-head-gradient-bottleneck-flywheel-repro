#!/usr/bin/env python
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from common import finalize_payload, pick_device, set_seed, write_json


@dataclass
class TrainConfig:
    vocab_size: int
    hidden_dim: int
    seq_len: int
    batch_size: int
    steps: int
    lr: float


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h, _ = self.rnn(h)
        h = self.norm(h)
        return self.head(h)


def sample_spamlang_batch(vocab_size: int, batch_size: int, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Partition vocabulary into content and spam halves.
    content = vocab_size // 2
    x = torch.randint(0, content, size=(batch_size, seq_len), device=device)

    # Next token: with p=0.7 deterministic content mapping, else random spam token.
    y = torch.empty_like(x)
    noise_mask = torch.rand(batch_size, seq_len, device=device) < 0.3

    mapped = (7 * x + 3) % content
    spam = torch.randint(content, vocab_size, size=(batch_size, seq_len), device=device)
    y.copy_(mapped)
    y[noise_mask] = spam[noise_mask]

    return x, y


def run_trial(cfg: TrainConfig, device: str, seed: int) -> Dict:
    set_seed(seed)
    model = TinyLM(cfg.vocab_size, cfg.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    losses: List[float] = []
    for _ in trange(cfg.steps, leave=False):
        x, y = sample_spamlang_batch(cfg.vocab_size, cfg.batch_size, cfg.seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))

    tail = losses[int(0.9 * len(losses)) :]
    return {
        "loss_curve": losses,
        "final_loss": float(losses[-1]),
        "tail_mean_loss": float(np.mean(tail)),
    }


def plot_curves(results: Dict[int, Dict], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for vocab_size, payload in results.items():
        curve = payload["loss_curve"]
        window = max(1, len(curve) // 50)
        smooth = np.convolve(curve, np.ones(window) / window, mode="valid")
        plt.plot(smooth, label=f"V={vocab_size}")
    plt.xlabel("Step (smoothed)")
    plt.ylabel("Train CE Loss")
    plt.title("SpamLang optimization vs vocabulary size (fixed hidden dim)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--plot", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    device = pick_device(args.device)

    all_results: Dict[int, Dict] = {}
    for i, vocab_size in enumerate(args.vocab_sizes):
        cfg = TrainConfig(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
        )
        all_results[vocab_size] = run_trial(cfg, device=device, seed=args.seed + i)

    plot_curves(all_results, args.plot)

    summary = []
    for v in sorted(all_results):
        summary.append(
            {
                "vocab_size": v,
                "final_loss": all_results[v]["final_loss"],
                "tail_mean_loss": all_results[v]["tail_mean_loss"],
            }
        )

    payload = {
        "paper": "arXiv:2603.10145",
        "experiment": "spamlang_vocab_scaling",
        "config": {
            "vocab_sizes": args.vocab_sizes,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
        },
        "results_by_vocab": summary,
        "notes": [
            "This targets the paper's claim that optimization worsens as vocabulary grows at fixed hidden size.",
            "Trend of interest: higher final loss for larger vocabulary under equal compute.",
        ],
        "plot": args.plot,
    }
    payload = finalize_payload(payload, start, device, args.seed)
    write_json(args.output, payload)

    print("Saved JSON:", args.output)
    print("Saved plot:", args.plot)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
