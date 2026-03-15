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
    head_rank: int
    seq_len: int
    batch_size: int
    steps: int
    lr: float
    eval_batches: int


class RankControlledLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, head_rank: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        rank = max(1, min(head_rank, hidden_dim))
        self.up = nn.Linear(hidden_dim, rank, bias=False)
        self.down = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h, _ = self.rnn(h)
        h = self.norm(h)
        return self.down(self.up(h))


def sample_spamlang_batch(vocab_size: int, batch_size: int, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    content = vocab_size // 2
    x = torch.randint(0, content, size=(batch_size, seq_len), device=device)

    y = torch.empty_like(x)
    noise_mask = torch.rand(batch_size, seq_len, device=device) < 0.3

    mapped = (7 * x + 3) % content
    spam = torch.randint(content, vocab_size, size=(batch_size, seq_len), device=device)
    y.copy_(mapped)
    y[noise_mask] = spam[noise_mask]
    return x, y


def evaluate_model(model: nn.Module, cfg: TrainConfig, device: str) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []

    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = sample_spamlang_batch(cfg.vocab_size, cfg.batch_size, cfg.seq_len, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean()

            losses.append(float(loss.item()))
            accs.append(float(acc.item()))

    model.train()
    return {
        "eval_loss": float(np.mean(losses)),
        "eval_acc": float(np.mean(accs)),
    }


def run_trial(cfg: TrainConfig, device: str, seed: int) -> Dict:
    set_seed(seed)
    model = RankControlledLM(cfg.vocab_size, cfg.hidden_dim, cfg.head_rank).to(device)
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
    eval_stats = evaluate_model(model, cfg, device)

    return {
        "final_loss": float(losses[-1]),
        "tail_mean_loss": float(np.mean(tail)),
        "eval_loss": eval_stats["eval_loss"],
        "eval_acc": eval_stats["eval_acc"],
    }


def plot_summary(rows: List[Dict], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ranks = [r["head_rank"] for r in rows]
    loss_means = [r["tail_mean_loss_mean"] for r in rows]
    loss_stds = [r["tail_mean_loss_std"] for r in rows]
    acc_means = [r["eval_acc_mean"] for r in rows]
    acc_stds = [r["eval_acc_std"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    axes[0].errorbar(ranks, loss_means, yerr=loss_stds, marker="o", capsize=4)
    axes[0].set_xlabel("Effective LM-head rank")
    axes[0].set_ylabel("Tail train CE loss")
    axes[0].set_title("Optimization vs LM-head rank")

    axes[1].errorbar(ranks, acc_means, yerr=acc_stds, marker="o", capsize=4)
    axes[1].set_xlabel("Effective LM-head rank")
    axes[1].set_ylabel("Eval token accuracy")
    axes[1].set_title("Generalization vs LM-head rank")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64, 96])
    parser.add_argument("--seeds", nargs="+", type=int, default=[21, 22, 23])
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--plot", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    device = pick_device(args.device)

    runs: List[Dict] = []
    for rank in args.ranks:
        for seed in args.seeds:
            cfg = TrainConfig(
                vocab_size=args.vocab_size,
                hidden_dim=args.hidden_dim,
                head_rank=rank,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                eval_batches=args.eval_batches,
            )
            result = run_trial(cfg, device=device, seed=seed)
            result["head_rank"] = rank
            result["seed"] = seed
            runs.append(result)

    summary_rows: List[Dict] = []
    for rank in sorted(set(args.ranks)):
        vals = [r for r in runs if r["head_rank"] == rank]
        tail_losses = np.array([r["tail_mean_loss"] for r in vals], dtype=np.float64)
        eval_accs = np.array([r["eval_acc"] for r in vals], dtype=np.float64)
        final_losses = np.array([r["final_loss"] for r in vals], dtype=np.float64)

        summary_rows.append(
            {
                "head_rank": int(rank),
                "num_seeds": int(len(vals)),
                "final_loss_mean": float(final_losses.mean()),
                "final_loss_std": float(final_losses.std()),
                "tail_mean_loss_mean": float(tail_losses.mean()),
                "tail_mean_loss_std": float(tail_losses.std()),
                "eval_acc_mean": float(eval_accs.mean()),
                "eval_acc_std": float(eval_accs.std()),
            }
        )

    plot_summary(summary_rows, args.plot)

    payload = {
        "paper": "arXiv:2603.10145",
        "experiment": "larger_scale_rank_ablation",
        "config": {
            "ranks": args.ranks,
            "seeds": args.seeds,
            "vocab_size": args.vocab_size,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_batches": args.eval_batches,
            "lr": args.lr,
        },
        "summary_by_rank": summary_rows,
        "per_run_metrics": runs,
        "plot": args.plot,
        "notes": [
            "Controls effective LM-head rank via factorized output head with fixed hidden size.",
            "Expected trend: higher rank should reduce loss and improve accuracy.",
        ],
    }

    payload = finalize_payload(payload, start, device, args.seed)
    write_json(args.output, payload)

    print("Saved JSON:", args.output)
    print("Saved plot:", args.plot)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
