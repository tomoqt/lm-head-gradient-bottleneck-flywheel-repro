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
class TrialConfig:
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


def evaluate_model(model: nn.Module, cfg: TrialConfig, device: str) -> Dict[str, float]:
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


def run_trial(cfg: TrialConfig, device: str, seed: int) -> Dict:
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

    eval_stats = evaluate_model(model, cfg, device)
    tail = losses[int(0.9 * len(losses)) :]

    return {
        "final_loss": float(losses[-1]),
        "tail_mean_loss": float(np.mean(tail)),
        "eval_loss": eval_stats["eval_loss"],
        "eval_acc": eval_stats["eval_acc"],
    }


def plot_trials(rows: List[Dict], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    xs = list(range(1, len(rows) + 1))
    ys = [r["tail_mean_loss"] for r in rows]
    labels = [f"r{r['head_rank']}-v{r['vocab_size']}" for r in rows]

    plt.figure(figsize=(10, 4.8))
    plt.plot(xs, ys, marker="o", linewidth=1.2)

    for i in range(0, len(rows), max(1, len(rows) // 12)):
        plt.annotate(labels[i], (xs[i], ys[i]), fontsize=7, alpha=0.8)

    plt.xlabel("Trial index")
    plt.ylabel("Tail train CE loss")
    plt.title("Budget stress sweep: trial-by-trial tail loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def summarize(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[int, int], List[Dict]] = {}
    for row in rows:
        key = (row["head_rank"], row["vocab_size"])
        groups.setdefault(key, []).append(row)

    out: List[Dict] = []
    for (rank, vocab), vals in sorted(groups.items()):
        tails = np.array([v["tail_mean_loss"] for v in vals], dtype=np.float64)
        accs = np.array([v["eval_acc"] for v in vals], dtype=np.float64)
        out.append(
            {
                "head_rank": int(rank),
                "vocab_size": int(vocab),
                "num_trials": int(len(vals)),
                "tail_mean_loss_mean": float(tails.mean()),
                "tail_mean_loss_std": float(tails.std()),
                "eval_acc_mean": float(accs.mean()),
                "eval_acc_std": float(accs.std()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-minutes", type=float, default=20.0)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64, 96])
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--plot", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    deadline = start + args.duration_minutes * 60.0
    device = pick_device(args.device)

    rows: List[Dict] = []
    trial_idx = 0

    while time.time() < deadline:
        rank = args.ranks[trial_idx % len(args.ranks)]
        vocab = args.vocab_sizes[(trial_idx // len(args.ranks)) % len(args.vocab_sizes)]
        seed = args.seed + trial_idx

        cfg = TrialConfig(
            vocab_size=vocab,
            hidden_dim=args.hidden_dim,
            head_rank=rank,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            eval_batches=args.eval_batches,
        )

        metrics = run_trial(cfg, device=device, seed=seed)
        row = {
            "trial_index": trial_idx + 1,
            "seed": seed,
            "head_rank": rank,
            "vocab_size": vocab,
            **metrics,
        }
        rows.append(row)
        print(row, flush=True)

        trial_idx += 1

    plot_trials(rows, args.plot)

    payload = {
        "paper": "arXiv:2603.10145",
        "experiment": "budget_exhaustive_stress_sweep",
        "config": {
            "duration_minutes": args.duration_minutes,
            "ranks": args.ranks,
            "vocab_sizes": args.vocab_sizes,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_batches": args.eval_batches,
            "lr": args.lr,
        },
        "num_trials": len(rows),
        "summary_by_rank_vocab": summarize(rows),
        "trials": rows,
        "plot": args.plot,
        "notes": [
            "Runs repeated rank-vocab trials until wall-clock budget is reached.",
            "Designed to keep expensive provisioned compute saturated while collecting additional empirical evidence.",
        ],
    }

    payload = finalize_payload(payload, start, device, args.seed)
    write_json(args.output, payload)
    print("Saved JSON:", args.output)
    print("Saved plot:", args.plot)


if __name__ == "__main__":
    main()
