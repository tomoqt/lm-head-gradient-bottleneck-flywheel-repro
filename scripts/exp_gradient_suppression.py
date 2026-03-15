#!/usr/bin/env python
import argparse
import math
import time
from typing import Iterable, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import finalize_payload, pick_device, set_seed, write_json


FALLBACK_TEXTS = [
    "Backpropagation through the LM head can attenuate useful directions.",
    "Large vocabularies induce severe low-rank constraints in gradient flow.",
    "We estimate suppression by projecting token-level logit gradients.",
    "This script is a minimal replication of the optimization bottleneck claim.",
]


def stream_texts(max_batches: int, batch_size: int) -> Iterable[List[str]]:
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        buf: List[str] = []
        emitted = 0
        for ex in ds:
            txt = ex["text"].strip()
            if not txt:
                continue
            buf.append(txt)
            if len(buf) == batch_size:
                yield buf
                buf = []
                emitted += 1
                if emitted >= max_batches:
                    return
        while emitted < max_batches:
            yield FALLBACK_TEXTS[:batch_size]
            emitted += 1
    except Exception:
        for _ in range(max_batches):
            yield FALLBACK_TEXTS[:batch_size]


def get_lm_head_weight(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach()
    out_emb = model.get_output_embeddings()
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise RuntimeError("Could not locate LM head weight matrix")
    return out_emb.weight.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--max-batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--sample-tokens-per-batch", type=int, default=96)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    set_seed(args.seed)
    device = pick_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    W = get_lm_head_weight(model).to(device=device, dtype=torch.float32)  # [V, d]

    # Build an orthonormal basis for col(W), enabling fast projection g -> QQ^T g.
    Q = torch.linalg.qr(W, mode="reduced").Q  # [V, r]

    suppressed: List[float] = []
    kept: List[float] = []

    for batch in tqdm(stream_texts(args.max_batches, args.batch_size), total=args.max_batches):
        enc = tokenizer(
            batch,
            max_length=args.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits.float()  # [B, T, V]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Exact CE gradient wrt logits: softmax(logits) - one_hot(target)
        probs = F.softmax(shift_logits, dim=-1)
        g = probs
        g.scatter_add_(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
            src=-torch.ones_like(shift_labels, dtype=g.dtype).unsqueeze(-1),
        )

        g_flat = g.reshape(-1, g.shape[-1])
        valid = (shift_labels.reshape(-1) != tokenizer.pad_token_id)
        g_flat = g_flat[valid]

        if g_flat.numel() == 0:
            continue

        n = g_flat.shape[0]
        take = min(n, args.sample_tokens_per_batch)
        idx = torch.randperm(n, device=device)[:take]
        g_tok = g_flat[idx]  # [N, V]

        coeff = g_tok @ Q  # [N, r]
        g_proj = coeff @ Q.transpose(0, 1)  # [N, V]

        g_norm = torch.linalg.norm(g_tok, dim=-1).clamp_min(1e-9)
        proj_norm = torch.linalg.norm(g_proj, dim=-1)
        rem_norm = torch.linalg.norm(g_tok - g_proj, dim=-1)

        kept.extend((proj_norm / g_norm).detach().cpu().tolist())
        suppressed.extend((rem_norm / g_norm).detach().cpu().tolist())

    def stats(x: List[float]) -> dict:
        t = torch.tensor(x)
        return {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "p50": float(t.median().item()),
            "p90": float(torch.quantile(t, 0.9).item()),
            "count": int(t.numel()),
        }

    payload = {
        "paper": "arXiv:2603.10145",
        "experiment": "gradient_suppression",
        "model": args.model,
        "config": {
            "max_batches": args.max_batches,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "sample_tokens_per_batch": args.sample_tokens_per_batch,
        },
        "results": {
            "suppressed_norm_ratio": stats(suppressed),
            "kept_norm_ratio": stats(kept),
        },
        "notes": [
            "Suppressed ratio is ||g - QQ^T g|| / ||g|| where Q spans col(W_out).",
            "If suppression is high, gradient flow through the output head discards most logit-gradient norm.",
        ],
    }

    payload = finalize_payload(payload, start, device, args.seed)
    write_json(args.output, payload)

    print("Gradient suppression mean:", payload["results"]["suppressed_norm_ratio"]["mean"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
