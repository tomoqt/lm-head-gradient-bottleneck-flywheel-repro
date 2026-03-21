# Reproduction: LM Head as a Gradient Bottleneck (arXiv:2603.10145)

This repository reproduces core claims from the paper under a strict total compute budget of **< $10** using Flywheel-provisioned compute.

## Scope

We target three claims that are feasible within budget:

1. **Gradient suppression through the LM head**
   - Estimate how much token-level logit gradient norm lies outside the LM-head reachable subspace.
2. **Optimization gets harder as vocabulary grows** (SpamLang-style synthetic setup)
   - Keep hidden size fixed and increase vocabulary; observe training loss trends.
3. **Higher effective LM-head rank improves optimization**
   - Hold hidden size and task fixed, vary output-head rank, and measure loss/accuracy trends over multiple seeds.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make all
```

Single-command runner:

```bash
bash scripts/run_repro.sh full
```

Use `quick` mode for a short smoke run:

```bash
bash scripts/run_repro.sh quick
```

Run larger-scale rank ablation:

```bash
python3 scripts/exp_rank_ablation_large.py \
  --output artifacts/rank_ablation_large.json \
  --plot artifacts/plots/rank_ablation_large.png
```

Artifacts:

- `artifacts/gradient_suppression.json`
- `artifacts/spamlang_bottleneck.json`
- `artifacts/plots/spamlang_bottleneck.png`
- `artifacts/rank_ablation_large.json`
- `artifacts/plots/rank_ablation_large.png`

## Flywheel Graph

Node IDs are in `flywheel/graph_manifest.json` for production and `flywheel/staging_graph_manifest.json` for Flywheel staging.

Workflow:

1. Acquire compute lease with hard cap <= 950 cents.
2. Run `make all` and optionally `make rank_ablation`.
3. Publish artifacts to empirical nodes.
4. Commit nodes with result summaries and repo commit SHA.

## Budgeting

Budget guardrail is set in `configs/replication.yaml` with target `$9.50` and hard max `$10.00`.

## Caveat

This is a **budgeted partial replication**. It focuses on directional evidence rather than exact reproduction of all large-scale pretraining experiments from the paper.
