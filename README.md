# Reproduction: LM Head as a Gradient Bottleneck (arXiv:2603.10145)

This repository reproduces core claims from the paper under a strict total compute budget of **< $10** using Flywheel-provisioned compute.

## Scope

We target two claims that are feasible within budget:

1. **Gradient suppression through the LM head**
   - Estimate how much token-level logit gradient norm lies outside the LM-head reachable subspace.
2. **Optimization gets harder as vocabulary grows** (SpamLang-style synthetic setup)
   - Keep hidden size fixed and increase vocabulary; observe training loss trends.

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

Artifacts:

- `artifacts/gradient_suppression.json`
- `artifacts/spamlang_bottleneck.json`
- `artifacts/plots/spamlang_bottleneck.png`

## Flywheel Graph

Node IDs are in `flywheel/graph_manifest.json`.

Workflow:

1. Acquire compute lease with hard cap <= 950 cents.
2. Run `make all`.
3. Publish artifacts to empirical nodes.
4. Commit nodes with result summaries and repo commit SHA.

## Budgeting

Budget guardrail is set in `configs/replication.yaml` with target `$9.50` and hard max `$10.00`.

## Caveat

This is a **budgeted partial replication**. It focuses on directional evidence rather than exact reproduction of all large-scale pretraining experiments from the paper.
