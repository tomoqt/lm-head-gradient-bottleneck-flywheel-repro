# Reproduction Outputs

Generated on 2026-03-15 using Flywheel-provisioned compute (`modal::modal_a10g`, region `us`).

## Key metrics

- Gradient suppression mean (GPT-2): `0.9943`
- SpamLang tail losses:
  - `V=256`: `2.0764`
  - `V=512`: `2.2957`
  - `V=1024`: `2.5117`
- Larger rank-ablation tail losses:
  - `r=16`: `2.5166`
  - `r=32`: `2.5174`
  - `r=64`: `2.5195`
  - `r=96`: `2.5213`

## Files

- `gradient_suppression.json`
- `spamlang_bottleneck.json`
- `plots/spamlang_bottleneck.png`
- `rank_ablation_large.json`
- `rank_ablation_large.log`
- `plots/rank_ablation_large.png`
- `budget_burn_trace.json`
- `budget_burn_summary.md`

Total Flywheel cumulative spend across runs in this graph: **950 cents (fully exhausted)**.

Budget-exhaustive node note: the high-cost vastai lease reached cap and auto-terminated; SSH to that host was connection-refused from this environment, so the node stores lease/budget accounting artifacts.
