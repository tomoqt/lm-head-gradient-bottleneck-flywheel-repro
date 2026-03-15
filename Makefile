PYTHON ?= python

.PHONY: setup gradient spamlang rank_ablation all

setup:
	$(PYTHON) -m pip install -r requirements.txt

gradient:
	$(PYTHON) scripts/exp_gradient_suppression.py \
		--model gpt2 \
		--output artifacts/gradient_suppression.json

spamlang:
	$(PYTHON) scripts/exp_spamlang_bottleneck.py \
		--output artifacts/spamlang_bottleneck.json \
		--plot artifacts/plots/spamlang_bottleneck.png

rank_ablation:
	$(PYTHON) scripts/exp_rank_ablation_large.py \
		--output artifacts/rank_ablation_large.json \
		--plot artifacts/plots/rank_ablation_large.png

all: gradient spamlang
