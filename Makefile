# PREFLIGHT
#
# `make demo` reproduces the entire demo video in one command. That is a gift
# to someone evaluating thirty projects, and it is also the honest test of
# whether the thing works on a machine that is not mine.

PY      ?= python
VENV    ?= .venv
BIN     := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN     := $(VENV)/Scripts
endif
PYTHON  := $(BIN)/python
VIDEO   ?= samples/demo.mp4
OUT     ?= preflight-out

.DEFAULT_GOAL := help
.PHONY: help setup corpus verify-data bench assets sample demo check fix drift test test-py test-ui \
        lint build clean docker docker-demo verify

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install Python and Node dependencies
	$(PY) -m venv $(VENV)
	$(PYTHON) -m pip install --quiet --upgrade pip
	$(PYTHON) -m pip install --quiet -e ".[asr,dev]"
	npm install --no-audit --no-fund
	$(MAKE) corpus

corpus: ## Author the policy corpus from its source script
	$(PYTHON) scripts/build_corpus.py

verify-data: corpus ## Check the data layer keeps its provenance promises (see PROVENANCE.md)
	$(PYTHON) scripts/verify_data.py

bench: ## Score the pipeline against the golden corpus (add ABLATION=1 for every layer)
	$(PYTHON) -m preflight.cli bench $(if $(ABLATION),--ablation,) --out $(OUT)/bench.json

assets: ## Generate the CC0 replacement audio bed
	$(PYTHON) scripts/make_assets.py

sample: assets ## Generate the narrated demo clip
	$(PYTHON) scripts/make_demo.py

check: ## Analyse $(VIDEO) and emit every artifact
	$(PYTHON) -m preflight.cli check $(VIDEO) --format all --out $(OUT)

fix: ## Compile and apply the remediation to $(VIDEO)
	$(PYTHON) -m preflight.cli fix $(VIDEO) --apply

## The verification loop. Red, then green, with the diff visible.
demo: corpus sample build ## Reproduce the full demo end to end
	@echo "=== BEFORE ==="
	-$(PYTHON) -m preflight.cli check $(VIDEO) --format all --out $(OUT)/before
	@echo "=== FIX ==="
	$(PYTHON) -m preflight.cli fix $(VIDEO) --apply
	@echo "=== AFTER ==="
	-$(PYTHON) -m preflight.cli check samples/demo.safe.mp4 --format all --out $(OUT)/after
	@$(PYTHON) -c "import json; \
	b=json.load(open('$(OUT)/before/report.json'))['scores']; \
	a=json.load(open('$(OUT)/after/report.json'))['scores']; \
	print(); \
	print(f\"  BEFORE  {b['overall']:>3}/100  {b['verdict'].replace('_',' ')}\"); \
	print(f\"  AFTER   {a['overall']:>3}/100  {a['verdict'].replace('_',' ')}\"); \
	print(f\"  DELTA   {a['overall']-b['overall']:+d}\"); print()"

drift: ## Demonstrate the Policy Drift Watcher on two corpus snapshots
	$(PYTHON) scripts/build_corpus.py
	$(PYTHON) -m preflight.cli snapshot --out data/policy-snapshots/2026-08.json
	$(PYTHON) scripts/simulate_drift.py
	-$(PYTHON) -m preflight.cli drift --against data/policy-snapshots/2026-08.json
	@$(PYTHON) scripts/build_corpus.py

test: test-py test-ui ## Run every test

test-py: ## Python tests
	$(PYTHON) -m pytest -q

test-ui: ## TypeScript tests, including the cross-language scoring contract
	npm test

verify: ## Regenerate the scoring vectors and prove both languages agree
	$(PYTHON) scripts/emit_scoring_vectors.py
	npm test

lint: ## Typecheck the UI
	npx tsc --noEmit -p tsconfig.json

build: ## Build the React bundle the HTML report embeds
	npm run build

docker: ## Build the container with models baked in
	docker build -t preflight:0.1.0 .

docker-demo: ## Run the full demo inside the container
	docker compose run --rm demo

clean: ## Remove generated artifacts, keeping the cache
	rm -rf $(OUT) dist samples/*.safe.* samples/edl.json samples/fix.sh samples/*.vtt
