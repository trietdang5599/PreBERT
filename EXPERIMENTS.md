# Experiment entry points

All commands below can be run from any directory. The shell launchers resolve
the repository and `.venv` automatically. Set `PYTHON=/path/to/python` to use a
different environment.

## PreBERT

`evaluation.py` is the public entry point. It expands a named mode and delegates
every run to the same cached experiment engine in `run_clustering_ablation.py`.

| Mode | Feature modes | Clustering | Encoder |
|---|---|---|---|
| `main` | `full` | Birch | BERT base |
| `preprocessing-ablation` | `raw`, `review-only`, `rating-only`, `full` | Birch | BERT base |
| `clustering-ablation` | `full` | K-Means, Birch, Bisecting K-Means, DBSCAN | BERT base |
| `encoder-ablation` | `full` | Birch | BERT base, ModernBERT, mmBERT |
| `custom` | CLI values | CLI values | CLI values |

Examples:

```bash
./scripts/run_prebert.sh main
./scripts/run_prebert.sh preprocessing-ablation --seeds 41 42 43
./scripts/run_prebert.sh clustering-ablation --datasets Small_All_Beauty_5_llama_filtered
./scripts/run_prebert.sh encoder-ablation --dry-run
```

A custom run remains possible without adding a preset:

```bash
./scripts/run_prebert.sh custom \
  --feature-modes full raw \
  --cluster-methods birch kmeans \
  --bert-models bert-base \
  --seeds 42
```

Dataset splitting and legacy artifact cleanup are centralized in
`helper/experiment_data.py`. Dataset/model/mode caches remain isolated by the
experiment engine, so an existing compatible BERT checkpoint is reused.

## LLM rating evaluation

Runtime environment defaults, model aliases, input-field mappings, and model
loading are centralized in `exp_llm/llm_settings.py`. The public Python entry
point is `python -m exp_llm`; implementation modules are not called directly.

| Shell preset | LLM modes |
|---|---|
| `main` | `pretrained`, `pretrained-processed` |
| `ablation` | all four combinations of original/filtered text and old/new rating |
| `baseline` | `pretrained` |
| `processed` | `pretrained-processed` |

Examples:

```bash
./scripts/run_llm.sh main
./scripts/run_llm.sh ablation --models llama3.2_1b llama3.2_3b
./scripts/run_llm.sh baseline --datasets data/Small_All_Beauty_5_llama_filtered.json
./scripts/run_llm.sh main --dry-run
```

Arguments intended for each individual `rating` run can be forwarded after
`--`:

```bash
./scripts/run_llm.sh main --dry-run -- --max-test-samples 20
```

Equivalent commands without the shell presets:

```bash
python -m exp_llm rating --dataset data/Small_All_Beauty_5_llama_filtered.json \
  --model llama3.2_1b --mode pretrained
python -m exp_llm rating-matrix --modes pretrained pretrained-processed
python -m exp_llm preprocess data/input.json --output data/output.json
python -m exp_llm semantic data/output.json --output-dir exp_llm/semantic_outputs/example
python -m exp_llm build-dataset --target-size 10000
```

## Other evaluation

Semantic-retention evaluation for all three filtered datasets:

```bash
./scripts/run_semantic_retention.sh --device mps
```

The unified dispatcher exposes all public launchers:

```bash
./scripts/run_evaluations.sh prebert main
./scripts/run_evaluations.sh llm ablation --dry-run
./scripts/run_evaluations.sh semantic --device auto
```
