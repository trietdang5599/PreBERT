# Experiment entry points

All commands below can be run from any directory. The shell launchers resolve
the repository and `.venv` automatically. Set `PYTHON=/path/to/python` to use a
different environment. Each file under `scripts/` has a `USER CONFIGURATION`
block at the top, so modes, datasets, models, seeds, devices, and output paths
can be selected by editing the launcher and then running it without arguments.

## Full thesis pipeline

The repository launchers are preconfigured with the complete experiment
matrix used in the thesis. Run the whole workflow in this order with:

```bash
./scripts/run_evaluations.sh
```

The dispatcher runs:

1. `run_preprocessing.sh`: build the dense Toys and Games subset, preprocess
   all three domains with Llama 3.2 3B Instruct, derive deterministic 80/10/10
   assignments from the original datasets, then map processed fields onto
   those preassigned rows.
2. `run_prebert.sh`: main results plus preprocessing, clustering, and encoder
   ablations.
3. `run_llm.sh`: 3 datasets x 4 pretrained LLMs x 4 text/rating modes.
4. `run_semantic_retention.sh`: semantic-retention evaluation for all three
   processed datasets.

Existing `*_llama_filtered.json` files can be reused when only regenerating
split mappings. The launcher is temporarily configured to rebuild them once
so the new hybrid segmentation is actually applied; afterwards set
`REUSE_EXISTING_PROCESSED=true` and `OVERWRITE_PROCESSED=false` to skip the
expensive LLM pass.
The preprocessing launcher defaults to hybrid segmentation: robust sentence
boundaries plus `;`, `:`, em dashes, and comma-prefixed contrastive connectors
such as `but`, `however`, and `whereas`. Set `SEGMENTATION_MODE=sentence` for
the sentence-only ablation.

Preprocessing uses deterministic safety rules around the LLM decision: explicit
rating evidence (including age suitability, music preference, and concrete
product defects) is protected, while raw ingredient/catalog lists and review
disclosures are removed and checked again before the output is written. Each
run also writes an audit CSV under `experiments/preprocessing_audits/` containing
every review shortened by at least 25% plus 50 deterministic random reviews.
Complete `audit_decision` and `audit_notes` manually, then use the unchanged
semantic-retention evaluator to inspect every similarity below `0.80`.

Each stage can also be run independently:

```bash
./scripts/run_preprocessing.sh
./scripts/run_prebert.sh
./scripts/run_llm.sh
./scripts/run_semantic_retention.sh
```

To use BERT only as a frozen pretrained encoder, set
`FINE_TUNE_BERT=false` in `scripts/run_prebert.sh`. This mode does not create a
random classification head: it uses pretrained BERT embeddings for clustering
and VADER for the coarse sentiment score. Its caches and result paths are
isolated with a `pretrained-only` suffix.

Fine-tuned runs use inverse-frequency class weights by default to prevent the
majority five-star class from dominating. Set `BALANCE_BERT_CLASSES=false` in
`scripts/run_prebert.sh` for an unweighted comparison. Schema-v4 metrics also
include balanced accuracy, macro-F1, and a binary confusion matrix.

## PreBERT

`experiments/evaluation.py` is the public entry point. It expands a named mode
and delegates every run to the cached engine in
`experiments/run_clustering_ablation.py`.

| Mode | Feature modes | Clustering | Encoder |
|---|---|---|---|
| `main` | `full` | Birch | ModernBERT |
| `preprocessing-ablation` | `raw`, `review-only`, `rating-only`, `full` | Birch | ModernBERT |
| `clustering-ablation` | `full` | K-Means, Birch, Bisecting K-Means, DBSCAN | ModernBERT |
| `encoder-ablation` | `full` | Birch | BERT base, ModernBERT, mmBERT |
| `custom` | CLI values | CLI values | CLI values |

ModernBERT (`answerdotai/ModernBERT-base`) is the default encoder for the main,
preprocessing-ablation, and clustering-ablation presets. The encoder ablation
still evaluates ModernBERT, BERT base, and mmBERT.

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

The preprocessing modes form a controlled 2x2 ablation while retaining the
same PreBERT architecture: `raw` uses original text/rating, `review-only` uses
filtered text/original rating, `rating-only` uses original text/adjusted
train-validation rating, and `full` uses filtered text/adjusted
train-validation rating. Every mode loads the precomputed physical splits and
evaluates the test split against the immutable `overall` field.

## LLM rating evaluation

Runtime environment defaults, model aliases, input-field mappings, and model
loading are centralized in `helper/llm_settings.py`. The public experiment entry
point is `python -m experiments`; dataset preparation uses the root-level
`preprocessing_reviews.py` command.
Rating splits are built from the original JSONL before fields from the matching
preprocessed JSONL are mapped onto each split.

| Shell preset | LLM modes |
|---|---|
| `main` | `pretrained`, `pretrained-processed` |
| `ablation` | all four legacy modes; metrics always use `overall` as ground truth |
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
python -m experiments rating --dataset data/Small_All_Beauty_5_llama_filtered.json \
  --model llama3.2_1b --mode pretrained
python -m experiments rating-matrix --modes pretrained pretrained-processed
python preprocessing_reviews.py preprocess data/input.json --output data/output.json
python preprocessing_reviews.py build-dataset --target-size 10000
python preprocessing_reviews.py split data/output_llama_filtered.json
python -m experiments semantic data/output.json \
  --output-dir experiments/semantic_outputs/example
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
