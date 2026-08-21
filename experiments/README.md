# LLM rating experiments

Experiment operations use `python -m experiments` with the commands `rating`,
`rating-matrix`, and `semantic`. Dataset building, preprocessing, and splitting
use the root-level `preprocessing_reviews.py` command.
The rating command splits a JSONL review dataset into stratified train,
validation, and test sets, asks a Hugging Face model to predict an integer
rating from 1 to 5, then reports RMSE and MAE.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Llama 3.1 is gated on Hugging Face. Accept its license on the model page, then
authenticate with `huggingface-cli login` or set `HF_TOKEN`.

The built-in aliases are:

- `qwen_3b` -> `Qwen/Qwen2.5-3B-Instruct` (instruction-tuned)
- `qwen2.5_0.5b` -> `Qwen/Qwen2.5-0.5B` (base)
- `qwen2.5_1.5b` -> `Qwen/Qwen2.5-1.5B` (base)
- `qwen2.5_3b` or `qwen_3b_base` -> `Qwen/Qwen2.5-3B` (base)
- `qwen2.5_7b` -> `Qwen/Qwen2.5-7B` (base)
- `llama3.1` -> `meta-llama/Llama-3.1-8B-Instruct`
- `llama3.2_1b` -> `meta-llama/Llama-3.2-1B-Instruct`
- `llama3.2_3b` -> `meta-llama/Llama-3.2-3B-Instruct`
- `llama3.2_1b_base` -> `meta-llama/Llama-3.2-1B` (base)
- `llama3.2_3b_base` -> `meta-llama/Llama-3.2-3B` (base)

Any other Hugging Face model id or local model directory can be passed to
`--model` directly.

## LLM preprocessing

The `python preprocessing_reviews.py preprocess` command replaces the legacy
Flair/VADER/spaCy notebook logic.
It splits each review into sentences and uses
`meta-llama/Llama-3.2-3B-Instruct` to retain only segments containing sentiment
or evaluative information. Relevance prompts use `KEEP`/`REMOVE` labels and
few-shot examples specialized for All Beauty, Digital Music, and Toys and
Games. The domain is inferred from the input filename; use `--domain` to set it
explicitly. It writes JSONL directly and preserves all original fields while
adding or replacing `filteredReviewText` and `overall_new`.

Validate a run without loading the model:

```bash
.venv/bin/python preprocessing_reviews.py preprocess \
  data/backup/Small_Toys_and_Games_5.json \
  --output data/Small_Toys_and_Games_5_llama_filtered.json \
  --dry-run
```

Run preprocessing:

```bash
.venv/bin/python preprocessing_reviews.py preprocess \
  data/backup/Small_Toys_and_Games_5.json \
  --output data/Small_Toys_and_Games_5_llama_filtered.json
```

The classifier defaults to a conservative `--remove-margin 0.5`: an uncertain
segment is kept, and `REMOVE` is accepted only when its score exceeds `KEEP` by
that margin. Set `--remove-margin 0` for more aggressive filtering. For example,
an explicit Toys and Games run can use `--domain toys_games`.

By default, the script also compares LLM polarity with the original rating and
reassesses directional conflicts using the item median, producing
`overall_new`. Pass `--no-adjust-ratings` to filter text without changing
ratings. If all segments are classified as non-evaluative, the original review
is retained so downstream experiments do not silently lose the sample; use
`--empty-review-policy empty` to keep an empty filtered value instead.

The output path must differ from the input path. Existing output is protected;
pass `--overwrite` explicitly to replace it. Llama 3.2 is gated, so accept its
Hugging Face license and authenticate before the first non-dry run.

By default, decoding is constrained to exactly one rating token from 1 to 5.
This prevents base models from returning explanations such as `Based on the
review...` instead of a rating. Use `--no-constrain-rating-output` only when you
want to inspect unconstrained free-text behavior; in that case `--max-new-tokens`
controls the response length and unparsable outputs fall back to rating 3.

Qwen's base models are pretrained language models, not instruction-tuned chat
models. They may ignore the instruction to return only one rating, which can
reduce `validParseRate`; prefer an Instruct alias for zero-shot evaluation.

The same caveat applies to the Llama aliases ending in `_base`. Prefer
`llama3.2_1b` or `llama3.2_3b` for zero-shot evaluation.

## Experiments

All rating metrics use the original `overall` field as ground truth. The modes
vary the review text supplied to the model; legacy mode names are retained so
existing commands and output paths continue to work.

Splits are created from the original dataset before processed text is attached.
For `*_llama_filtered.json`, the evaluator automatically finds an original file
with the same `(reviewerID, asin, unixReviewTime, overall)` interactions and
maps `filteredReviewText` onto the already-created splits, so preprocessing does
not need to be run again. Use `--split-dataset path/to/original.json` when the
source file cannot be inferred from its name. For processed modes, prediction
receives only the mapped `filteredReviewText`; test labels are verified against
the source dataset's original `overall`, and `overall_new` is never used as test
ground truth.

Create reusable physical splits without rerunning preprocessing:

```bash
python preprocessing_reviews.py split \
  data/Small_All_Beauty_5_llama_filtered.json
```

This writes `train.json`, `val.json`, and `test.json` under
`data/splits/<dataset>/`. Train and validation preserve processed fields; test
keeps the original `overall` and removes `overall_new`.

| Effect | Mode | Text field | Rating field |
|---|---|---|---|
| Baseline | `pretrained` | `reviewText` | `overall` |
| Filtered text only | `pretrained-processed-mix` | `filteredReviewText` | `overall` |
| Legacy baseline alias | `pretrained-rating-only` | `reviewText` | `overall` |
| Filtered text | `pretrained-processed` | `filteredReviewText` | `overall` |

Pretrained evaluation using `reviewText` and `overall`:

```bash
python3 -m experiments rating \
  --dataset data/Small_All_Beauty_5_llama_filtered.json \
  --model qwen_3b \
  --mode pretrained
```

Pretrained evaluation using `filteredReviewText` with `overall` as ground truth:

```bash
python3 -m experiments rating \
  --dataset data/Small_All_Beauty_5_llama_filtered.json \
  --model qwen_3b \
  --mode pretrained-processed
```

Smaller Llama 3.2 1B Instruct model:

```bash
python3 -m experiments rating \
  --dataset data/Small_All_Beauty_5_llama_filtered.json \
  --model llama3.2_1b \
  --mode pretrained
```

## Evaluate all pretrained models

Run the full zero-shot matrix sequentially. Dataset, model, and mode lists are
CLI options; defaults come from `helper/llm_settings.py`:

```bash
.venv/bin/python -m experiments rating-matrix
```

The recommended preset launchers are:

```bash
./scripts/run_llm.sh main       # pretrained vs pretrained-processed
./scripts/run_llm.sh ablation   # complete 2x2 preprocessing ablation
./scripts/run_llm.sh baseline
./scripts/run_llm.sh processed
```

Existing runs with a `metrics.json` file are skipped. Use `--force` to rerun
them or `--dry-run` to preview the commands.
The default inference batch size is 1 to accommodate the 7B model; increase it
when memory permits. Llama models require an accepted Hugging Face license and
an authenticated account.

Options for each `rating` run can be appended after `--`, for example:

```bash
.venv/bin/python -m experiments rating-matrix --dry-run -- \
  --max-test-samples 20
```

For a quick pipeline check, restrict the sample counts:

```bash
python3 -m experiments rating \
  --dataset data/Small_All_Beauty_5_llama_filtered.json \
  --model qwen_3b \
  --mode pretrained \
  --max-test-samples 20
```

## Outputs

Each run writes to
`experiments/outputs/<dataset>/<model>/<mode>/`:

- `run_config.json`: model, field mapping, seed, ratios, and split sizes
- `splits/*.jsonl`: the exact reproducible train/validation/test subsets
- `predictions.jsonl`: `predictRating`, `groundTruth`, parse status, and raw output
- `wrong_predictions.csv`: incorrect predictions, review text, signed difference
  (`predictRating - groundTruth`), and absolute difference, sorted by largest error
- `metrics.json`: RMSE, MAE, sample count, wrong-prediction rate, and valid parse rate

An unparsable model response is recorded with `validParse: false` and assigned
the neutral fallback rating 3, so `predictRating` always remains an integer in
the required range and failures are included in RMSE/MAE.
