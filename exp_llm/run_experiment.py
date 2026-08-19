#!/usr/bin/env python3
"""Evaluate pretrained language models for review rating prediction.

The input files in ``data/`` are JSON Lines files: one JSON object per line.
This script deliberately uses only the review text as model input. The rating is
used only as ground truth during evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error

try:
    from exp_llm.llm_settings import (
        MODE_DESCRIPTIONS,
        MODE_FIELDS,
        configure_runtime_environment,
        load_model_and_tokenizer,
        resolve_model_id,
    )
except ModuleNotFoundError:  # Direct execution from inside exp_llm/.
    from llm_settings import (
        MODE_DESCRIPTIONS,
        MODE_FIELDS,
        configure_runtime_environment,
        load_model_and_tokenizer,
        resolve_model_id,
    )


configure_runtime_environment()

SYSTEM_PROMPT = (
    "You are a rating classifier, not a conversational assistant. Predict the "
    "Amazon rating expressed by the review. STRICT OUTPUT CONTRACT: return "
    "exactly one ASCII digit from this set: 1, 2, 3, 4, 5. Never output words, "
    "an explanation, punctuation, a label, or more than one character."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained LLMs on review rating prediction."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path")
    parser.add_argument(
        "--model",
        default="qwen_3b",
        help=(
            "Model alias (for example qwen_3b, qwen2.5_3b, llama3.2_1b) or "
            "any Hugging Face model id/path"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_FIELDS),
        default="pretrained",
        help="Experiment type and fields to use",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("exp_llm/outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--constrain-rating-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict decoding to exactly one rating from 1 to 5 (enabled by "
            "default; disable with --no-constrain-rating-output)"
        ),
    )
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load the pretrained model in 4-bit for inference (CUDA only)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the Hugging Face model repository",
    )
    args = parser.parse_args(argv)

    ratios = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratios, 1.0, abs_tol=1e-8):
        parser.error("--train-ratio + --val-ratio + --test-ratio must equal 1")
    if min(args.train_ratio, args.val_ratio, args.test_ratio) <= 0:
        parser.error("all split ratios must be greater than zero")
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    for name in (
        "max_new_tokens",
        "inference_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.max_length < 8:
        parser.error("--max-length must be at least 8")
    return args


def read_jsonl(path: Path, text_field: str, rating_field: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

            text = raw.get(text_field)
            rating = raw.get(rating_field)
            if not isinstance(text, str) or not text.strip():
                skipped["missing_text"] += 1
                continue
            try:
                numeric_rating = float(rating)
            except (TypeError, ValueError):
                skipped["invalid_rating"] += 1
                continue
            integer_rating = int(numeric_rating)
            if numeric_rating != integer_rating or integer_rating not in range(1, 6):
                skipped["invalid_rating"] += 1
                continue
            rows.append(
                {
                    "source_index": line_number - 1,
                    "text": text.strip(),
                    "rating": integer_rating,
                }
            )

    if not rows:
        raise ValueError(
            f"No valid examples found using fields {text_field!r} and {rating_field!r}"
        )
    if skipped:
        print(f"Skipped rows: {dict(skipped)}")
    return rows


def stratified_split(
    rows: Sequence[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split each rating class into train/validation/test."""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["rating"]].append(row)

    rng = random.Random(seed)
    splits: list[list[dict[str, Any]]] = [[], [], []]
    for rating in sorted(groups):
        group = groups[rating][:]
        rng.shuffle(group)
        size = len(group)
        train_end = round(size * train_ratio)
        val_end = train_end + round(size * val_ratio)
        # Keep all three subsets represented when a rating class permits it.
        if size >= 3:
            train_end = min(max(train_end, 1), size - 2)
            val_end = min(max(val_end, train_end + 1), size - 1)
        splits[0].extend(group[:train_end])
        splits[1].extend(group[train_end:val_end])
        splits[2].extend(group[val_end:])

    for split in splits:
        rng.shuffle(split)
    return splits[0], splits[1], splits[2]


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows if limit is None else rows[:limit]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_wrong_predictions_csv(
    path: Path, predictions: Sequence[dict[str, Any]]
) -> int:
    """Write incorrectly predicted reviews and their rating differences to CSV."""
    wrong_predictions = []
    for row in predictions:
        signed_difference = row["predictRating"] - row["groundTruth"]
        if signed_difference == 0:
            continue
        wrong_predictions.append(
            {
                "source_index": row["source_index"],
                "reviewText": row["reviewText"],
                "groundTruth": row["groundTruth"],
                "predictRating": row["predictRating"],
                "signedDifference": signed_difference,
                "absoluteDifference": abs(signed_difference),
                "validParse": row["validParse"],
                "rawOutput": row["rawOutput"],
            }
        )

    # Put the largest errors first to make error analysis more convenient.
    wrong_predictions.sort(
        key=lambda row: (-row["absoluteDifference"], row["source_index"])
    )
    fieldnames = [
        "source_index",
        "reviewText",
        "groundTruth",
        "predictRating",
        "signedDifference",
        "absoluteDifference",
        "validParse",
        "rawOutput",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig lets Excel display Unicode review text without extra setup.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wrong_predictions)
    return len(wrong_predictions)


def user_prompt(review_text: str) -> str:
    return (
        "Classify the review using this rating scale:\n"
        # "1 = very negative\n"
        # "2 = negative\n"
        # "3 = neutral or mixed\n"
        # "4 = positive\n"
        # "5 = very positive\n\n"
        "Your entire response must be exactly one digit: 1, 2, 3, 4, or 5.\n"
        f"Review:\n{review_text}\n\n"
        "Answer (one digit only):"
    )


def render_prompt(tokenizer: Any, review_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(review_text)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"User: {user_prompt(review_text)}\n\nAssistant:"
    )


RATING_PATTERN = re.compile(r"(?<!\d)([1-5])(?!\d)")


def parse_rating(generated_text: str) -> tuple[int, bool]:
    match = RATING_PATTERN.search(generated_text.strip())
    if match:
        return int(match.group(1)), True
    # Always produce the required integer while making failures visible in metrics.
    return 3, False


def rating_token_id_map(tokenizer: Any) -> dict[int, int]:
    """Map single-token representations of ratings to integer ratings."""
    token_map: dict[int, int] = {}
    missing_ratings = []
    for rating in range(1, 6):
        found_single_token = False
        # Support tokenizers that encode a digit differently after whitespace.
        for candidate in (str(rating), f" {rating}"):
            token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 1:
                token_map[token_ids[0]] = rating
                found_single_token = True
        if not found_single_token:
            missing_ratings.append(rating)
    if missing_ratings:
        raise RuntimeError(
            "Cannot constrain rating output because these ratings are not "
            f"single tokenizer tokens: {missing_ratings}. Run with "
            "--no-constrain-rating-output to use normal generation."
        )
    return token_map


def batched(rows: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def predict(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    rating_tokens = rating_token_id_map(tokenizer) if args.constrain_rating_output else {}
    allowed_rating_token_ids = list(rating_tokens)
    results: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(batched(rows, args.inference_batch_size), start=1):
        prompts = [render_prompt(tokenizer, row["text"]) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.constrain_rating_output:
            generation_kwargs.update(
                max_new_tokens=1,
                min_new_tokens=1,
                prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: (
                    allowed_rating_token_ids
                ),
            )
        else:
            generation_kwargs["max_new_tokens"] = args.max_new_tokens
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for position, (row, raw_output) in enumerate(zip(batch, texts)):
            if args.constrain_rating_output:
                prediction = rating_tokens[int(new_tokens[position, 0].item())]
                valid = True
            else:
                prediction, valid = parse_rating(raw_output)
            results.append(
                {
                    "source_index": row["source_index"],
                    "reviewText": row["text"],
                    "groundTruth": row["rating"],
                    "predictRating": prediction,
                    "validParse": valid,
                    "rawOutput": raw_output.strip(),
                }
            )
        print(f"Inference batch {batch_number}: {len(results)}/{len(rows)}", flush=True)
    return results


def calculate_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate RMSE using the same implementation as train.py."""
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    squared_errors = (y_true_np - y_pred_np) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return float(rmse)


def calculate_metrics(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot calculate metrics for an empty test set")
    targets = [row["groundTruth"] for row in predictions]
    predicts = [row["predictRating"] for row in predictions]
    num_wrong_predictions = sum(
        target != prediction for target, prediction in zip(targets, predicts)
    )
    return {
        "rmse": calculate_rmse(targets, predicts),
        "mae": float(mean_absolute_error(targets, predicts)),
        "numTestSamples": len(predictions),
        "numWrongPredictions": num_wrong_predictions,
        "wrongPredictionRate": num_wrong_predictions / len(predictions),
        "validParseRate": sum(row["validParse"] for row in predictions) / len(predictions),
        "fallbackPrediction": 3,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_id = resolve_model_id(args.model)
    text_field, rating_field = MODE_FIELDS[args.mode]
    rows = read_jsonl(args.dataset, text_field, rating_field)
    train_rows, val_rows, test_rows = stratified_split(
        rows, args.train_ratio, args.val_ratio, args.seed
    )
    train_rows = limit_rows(train_rows, args.max_train_samples)
    val_rows = limit_rows(val_rows, args.max_val_samples)
    test_rows = limit_rows(test_rows, args.max_test_samples)

    dataset_name = args.dataset.stem
    safe_model_name = model_id.replace("/", "--")
    run_dir = args.output_dir / dataset_name / safe_model_name / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir = run_dir / "splits"
    write_jsonl(split_dir / "train.jsonl", train_rows)
    write_jsonl(split_dir / "validation.jsonl", val_rows)
    write_jsonl(split_dir / "test.jsonl", test_rows)
    metadata = {
        "dataset": str(args.dataset),
        "model": model_id,
        "mode": args.mode,
        "modeDescription": MODE_DESCRIPTIONS[args.mode],
        "constrainRatingOutput": args.constrain_rating_output,
        "textField": text_field,
        "ratingField": rating_field,
        "seed": args.seed,
        "splitRatios": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": args.test_ratio,
        },
        "splitSizes": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
        },
        "hyperparameters": {
            "maxLength": args.max_length,
            "inferenceBatchSize": args.inference_batch_size,
            "maxNewTokens": args.max_new_tokens,
            "use4Bit": args.use_4bit,
        },
        "ratingDistribution": dict(sorted(Counter(row["rating"] for row in rows).items())),
    }
    write_json(run_dir / "run_config.json", metadata)
    print(json.dumps(metadata, indent=2))

    model, tokenizer = load_model_and_tokenizer(args, model_id)
    predictions = predict(args, model, tokenizer, test_rows)
    metrics = calculate_metrics(predictions)
    metrics.update({"model": model_id, "mode": args.mode, "dataset": str(args.dataset)})
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    wrong_count = write_wrong_predictions_csv(
        run_dir / "wrong_predictions.csv", predictions
    )
    write_json(run_dir / "metrics.json", metrics)
    print("\nResults")
    print(json.dumps(metrics, indent=2))
    print(f"Wrong predictions collected: {wrong_count}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
