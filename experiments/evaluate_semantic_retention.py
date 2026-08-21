#!/usr/bin/env python3
"""Measure how much semantics preprocessing preserves while shortening reviews."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from helper.llm_settings import (
        DEFAULT_SEMANTIC_MODEL,
        configure_runtime_environment,
    )
except ModuleNotFoundError:  # Direct execution from inside experiments/.
    from helper.llm_settings import (
        DEFAULT_SEMANTIC_MODEL,
        configure_runtime_environment,
    )

configure_runtime_environment()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = DEFAULT_SEMANTIC_MODEL


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare filteredReviewText with the original reviewText and/or "
            "the reviewer-written summary using normalized embedding cosine similarity."
        )
    )
    parser.add_argument("input", type=Path, help="Input JSON Lines dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processed-field", default="filteredReviewText")
    parser.add_argument(
        "--reference-fields",
        nargs="+",
        default=["reviewText", "summary"],
        help="Reference text fields to compare against (default: reviewText summary)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only-changed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate only rows where processed text differs from reviewText "
            "after whitespace normalization (default: true)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_length < 8:
        parser.error("--max-length must be at least 8")
    if not -1 <= args.threshold <= 1:
        parser.error("--threshold must be between -1 and 1")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def text_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalized_text(value: Any) -> str:
    return " ".join(text_value(value).split())


def batched(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class TextEmbedder:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    @staticmethod
    def _mean_pool(
        hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for text_batch in batched(texts, self.batch_size):
            encoded = self.tokenizer(
                list(text_batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                model_output = self.model(**encoded)
                embeddings = self._mean_pool(
                    model_output.last_hidden_state,
                    encoded["attention_mask"],
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
            outputs.append(embeddings.cpu())
        return torch.cat(outputs, dim=0)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int = 2000
) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if len(values) else math.nan
        return value, value
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def metric_summary(
    values: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "count": 0,
            "mean": None,
            "mean_ci95_low": None,
            "mean_ci95_high": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
            f"rate_at_or_above_{threshold:.2f}": None,
        }
    ci_low, ci_high = bootstrap_mean_ci(finite, rng)
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "mean_ci95_low": ci_low,
        "mean_ci95_high": ci_high,
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        f"rate_at_or_above_{threshold:.2f}": float((finite >= threshold).mean()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_rows = read_jsonl(args.input, args.max_samples)
    indexed_rows = list(enumerate(input_rows))
    changed_count = sum(
        normalized_text(row.get("reviewText"))
        != normalized_text(row.get(args.processed_field))
        for row in input_rows
    )
    if args.only_changed:
        indexed_rows = [
            (index, row)
            for index, row in indexed_rows
            if normalized_text(row.get("reviewText"))
            != normalized_text(row.get(args.processed_field))
        ]
    if not indexed_rows:
        raise ValueError(
            f"No changed reviews found between reviewText and {args.processed_field}"
        )
    source_indices, rows = zip(*indexed_rows)
    rows = list(rows)
    device = select_device(args.device)
    print(f"Input rows: {len(input_rows)}")
    print(f"Changed rows: {changed_count}")
    print(f"Rows evaluated: {len(rows)}")
    print(f"Embedding model: {args.model}")
    print(f"Device: {device}")

    processed = [text_value(row.get(args.processed_field)) for row in rows]
    embedder = TextEmbedder(args.model, device, args.batch_size, args.max_length)
    processed_embeddings = embedder.encode(processed)

    result = pd.DataFrame(
        {
            "source_row_index": source_indices,
            "reviewerID": [row.get("reviewerID") for row in rows],
            "asin": [row.get("asin") for row in rows],
            "original_word_count": [
                len(text_value(row.get("reviewText")).split()) for row in rows
            ],
            "processed_word_count": [len(text.split()) for text in processed],
        }
    )
    original_lengths = result["original_word_count"].to_numpy(dtype=np.float64)
    processed_lengths = result["processed_word_count"].to_numpy(dtype=np.float64)
    result["retained_word_ratio"] = np.divide(
        processed_lengths,
        original_lengths,
        out=np.zeros_like(processed_lengths),
        where=original_lengths > 0,
    )
    result["removed_word_ratio"] = 1.0 - result["retained_word_ratio"]

    summary: dict[str, Any] = {
        "input": str(args.input),
        "model": args.model,
        "device": str(device),
        "processed_field": args.processed_field,
        "selection": "changed_only" if args.only_changed else "all_reviews",
        "threshold": args.threshold,
        "input_rows": len(input_rows),
        "changed_rows": changed_count,
        "rows_evaluated": len(rows),
        "mean_retained_word_ratio": float(result["retained_word_ratio"].mean()),
        "mean_removed_word_ratio": float(result["removed_word_ratio"].mean()),
        "empty_processed_rate": float((result["processed_word_count"] == 0).mean()),
        "references": {},
    }
    rng = np.random.default_rng(args.seed)
    for field in args.reference_fields:
        references = [text_value(row.get(field)) for row in rows]
        reference_embeddings = embedder.encode(references)
        similarities = F.cosine_similarity(
            reference_embeddings,
            processed_embeddings,
            dim=1,
        ).numpy()
        similarities = np.clip(similarities, -1.0, 1.0)
        empty_reference = np.array([not text for text in references])
        empty_processed = np.array([not text for text in processed])
        similarities[empty_reference | empty_processed] = np.nan
        column = f"semantic_similarity_{field}"
        result[column] = similarities
        removed = result["removed_word_ratio"].to_numpy()
        summary["references"][field] = {
            "evaluated_reviews": metric_summary(similarities, args.threshold, rng),
            "removed_at_least_10_percent": metric_summary(
                similarities[removed >= 0.10], args.threshold, rng
            ),
            "removed_at_least_25_percent": metric_summary(
                similarities[removed >= 0.25], args.threshold, rng
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "semantic_retention_per_review.csv"
    summary_path = args.output_dir / "semantic_retention_summary.json"
    result.to_csv(detail_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
