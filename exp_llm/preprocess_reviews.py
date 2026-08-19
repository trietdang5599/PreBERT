#!/usr/bin/env python3
"""Filter non-evaluative review segments with an instruction-tuned LLM.

This replaces the former notebook pipeline that mixed Flair, VADER, spaCy,
temporary CSV files, and unrelated cleanup operations. Input and output are
JSON Lines files. Every source record is preserved and receives the fields
``filteredReviewText`` and ``overall_new``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Protocol, Sequence

try:
    from exp_llm.llm_settings import (
        DEFAULT_PREPROCESSING_MODEL,
        configure_runtime_environment,
    )
except ModuleNotFoundError:  # Direct execution from inside exp_llm/.
    from llm_settings import (
        DEFAULT_PREPROCESSING_MODEL,
        configure_runtime_environment,
    )

configure_runtime_environment()

DEFAULT_MODEL = DEFAULT_PREPROCESSING_MODEL
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|(?=[A-Z]))|[\r\n]+")

RELEVANCE_INSTRUCTION = """Decide whether the review segment contains sentiment or evaluative information.
Count opinions, emotions, preferences, satisfaction, dissatisfaction, product quality, performance, usability, value, and personal experience that helps infer a 1-5 rating.
Do not count shipping dates, product identifiers, neutral descriptions, or background facts with no evaluation.
Return exactly one digit: 1 if the segment is relevant, otherwise 0.

Segment:
{text}

Answer:"""

POLARITY_INSTRUCTION = """Classify the overall sentiment expressed by this Amazon review.
Return exactly one digit using this scale:
0 = negative
1 = neutral or mixed
2 = positive

Review:
{text}

Answer:"""


class ReviewClassifier(Protocol):
    def classify_relevance(self, texts: Sequence[str]) -> list[bool]: ...

    def classify_polarity(self, texts: Sequence[str]) -> list[int]: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Llama 3.2 3B Instruct to remove non-evaluative review segments "
            "and optionally reassess inconsistent ratings."
        )
    )
    parser.add_argument("input", type=Path, help="Source JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-field", default="reviewText")
    parser.add_argument("--rating-field", default="overall")
    parser.add_argument("--item-field", default="asin")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--review-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: auto, preferring CUDA then MPS)",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--empty-review-policy",
        choices=("keep-original", "empty"),
        default="keep-original",
        help=(
            "What to store when every segment is classified as non-evaluative "
            "(default: preserve the original review)"
        ),
    )
    parser.add_argument(
        "--adjust-ratings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adjust ratings whose binary direction conflicts with LLM polarity",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and print configuration without loading the model",
    )
    args = parser.parse_args(argv)

    if args.batch_size <= 0 or args.review_batch_size <= 0:
        parser.error("batch sizes must be greater than zero")
    if args.max_length < 32:
        parser.error("--max-length must be at least 32")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be greater than zero")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ; write to a new file first")
    if args.output.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"output already exists: {args.output} (use --overwrite)")
    return args


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def split_review(text: Any) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    normalized = re.sub(r"[ \t]+", " ", text.replace("\\/", "/")).strip()
    return [segment.strip() for segment in SENTENCE_BOUNDARY.split(normalized) if segment.strip()]


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def rating_value(row: dict[str, Any], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 1 <= value <= 5 else None


def item_medians(
    rows: Sequence[dict[str, Any]], item_field: str, rating_field: str
) -> dict[str, float]:
    ratings: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        item = row.get(item_field)
        rating = rating_value(row, rating_field)
        if item is not None and rating is not None:
            ratings[str(item)].append(rating)
    return {item: float(median(values)) for item, values in ratings.items()}


def adjusted_rating(original: float, item_median: float, polarity: int) -> float:
    rating_is_positive = original >= 4
    polarity_is_positive = polarity == 2
    polarity_is_negative = polarity == 0
    disagrees = (polarity_is_positive and not rating_is_positive) or (
        polarity_is_negative and rating_is_positive
    )
    return float(round((original + item_median) / 2)) if disagrees else original


def preprocess_batch(
    rows: Sequence[dict[str, Any]],
    classifier: ReviewClassifier,
    medians: dict[str, float],
    *,
    text_field: str,
    rating_field: str,
    item_field: str,
    empty_review_policy: str,
    adjust_ratings: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    split_rows = [split_review(row.get(text_field)) for row in rows]
    segments = [segment for parts in split_rows for segment in parts]
    decisions = classifier.classify_relevance(segments) if segments else []
    if len(decisions) != len(segments):
        raise RuntimeError("Relevance classifier returned an unexpected result count")

    cursor = 0
    filtered_texts: list[str] = []
    removed_counts: list[int] = []
    for parts in split_rows:
        part_decisions = decisions[cursor : cursor + len(parts)]
        cursor += len(parts)
        kept = [part for part, keep in zip(parts, part_decisions) if keep]
        removed_counts.append(len(parts) - len(kept))
        if not kept and parts and empty_review_policy == "keep-original":
            kept = parts
        filtered_texts.append(" ".join(kept))

    polarities = (
        classifier.classify_polarity(filtered_texts) if adjust_ratings else [1] * len(rows)
    )
    if len(polarities) != len(rows):
        raise RuntimeError("Polarity classifier returned an unexpected result count")

    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row, filtered_text, removed, polarity in zip(
        rows, filtered_texts, removed_counts, polarities
    ):
        result = dict(row)
        result["filteredReviewText"] = filtered_text
        original = rating_value(row, rating_field)
        new_rating = original
        if adjust_ratings and original is not None:
            item = str(row.get(item_field))
            new_rating = adjusted_rating(original, medians.get(item, original), polarity)
            if new_rating != original:
                stats["ratings_adjusted"] += 1
        result["overall_new"] = new_rating if new_rating is not None else row.get(rating_field)
        output.append(result)
        stats["reviews"] += 1
        stats["segments_removed"] += removed
        stats["empty_filtered_reviews"] += int(not filtered_text)
    return output, stats


class LlamaReviewClassifier:
    def __init__(
        self,
        model_id: str,
        *,
        batch_size: int,
        max_length: int,
        device: str,
        trust_remote_code: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install dependencies with: pip install -r exp_llm/requirements.txt"
            ) from exc

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "--device mps was requested, but torch.backends.mps.is_available() "
                "is False. Use an arm64 Python/PyTorch build on macOS 13 or newer."
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        requested_device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if requested_device.type == "cuda":
            model_kwargs.update(
                torch_dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16,
                device_map="auto",
            )
        elif requested_device.type == "mps":
            model_kwargs.update(torch_dtype=torch.float16, low_cpu_mem_usage=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if requested_device.type != "cuda":
            self.model.to(requested_device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _render_prompt(self, instruction: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a strict text classifier. Follow the output contract.",
            },
            {"role": "user", "content": instruction},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"System: {messages[0]['content']}\n\nUser: {instruction}\n\nAssistant:"

    def _label_tokens(self, labels: Sequence[str]) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for label in labels:
            token_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(token_ids) != 1:
                raise RuntimeError(f"Label {label!r} is not a single token")
            mapping[int(token_ids[0])] = label
        if len(mapping) != len(labels):
            raise RuntimeError("Classification labels do not have unique token IDs")
        return mapping

    def _classify(self, prompts: Sequence[str], labels: Sequence[str]) -> list[str]:
        if not prompts:
            return []
        label_tokens = self._label_tokens(labels)
        label_ids = list(label_tokens)
        rendered = [self._render_prompt(prompt) for prompt in prompts]
        # Adjacent prompts of similar length need much less padding. Keep their
        # original indices so classification results still align with inputs.
        indexed_prompts = sorted(enumerate(rendered), key=lambda pair: len(pair[1]))
        results: list[str | None] = [None] * len(rendered)
        output_embeddings = self.model.get_output_embeddings()
        base_model = getattr(self.model, self.model.base_model_prefix)
        candidate_ids = self.torch.tensor(
            label_ids,
            dtype=self.torch.long,
            device=self.device,
        )
        candidate_weights = output_embeddings.weight.index_select(
            0, candidate_ids
        ).detach()
        output_bias = getattr(output_embeddings, "bias", None)
        candidate_bias = (
            output_bias.index_select(0, candidate_ids).detach()
            if output_bias is not None
            else None
        )

        for indexed_batch in batched(indexed_prompts, self.batch_size):
            original_indices = [index for index, _prompt in indexed_batch]
            prompt_batch = [prompt for _index, prompt in indexed_batch]
            encoded = self.tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                # Run only the transformer backbone, disable the unused KV
                # cache, and project the final hidden state onto 2-3 label rows
                # instead of every token in Llama's full vocabulary.
                hidden = base_model(
                    **encoded,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, -1, :]
                candidate_scores = hidden @ candidate_weights.transpose(0, 1)
                if candidate_bias is not None:
                    candidate_scores += candidate_bias
                selected = candidate_scores.argmax(dim=-1).cpu().tolist()
            for original_index, label_index in zip(original_indices, selected):
                results[original_index] = label_tokens[label_ids[label_index]]

        if any(result is None for result in results):
            raise RuntimeError("Classifier failed to produce all requested labels")
        return [str(result) for result in results]

    def classify_relevance(self, texts: Sequence[str]) -> list[bool]:
        prompts = [RELEVANCE_INSTRUCTION.format(text=text) for text in texts]
        return [label == "1" for label in self._classify(prompts, ("0", "1"))]

    def classify_polarity(self, texts: Sequence[str]) -> list[int]:
        prompts = [POLARITY_INSTRUCTION.format(text=text or "[empty]") for text in texts]
        return [int(label) for label in self._classify(prompts, ("0", "1", "2"))]


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = read_jsonl(args.input, args.max_samples)
    segment_counts = [len(split_review(row.get(args.text_field))) for row in rows]
    missing_text = sum(count == 0 for count in segment_counts)
    classification_count = sum(segment_counts) + (len(rows) if args.adjust_ratings else 0)
    print("LLM review preprocessing")
    print(f"  input: {args.input}")
    print(f"  output: {args.output}")
    print(f"  model: {args.model}")
    print(f"  rows: {len(rows)}")
    print(f"  rows without text: {missing_text}")
    print(f"  review segments: {sum(segment_counts)}")
    print(f"  LLM classifications: {classification_count}")
    print(f"  adjust ratings: {args.adjust_ratings}")
    print(f"  requested device: {args.device}")
    if args.dry_run:
        return

    classifier = LlamaReviewClassifier(
        args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"  active device: {classifier.device}")
    medians = item_medians(rows, args.item_field, args.rating_field)
    processed: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    completed = 0
    started_at = time.perf_counter()
    for row_batch in batched(rows, args.review_batch_size):
        output_batch, stats = preprocess_batch(
            row_batch,
            classifier,
            medians,
            text_field=args.text_field,
            rating_field=args.rating_field,
            item_field=args.item_field,
            empty_review_policy=args.empty_review_policy,
            adjust_ratings=args.adjust_ratings,
        )
        processed.extend(output_batch)
        totals.update(stats)
        completed += len(row_batch)
        elapsed = time.perf_counter() - started_at
        reviews_per_second = completed / elapsed
        remaining_seconds = (len(rows) - completed) / reviews_per_second
        print(
            f"Processed {completed}/{len(rows)} | {reviews_per_second:.2f} reviews/s "
            f"| ETA {remaining_seconds / 60:.1f} min",
            flush=True,
        )

    write_jsonl_atomic(args.output, processed)
    print("\nResults")
    print(json.dumps(dict(totals), indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
