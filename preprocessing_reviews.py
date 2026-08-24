#!/usr/bin/env python3
"""Build, preprocess, and split Amazon review datasets.

The module combines dense k-core subset construction, LLM review filtering,
rating reassessment, and deterministic splitting. Test splits retain the
original ``overall`` and omit ``overall_new``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import math
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from helper.llm_settings import (
        DEFAULT_PREPROCESSING_MODEL,
        configure_runtime_environment,
    )
except ModuleNotFoundError:
    from helper.llm_settings import (
        DEFAULT_PREPROCESSING_MODEL,
        configure_runtime_environment,
    )

configure_runtime_environment()

DEFAULT_MODEL = DEFAULT_PREPROCESSING_MODEL
CLAUSE_BOUNDARY = re.compile(
    r"(?<=[;:])\s+|(?<=[—–])\s*|"
    r"(?<=,)\s+(?=(?:but|however|yet|while|whereas|although|though|"
    r"even though|instead|still|unfortunately|fortunately)\b)|"
    r"(?<=,)\s+(?=(?:and|so)\s+(?:I|we|it|they|this|that|the\s+"
    r"(?:product|item|toy|game|song|album))\b)",
    re.IGNORECASE,
)
ABBREVIATIONS = {
    "dr.", "e.g.", "etc.", "i.e.", "jr.", "mr.", "mrs.", "ms.",
    "no.", "prof.", "sr.", "st.", "u.k.", "u.s.", "vs.",
}

RELEVANCE_INSTRUCTION = """Decide whether the Amazon review segment should be kept for rating prediction.

KEEP segments that provide direct evidence for the rating: sentiment, preference,
recommended age or audience, engagement, repeated use, quality, performance,
usability, benefits, defects, value, comparisons, or recommendations. Concrete
evidence such as poor drilling, a broken part, loving a genre, or suitability
for a wedding is evaluative even without a rating word.

REMOVE content that does not provide direct evidence for the product rating: raw copied
ingredient/track/variant/specification lists; free-product or sponsored-review
disclosures; seller/shipping/packaging/availability-only details; signatures
and identifiers; URLs and promotions for other products; review-writing
commentary; unrelated personal history; copied manufacturer descriptions; and
redundant background. Ingredient effects and other reviewer interpretations
are evaluation and must be kept. Price and purchase details are kept only when
they support a value judgment such as overpriced, a bargain, or worth the cost.

Judge the TARGET itself. Context only resolves references and must not turn
metadata into rating evidence. Mentioning the product does not by itself make a
segment evaluative. Generic facts such as available variants, release dates, or
what comes in the box should be removed. If a sentence mixes background and
evaluation, keep only its evaluative target clause when segmentation permits.
Preserve negation, explicit star ratings, update statements, and reasons for a
positive, negative, or mixed opinion. When uncertain, KEEP only if removing the
target could change the rating or its explanation; otherwise REMOVE.

Domain-specific examples:
{few_shots}

Full sentence context:
{context}

Target segment:
{text}

Return exactly one label: KEEP or REMOVE.
Answer:"""

POLARITY_INSTRUCTION = """Classify the overall sentiment expressed by this Amazon review.
Return exactly one digit using this scale:
0 = negative
1 = neutral or mixed
2 = positive

Review:
{text}

Answer:"""


DOMAIN_RELEVANCE_EXAMPLES = {
    "all_beauty": (
        ("It smells wonderful and leaves my skin soft for hours.", "KEEP"),
        ("The applicator is awkward and wastes a lot of product.", "KEEP"),
        ("Salicylic acid helped clear my existing breakouts.", "KEEP"),
        (
            "Ingredients: Water, Glycerin, Cetearyl Alcohol, Fragrance, Phenoxyethanol.",
            "REMOVE",
        ),
        ("I received this product free in exchange for an honest review.", "REMOVE"),
        ("The bottle contains 8 fluid ounces.", "REMOVE"),
        ("I ordered it on May 3 and it arrived in a cardboard box.", "REMOVE"),
    ),
    "digital_music": (
        ("The vocals are powerful, but the production sounds muddy.", "KEEP"),
        ("I keep replaying this album because every track is memorable.", "KEEP"),
        ("I love the 80's.", "KEEP"),
        ("This beautiful song is perfect for a wedding first dance.", "KEEP"),
        ("Track list: Intro, First Light, Home Again, Finale.", "REMOVE"),
        ("The album contains twelve tracks and was released in 2012.", "REMOVE"),
        ("The MP3 file is four minutes long.", "REMOVE"),
    ),
    "toys_games": (
        ("He finds it interesting and plays with it often.", "KEEP"),
        (
            "It does a nice job in helping to develop both motor skills and mental acuity.",
            "KEEP",
        ),
        ("The holes were poorly drilled and the plywood splintered.", "KEEP"),
        (
            "This is too difficult for a five-year-old but works for ages seven and up.",
            "KEEP",
        ),
        ("Included items: red card, blue card, six tokens, instruction sheet.", "REMOVE"),
        ("I received this toy free from the manufacturer for review.", "REMOVE"),
        ("This was purchased for a 2 3/4 year old boy.", "REMOVE"),
        ("The box has a blue product code printed on the side.", "REMOVE"),
    ),
    "general": (
        ("I use it every day and it works exactly as expected.", "KEEP"),
        ("It broke after two uses and was a waste of money.", "KEEP"),
        ("I received a free sample in exchange for my honest opinion.", "REMOVE"),
        ("The package arrived on Tuesday.", "REMOVE"),
        ("The item number is printed below the barcode.", "REMOVE"),
    ),
}

DISCLOSURE_PATTERNS = (
    re.compile(
        r"\b(?:I|we)\s+(?:received|got|was given|were given|was sent|were sent)\b"
        r".{0,100}\b(?:in exchange for|for (?:an? )?(?:honest )?review|"
        r"review purposes?|free (?:sample|product|item|toy|copy)|"
        r"complimentary (?:sample|product|item|toy|copy))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:this|the)\s+(?:product|item|toy|album|copy)\s+was\s+"
        r"(?:provided|supplied|sent)\b.{0,60}\b(?:for review|free of charge)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:I|we)\b.{0,60}\b(?:bzzagent|momselect|vine voice|review program)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:I was not compensated|opinions? (?:are|remain) my own)\b",
        re.IGNORECASE,
    ),
)
RAW_LIST_HEADING = re.compile(
    r"\b(?:active\s+|inactive\s+)?ingredients?\b\s*(?:list\s*:|:)|"
    r"\b(?:track|product|variant|scent|specification|package contents?|included items?)"
    r"\s+list\b\s*:|\bhere (?:are|is)\b.{0,15}\bingredients?\b.{0,100}:\s*$",
    re.IGNORECASE,
)
PRODUCT_LIST_INTRO = re.compile(
    r"\b(?:list (?:to help|of)|available (?:colors|scents|variants)|"
    r"choose (?:your|a) favorite|included items?|package contents?|track list)\b",
    re.IGNORECASE,
)
INGREDIENT_VOCABULARY = re.compile(
    r"\b(?:water|aqua|acid|alcohol|glycerin|glyceryl|sodium|potassium|extract|"
    r"oil|fragrance|parfum|silica|stearate|sulfate|chloride|tocopher\w*|"
    r"phenoxyethanol|dimethicone|cellulose|sorbitol|colorant|ci\s*\d+)\b",
    re.IGNORECASE,
)
EVALUATIVE_SIGNAL = re.compile(
    r"\b(?:love|like|hate|dislike|favorite|best|worst|great|good|bad|beautiful|"
    r"excellent|amazing|awesome|perfect|poor|cheap|expensive|overpriced|"
    r"disappoint\w*|recommend\w*|worth|works?|worked|working|durable|sturdy|"
    r"easy|difficult|hard|soft|smooth|comfortable|uncomfortable|useful|useless|"
    r"fun|boring|enjoy\w*|effective|quality|smells?|sounds?|tastes?|plays?)\b",
    re.IGNORECASE,
)
STRONG_EVALUATIVE_SIGNAL = re.compile(
    r"\b(?:hate|dislike|best|worst|poorly|overpriced|disappoint\w*|ashamed|"
    r"recommend\w*|worth(?:less)?|waste|broke|broken|splinter\w*|irritat\w*|"
    r"defect\w*|flaw\w*|too young|too old|not (?:work|working|worth)|"
    r"perfect for|suitable|appropriate|purchase again|buy again|"
    r"[1-5]\s*(?:/\s*5|stars?))\b",
    re.IGNORECASE,
)
EXPERIENCE_SIGNAL = re.compile(
    r"\b(?:I|we|my|our|me|us|used?|using|tried|bought|purchased|received|"
    r"child|kid|son|daughter|husband|wife|friend)\b",
    re.IGNORECASE,
)

DOMAIN_ALIASES = {
    "ab": "all_beauty",
    "all-beauty": "all_beauty",
    "dm": "digital_music",
    "digital-music": "digital_music",
    "tg": "toys_games",
    "toys-and-games": "toys_games",
}


@dataclass(frozen=True)
class ReviewSegment:
    text: str
    context: str
    forced_relevance: bool | None = None
    decision_reason: str | None = None


class ReviewClassifier(Protocol):
    def classify_relevance(self, segments: Sequence[ReviewSegment]) -> list[bool]: ...

    def classify_polarity(self, texts: Sequence[str]) -> list[int]: ...


def parse_preprocess_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Llama 3.2 3B Instruct to remove non-evaluative review segments "
            "and optionally reassess inconsistent ratings."
        )
    )
    parser.add_argument("input", type=Path, help="Source JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--domain",
        choices=("auto", "all_beauty", "digital_music", "toys_games", "general"),
        default="auto",
        help="Few-shot example domain (default: infer from the input filename)",
    )
    parser.add_argument("--text-field", default="reviewText")
    parser.add_argument("--rating-field", default="overall")
    parser.add_argument("--item-field", default="asin")
    parser.add_argument(
        "--segmentation-mode",
        choices=("sentence", "hybrid"),
        default="hybrid",
        help=(
            "sentence=terminal punctuation/newlines only; hybrid additionally "
            "splits strong punctuation and selected contrastive conjunctions"
        ),
    )
    parser.add_argument(
        "--minimum-clause-words",
        type=int,
        default=2,
        help="Merge shorter clause fragments with a neighbour (default: 2)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--review-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--remove-margin",
        type=float,
        default=0.05,
        help=(
            "Require the REMOVE logit to exceed KEEP by this margin; larger "
            "values preserve more uncertain segments (default: 0.05)"
        ),
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=50,
        help="Random reviews added to the preprocessing audit (default: 50)",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Audit CSV path (default: <output>_preprocessing_audit.csv)",
    )
    parser.add_argument("--audit-seed", type=int, default=42)
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
    if args.remove_margin < 0:
        parser.error("--remove-margin must be non-negative")
    if args.minimum_clause_words < 1:
        parser.error("--minimum-clause-words must be positive")
    if args.audit_sample_size < 0:
        parser.error("--audit-sample-size must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be greater than zero")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ; write to a new file first")
    if args.output.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"output already exists: {args.output} (use --overwrite)")
    return args


def infer_domain(path: Path, requested: str = "auto") -> str:
    """Resolve the few-shot domain explicitly or from a dataset filename."""
    if requested != "auto":
        return DOMAIN_ALIASES.get(requested, requested)
    normalized = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    if "all_beauty" in normalized:
        return "all_beauty"
    if "digital_music" in normalized:
        return "digital_music"
    if "toys_and_games" in normalized or "toys_games" in normalized:
        return "toys_games"
    return "general"


def relevance_few_shots(domain: str) -> str:
    examples = DOMAIN_RELEVANCE_EXAMPLES.get(
        DOMAIN_ALIASES.get(domain, domain), DOMAIN_RELEVANCE_EXAMPLES["general"]
    )
    return "\n\n".join(
        f"Example segment: {text}\nDecision: {label}" for text, label in examples
    )


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


def is_disclosure(text: str) -> bool:
    return any(pattern.search(text) for pattern in DISCLOSURE_PATTERNS)


def is_raw_ingredient_list(text: str) -> bool:
    comma_parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(comma_parts) < 5:
        return False
    short_part_rate = sum(
        len(part.split()) <= 8 for part in comma_parts
    ) / len(comma_parts)
    return (
        len(INGREDIENT_VOCABULARY.findall(text)) >= 3
        and short_part_rate >= 0.70
    )


def is_catalog_entry(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\*?[A-Z][A-Za-z'& /-]{2,28}(?:\s+\([^)]{2,35}\))?\s+"
            r"[A-Z][^.!?]{2,100}[.!?]?",
            text.strip(),
        )
    ) and not bool(re.search(r"\b(?:I|we|my|our)\b", text, re.IGNORECASE))


def prohibited_content_labels(text: Any) -> list[str]:
    """Detect content that must never survive preprocessing."""
    if not isinstance(text, str) or not text.strip():
        return []
    labels: list[str] = []
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if unit.strip()
    ]
    if any(is_disclosure(unit) for unit in units):
        labels.append("disclosure")
    if any(is_raw_ingredient_list(unit) for unit in units):
        labels.append("raw_ingredient_list")
    catalog_count = sum(is_catalog_entry(unit) for unit in units)
    if PRODUCT_LIST_INTRO.search(text) and catalog_count >= 8:
        labels.append("raw_product_list")
    return labels


def annotate_segments(segments: Sequence[ReviewSegment]) -> list[ReviewSegment]:
    """Apply conservative, deterministic decisions around the LLM classifier."""
    catalog_indices = {
        index for index, segment in enumerate(segments) if is_catalog_entry(segment.text)
    }
    copied_catalog = len(catalog_indices) >= 8 and any(
        PRODUCT_LIST_INTRO.search(segment.text) for segment in segments
    )
    annotated: list[ReviewSegment] = []
    for index, segment in enumerate(segments):
        text = segment.text
        if is_disclosure(text):
            decision, reason = False, "disclosure"
        elif is_raw_ingredient_list(text):
            decision, reason = False, "raw_ingredient_list"
        elif RAW_LIST_HEADING.search(text):
            decision, reason = False, "raw_list_heading"
        elif copied_catalog and (
            index in catalog_indices or PRODUCT_LIST_INTRO.search(text)
        ):
            decision, reason = False, "raw_product_list"
        elif STRONG_EVALUATIVE_SIGNAL.search(text):
            decision, reason = True, "strong_evaluative_signal"
        elif EVALUATIVE_SIGNAL.search(text) and EXPERIENCE_SIGNAL.search(text):
            # Broad words such as "good", "like", and "fun" are not enough
            # by themselves: copied descriptions and variant catalogues often
            # contain them. Only force KEEP when they are tied to experience.
            decision, reason = True, "experiential_evaluative_signal"
        else:
            decision, reason = None, None
        annotated.append(
            ReviewSegment(
                text=text,
                context=segment.context,
                forced_relevance=decision,
                decision_reason=reason,
            )
        )
    return annotated


def _is_abbreviation(sentence_prefix: str) -> bool:
    match = re.search(r"([A-Za-z][A-Za-z.]*)$", sentence_prefix)
    if match is None:
        return False
    token = match.group(1).lower()
    return token in ABBREVIATIONS or bool(
        re.fullmatch(r"(?:[a-z]\.){2,}", token)
    )


def _sentence_chunks(paragraph: str) -> list[str]:
    """Split terminal punctuation without breaking decimals or abbreviations."""
    chunks: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:[\"')\]]*)?(?=\s+|$)", paragraph):
        punctuation = match.group(0)
        if punctuation.startswith(".") and _is_abbreviation(
            paragraph[start : match.start() + 1]
        ):
            following = paragraph[match.end() :].lstrip()
            next_word = re.match(r"[\"'(\[]*([A-Za-z]+)", following)
            sentence_starters = {
                "a", "he", "i", "it", "she", "that", "the", "they",
                "this", "we",
            }
            if next_word is None or next_word.group(1).lower() not in sentence_starters:
                continue
        end = match.end()
        chunk = paragraph[start:end].strip()
        if chunk:
            chunks.append(chunk)
        whitespace = re.match(r"\s+", paragraph[end:])
        start = end + (whitespace.end() if whitespace else 0)
    remainder = paragraph[start:].strip()
    if remainder:
        chunks.append(remainder)
    return chunks


def _merge_short_clauses(parts: Sequence[str], minimum_words: int) -> list[str]:
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and len(part.split()) < minimum_words:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    if (
        len(merged) > 1
        and len(merged[0].split()) < minimum_words
        and not merged[0].endswith(":")
    ):
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)
    return merged


def segment_review(
    text: Any,
    *,
    mode: str = "hybrid",
    minimum_clause_words: int = 2,
) -> list[ReviewSegment]:
    if not isinstance(text, str) or not text.strip():
        return []
    if mode not in {"sentence", "hybrid"}:
        raise ValueError("segmentation mode must be 'sentence' or 'hybrid'")
    normalized = re.sub(r"[ \t]+", " ", text.replace("\\/", "/")).strip()
    sentences = [
        sentence
        for paragraph in re.split(r"[\r\n]+", normalized)
        if paragraph.strip()
        for sentence in _sentence_chunks(paragraph.strip())
    ]
    segments: list[ReviewSegment] = []
    for sentence in sentences:
        parts = (
            _merge_short_clauses(
                CLAUSE_BOUNDARY.split(sentence), minimum_clause_words
            )
            if mode == "hybrid"
            else [sentence]
        )
        segments.extend(
            ReviewSegment(text=part, context=sentence) for part in parts
        )
    return annotate_segments(segments)


def split_review(
    text: Any,
    *,
    mode: str = "hybrid",
    minimum_clause_words: int = 2,
) -> list[str]:
    """Compatibility wrapper returning only segment text."""
    return [
        segment.text
        for segment in segment_review(
            text,
            mode=mode,
            minimum_clause_words=minimum_clause_words,
        )
    ]


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


def prepare_review_segments(
    rows: Sequence[dict[str, Any]],
    *,
    text_field: str,
    segmentation_mode: str,
    minimum_clause_words: int,
) -> list[list[ReviewSegment]]:
    """Step 1: normalize, split, and deterministically annotate each review."""
    return [
        segment_review(
            row.get(text_field),
            mode=segmentation_mode,
            minimum_clause_words=minimum_clause_words,
        )
        for row in rows
    ]


def resolve_relevance_decisions(
    split_rows: Sequence[Sequence[ReviewSegment]],
    classifier: ReviewClassifier,
) -> tuple[list[list[bool]], Counter[str]]:
    """Step 2: combine deterministic rules with LLM relevance decisions."""
    segments = [segment for parts in split_rows for segment in parts]
    unresolved = [
        segment for segment in segments if segment.forced_relevance is None
    ]
    model_decisions = classifier.classify_relevance(unresolved) if unresolved else []
    if len(model_decisions) != len(unresolved):
        raise RuntimeError("Relevance classifier returned an unexpected result count")

    flat_decisions: list[bool] = []
    model_cursor = 0
    for segment in segments:
        if segment.forced_relevance is None:
            flat_decisions.append(model_decisions[model_cursor])
            model_cursor += 1
        else:
            flat_decisions.append(segment.forced_relevance)

    decisions_by_review: list[list[bool]] = []
    cursor = 0
    for parts in split_rows:
        decisions_by_review.append(flat_decisions[cursor : cursor + len(parts)])
        cursor += len(parts)

    stats: Counter[str] = Counter()
    stats["segments_forced_keep"] = sum(
        segment.forced_relevance is True for segment in segments
    )
    stats["segments_forced_remove"] = sum(
        segment.forced_relevance is False for segment in segments
    )
    stats["segments_llm_classified"] = len(unresolved)
    return decisions_by_review, stats


def build_filtered_reviews(
    split_rows: Sequence[Sequence[ReviewSegment]],
    decisions_by_review: Sequence[Sequence[bool]],
    *,
    empty_review_policy: str,
) -> tuple[list[str], list[int]]:
    """Step 3: rebuild review text from the segments selected for retention."""
    if len(split_rows) != len(decisions_by_review):
        raise RuntimeError("Review decisions do not align with input reviews")
    filtered_texts: list[str] = []
    removed_counts: list[int] = []
    for parts, part_decisions in zip(split_rows, decisions_by_review):
        if len(parts) != len(part_decisions):
            raise RuntimeError("Segment decisions do not align with review segments")
        kept: list[str] = []
        for index, (part, keep) in enumerate(zip(parts, part_decisions)):
            if not keep:
                continue
            text = part.text
            if index > 0 and not part_decisions[index - 1]:
                text = re.sub(
                    r"^(?:and|but|however|yet|while|whereas|although|though|"
                    r"instead|still|unfortunately|fortunately)\b\s*[,;:]?\s*",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )
            if index + 1 < len(part_decisions) and not part_decisions[index + 1]:
                text = re.sub(r"\s*[,;:—–]\s*$", "", text)
            if text:
                kept.append(text)
        removed_counts.append(len(parts) - len(kept))
        if not kept and parts and empty_review_policy == "keep-original":
            # Never restore deterministic prohibited content. Unknown segments
            # may be restored to avoid manufacturing an empty training review.
            kept = [
                part.text
                for part in parts
                if part.forced_relevance is not False
                and not prohibited_content_labels(part.text)
            ]
        filtered_texts.append(" ".join(kept))
    return filtered_texts, removed_counts


def validate_filtered_reviews(filtered_texts: Sequence[str]) -> None:
    """Step 4: enforce preprocessing invariants before rating adjustment."""
    violations = [
        (index, prohibited_content_labels(text))
        for index, text in enumerate(filtered_texts)
        if prohibited_content_labels(text)
    ]
    if violations:
        preview = ", ".join(
            f"batch row {index}: {'/'.join(labels)}"
            for index, labels in violations[:5]
        )
        raise RuntimeError(f"Prohibited content survived preprocessing: {preview}")


def reassess_filtered_review_ratings(
    rows: Sequence[dict[str, Any]],
    filtered_texts: Sequence[str],
    classifier: ReviewClassifier,
    medians: dict[str, float],
    *,
    rating_field: str,
    item_field: str,
    adjust_ratings: bool,
) -> tuple[list[Any], int]:
    """Step 5: reassess ratings using only the already-filtered review text."""
    if len(rows) != len(filtered_texts):
        raise RuntimeError("Filtered reviews do not align with input rows")
    if not adjust_ratings:
        return [row.get(rating_field) for row in rows], 0
    polarities = classifier.classify_polarity(filtered_texts)
    if len(polarities) != len(rows):
        raise RuntimeError("Polarity classifier returned an unexpected result count")

    ratings: list[Any] = []
    adjusted_count = 0
    for row, polarity in zip(rows, polarities):
        original = rating_value(row, rating_field)
        if original is None:
            ratings.append(row.get(rating_field))
            continue
        item = str(row.get(item_field))
        new_rating = adjusted_rating(original, medians.get(item, original), polarity)
        ratings.append(new_rating)
        adjusted_count += int(new_rating != original)
    return ratings, adjusted_count


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
    segmentation_mode: str = "hybrid",
    minimum_clause_words: int = 2,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    split_rows = prepare_review_segments(
        rows,
        text_field=text_field,
        segmentation_mode=segmentation_mode,
        minimum_clause_words=minimum_clause_words,
    )
    decisions_by_review, stats = resolve_relevance_decisions(split_rows, classifier)
    filtered_texts, removed_counts = build_filtered_reviews(
        split_rows,
        decisions_by_review,
        empty_review_policy=empty_review_policy,
    )
    validate_filtered_reviews(filtered_texts)
    reassessed_ratings, adjusted_count = reassess_filtered_review_ratings(
        rows,
        filtered_texts,
        classifier,
        medians,
        rating_field=rating_field,
        item_field=item_field,
        adjust_ratings=adjust_ratings,
    )

    output: list[dict[str, Any]] = []
    stats["ratings_adjusted"] += adjusted_count
    for row, filtered_text, removed, new_rating in zip(
        rows, filtered_texts, removed_counts, reassessed_ratings
    ):
        result = dict(row)
        result["filteredReviewText"] = filtered_text
        result["overall_new"] = new_rating
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
        domain: str,
        remove_margin: float,
        device: str,
        trust_remote_code: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.domain = DOMAIN_ALIASES.get(domain, domain)
        self.remove_margin = remove_margin
        self.relevance_examples = relevance_few_shots(self.domain)
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
        # Preserve the target segment and answer suffix when an unusually long
        # sentence makes the few-shot prompt exceed max_length.
        self.tokenizer.truncation_side = "left"

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

    def _classify(
        self,
        prompts: Sequence[str],
        labels: Sequence[str],
        *,
        preferred_label: str | None = None,
        decision_margin: float = 0.0,
    ) -> list[str]:
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
                selected = candidate_scores.argmax(dim=-1)
                if preferred_label is not None and decision_margin > 0:
                    preferred_token_id = next(
                        token_id
                        for token_id, label in label_tokens.items()
                        if label == preferred_label
                    )
                    preferred_index = label_ids.index(preferred_token_id)
                    preferred_scores = candidate_scores[:, preferred_index]
                    best_scores = candidate_scores.gather(
                        1, selected.unsqueeze(1)
                    ).squeeze(1)
                    uncertain = (best_scores - preferred_scores) < decision_margin
                    selected = self.torch.where(
                        uncertain,
                        self.torch.full_like(selected, preferred_index),
                        selected,
                    )
                selected_indices = selected.cpu().tolist()
            for original_index, label_index in zip(
                original_indices, selected_indices
            ):
                results[original_index] = label_tokens[label_ids[label_index]]

        if any(result is None for result in results):
            raise RuntimeError("Classifier failed to produce all requested labels")
        return [str(result) for result in results]

    def classify_relevance(self, segments: Sequence[ReviewSegment]) -> list[bool]:
        prompts = [
            RELEVANCE_INSTRUCTION.format(
                text=segment.text,
                context=segment.context,
                few_shots=self.relevance_examples,
            )
            for segment in segments
        ]
        labels = self._classify(
            prompts,
            ("KEEP", "REMOVE"),
            preferred_label="KEEP",
            decision_margin=self.remove_margin,
        )
        return [label == "KEEP" for label in labels]

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


def write_preprocessing_audit(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    source_field: str,
    processed_field: str = "filteredReviewText",
    random_sample_size: int = 50,
    seed: int = 42,
) -> dict[str, int]:
    """Write all heavily shortened reviews plus a reproducible random sample."""
    heavy_indices: set[int] = set()
    for index, row in enumerate(rows):
        original_words = len(str(row.get(source_field) or "").split())
        processed_words = len(str(row.get(processed_field) or "").split())
        removed_ratio = (
            1.0 - processed_words / original_words if original_words else 0.0
        )
        if removed_ratio >= 0.25:
            heavy_indices.add(index)
    rng = np.random.default_rng(seed)
    sample_count = min(random_sample_size, len(rows))
    random_indices = set(
        int(value)
        for value in rng.choice(len(rows), size=sample_count, replace=False)
    )
    selected = sorted(heavy_indices | random_indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    fields = (
        "source_row_index",
        "reviewerID",
        "asin",
        "overall",
        "audit_reason",
        "removed_word_ratio",
        "reviewText",
        "filteredReviewText",
        "audit_decision",
        "audit_notes",
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in selected:
                row = rows[index]
                original = str(row.get(source_field) or "")
                processed = str(row.get(processed_field) or "")
                original_words = len(original.split())
                processed_words = len(processed.split())
                reasons = []
                if index in heavy_indices:
                    reasons.append("removed_at_least_25_percent")
                if index in random_indices:
                    reasons.append("random_overkeeping_audit")
                writer.writerow(
                    {
                        "source_row_index": index,
                        "reviewerID": row.get("reviewerID"),
                        "asin": row.get("asin"),
                        "overall": row.get("overall"),
                        "audit_reason": ";".join(reasons),
                        "removed_word_ratio": (
                            1.0 - processed_words / original_words
                            if original_words
                            else 0.0
                        ),
                        "reviewText": original,
                        "filteredReviewText": processed,
                        "audit_decision": "",
                        "audit_notes": "",
                    }
                )
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return {
        "heavy_removal_reviews": len(heavy_indices),
        "random_reviews": len(random_indices),
        "audit_rows": len(selected),
    }


def preprocess_main(argv: Sequence[str] | None = None) -> None:
    args = parse_preprocess_args(argv)
    rows = read_jsonl(args.input, args.max_samples)
    domain = infer_domain(args.input, args.domain)
    preview_segments = [
        segment_review(
            row.get(args.text_field),
            mode=args.segmentation_mode,
            minimum_clause_words=args.minimum_clause_words,
        )
        for row in rows
    ]
    segment_counts = [len(parts) for parts in preview_segments]
    missing_text = sum(count == 0 for count in segment_counts)
    relevance_classifications = sum(
        segment.forced_relevance is None
        for parts in preview_segments
        for segment in parts
    )
    rating_classifications = len(rows) if args.adjust_ratings else 0
    print("LLM review preprocessing")
    print(f"  input: {args.input}")
    print(f"  output: {args.output}")
    print(f"  model: {args.model}")
    print(f"  few-shot domain: {domain}")
    print(f"  REMOVE decision margin: {args.remove_margin}")
    print(f"  segmentation mode: {args.segmentation_mode}")
    print(f"  minimum clause words: {args.minimum_clause_words}")
    print(f"  preprocessing audit random sample: {args.audit_sample_size}")
    print(f"  rows: {len(rows)}")
    print(f"  rows without text: {missing_text}")
    print(f"  review segments: {sum(segment_counts)}")
    print(f"  LLM relevance classifications: {relevance_classifications}")
    print(f"  LLM rating classifications: {rating_classifications}")
    print(
        "  total LLM classifications: "
        f"{relevance_classifications + rating_classifications}"
    )
    print(f"  adjust ratings: {args.adjust_ratings}")
    print(f"  requested device: {args.device}")
    if args.dry_run:
        return

    classifier = LlamaReviewClassifier(
        args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        domain=domain,
        remove_margin=args.remove_margin,
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
            segmentation_mode=args.segmentation_mode,
            minimum_clause_words=args.minimum_clause_words,
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
    audit_path = args.audit_output or args.output.with_name(
        f"{args.output.stem}_preprocessing_audit.csv"
    )
    audit_stats = write_preprocessing_audit(
        audit_path,
        processed,
        source_field=args.text_field,
        random_sample_size=args.audit_sample_size,
        seed=args.audit_seed,
    )
    print("\nResults")
    print(json.dumps(dict(totals), indent=2))
    print(f"Saved: {args.output}")
    print(f"Audit: {audit_path} ({json.dumps(audit_stats)})")


DEFAULT_SOURCE_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Toys_and_Games_5.json.gz"
)
REQUIRED_FIELDS = {"reviewerID", "asin", "overall", "reviewText"}

# Target interaction densities measured from the full datasets in the thesis.
# The profile sampler scales the corresponding user/item counts to --target-size.
SAMPLING_PROFILES = {
    "digital-music": {"reviews_per_user": 1.88, "reviews_per_item": 3.47},
    "toys-games": {"reviews_per_user": 1.95, "reviews_per_item": 13.13},
}


def parse_build_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed-size subset from an Amazon review archive."
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=Path("data/raw/reviews_Toys_and_Games_5.json.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Small_Toys_and_Games_5_dense10k.json"),
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--target-size", type=int, default=10_000)
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument(
        "--sampling-profile",
        choices=("dense", *SAMPLING_PROFILES),
        default="dense",
        help=(
            "Subset strategy: dense preserves the legacy k-core behavior; "
            "domain profiles target the full dataset's interaction density."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.target_size <= 0 or args.k_core <= 0:
        parser.error("--target-size and --k-core must be positive")
    if args.sampling_profile == "dense" and args.target_size < args.k_core * args.k_core:
        parser.error("--target-size is too small for the requested k-core")
    return args


def download_file(url: str, destination: Path) -> None:
    """Download atomically and reuse an existing non-empty cache."""
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            print(f"Downloading: {url}")
            request = urllib.request.Request(
                url, headers={"User-Agent": "PreBERT-research/1.0"}
            )
            with urllib.request.urlopen(request) as response:
                shutil.copyfileobj(response, target, length=1024 * 1024)
        temporary.replace(destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    print(f"Saved source: {destination}")


def _parse_loose_json(line: bytes, line_number: int) -> dict[str, Any]:
    text = line.decode("utf-8").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # The 2014 release is documented as "loose JSON" (Python dict syntax).
        value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object at source line {line_number}")
    missing = REQUIRED_FIELDS.difference(value)
    if missing:
        raise ValueError(f"Missing {sorted(missing)} at source line {line_number}")
    return value


def load_reviews(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                rows.append(_parse_loose_json(line, line_number))
    frame = pd.DataFrame(rows)
    frame["reviewerID"] = frame["reviewerID"].astype(str)
    frame["asin"] = frame["asin"].astype(str)
    frame = frame.drop_duplicates(
        subset=["reviewerID", "asin", "unixReviewTime", "reviewText"],
        keep="first",
    ).reset_index(drop=True)
    print(f"Loaded {len(frame):,} unique reviews")
    return frame


def load_existing_subset(path: Path) -> pd.DataFrame:
    """Load a previously generated strict-JSONL subset without changing IDs."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"Existing subset is empty: {path}")
    frame = pd.DataFrame(records)
    frame["reviewerID"] = frame["reviewerID"].astype(str)
    frame["asin"] = frame["asin"].astype(str)
    return frame


def iterative_k_core(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return the maximal user-item k-core contained in ``frame``."""
    core = frame.copy()
    while not core.empty:
        user_counts = core["reviewerID"].value_counts()
        item_counts = core["asin"].value_counts()
        keep = core["reviewerID"].map(user_counts).ge(k) & core["asin"].map(
            item_counts
        ).ge(k)
        if bool(keep.all()):
            break
        core = core.loc[keep].copy()
    return core.reset_index(drop=True)


def _remove_edges(
    frame: pd.DataFrame, count: int, k: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Greedily remove up to ``count`` edges without breaking the k-core."""
    if count <= 0:
        return frame
    user_degree = frame["reviewerID"].value_counts().to_dict()
    item_degree = frame["asin"].value_counts().to_dict()
    removed: list[int] = []
    for index in rng.permutation(frame.index.to_numpy()):
        user = frame.at[index, "reviewerID"]
        item = frame.at[index, "asin"]
        if user_degree[user] > k and item_degree[item] > k:
            removed.append(int(index))
            user_degree[user] -= 1
            item_degree[item] -= 1
            if len(removed) == count:
                break
    return frame.drop(index=removed).reset_index(drop=True)


def _remove_degree_k_nodes(
    frame: pd.DataFrame,
    max_edges: int,
    k: int,
    rng: np.random.Generator,
    node_column: str,
    neighbor_column: str,
) -> pd.DataFrame:
    """Remove whole degree-k nodes while all neighboring nodes remain valid."""
    removable_budget = max_edges - (max_edges % k)
    if removable_budget < k:
        return frame

    node_degree = frame[node_column].value_counts()
    neighbor_degree = frame[neighbor_column].value_counts().to_dict()
    candidates = node_degree[node_degree.eq(k)].index.to_numpy(dtype=object)
    rng.shuffle(candidates)
    grouped_indices = frame.groupby(node_column, sort=False).indices
    removed: list[int] = []

    for node in candidates:
        if len(removed) + k > removable_budget:
            break
        indices = np.asarray(grouped_indices[node], dtype=int)
        impacts = frame.iloc[indices][neighbor_column].value_counts().to_dict()
        if all(neighbor_degree[value] - amount >= k for value, amount in impacts.items()):
            removed.extend(frame.index[indices].tolist())
            for value, amount in impacts.items():
                neighbor_degree[value] -= amount

    return frame.drop(index=removed).reset_index(drop=True)


def shrink_k_core(
    core: pd.DataFrame, target_size: int, k: int, seed: int
) -> pd.DataFrame:
    """Shrink a k-core to exactly ``target_size`` while retaining its invariant."""
    if len(core) < target_size:
        raise ValueError("The candidate k-core is smaller than the requested target")
    if len(core) == target_size:
        return core.reset_index(drop=True)

    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt)
        work = core.copy().reset_index(drop=True)
        while len(work) > target_size:
            excess = len(work) - target_size
            before = len(work)

            # Resolve the remainder first so later degree-k node removals can
            # reduce the graph in exact multiples of k.
            remainder = excess % k
            if remainder:
                work = _remove_edges(work, remainder, k, rng)
                if len(work) < before:
                    continue

            work = _remove_degree_k_nodes(
                work, excess, k, rng, "reviewerID", "asin"
            )
            if len(work) < before:
                continue
            work = _remove_degree_k_nodes(
                work, excess, k, rng, "asin", "reviewerID"
            )
            if len(work) < before:
                continue

            # Dense regions may have no degree-k nodes yet; consume their
            # surplus edges and expose removable boundary nodes.
            work = _remove_edges(work, excess, k, rng)
            if len(work) == before:
                break

        if len(work) == target_size:
            validate_subset(work, target_size, k)
            return work.sample(frac=1, random_state=seed).reset_index(drop=True)
    raise RuntimeError(
        "Could not shrink the candidate to the exact target while preserving "
        "the k-core. Try a different --seed or a larger candidate item limit."
    )


def select_dense_subset(
    reviews: pd.DataFrame, target_size: int, k: int, seed: int
) -> pd.DataFrame:
    """Find a compact high-activity candidate core, then prune it exactly."""
    ranked_items = reviews["asin"].value_counts().index
    limits = [250, 500, 1_000, 2_000, 4_000, 8_000, len(ranked_items)]
    limits = sorted({min(limit, len(ranked_items)) for limit in limits})
    reserve = max(1_000, target_size // 10)
    fallback: pd.DataFrame | None = None

    for limit in limits:
        candidate = reviews.loc[reviews["asin"].isin(ranked_items[:limit])]
        core = iterative_k_core(candidate, k)
        print(
            f"Top {limit:,} items -> {len(core):,} reviews, "
            f"{core['reviewerID'].nunique():,} users, {core['asin'].nunique():,} items"
        )
        if len(core) >= target_size:
            fallback = core
        if len(core) >= target_size + reserve:
            return shrink_k_core(core, target_size, k, seed)

    if fallback is not None:
        return shrink_k_core(fallback, target_size, k, seed)
    raise RuntimeError(
        f"No {k}-core with at least {target_size:,} reviews could be constructed"
    )


def profile_target_counts(profile: str, target_size: int) -> tuple[int, int]:
    """Return the user/item counts implied by a full-dataset profile."""
    try:
        values = SAMPLING_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown sampling profile: {profile}") from exc
    users = round(target_size / values["reviews_per_user"])
    items = round(target_size / values["reviews_per_item"])
    if min(users, items) <= 0:
        raise ValueError("Profile target counts must be positive")
    return users, items


def _top_entities(frame: pd.DataFrame, column: str, count: int) -> set[str]:
    values = frame[column].value_counts(sort=True)
    if len(values) < count:
        raise RuntimeError(
            f"Only {len(values):,} distinct {column} values are available; "
            f"profile requires {count:,}"
        )
    return set(values.head(count).index.astype(str))


def _profile_candidate(
    reviews: pd.DataFrame, target_users: int, target_items: int
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Construct an interaction pool with the requested entity cardinalities."""
    # Alternate user and item selection. Re-ranking after every restriction
    # avoids requesting entities that have no interactions in the final pool.
    items = _top_entities(reviews, "asin", target_items)
    users: set[str] = set()
    candidate = reviews
    for _ in range(8):
        candidate = reviews.loc[reviews["asin"].isin(items)]
        users = _top_entities(candidate, "reviewerID", target_users)
        candidate = reviews.loc[reviews["reviewerID"].isin(users)]
        items = _top_entities(candidate, "asin", target_items)

    candidate = reviews.loc[
        reviews["reviewerID"].isin(users) & reviews["asin"].isin(items)
    ].copy()
    active_users = set(candidate["reviewerID"].astype(str).unique())
    active_items = set(candidate["asin"].astype(str).unique())
    if len(active_users) != target_users or len(active_items) != target_items:
        raise RuntimeError(
            "Could not form a profile cohort with the requested user/item "
            "counts. Try a larger source dataset or a smaller --target-size."
        )
    return candidate, active_users, active_items


def _coverage_sample(
    candidate: pd.DataFrame,
    users: set[str],
    items: set[str],
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    """Sample exactly target_size interactions while retaining every entity."""
    if len(candidate) < target_size:
        raise RuntimeError(
            f"Profile candidate has {len(candidate):,} reviews; need {target_size:,}"
        )
    rng = np.random.default_rng(seed)
    shuffled = candidate.iloc[rng.permutation(len(candidate))].copy()
    selected_indices: set[Any] = set()

    # First cover all lower-cardinality entities, then cover entities not yet
    # represented. Selecting randomly among each entity's candidate edges keeps
    # the procedure deterministic but avoids input-order bias.
    for column, entities in (("asin", items), ("reviewerID", users)):
        for entity in sorted(entities):
            matching = shuffled.index[shuffled[column].astype(str).eq(entity)]
            if not any(index in selected_indices for index in matching):
                selected_indices.add(matching[0])

    if len(selected_indices) > target_size:
        raise RuntimeError(
            "Profile requires more entity-coverage edges than --target-size"
        )
    remaining = [index for index in shuffled.index if index not in selected_indices]
    selected_indices.update(remaining[: target_size - len(selected_indices)])
    result = shuffled.loc[
        [index for index in shuffled.index if index in selected_indices]
    ].copy()
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def select_profile_subset(
    reviews: pd.DataFrame, target_size: int, profile: str, seed: int
) -> pd.DataFrame:
    """Create a fixed-size subset matching a domain's user/item densities."""
    target_users, target_items = profile_target_counts(profile, target_size)
    candidate, users, items = _profile_candidate(reviews, target_users, target_items)
    subset = _coverage_sample(candidate, users, items, target_size, seed)
    actual_users = subset["reviewerID"].nunique()
    actual_items = subset["asin"].nunique()
    if actual_users != target_users or actual_items != target_items:
        raise AssertionError("Profile sampler did not retain every selected entity")
    return subset


def dataset_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    user_counts = frame["reviewerID"].value_counts()
    item_counts = frame["asin"].value_counts()
    return {
        "reviews": int(len(frame)),
        "users": int(len(user_counts)),
        "items": int(len(item_counts)),
        "density": float(len(frame) / (len(user_counts) * len(item_counts))),
        "mean_reviews_per_user": float(user_counts.mean()),
        "mean_reviews_per_item": float(item_counts.mean()),
        "min_reviews_per_user": int(user_counts.min()),
        "median_reviews_per_user": float(user_counts.median()),
        "min_reviews_per_item": int(item_counts.min()),
        "median_reviews_per_item": float(item_counts.median()),
        "rating_distribution": {
            str(key): int(value)
            for key, value in frame["overall"].value_counts().sort_index().items()
        },
    }


def validate_subset(frame: pd.DataFrame, target_size: int, k: int) -> None:
    if len(frame) != target_size:
        raise AssertionError(f"Expected {target_size} rows, found {len(frame)}")
    if frame["reviewerID"].value_counts().min() < k:
        raise AssertionError("User k-core invariant was violated")
    if frame["asin"].value_counts().min() < k:
        raise AssertionError("Item k-core invariant was violated")
    missing = REQUIRED_FIELDS.difference(frame.columns)
    if missing:
        raise AssertionError(f"Output is missing required fields: {sorted(missing)}")


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            for record in records:
                clean_record: dict[str, Any] = {}
                for key, value in record.items():
                    if isinstance(value, np.generic):
                        value = value.item()
                    if isinstance(value, float) and not math.isfinite(value):
                        value = None
                    clean_record[key] = value
                target.write(
                    json.dumps(clean_record, ensure_ascii=False, allow_nan=False) + "\n"
                )
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_sampling_report(
    subset: pd.DataFrame, args: argparse.Namespace, report_path: Path
) -> dict[str, Any]:
    report = dataset_statistics(subset)
    report.update(
        {
            "source_url": args.source_url,
            "source_cache": str(args.raw_cache),
            "output": str(args.output),
            "seed": args.seed,
            "k_core": args.k_core,
            "sampling_profile": args.sampling_profile,
        }
    )
    if args.sampling_profile != "dense":
        target_users, target_items = profile_target_counts(
            args.sampling_profile, args.target_size
        )
        report["profile_targets"] = {
            "users": target_users,
            "items": target_items,
            "reviews_per_user": SAMPLING_PROFILES[args.sampling_profile][
                "reviews_per_user"
            ],
            "reviews_per_item": SAMPLING_PROFILES[args.sampling_profile][
                "reviews_per_item"
            ],
        }
        report["profile_deviation"] = {
            "users": int(report["users"] - target_users),
            "items": int(report["items"] - target_items),
            "reviews_per_user": float(
                report["mean_reviews_per_user"]
                - SAMPLING_PROFILES[args.sampling_profile]["reviews_per_user"]
            ),
            "reviews_per_item": float(
                report["mean_reviews_per_item"]
                - SAMPLING_PROFILES[args.sampling_profile]["reviews_per_item"]
            ),
        }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_dataset_main(argv: Sequence[str] | None = None) -> None:
    args = parse_build_args(argv)
    report_path = args.output.with_suffix(".sampling_report.json")

    # Make the command safe to rerun from a notebook. If the exact subset is
    # already valid, keep it and recreate a missing/stale report instead of
    # failing with argparse exit code 2.
    if args.output.is_file() and not args.overwrite:
        subset = load_existing_subset(args.output)
        if args.sampling_profile == "dense":
            validate_subset(subset, args.target_size, args.k_core)
        else:
            target_users, target_items = profile_target_counts(
                args.sampling_profile, args.target_size
            )
            if subset["reviewerID"].nunique() != target_users or subset["asin"].nunique() != target_items:
                raise ValueError("Existing subset does not match the sampling profile")
        report = write_sampling_report(subset, args, report_path)
        print(f"Using existing valid subset: {args.output}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved report: {report_path}")
        return

    download_file(args.source_url, args.raw_cache)
    reviews = load_reviews(args.raw_cache)
    if args.sampling_profile == "dense":
        subset = select_dense_subset(reviews, args.target_size, args.k_core, args.seed)
        validate_subset(subset, args.target_size, args.k_core)
    else:
        subset = select_profile_subset(
            reviews, args.target_size, args.sampling_profile, args.seed
        )
    write_jsonl_atomic(args.output, subset.to_dict(orient="records"))

    report = write_sampling_report(subset, args, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved subset: {args.output}")
    print(f"Saved report: {report_path}")


SPLIT_MAPPING_FIELDS = ("reviewerID", "asin", "unixReviewTime", "overall")
SPLIT_PROFILES = {
    "8-1-1": (0.8, 0.1),
    "7-1-2": (0.7, 0.1),
}


def split_output_directory(dataset_path: Path, seed: int, profile: str) -> Path:
    """Return the stable physical location for a named split profile."""
    root = Path("data/splits") / dataset_path.stem
    # Retain the existing 8/1/1 layout for compatibility with prior results.
    if profile == "8-1-1":
        return root / f"seed_{seed}"
    return root / f"ratio_{profile.replace('-', '_')}" / f"seed_{seed}"


def _split_mapping_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in SPLIT_MAPPING_FIELDS)


def _mapping_signature(
    rows: Sequence[dict[str, Any]],
) -> Counter[tuple[Any, ...]]:
    return Counter(_split_mapping_key(row) for row in rows)


def _source_candidates(processed_path: Path) -> list[Path]:
    suffix = "_llama_filtered"
    if not processed_path.stem.endswith(suffix):
        return [processed_path]
    base = processed_path.stem[: -len(suffix)]
    return list(
        dict.fromkeys(
            (
                processed_path.with_name(f"{base}.json"),
                processed_path.with_name(f"{base}_dense10k.json"),
                processed_path.parent / "backup" / f"{base}.json",
                processed_path.parent / "backup" / f"{base}_dense10k.json",
            )
        )
    )


def resolve_split_source(
    processed_path: Path,
    processed_rows: Sequence[dict[str, Any]],
    explicit_source: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Resolve an original dataset containing exactly the processed reviews."""
    signature = _mapping_signature(processed_rows)
    candidates = [explicit_source] if explicit_source else _source_candidates(processed_path)
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        rows = read_jsonl(candidate)
        if _mapping_signature(rows) == signature:
            return candidate, rows
    checked = ", ".join(str(path) for path in candidates)
    raise ValueError(
        f"No original dataset matches {processed_path}. Checked: {checked}. "
        "Pass the correct file with --source."
    )


def _occurrence_map(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[tuple[Any, ...], int], dict[str, Any]]:
    occurrences: Counter[tuple[Any, ...]] = Counter()
    mapped = {}
    for row in rows:
        key = _split_mapping_key(row)
        occurrence = occurrences[key]
        occurrences[key] += 1
        mapped[(key, occurrence)] = row
    return mapped


def split_preprocessed_rows(
    source_rows: Sequence[dict[str, Any]],
    processed_rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_rating_field: str = "overall",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split original rows first, then attach already-preprocessed records."""
    processed_map = _occurrence_map(processed_rows)
    occurrences: Counter[tuple[Any, ...]] = Counter()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    labels = []
    for source in source_rows:
        key = _split_mapping_key(source)
        occurrence = occurrences[key]
        occurrences[key] += 1
        try:
            processed = processed_map[(key, occurrence)]
        except KeyError as exc:
            raise ValueError(f"Missing processed row for {key!r} occurrence {occurrence}") from exc
        rating = float(source.get("overall"))
        integer_rating = int(rating)
        if rating != integer_rating or integer_rating not in range(1, 6):
            raise ValueError(f"Invalid original overall value: {source.get('overall')!r}")
        pairs.append((source, processed))
        labels.append(integer_rating)

    train_size = round(len(pairs) * train_ratio)
    validation_size = round(len(pairs) * validation_ratio)
    train_pairs, holdout_pairs, _, holdout_labels = train_test_split(
        pairs,
        labels,
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )
    validation_pairs, test_pairs = train_test_split(
        holdout_pairs,
        train_size=validation_size,
        random_state=seed + 1,
        stratify=holdout_labels,
    )
    splits = [train_pairs, validation_pairs, test_pairs]

    outputs: list[list[dict[str, Any]]] = []
    for split_index, split in enumerate(splits):
        records = []
        for source, processed in split:
            record = dict(processed)
            record["overall"] = source["overall"]
            if split_index == 2 and test_rating_field == "overall":
                record.pop("overall_new", None)
            records.append(record)
        outputs.append(records)
    return outputs[0], outputs[1], outputs[2]


def parse_split_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic 80/10/10 splits from a processed JSONL file."
    )
    parser.add_argument("input", type=Path, help="Existing *_llama_filtered JSONL")
    parser.add_argument("--source", type=Path, help="Original pre-preprocessing JSONL")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-profile",
        choices=tuple(SPLIT_PROFILES),
        default="8-1-1",
        help="Named split ratio: 8-1-1 (80/10/10) or 7-1-2 (70/10/20).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--test-rating-field",
        choices=("overall", "overall_new"),
        default="overall",
        help=(
            "Rating field retained for test evaluation. Default removes "
            "overall_new so the original overall is the only test label."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.train_ratio, args.validation_ratio = SPLIT_PROFILES[args.split_profile]
    test_ratio = 1.0 - args.train_ratio - args.validation_ratio
    if min(args.train_ratio, args.validation_ratio, test_ratio) <= 0:
        parser.error("train, validation, and test ratios must all be positive")
    if not math.isclose(args.train_ratio + args.validation_ratio + test_ratio, 1.0):
        parser.error("split ratios must sum to 1")
    return args


def split_main(argv: Sequence[str] | None = None) -> None:
    args = parse_split_args(argv)
    processed_rows = read_jsonl(args.input)
    source_path, source_rows = resolve_split_source(
        args.input, processed_rows, args.source
    )
    train_rows, validation_rows, test_rows = split_preprocessed_rows(
        source_rows,
        processed_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_rating_field=args.test_rating_field,
    )
    output_dir = args.output_dir or split_output_directory(
        args.input, args.seed, args.split_profile
    )
    paths = {
        "train": output_dir / "train.json",
        "validation": output_dir / "val.json",
        "test": output_dir / "test.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Split output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    write_jsonl_atomic(paths["train"], train_rows)
    write_jsonl_atomic(paths["validation"], validation_rows)
    write_jsonl_atomic(paths["test"], test_rows)
    print(f"Split source: {source_path}")
    print(
        f"Saved train={len(train_rows)}, validation={len(validation_rows)}, "
        f"test={len(test_rows)} to {output_dir} "
        f"(profile: {args.split_profile}; test rating field: {args.test_rating_field})"
    )


COMMANDS = ("preprocess", "build-dataset", "split")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in COMMANDS:
        command = arguments.pop(0)
        handlers = {
            "preprocess": preprocess_main,
            "build-dataset": build_dataset_main,
            "split": split_main,
        }
        handlers[command](arguments)
        return
    if arguments and arguments[0] not in {"-h", "--help"}:
        # Preserve the former direct interface: preprocessing_reviews.py INPUT ...
        preprocess_main(arguments)
        return
    parser = argparse.ArgumentParser(
        description="Build, preprocess, and split Amazon review datasets."
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.parse_args(arguments)


if __name__ == "__main__":
    main()
