#!/usr/bin/env python3
"""Download Amazon Toys & Games 5-core and build an exact dense subset.

The common failure mode when taking 10,000 random reviews is that nearly every
user and item appears only once.  This module instead preserves a bipartite
k-core: every selected reviewer and product has at least ``k`` interactions.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_SOURCE_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Toys_and_Games_5.json.gz"
)
REQUIRED_FIELDS = {"reviewerID", "asin", "overall", "reviewText"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exact 10k dense subset from Amazon Toys & Games 5-core."
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=Path("data/raw/reviews_Toys_and_Games_5.json.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Small_Toys_and_Games_5.json"),
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--target-size", type=int, default=10_000)
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.target_size <= 0 or args.k_core <= 0:
        parser.error("--target-size and --k-core must be positive")
    if args.target_size < args.k_core * args.k_core:
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


def dataset_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    user_counts = frame["reviewerID"].value_counts()
    item_counts = frame["asin"].value_counts()
    return {
        "reviews": int(len(frame)),
        "users": int(len(user_counts)),
        "items": int(len(item_counts)),
        "density": float(len(frame) / (len(user_counts) * len(item_counts))),
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
        }
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report_path = args.output.with_suffix(".sampling_report.json")

    # Make the command safe to rerun from a notebook. If the exact subset is
    # already valid, keep it and recreate a missing/stale report instead of
    # failing with argparse exit code 2.
    if args.output.is_file() and not args.overwrite:
        subset = load_existing_subset(args.output)
        validate_subset(subset, args.target_size, args.k_core)
        report = write_sampling_report(subset, args, report_path)
        print(f"Using existing valid subset: {args.output}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved report: {report_path}")
        return

    download_file(args.source_url, args.raw_cache)
    reviews = load_reviews(args.raw_cache)
    subset = select_dense_subset(reviews, args.target_size, args.k_core, args.seed)
    validate_subset(subset, args.target_size, args.k_core)
    write_jsonl_atomic(args.output, subset.to_dict(orient="records"))

    report = write_sampling_report(subset, args, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved subset: {args.output}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
