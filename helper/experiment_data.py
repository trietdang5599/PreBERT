"""Shared dataset and artifact helpers for PreBERT experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

def _read_jsonl_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def create_dataframes(
    json_file: str | Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the immutable pre-preprocessing train/validation/test assignment.

    ``train_ratio``, ``valid_ratio``, ``test_ratio``, and ``seed`` remain in
    the signature for API compatibility. Splitting is owned exclusively by
    ``preprocessing_reviews.py split`` and is never repeated during an
    experiment.
    """
    del train_ratio, valid_ratio, test_ratio, seed
    dataset_path = Path(json_file)
    split_dir = dataset_path.parent / "splits" / dataset_path.stem
    paths = {
        "train": split_dir / "train.json",
        "validation": split_dir / "val.json",
        "test": split_dir / "test.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Precomputed dataset split(s) not found: " + ", ".join(missing)
        )

    rows = {name: _read_jsonl_records(path) for name, path in paths.items()}
    frames = {name: pd.DataFrame(values) for name, values in rows.items()}
    required = {"reviewerID", "asin", "overall", "reviewText", "filteredReviewText"}
    for name, frame in frames.items():
        absent = required.difference(frame.columns)
        if absent:
            raise ValueError(
                f"{paths[name]} is missing required field(s): {sorted(absent)}"
            )
        if frame.empty:
            raise ValueError(f"{paths[name]} is empty")

    for name in ("train", "validation"):
        if (
            "overall_new" not in frames[name]
            or frames[name]["overall_new"].isna().any()
        ):
            raise ValueError(f"{paths[name]} must contain non-null overall_new")
    if "overall_new" in frames["test"].columns:
        raise ValueError(
            f"{paths['test']} must not contain overall_new; test ground truth is overall"
        )

    train_frame = frames["train"].reset_index(drop=True)
    valid_frame = frames["validation"].reset_index(drop=True)
    test_frame = frames["test"].reset_index(drop=True)
    frame = pd.concat(
        [train_frame, valid_frame, test_frame], ignore_index=True, sort=False
    )
    print(
        f"Loaded precomputed splits from {split_dir}: train={len(train_frame)}, "
        f"valid={len(valid_frame)}, test={len(test_frame)}"
    )
    return frame, train_frame, valid_frame, test_frame


def remove_generated_artifacts(remove_bert_checkpoint: bool = True) -> None:
    """Remove legacy global caches that cannot safely cross experiments."""
    split_names = ("train", "valid", "test", "vaild")
    feature_prefixes = (
        "allFeatureReview_",
        "reviewer_feature_",
        "item_feature_",
        "u_deep_",
        "i_deep_",
        "z_item_",
        "z_reviewer_",
        "transformed_udeep_",
        "transformed_ideep_",
    )
    paths = [
        Path("feature") / f"{prefix}{split}.csv"
        for prefix in feature_prefixes
        for split in split_names
    ]
    paths.extend(
        Path("data") / f"final_data_feature_DeepBERT_{split}.csv"
        for split in split_names
    )
    paths.append(Path("feature/interactions_train.csv"))
    for split in split_names:
        paths.extend(
            (
                Path("chkpt") / f"svd_{split}.pt",
                Path("chkpt") / f"fm_checkpoint_{split}.pkl",
                Path("chkpt") / f"encoded_features_{split}.npz",
            )
        )
    paths.append(Path("chkpt/DeepBERT.pt"))
    if remove_bert_checkpoint:
        paths.append(Path("chkpt/bert_last_checkpoint.pt"))

    removed = 0
    for path in paths:
        if path.is_file():
            path.unlink()
            removed += 1
    print(f"Removed {removed} generated artifact(s) from the previous run")
