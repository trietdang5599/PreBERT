"""Shared dataset and artifact helpers for PreBERT experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helper.utils import read_data


DATA_COLUMNS = (
    "reviewerID",
    "asin",
    "overall",
    "overall_new",
    "reviewText",
    "filteredReviewText",
)


def create_dataframes(
    json_file: str | Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a dataset and create deterministic warm-start splits."""
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train, validation, and test ratios must sum to 1")
    if min(train_ratio, valid_ratio, test_ratio) <= 0:
        raise ValueError("train, validation, and test ratios must be positive")
    data = read_data(str(json_file))
    frame = pd.DataFrame(data, columns=DATA_COLUMNS)

    rng = np.random.default_rng(seed)
    user_values = frame["reviewerID"].astype(str).to_numpy()
    item_values = frame["asin"].astype(str).to_numpy()
    user_counts = pd.Series(user_values).value_counts().to_dict()
    item_counts = pd.Series(item_values).value_counts().to_dict()
    target_holdout = int(round((valid_ratio + test_ratio) * len(frame)))
    holdout_indices: list[int] = []

    for index in rng.permutation(len(frame)):
        user = user_values[index]
        item = item_values[index]
        if user_counts[user] > 1 and item_counts[item] > 1:
            holdout_indices.append(int(index))
            user_counts[user] -= 1
            item_counts[item] -= 1
            if len(holdout_indices) >= target_holdout:
                break

    if len(holdout_indices) < target_holdout:
        print(
            f"Warm-start split retained only {len(holdout_indices)}/{target_holdout} "
            "requested validation/test rows because the sampled dataset contains "
            "cold-start users/items."
        )
    if len(holdout_indices) < 2:
        raise ValueError("Not enough warm-start interactions for validation and test")

    rng.shuffle(holdout_indices)
    valid_share = valid_ratio / (valid_ratio + test_ratio)
    valid_size = max(1, int(round(len(holdout_indices) * valid_share)))
    valid_indices = holdout_indices[:valid_size]
    test_indices = holdout_indices[valid_size:]
    train_indices = np.setdiff1d(
        np.arange(len(frame)),
        np.asarray(holdout_indices),
        assume_unique=False,
    )

    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    valid_frame = frame.iloc[valid_indices].reset_index(drop=True)
    test_frame = frame.iloc[test_indices].reset_index(drop=True)

    train_users = set(train_frame["reviewerID"].astype(str))
    train_items = set(train_frame["asin"].astype(str))
    for split_name, split_frame in (("valid", valid_frame), ("test", test_frame)):
        if not set(split_frame["reviewerID"].astype(str)).issubset(train_users):
            raise RuntimeError(f"{split_name} contains users absent from train")
        if not set(split_frame["asin"].astype(str)).issubset(train_items):
            raise RuntimeError(f"{split_name} contains items absent from train")

    print(
        f"Split sizes: train={len(train_frame)}, valid={len(valid_frame)}, "
        f"test={len(test_frame)} (100% warm-start validation/test)"
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
