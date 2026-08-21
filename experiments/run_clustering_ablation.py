#!/usr/bin/env python3
"""Run the PreBERT clustering ablation with k-topic fixed at 40.

BERT training, train-review BERT embeddings, and coarse sentiment scores are
cached once per dataset. Each clustering method only rebuilds the features
that depend on topic clusters and retrains the small downstream regressor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Let unsupported Metal operations fall back to CPU instead of aborting.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch

from helper.device import get_device, release_device_cache
from helper.experiment_data import create_dataframes, remove_generated_artifacts
from helper.utils import csv_to_dataloader
from train import prepare_deepbert_splits, test, test_rsme, train_deepbert


NUM_TOPICS = 40
FEATURE_PIPELINE_VERSION = 4
DEFAULT_BERT_MODEL = "answerdotai/ModernBERT-base"
LEGACY_BERT_MODEL = "bert-base-uncased"
BERT_MODEL_ALIASES = {
    "bert-base": LEGACY_BERT_MODEL,
    "bert-base-uncased": LEGACY_BERT_MODEL,
    "modernbert-base": DEFAULT_BERT_MODEL,
    "mmbert-base": "jhu-clsp/mmBERT-base",
}
DEFAULT_DATASETS = [
    "Small_All_Beauty_5_llama_filtered",
    "Small_Digital_Music_5_llama_filtered",
    "Small_Toys_and_Games_5_llama_filtered",
]
CLUSTER_METHODS = {
    "kmeans": ("KMeans", "kmeans"),
    "birch": ("Birch", "birch"),
    # "bisecting-kmeans": ("BisectingKMeans", "bisecting_kmeans"),
    "bisectingkmeans": ("BisectingKMeans", "bisecting_kmeans"),
    "dbscan": ("DBSCAN", "dbscan"),
    "dscan": ("DBSCAN", "dbscan"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate KMeans, Birch, BisectingKMeans, and DBSCAN in PreBERT at "
            "k-topic=40 while training BERT only once per dataset."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        metavar="DATASET",
        help="Dataset stem(s) under data/, without .json.",
    )
    parser.add_argument(
        "--cluster-methods",
        nargs="+",
        default=["kmeans", "birch", "bisectingkmeans", "dscan"],
        metavar="METHOD",
        help="Any of: kmeans, birch, bisectingkmeans, dbscan.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum downstream-regressor epochs (default: 100).",
    )
    parser.add_argument("--num-words", type=int, default=100)
    parser.add_argument(
        "--max-topics-per-word",
        type=int,
        default=2,
        help="Maximum topic vocabularies that may share one word (default: 2).",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("full", "review-only", "rating-only", "raw"),
        default="full",
        help=(
            "Use all processed components, review features only, original "
            "review text with adjusted ratings, or raw review/rating fields."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bert-model",
        default=DEFAULT_BERT_MODEL,
        help=(
            "Encoder alias or Hugging Face model ID. Aliases: bert-base, "
            "modernbert-base, mmbert-base."
        ),
    )
    parser.add_argument(
        "--fine-tune-bert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fine-tune BERT for sentiment (default). With "
            "--no-fine-tune-bert, use a frozen pretrained encoder and VADER "
            "for coarse sentiment."
        ),
    )
    parser.add_argument(
        "--balance-bert-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights for BERT fine-tuning.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("chkpt/clustering_ablation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/clustering_ablation"),
    )
    parser.add_argument(
        "--force-bert",
        action="store_true",
        help="Discard dataset-specific BERT caches and fine-tune BERT again.",
    )
    parser.add_argument(
        "--force-results",
        action="store_true",
        help="Rerun completed clustering methods while retaining BERT caches.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (
        args.batch_size <= 0
        or args.epochs <= 0
        or args.num_words <= 0
        or args.max_topics_per_word <= 0
    ):
        parser.error(
            "--batch-size, --epochs, --num-words, and "
            "--max-topics-per-word must be positive"
        )
    if args.max_topics_per_word > NUM_TOPICS:
        parser.error(f"--max-topics-per-word cannot exceed {NUM_TOPICS}")
    unknown = [
        method for method in args.cluster_methods if method.lower() not in CLUSTER_METHODS
    ]
    if unknown:
        parser.error(
            f"unsupported clustering method(s): {', '.join(unknown)}; "
            "choose kmeans, birch, bisectingkmeans, or dbscan"
        )
    args.bert_model = BERT_MODEL_ALIASES.get(
        args.bert_model.lower(), args.bert_model
    )
    return args


def normalized_methods(values: list[str]) -> list[tuple[str, str]]:
    methods: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        canonical, slug = CLUSTER_METHODS[value.lower()]
        if slug not in seen:
            methods.append((canonical, slug))
            seen.add(slug)
    return methods


def bert_model_slug(model_id: str) -> str:
    slug = "-".join(
        part for part in "".join(
            character.lower() if character.isalnum() else " "
            for character in model_id
        ).split()
    )
    if not slug:
        raise ValueError(f"Cannot derive a path-safe encoder name from {model_id!r}")
    return slug


def encoder_suffix(model_id: str) -> str:
    return "" if model_id == DEFAULT_BERT_MODEL else f"_bert-{bert_model_slug(model_id)}"


def encoder_cache_dir(
    cache_root: Path,
    dataset: str,
    feature_mode: str,
    model_id: str,
    fine_tune_bert: bool = True,
    balance_bert_classes: bool = True,
) -> Path:
    path = cache_root / dataset
    if feature_mode != "full":
        path = path / feature_mode
    # Preserve the historical BERT-base cache at the dataset root. Every
    # newer encoder receives its own directory, including the new default
    # ModernBERT, so changing defaults cannot overwrite an old checkpoint.
    if model_id != LEGACY_BERT_MODEL:
        path = path / "encoders" / bert_model_slug(model_id)
    if not fine_tune_bert:
        path = path / "pretrained-only"
    elif not balance_bert_classes:
        path = path / "unbalanced-classes"
    return path


def dataset_path(value: str) -> tuple[str, Path]:
    path = Path(value)
    if path.suffix == ".json" or path.parent != Path("."):
        resolved = path
        name = path.stem
    else:
        resolved = Path("data") / f"{value}.json"
        name = value
    if not name.endswith("_llama_filtered"):
        raise ValueError(
            f"Ablation datasets must end with '_llama_filtered': {resolved}"
        )
    return name, resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_fingerprint(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    feature_mode: str = "full",
) -> str:
    """Identify the exact data used for BERT fitting and model selection."""
    digest = hashlib.sha256(f"split-seed={seed}\n".encode())
    if feature_mode == "raw":
        fields = ("reviewerID", "asin", "overall", "reviewText")
    elif feature_mode == "review-only":
        fields = ("reviewerID", "asin", "overall", "filteredReviewText")
    elif feature_mode == "rating-only":
        fields = ("reviewerID", "asin", "overall_new", "reviewText")
    else:
        fields = ("reviewerID", "asin", "overall_new", "filteredReviewText")
    for split_name, frame in (("train", train_df), ("valid", valid_df)):
        digest.update(f"[{split_name}]\n".encode())
        for values in frame.loc[:, fields].itertuples(index=False, name=None):
            line = json.dumps(
                [None if pd.isna(value) else value for value in values],
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
    test_text_field = (
        "reviewText" if feature_mode in {"raw", "rating-only"}
        else "filteredReviewText"
    )
    digest.update(b"[test]\n")
    for values in test_df.loc[
        :, ("reviewerID", "asin", "overall", test_text_field)
    ].itertuples(index=False, name=None):
        line = json.dumps(
            [None if pd.isna(value) else value for value in values],
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def reset_bert_cache(cache_dir: Path) -> None:
    """Remove only artifacts owned by this dataset-specific cache."""
    for filename in (
        "bert_last_checkpoint.pt",
        "bert_train_embeddings.npy",
        f"bert_train_embeddings_v{FEATURE_PIPELINE_VERSION}.npy",
        "bert_coarse_scores.npy",
        "manifest.json",
    ):
        (cache_dir / filename).unlink(missing_ok=True)


def remove_cluster_artifacts(checkpoint_name: str) -> None:
    """Reset topic-dependent files, retaining BERT, SVD, and FM caches."""
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
    paths.append(Path("chkpt") / f"{checkpoint_name}.pt")
    for path in paths:
        path.unlink(missing_ok=True)


def result_path(
    output_dir: Path,
    dataset: str,
    method_slug: str,
    *,
    num_words: int,
    max_topics_per_word: int,
    feature_mode: str,
    bert_model: str,
    fine_tune_bert: bool,
    balance_bert_classes: bool,
    seed: int,
) -> Path:
    experiment = (
        f"k{NUM_TOPICS}_words{num_words}_tpw{max_topics_per_word}_"
        f"{feature_mode}{encoder_suffix(bert_model)}"
        f"{'_pretrained-only' if not fine_tune_bert else ''}_seed{seed}_"
        f"{'unbalanced_' if fine_tune_bert and not balance_bert_classes else ''}"
        f"v{FEATURE_PIPELINE_VERSION}"
    )
    return output_dir / dataset / experiment / method_slug / "metrics.json"


def valid_cached_result(
    path: Path,
    fingerprint: str,
    *,
    num_words: int,
    epochs: int,
    batch_size: int,
    seed: int,
    max_topics_per_word: int,
    feature_mode: str,
    bert_model: str,
    fine_tune_bert: bool,
    balance_bert_classes: bool,
) -> dict[str, Any] | None:
    result = read_json(path)
    expected = {
        "schemaVersion": FEATURE_PIPELINE_VERSION,
        "splitFingerprint": fingerprint,
        "topics": NUM_TOPICS,
        "numWords": num_words,
        "maxTopicsPerWord": max_topics_per_word,
        "featureMode": feature_mode,
        "epochs": epochs,
        "batchSize": batch_size,
        "seed": seed,
        "groundTruthField": "overall",
        "splitBeforePreprocessing": True,
        "bertClassBalanced": balance_bert_classes if fine_tune_bert else False,
    }
    expected["bertModel"] = bert_model
    if result is None or any(
        result.get(key) != value for key, value in expected.items()
    ):
        return None
    # Results created before this option existed were always fine-tuned.
    if result.get("bertFineTuned", True) != fine_tune_bert:
        return None
    required = ("accuracy", "balancedAccuracy", "macroF1", "rmse", "mae")
    return result if all(key in result for key in required) else None


def main() -> None:
    args = parse_args()
    methods = normalized_methods(args.cluster_methods)
    resolved_datasets = [dataset_path(value) for value in args.datasets]

    print(f"Device: {get_device()}")
    print(f"k-topic: {NUM_TOPICS}")
    print(
        f"Topic vocabulary: {args.num_words} words/topic, shared by at most "
        f"{args.max_topics_per_word} topics"
    )
    print(f"Feature mode: {args.feature_mode}")
    print(f"BERT encoder: {args.bert_model}")
    print(f"BERT fine-tuning: {args.fine_tune_bert}")
    print(f"BERT class-balanced loss: {args.balance_bert_classes}")
    print("Clustering methods: " + ", ".join(name for name, _ in methods))
    uses_bert = True
    print("BERT policy: one checkpoint + embeddings + coarse scores per dataset")
    if args.dry_run:
        for dataset, path in resolved_datasets:
            cache_dir = encoder_cache_dir(
                args.cache_root,
                dataset,
                args.feature_mode,
                args.bert_model,
                args.fine_tune_bert,
                args.balance_bert_classes,
            )
            print(f"  {dataset}: {path} -> {cache_dir}")
        return

    summaries: list[dict[str, Any]] = []
    for dataset, json_path in resolved_datasets:
        if not json_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {json_path}")

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset}")
        print("=" * 80)
        set_seed(args.seed)
        _, train_df, valid_df, test_df = create_dataframes(
            str(json_path), seed=args.seed
        )
        fingerprint = split_fingerprint(
            train_df,
            valid_df,
            test_df,
            seed=args.seed,
            feature_mode=args.feature_mode,
        )
        # Modes using original review text need separately fine-tuned BERT
        # checkpoints. Keep them isolated from the filtered-text cache.
        cache_dir = encoder_cache_dir(
            args.cache_root,
            dataset,
            args.feature_mode,
            args.bert_model,
            args.fine_tune_bert,
            args.balance_bert_classes,
        )
        manifest_path = cache_dir / "manifest.json"
        manifest = read_json(manifest_path)
        cache_matches = (
            manifest is not None
            and manifest.get("splitFingerprint") == fingerprint
            and manifest.get("featureMode") == args.feature_mode
            and manifest.get("bertModel", LEGACY_BERT_MODEL) == args.bert_model
            and manifest.get("featurePipelineVersion") == FEATURE_PIPELINE_VERSION
            # Legacy manifests always represent the former fine-tuned mode.
            and manifest.get("bertFineTuned", True) == args.fine_tune_bert
            and manifest.get("bertClassBalanced", False)
            == (args.balance_bert_classes if args.fine_tune_bert else False)
            and (
                not args.fine_tune_bert
                or (cache_dir / "bert_last_checkpoint.pt").is_file()
            )
        )
        cache_reset = args.force_bert or not cache_matches
        if cache_reset:
            reset_bert_cache(cache_dir)
            print(
                "BERT cache is missing/stale; the first method will prepare "
                "the configured encoder once."
            )
        else:
            print(f"Reusing dataset BERT cache: {cache_dir}")

        manifest_value = {
            "dataset": dataset,
            "datasetPath": str(json_path.resolve()),
            "splitDirectory": str(
                (json_path.parent / "splits" / json_path.stem).resolve()
            ),
            "splitFingerprint": fingerprint,
            "featureMode": args.feature_mode,
            "bertModel": args.bert_model,
            "bertFineTuned": args.fine_tune_bert,
            "bertClassBalanced": (
                args.balance_bert_classes if args.fine_tune_bert else False
            ),
            "seed": args.seed,
            "trainRows": len(train_df),
            "validRows": len(valid_df),
            "testRows": len(test_df),
            "featurePipelineVersion": FEATURE_PIPELINE_VERSION,
            "groundTruthField": "overall",
            "splitBeforePreprocessing": True,
        }
        write_json_atomic(manifest_path, manifest_value)

        pending = []
        for method_name, method_slug in methods:
            path = result_path(
                args.output_dir,
                dataset,
                method_slug,
                num_words=args.num_words,
                max_topics_per_word=args.max_topics_per_word,
                feature_mode=args.feature_mode,
                bert_model=args.bert_model,
                fine_tune_bert=args.fine_tune_bert,
                balance_bert_classes=args.balance_bert_classes,
                seed=args.seed,
            )
            cached_result = None
            if not args.force_results and not cache_reset:
                cached_result = valid_cached_result(
                    path,
                    fingerprint,
                    num_words=args.num_words,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    max_topics_per_word=args.max_topics_per_word,
                    feature_mode=args.feature_mode,
                    bert_model=args.bert_model,
                    fine_tune_bert=args.fine_tune_bert,
                    balance_bert_classes=args.balance_bert_classes,
                )
            if cached_result is None:
                pending.append((method_name, method_slug, path))
            else:
                print(f"Skipping completed result: {dataset} | {method_name}")
                summaries.append(cached_result)

        if not pending:
            continue

        # These legacy paths are global rather than dataset-scoped. Clear them
        # once at the dataset boundary, including SVD/FM trained on old data.
        remove_generated_artifacts(remove_bert_checkpoint=False)

        for method_name, method_slug, metrics_path in pending:
            print("\n" + "-" * 80)
            print(f"Cluster: {method_name} | k-topic: {NUM_TOPICS}")
            print("-" * 80)
            set_seed(args.seed)
            checkpoint_name = (
                f"PreBERT_ablation_{dataset}_{method_slug}_k{NUM_TOPICS}_"
                f"{bert_model_slug(args.bert_model)}"
            )
            remove_cluster_artifacts(checkpoint_name)
            started_at = time.perf_counter()

            feature_paths = prepare_deepbert_splits(
                train_df,
                valid_df,
                test_df,
                NUM_TOPICS,
                args.num_words,
                cluster_method=method_name,
                bert_cache_dir=cache_dir,
                embeddings_cache_path=cache_dir
                / f"bert_train_embeddings_v{FEATURE_PIPELINE_VERSION}.npy",
                coarse_cache_path=cache_dir / "bert_coarse_scores.npy",
                max_topics_per_word=args.max_topics_per_word,
                feature_mode=args.feature_mode,
                cluster_seed=args.seed,
                bert_model=args.bert_model,
                bert_fine_tuning=args.fine_tune_bert,
                balance_bert_classes=args.balance_bert_classes,
            )
            train_loader = csv_to_dataloader(
                feature_paths["train"], args.batch_size, shuffle=True
            )
            valid_loader = csv_to_dataloader(
                feature_paths["valid"], args.batch_size, shuffle=False
            )
            test_loader = csv_to_dataloader(
                feature_paths["test"], args.batch_size, shuffle=False
            )
            model = train_deepbert(
                train_loader,
                valid_loader,
                NUM_TOPICS,
                args.batch_size,
                args.epochs,
                checkpoint_name,
                log_interval=100,
            )
            classification_metrics = test(
                model, test_loader, return_details=True
            )
            rmse, mae = test_rsme(model, test_loader)
            elapsed = time.perf_counter() - started_at
            metrics = {
                "schemaVersion": FEATURE_PIPELINE_VERSION,
                "dataset": dataset,
                "splitDirectory": str(
                    (json_path.parent / "splits" / json_path.stem).resolve()
                ),
                "method": "PreBERT",
                "clusterMethod": method_name,
                "topics": NUM_TOPICS,
                "numWords": args.num_words,
                "maxTopicsPerWord": args.max_topics_per_word,
                "featureMode": args.feature_mode,
                "bertModel": args.bert_model,
                "bertFineTuned": args.fine_tune_bert,
                "bertClassBalanced": (
                    args.balance_bert_classes if args.fine_tune_bert else False
                ),
                "coarseSentimentMethod": (
                    "fine-tuned-bert" if args.fine_tune_bert else "vader"
                ),
                "epochs": args.epochs,
                "batchSize": args.batch_size,
                "seed": args.seed,
                "trainRows": len(train_df),
                "validRows": len(valid_df),
                "testRows": len(test_df),
                "splitFingerprint": fingerprint,
                **classification_metrics,
                "rmse": float(rmse),
                "mae": float(mae),
                "elapsedSeconds": elapsed,
                "bertCheckpoint": (
                    str(cache_dir / "bert_last_checkpoint.pt")
                    if uses_bert and args.fine_tune_bert
                    else None
                ),
                "bertEmbeddings": (
                    str(
                        cache_dir
                        / f"bert_train_embeddings_v{FEATURE_PIPELINE_VERSION}.npy"
                    )
                    if uses_bert
                    else None
                ),
                "bertCoarseScores": (
                    str(cache_dir / "bert_coarse_scores.npy") if uses_bert else None
                ),
                "clusterApplied": True,
                "textField": (
                    "reviewText"
                    if args.feature_mode in {"raw", "rating-only"}
                    else "filteredReviewText"
                ),
                "ratingField": "overall",
                "trainingRatingField": (
                    "overall_new"
                    if args.feature_mode in {"rating-only", "full"}
                    else "overall"
                ),
                "validationRatingField": (
                    "overall_new"
                    if args.feature_mode in {"rating-only", "full"}
                    else "overall"
                ),
                "groundTruthField": "overall",
                "splitBeforePreprocessing": True,
                "featureDiagnostics": feature_paths["featureDiagnostics"],
            }
            write_json_atomic(metrics_path, metrics)
            summaries.append(metrics)
            print(
                f"Saved {metrics_path} | RMSE={rmse:.6f} | MAE={mae:.6f} | "
                f"Accuracy={classification_metrics['accuracy']:.6f}, "
                f"balanced={classification_metrics['balancedAccuracy']:.6f}"
            )
            del model, train_loader, valid_loader, test_loader
            release_device_cache(get_device())

    if summaries:
        summary_frame = pd.DataFrame(summaries)
        summary_frame = summary_frame.sort_values(
            ["dataset", "rmse", "mae", "clusterMethod"]
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / (
            f"clustering_ablation_k{NUM_TOPICS}_words{args.num_words}_"
            f"tpw{args.max_topics_per_word}_{args.feature_mode}"
            f"{encoder_suffix(args.bert_model)}_"
            f"{'pretrained-only_' if not args.fine_tune_bert else ''}"
            f"seed{args.seed}_v{FEATURE_PIPELINE_VERSION}.csv"
        )
        summary_frame.to_csv(summary_path, index=False)
        print(f"\nAggregate results: {summary_path}")


if __name__ == "__main__":
    main()
