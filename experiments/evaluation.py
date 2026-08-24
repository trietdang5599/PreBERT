#!/usr/bin/env python3
"""Unified command-line entry point for PreBERT experiments.

The heavy work remains in ``experiments/run_clustering_ablation.py``. This file
defines reproducible presets so main results and ablations use exactly the same
split, cache, feature extraction, training, and metrics implementation.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


DEFAULT_DATASETS = (
    "Small_All_Beauty_5_llama_filtered",
    "Small_Digital_Music_5_llama_filtered",
    "Small_Toys_and_Games_5_llama_filtered",
)
FEATURE_MODES = ("full", "review-only", "rating-only", "raw")
REC_FEATURE_ABLATIONS = ("full", "without-review", "without-rating")
CLUSTER_METHODS = ("kmeans", "birch", "bisectingkmeans", "dbscan")
BERT_MODELS = ("modernbert-base", "bert-base", "mmbert-base")
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExperimentPreset:
    feature_modes: tuple[str, ...]
    cluster_methods: tuple[str, ...]
    bert_models: tuple[str, ...]
    rec_feature_ablations: tuple[str, ...] = ("full",)


PRESETS = {
    "main": ExperimentPreset(("full",), ("birch",), ("modernbert-base",)),
    "preprocessing-ablation": ExperimentPreset(
        ("raw", "review-only", "rating-only", "full"),
        ("birch",),
        ("modernbert-base",),
    ),
    "clustering-ablation": ExperimentPreset(
        ("full",), CLUSTER_METHODS, ("modernbert-base",)
    ),
    "encoder-ablation": ExperimentPreset(("full",), ("birch",), BERT_MODELS),
    "rec-feature-ablation": ExperimentPreset(
        ("full",), ("birch",), ("modernbert-base",), REC_FEATURE_ABLATIONS
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main PreBERT results or a named ablation preset."
    )
    parser.add_argument(
        "--mode",
        choices=(*PRESETS, "custom"),
        default="main",
        help=(
            "main=full preprocessing with Birch; preprocessing-ablation=the "
            "four input modes; clustering-ablation=the clustering methods; "
            "encoder-ablation=BERT backbone variants; custom=explicit lists."
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--split-profile",
        choices=("8-1-1", "7-1-2"),
        default="8-1-1",
        help="Physical train/validation/test split ratio to load.",
    )
    parser.add_argument(
        "--ground-truth-field",
        choices=("overall", "overall_new"),
        default="overall",
        help="Test label used for metrics (default: overall).",
    )
    parser.add_argument(
        "--feature-modes", nargs="+", choices=FEATURE_MODES, default=["full"]
    )
    parser.add_argument(
        "--rec-feature-ablations",
        nargs="+",
        choices=REC_FEATURE_ABLATIONS,
        default=["full"],
        help="PreBERT-Rec feature branches to retain for custom mode.",
    )
    parser.add_argument(
        "--cluster-methods", nargs="+", choices=CLUSTER_METHODS, default=["birch"]
    )
    parser.add_argument("--bert-models", nargs="+", default=["modernbert-base"])
    parser.add_argument(
        "--fine-tune-bert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fine-tune BERT (default) or use its frozen pretrained encoder.",
    )
    parser.add_argument(
        "--balance-bert-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights during BERT fine-tuning.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--regressor-architecture",
        choices=("linear", "fusion-mlp"),
        default="linear",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument(
        "--num-topics",
        "--k-topic",
        dest="num_topics",
        type=int,
        default=40,
        help="Number of topic clusters (default: 40).",
    )
    parser.add_argument("--num-words", type=int, default=100)
    parser.add_argument("--max-topics-per-word", type=int, default=2)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("chkpt/clustering_ablation")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/clustering_ablation")
    )
    parser.add_argument("--force-bert", action="store_true")
    parser.add_argument(
        "--standardize-deep-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit StandardScaler on train entity features before FM fusion.",
    )
    parser.add_argument("--force-results", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain non-negative integers")
    if min(
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.num_topics,
        args.num_words,
        args.max_topics_per_word,
        args.mlp_hidden_dim,
    ) <= 0 or args.weight_decay < 0 or not 0 <= args.mlp_dropout < 1:
        parser.error(
            "numeric options must be positive, except --weight-decay which "
            "may be zero"
        )
    return args


def selected_matrix(args: argparse.Namespace) -> ExperimentPreset:
    if args.mode != "custom":
        return PRESETS[args.mode]
    return ExperimentPreset(
        tuple(dict.fromkeys(args.feature_modes)),
        tuple(dict.fromkeys(args.cluster_methods)),
        tuple(dict.fromkeys(args.bert_models)),
        tuple(dict.fromkeys(args.rec_feature_ablations)),
    )


def build_command(
    args: argparse.Namespace,
    feature_mode: str,
    cluster_methods: tuple[str, ...],
    bert_model: str,
    seed: int,
    rec_feature_ablation: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.run_clustering_ablation",
        "--datasets",
        *args.datasets,
        "--feature-mode",
        feature_mode,
        "--cluster-methods",
        *cluster_methods,
        "--bert-model",
        bert_model,
        "--seed",
        str(seed),
        "--split-profile",
        args.split_profile,
        "--ground-truth-field",
        args.ground_truth_field,
        "--rec-feature-ablation",
        rec_feature_ablation,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--regressor-architecture",
        args.regressor_architecture,
        "--mlp-hidden-dim",
        str(args.mlp_hidden_dim),
        "--mlp-dropout",
        str(args.mlp_dropout),
        "--num-topics",
        str(args.num_topics),
        "--num-words",
        str(args.num_words),
        "--max-topics-per-word",
        str(args.max_topics_per_word),
        "--cache-root",
        str(args.cache_root),
        "--output-dir",
        str(args.output_dir),
    ]
    command.append(
        "--standardize-deep-features"
        if args.standardize_deep_features
        else "--no-standardize-deep-features"
    )
    if args.force_bert:
        command.append("--force-bert")
    command.append(
        "--fine-tune-bert" if args.fine_tune_bert else "--no-fine-tune-bert"
    )
    command.append(
        "--balance-bert-classes"
        if args.balance_bert_classes
        else "--no-balance-bert-classes"
    )
    if args.force_results:
        command.append("--force-results")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    args = parse_args()
    matrix = selected_matrix(args)
    jobs = [
        (feature_mode, bert_model, seed, rec_feature_ablation)
        for seed in dict.fromkeys(args.seeds)
        for feature_mode in matrix.feature_modes
        for bert_model in matrix.bert_models
        for rec_feature_ablation in matrix.rec_feature_ablations
    ]

    print("PreBERT experiment matrix")
    print(f"  mode: {args.mode}")
    print(f"  datasets: {', '.join(args.datasets)}")
    print(f"  feature modes: {', '.join(matrix.feature_modes)}")
    print(f"  clustering: {', '.join(matrix.cluster_methods)}")
    print(f"  BERT encoders: {', '.join(matrix.bert_models)}")
    print(f"  k-topic: {args.num_topics}")
    print(f"  downstream LR / weight decay: {args.learning_rate:g} / {args.weight_decay:g}")
    print(f"  regressor: {args.regressor_architecture}")
    print(f"  PreBERT-Rec features: {', '.join(matrix.rec_feature_ablations)}")
    print(f"  seeds: {', '.join(map(str, args.seeds))}")
    print(f"  split profile: {args.split_profile}")
    print(f"  test ground truth: {args.ground_truth_field}")
    print(f"  jobs: {len(jobs)}")

    failures = 0
    for index, (feature_mode, bert_model, seed, rec_feature_ablation) in enumerate(jobs, start=1):
        command = build_command(
            args,
            feature_mode,
            matrix.cluster_methods,
            bert_model,
            seed,
            rec_feature_ablation,
        )
        print(
            f"\n[{index}/{len(jobs)}] feature={feature_mode} | "
            f"encoder={bert_model} | seed={seed}"
            f" | rec-features={rec_feature_ablation}"
        )
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode == 0:
            continue
        failures += 1
        if not args.keep_going:
            raise subprocess.CalledProcessError(result.returncode, command)

    if failures:
        raise SystemExit(f"{failures} PreBERT job(s) failed")


if __name__ == "__main__":
    main()
