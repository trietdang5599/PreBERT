#!/usr/bin/env python3
"""Unified command-line entry point for PreBERT experiments.

The heavy work remains in ``run_clustering_ablation.py``. This file defines
reproducible presets so main results and ablations use exactly the same split,
cache, feature extraction, training, and metrics implementation.
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
CLUSTER_METHODS = ("kmeans", "birch", "bisectingkmeans", "dbscan")
BERT_MODELS = ("bert-base", "modernbert-base", "mmbert-base")
REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentPreset:
    feature_modes: tuple[str, ...]
    cluster_methods: tuple[str, ...]
    bert_models: tuple[str, ...]


PRESETS = {
    "main": ExperimentPreset(("full",), ("birch",), ("bert-base",)),
    "preprocessing-ablation": ExperimentPreset(
        ("raw", "review-only", "rating-only", "full"),
        ("birch",),
        ("bert-base",),
    ),
    "clustering-ablation": ExperimentPreset(
        ("full",), CLUSTER_METHODS, ("bert-base",)
    ),
    "encoder-ablation": ExperimentPreset(("full",), ("birch",), BERT_MODELS),
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
        "--feature-modes", nargs="+", choices=FEATURE_MODES, default=["full"]
    )
    parser.add_argument(
        "--cluster-methods", nargs="+", choices=CLUSTER_METHODS, default=["birch"]
    )
    parser.add_argument("--bert-models", nargs="+", default=["bert-base"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-words", type=int, default=100)
    parser.add_argument("--max-topics-per-word", type=int, default=2)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("chkpt/clustering_ablation")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/clustering_ablation")
    )
    parser.add_argument("--force-bert", action="store_true")
    parser.add_argument("--force-results", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain non-negative integers")
    if min(
        args.batch_size,
        args.epochs,
        args.num_words,
        args.max_topics_per_word,
    ) <= 0:
        parser.error("numeric training options must be positive")
    return args


def selected_matrix(args: argparse.Namespace) -> ExperimentPreset:
    if args.mode != "custom":
        return PRESETS[args.mode]
    return ExperimentPreset(
        tuple(dict.fromkeys(args.feature_modes)),
        tuple(dict.fromkeys(args.cluster_methods)),
        tuple(dict.fromkeys(args.bert_models)),
    )


def build_command(
    args: argparse.Namespace,
    feature_mode: str,
    cluster_methods: tuple[str, ...],
    bert_model: str,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "run_clustering_ablation.py"),
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
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--num-words",
        str(args.num_words),
        "--max-topics-per-word",
        str(args.max_topics_per_word),
        "--cache-root",
        str(args.cache_root),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.force_bert:
        command.append("--force-bert")
    if args.force_results:
        command.append("--force-results")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    args = parse_args()
    matrix = selected_matrix(args)
    jobs = [
        (feature_mode, bert_model, seed)
        for seed in dict.fromkeys(args.seeds)
        for feature_mode in matrix.feature_modes
        for bert_model in matrix.bert_models
    ]

    print("PreBERT experiment matrix")
    print(f"  mode: {args.mode}")
    print(f"  datasets: {', '.join(args.datasets)}")
    print(f"  feature modes: {', '.join(matrix.feature_modes)}")
    print(f"  clustering: {', '.join(matrix.cluster_methods)}")
    print(f"  BERT encoders: {', '.join(matrix.bert_models)}")
    print(f"  seeds: {', '.join(map(str, args.seeds))}")
    print(f"  jobs: {len(jobs)}")

    failures = 0
    for index, (feature_mode, bert_model, seed) in enumerate(jobs, start=1):
        command = build_command(
            args, feature_mode, matrix.cluster_methods, bert_model, seed
        )
        print(
            f"\n[{index}/{len(jobs)}] feature={feature_mode} | "
            f"encoder={bert_model} | seed={seed}"
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
