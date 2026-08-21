"""Unified CLI for LLM dataset preparation and evaluation.

Examples::

    python -m experiments rating --dataset data/example.json --model llama3.2_1b
    python -m experiments rating-matrix --modes pretrained pretrained-processed
    python preprocessing_reviews.py preprocess data/input.json --output data/output.json
    python -m experiments semantic data/output.json --output-dir results/semantic
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from helper.llm_settings import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    GROUND_TRUTH_FIELD,
    MODE_FIELDS,
    resolve_model_id,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
COMMANDS = (
    "rating",
    "rating-matrix",
    "semantic",
)


def matrix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments rating-matrix",
        description="Evaluate a matrix of pretrained LLMs, datasets, and modes.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        default=list(DEFAULT_DATASETS),
        help="JSONL datasets (default: the three *_llama_filtered datasets)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model aliases or Hugging Face model IDs",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_FIELDS),
        default=list(MODE_FIELDS),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra rating arguments after --",
    )
    return parser


def expected_metrics_path(
    output_dir: Path, dataset: Path, model_id: str, mode: str
) -> Path:
    return (
        output_dir
        / dataset.stem
        / model_id.replace("/", "--")
        / mode
        / "metrics.json"
    )


def is_completed_rating_run(metrics_path: Path) -> bool:
    """Return true only for results using the current ground-truth contract."""
    config_path = metrics_path.with_name("run_config.json")
    if not metrics_path.is_file() or not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        config.get("ratingField") == GROUND_TRUTH_FIELD
        and config.get("splitBeforePreprocessing") is True
    )


def rating_command(
    args: argparse.Namespace,
    dataset: Path,
    model_alias: str,
    mode: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments",
        "rating",
        "--dataset",
        str(dataset),
        "--model",
        model_alias,
        "--mode",
        mode,
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
        "--max-length",
        str(args.max_length),
        "--inference-batch-size",
        str(args.inference_batch_size),
    ]
    extra_args = (
        args.extra_args[1:]
        if args.extra_args[:1] == ["--"]
        else args.extra_args
    )
    return command + extra_args


def run_rating_matrix(argv: Sequence[str]) -> None:
    parser = matrix_parser()
    args = parser.parse_args(argv)
    if args.max_length < 8:
        parser.error("--max-length must be at least 8")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be greater than zero")

    datasets = tuple(dict.fromkeys(args.datasets))
    models = tuple(
        (alias, resolve_model_id(alias)) for alias in dict.fromkeys(args.models)
    )
    modes = tuple(dict.fromkeys(args.modes))
    missing = [str(path) for path in datasets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing datasets: {', '.join(missing)}")

    total = len(datasets) * len(models) * len(modes)
    print("Pretrained LLM evaluation matrix")
    print(f"  datasets: {len(datasets)}")
    print(f"  models: {', '.join(alias for alias, _ in models)}")
    print(f"  modes: {', '.join(modes)}")
    print(f"  experiments: {total}")
    print("  execution: sequential (one model in memory at a time)")

    started = time.monotonic()
    completed = skipped = failed = position = 0
    for dataset in datasets:
        for model_alias, model_id in models:
            for mode in modes:
                position += 1
                metrics_path = expected_metrics_path(
                    args.output_dir, dataset, model_id, mode
                )
                if is_completed_rating_run(metrics_path) and not args.force:
                    print(f"\n[{position}/{total}] Skip completed: {metrics_path}")
                    skipped += 1
                    continue

                command = rating_command(args, dataset, model_alias, mode)
                print(
                    f"\n[{position}/{total}] {dataset.stem} | "
                    f"{model_alias} | {mode}"
                )
                print(shlex.join(command), flush=True)
                if args.dry_run:
                    continue

                result = subprocess.run(command, cwd=REPO_ROOT, check=False)
                if result.returncode == 0:
                    completed += 1
                    continue
                failed += 1
                if not args.keep_going:
                    raise subprocess.CalledProcessError(result.returncode, command)

    elapsed_minutes = (time.monotonic() - started) / 60
    if args.dry_run:
        print(
            f"\nWould run: {total - skipped}; skipped: {skipped}; "
            f"elapsed: {elapsed_minutes:.1f} minutes"
        )
    else:
        print(
            f"\nCompleted: {completed}; skipped: {skipped}; failed: {failed}; "
            f"elapsed: {elapsed_minutes:.1f} minutes"
        )
    if failed:
        raise SystemExit(1)


def run_rating(argv: Sequence[str]) -> None:
    from experiments.run_experiment import main

    main(argv)


def run_semantic(argv: Sequence[str]) -> None:
    from experiments.evaluate_semantic_retention import main

    main(argv)


def root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments",
        description="Unified experiment CLI.",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = root_parser().parse_args(argv)
    handlers = {
        "rating": run_rating,
        "rating-matrix": run_rating_matrix,
        "semantic": run_semantic,
    }
    handlers[args.command](args.arguments)


if __name__ == "__main__":
    main()
