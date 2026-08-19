"""Runtime device selection shared by training and feature extraction."""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """Select CUDA, Apple Metal (MPS), or CPU in that order.

    Set ``PREBERT_DEVICE`` to ``cuda``, ``mps``, or ``cpu`` to override the
    automatic selection.
    """
    requested = os.getenv("PREBERT_DEVICE", "auto").lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("PREBERT_DEVICE must be one of: auto, cuda, mps, cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PREBERT_DEVICE=cuda, but CUDA is unavailable")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "PREBERT_DEVICE=mps, but MPS is unavailable. Check that Python "
                "and PyTorch are arm64 and run on macOS 13 or newer."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
