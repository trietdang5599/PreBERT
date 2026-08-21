"""Shared runtime, model, and input-mode settings for LLM experiments."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


MODEL_ALIASES = {
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5_1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5_3b": "Qwen/Qwen2.5-3B",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B",
    "qwen_3b_base": "Qwen/Qwen2.5-3B",
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2_1b_base": "meta-llama/Llama-3.2-1B",
    "llama3.2_3b_base": "meta-llama/Llama-3.2-3B",
}

GROUND_TRUTH_FIELD = "overall"

# Evaluation always measures predictions against the original Amazon rating.
# Mode names are retained for compatibility with existing commands and output
# paths; they now vary only the review text presented to the model.
MODE_FIELDS = {
    "pretrained": ("reviewText", GROUND_TRUTH_FIELD),
    "pretrained-processed": ("filteredReviewText", GROUND_TRUTH_FIELD),
    "pretrained-processed-mix": ("filteredReviewText", GROUND_TRUTH_FIELD),
    "pretrained-rating-only": ("reviewText", GROUND_TRUTH_FIELD),
}

MODE_DESCRIPTIONS = {
    "pretrained": "baseline: original review text + original rating",
    "pretrained-processed-mix": "text-only ablation: filtered text + original rating",
    "pretrained-rating-only": "legacy alias: original text + original ground truth",
    "pretrained-processed": "filtered text + original ground truth",
}

DEFAULT_DATASETS = (
    Path("data/Small_All_Beauty_5_llama_filtered.json"),
    Path("data/Small_Digital_Music_5_llama_filtered.json"),
    Path("data/Small_Toys_and_Games_5_llama_filtered.json"),
)

DEFAULT_MODELS = (
    "qwen2.5_3b",
    "qwen-3b",
    "llama3.2_1b",
    "llama3.2_3b",
)

DEFAULT_PREPROCESSING_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def configure_runtime_environment() -> None:
    """Set safe defaults before importing PyTorch/Transformers."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_model_id(value: str) -> str:
    return MODEL_ALIASES.get(value.lower(), value)


def configure_greedy_generation(model: Any) -> None:
    """Remove sampling-only flags inherited from model checkpoints."""
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = False
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
    ):
        if hasattr(generation_config, field):
            setattr(generation_config, field, None)


def load_model_and_tokenizer(args: Any, model_id: str) -> tuple[Any, Any]:
    """Load one causal LLM using a device-appropriate memory policy."""
    configure_runtime_environment()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLM dependencies. Run: pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--use-4bit requires an NVIDIA CUDA device")
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            ),
            device_map="auto",
        )
    elif torch.cuda.is_available():
        model_kwargs.update(
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            device_map="auto",
        )
    elif torch.backends.mps.is_available():
        model_kwargs.update(torch_dtype=torch.float16, low_cpu_mem_usage=True)

    generation_logger = logging.getLogger(
        "transformers.generation.configuration_utils"
    )
    previous_level = generation_logger.level
    generation_logger.setLevel(logging.ERROR)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    finally:
        generation_logger.setLevel(previous_level)

    if not args.use_4bit and not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    configure_greedy_generation(model)
    return model, tokenizer
