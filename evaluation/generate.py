import argparse
import json
import sys

from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports
# ============================================================

from model.gpt import GPT

from tokenizer.bpe_tokenizer import (
    BPETokenizer,
)

from training.adamw import (
    AdamW,
)

from training.checkpoint import (
    load_checkpoint,
)

from evaluation.generation import (
    generate_text,
)


# ============================================================
# Paths
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.1.json"
)


TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_50m_v0.2"
)


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "general_pretraining"
    / "general_pretraining_full_v0.1"
    / "mathgpt_step_008197.npz"
)


# ============================================================
# Helpers
# ============================================================

def load_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Generate text using "
            "MathGPT-General-v0.1."
        )
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            DEFAULT_CHECKPOINT_PATH
        ),
        help=(
            "Checkpoint path."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "greedy",
            "sample",
        ],
        default="sample",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.is_absolute():

        checkpoint_path = (
            PROJECT_ROOT
            /
            checkpoint_path
        )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint missing: "
            f"{checkpoint_path}"
        )

    # ========================================================
    # Config
    # ========================================================

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        BPETokenizer.from_files(
            config_path=(
                TOKENIZER_DIR
                /
                "tokenizer_config.json"
            ),

            merges_path=(
                TOKENIZER_DIR
                /
                "merges.json"
            ),

            vocab_path=(
                TOKENIZER_DIR
                /
                "vocab.json"
            ),
        )
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "MATHGPT-GENERAL-v0.1"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Loading model..."
    )

    model = GPT.from_config(
        model_config,
        seed=42,
    )

    # ========================================================
    # Temporary AdamW object
    #
    # Our current checkpoint loader restores both
    # model + optimizer state.
    #
    # Generation does NOT use AdamW after loading.
    # ========================================================

    optimizer = AdamW(
        model=model,

        learning_rate=(
            training_config[
                "optimizer"
            ][
                "learning_rate"
            ]
        ),

        beta1=(
            training_config[
                "optimizer"
            ][
                "beta1"
            ]
        ),

        beta2=(
            training_config[
                "optimizer"
            ][
                "beta2"
            ]
        ),

        epsilon=(
            training_config[
                "optimizer"
            ][
                "epsilon"
            ]
        ),

        weight_decay=(
            training_config[
                "optimizer"
            ][
                "weight_decay"
            ]
        ),
    )

    # ========================================================
    # Restore final general model
    # ========================================================

    metadata = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        strict=True,
    )

    step = (
        metadata[
            "optimizer_step_count"
        ]
    )

    parameter_count = (
        metadata[
            "parameter_count"
        ]
    )

    if step != 8197:

        print()

        print(
            "WARNING:"
        )

        print(
            f"Checkpoint is step {step:,}, "
            "not the official step-8197 "
            "General-v0.1 checkpoint."
        )

    print(
        f"Checkpoint: "
        f"{checkpoint_path.name}"
    )

    print(
        f"Training step: "
        f"{step:,}"
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Vocabulary: "
        f"{model.vocab_size:,}"
    )

    print(
        f"Context length: "
        f"{model.context_length}"
    )

    # ========================================================
    # Generation settings
    # ========================================================

    print()

    print(
        "Generation Settings"
    )

    print(
        "-------------------"
    )

    print(
        f"Strategy: "
        f"{args.strategy}"
    )

    if args.strategy == "sample":

        print(
            f"Temperature: "
            f"{args.temperature}"
        )

        print(
            f"Top-k: "
            f"{args.top_k}"
        )

        print(
            f"Top-p: "
            f"{args.top_p}"
        )

        print(
            f"Seed: "
            f"{args.seed}"
        )

    print(
        f"Max new tokens: "
        f"{args.max_new_tokens}"
    )

    # ========================================================
    # Generate
    # ========================================================

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=(
            args.max_new_tokens
        ),
        strategy=(
            args.strategy
        ),
        temperature=(
            args.temperature
        ),
        top_k=(
            args.top_k
        ),
        top_p=(
            args.top_p
        ),
        seed=(
            args.seed
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "PROMPT"
    )

    print(
        "=" * 72
    )

    print()

    print(
        result[
            "prompt"
        ]
    )

    print()
    print(
        "=" * 72
    )

    print(
        "GENERATED CONTINUATION"
    )

    print(
        "=" * 72
    )

    print()

    print(
        result[
            "generated_text"
        ]
    )

    print()
    print(
        "=" * 72
    )

    print(
        "FULL TEXT"
    )

    print(
        "=" * 72
    )

    print()

    print(
        result[
            "text"
        ]
    )

    print()

    print(
        f"Generated tokens: "
        f"{result['generated_tokens']}"
    )

    print(
        f"Stopped on EOS: "
        f"{result['stopped_on_eos']}"
    )

    print()
    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()