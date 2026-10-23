import argparse
import json
import resource
import sys
import time
from pathlib import Path

from bpe_trainer import BPETrainer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "tokenizer_training_corpus.txt"
)


TOKENIZER_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "tokenizer_config_v0.1.json"
)


# ============================================================
# Tokenizer constants
# ============================================================

BYTE_OFFSET = 3

SPECIAL_TOKEN_COUNT = 3

BYTE_TOKEN_COUNT = 256

BASE_VOCAB_SIZE = (
    SPECIAL_TOKEN_COUNT
    + BYTE_TOKEN_COUNT
)

FINAL_TARGET_MERGES = 15_741

FINAL_TARGET_VOCAB_SIZE = 16_000


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Read command-line options.

    Example:

    python tokenizer/train_tokenizer.py --merges 100

    or:

    python tokenizer/train_tokenizer.py --merges 15741
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train the MathGPT byte-level "
            "BPE tokenizer."
        )
    )

    parser.add_argument(
        "--merges",
        type=int,
        default=FINAL_TARGET_MERGES,
        help=(
            "Number of BPE merges to learn. "
            "Default: 15741"
        ),
    )

    return parser.parse_args()


# ============================================================
# Memory measurement
# ============================================================

def get_peak_memory_mb():
    """
    Return peak process memory usage in MB.

    macOS:
        ru_maxrss is reported in bytes.

    Linux:
        ru_maxrss is reported in kilobytes.
    """

    usage = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if sys.platform == "darwin":

        return (
            usage
            / (1024 * 1024)
        )

    return (
        usage
        / 1024
    )


# ============================================================
# Output directory
# ============================================================

def get_output_directory(num_merges):
    """
    Use a permanent directory for the real
    16K tokenizer.

    Test runs use separate directories so they
    cannot overwrite the final tokenizer.
    """

    artifacts_root = (
        PROJECT_ROOT
        / "tokenizer"
        / "artifacts"
    )

    if num_merges == FINAL_TARGET_MERGES:

        return (
            artifacts_root
            / "tokenizer_10m_v0.1"
        )

    return (
        artifacts_root
        / f"tokenizer_10m_test_{num_merges}"
    )


# ============================================================
# Configuration
# ============================================================

def load_tokenizer_config():
    """
    Load the tokenizer configuration.
    """

    if not TOKENIZER_CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Tokenizer config not found: "
            f"{TOKENIZER_CONFIG_PATH}"
        )

    with open(
        TOKENIZER_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    return config


# ============================================================
# Corpus loading
# ============================================================

def load_corpus():
    """
    Load the complete tokenizer training corpus.
    """

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"Tokenizer corpus not found: "
            f"{CORPUS_PATH}"
        )

    print()
    print(
        "Loading tokenizer training corpus..."
    )

    print(
        "------------------------------------"
    )

    start_time = (
        time.perf_counter()
    )

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read()

    load_time = (
        time.perf_counter()
        - start_time
    )

    utf8_bytes = len(
        text.encode("utf-8")
    )

    print(
        f"Characters: "
        f"{len(text):,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{utf8_bytes:,}"
    )

    print(
        f"Load time: "
        f"{load_time:.2f} seconds"
    )

    print(
        f"Peak memory after load: "
        f"{get_peak_memory_mb():,.2f} MB"
    )

    return text


# ============================================================
# Training
# ============================================================

def train_tokenizer(num_merges):
    """
    Train a MathGPT BPE tokenizer and save
    the learned vocabulary and merge rules.
    """

    if num_merges <= 0:

        raise ValueError(
            "Number of merges must be "
            "greater than zero."
        )

    tokenizer_config = (
        load_tokenizer_config()
    )

    text = load_corpus()

    output_directory = (
        get_output_directory(
            num_merges
        )
    )

    target_vocab_size = (
        BASE_VOCAB_SIZE
        + num_merges
    )

    print()
    print("=" * 60)

    print(
        "MathGPT Tokenizer Training"
    )

    print("=" * 60)

    print(
        f"Corpus: "
        f"{CORPUS_PATH}"
    )

    print(
        f"Requested merges: "
        f"{num_merges:,}"
    )

    print(
        f"Base vocabulary size: "
        f"{BASE_VOCAB_SIZE:,}"
    )

    print(
        f"Expected vocabulary size: "
        f"{target_vocab_size:,}"
    )

    print(
        f"Output directory: "
        f"{output_directory}"
    )

    print()

    # --------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------

    trainer = BPETrainer(
        byte_offset=BYTE_OFFSET,
        num_merges=num_merges,
    )

    # --------------------------------------------------------
    # Train tokenizer
    # --------------------------------------------------------

    start_time = (
        time.perf_counter()
    )

    trainer.train(
        [text]
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Collect final statistics
    # --------------------------------------------------------

    merges_completed = len(
        trainer.merges
    )

    final_vocab_size = (
        trainer.base_vocab_size
        + merges_completed
    )

    peak_memory_mb = (
        get_peak_memory_mb()
    )

    print()
    print("=" * 60)

    print(
        "Training Summary"
    )

    print("=" * 60)

    print(
        f"Merges requested: "
        f"{num_merges:,}"
    )

    print(
        f"Merges completed: "
        f"{merges_completed:,}"
    )

    print(
        f"Final vocabulary size: "
        f"{final_vocab_size:,}"
    )

    print()

    print(
        f"Training time: "
        f"{training_time:,.2f} seconds"
    )

    print(
        f"Training time: "
        f"{training_time / 60:,.2f} minutes"
    )

    print(
        f"Training time: "
        f"{training_time / 3600:,.2f} hours"
    )

    if merges_completed > 0:

        print(
            f"Average seconds/merge: "
            f"{training_time / merges_completed:.4f}"
        )

    print(
        f"Peak memory: "
        f"{peak_memory_mb:,.2f} MB"
    )

    print()

    # --------------------------------------------------------
    # Add training metadata to tokenizer config
    # --------------------------------------------------------

    tokenizer_config[
        "training_corpus"
    ] = {
        "name": (
            "tokenizer_training_corpus_10m"
        ),

        "characters": len(
            text
        ),

        "utf8_bytes": len(
            text.encode("utf-8")
        ),

        "general_ratio": 0.70,

        "math_ratio": 0.30,
    }

    tokenizer_config[
        "training"
    ] = {
        "requested_merges": (
            num_merges
        ),

        "completed_merges": (
            merges_completed
        ),

        "base_vocab_size": (
            trainer.base_vocab_size
        ),

        "final_vocab_size": (
            final_vocab_size
        ),

        "training_seconds": (
            training_time
        ),

        "training_minutes": (
            training_time / 60
        ),

        "training_hours": (
            training_time / 3600
        ),

        "peak_memory_mb": (
            peak_memory_mb
        ),
    }

    # --------------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------------

    trainer.save(
        output_directory=(
            output_directory
        ),

        tokenizer_config=(
            tokenizer_config
        ),
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "Tokenizer Training Complete"
    )

    print("=" * 60)

    print(
        f"Artifacts saved to:"
    )

    print(
        output_directory
    )

    print()

    if (
        merges_completed
        != num_merges
    ):

        print(
            "WARNING: Trainer did not "
            "complete all requested merges."
        )

    else:

        print(
            "SUCCESS: All requested "
            "merges were completed."
        )

    if (
        num_merges
        == FINAL_TARGET_MERGES
    ):

        print()

        print(
            "Final tokenizer validation"
        )

        print(
            "--------------------------"
        )

        print(
            f"Expected vocabulary: "
            f"{FINAL_TARGET_VOCAB_SIZE:,}"
        )

        print(
            f"Actual vocabulary: "
            f"{final_vocab_size:,}"
        )

        if (
            final_vocab_size
            == FINAL_TARGET_VOCAB_SIZE
        ):

            print(
                "SUCCESS: Final vocabulary "
                "size is exactly 16,000."
            )

        else:

            print(
                "WARNING: Final vocabulary "
                "size is not 16,000."
            )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    train_tokenizer(
        num_merges=args.merges
    )


if __name__ == "__main__":
    main()