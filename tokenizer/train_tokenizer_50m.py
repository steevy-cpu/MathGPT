import argparse
import json
import resource
import shutil
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
    / "tokenizer_training_corpus_50m_v0.2.txt"
)

CORPUS_REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_50m_corpus_report.json"
)

# We keep the original tokenizer specification because the
# fundamental design is unchanged:
#
#   byte-level BPE
#   3 special tokens
#   16,000 logical vocabulary
#
# This script updates the saved artifact metadata to v0.2.
TOKENIZER_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "tokenizer_config_v0.1.json"
)

ARTIFACTS_ROOT = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
)

BASELINE_DIRECTORY = (
    ARTIFACTS_ROOT
    / "tokenizer_10m_v0.1"
)

FINAL_OUTPUT_DIRECTORY = (
    ARTIFACTS_ROOT
    / "tokenizer_50m_v0.2"
)

TRAINING_REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_50m_v0.2_training_report.json"
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

# vocab.json contains byte/BPE tokens only.
# The 3 special tokens are stored separately in the config.
FINAL_SERIALIZED_VOCAB_SIZE = (
    BYTE_TOKEN_COUNT
    + FINAL_TARGET_MERGES
)

EXPECTED_CORPUS_CHARACTERS = 50_000_000

EXPECTED_GENERAL_CHARACTERS = 35_000_000
EXPECTED_MATH_CHARACTERS = 15_000_000


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Examples:

        Smoke test:
            python tokenizer/train_tokenizer_50m.py --merges 100

        Final tokenizer:
            python tokenizer/train_tokenizer_50m.py --merges 15741

        Explicitly replace an existing TEST output:
            python tokenizer/train_tokenizer_50m.py \
                --merges 100 \
                --overwrite

    The final tokenizer directory is also protected from
    accidental overwrite unless --overwrite is explicitly used.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train the MathGPT tokenizer_50m_v0.2 "
            "byte-level BPE tokenizer."
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of an existing output "
            "directory. Use carefully."
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
# Output directory safety
# ============================================================

def get_output_directory(
    num_merges,
):
    """
    Full 15,741-merge run:
        tokenizer/artifacts/tokenizer_50m_v0.2

    Test runs:
        tokenizer/artifacts/tokenizer_50m_v0.2_test_<merges>
    """

    if (
        num_merges
        == FINAL_TARGET_MERGES
    ):
        return (
            FINAL_OUTPUT_DIRECTORY
        )

    return (
        ARTIFACTS_ROOT
        / (
            "tokenizer_50m_v0.2_test_"
            f"{num_merges}"
        )
    )


def prepare_output_directory(
    output_directory,
    overwrite,
):
    """
    Protect both the 10M baseline and existing v0.2 artifacts.
    """

    resolved_output = (
        output_directory.resolve()
    )

    resolved_baseline = (
        BASELINE_DIRECTORY.resolve()
    )

    if (
        resolved_output
        == resolved_baseline
    ):
        raise RuntimeError(
            "SAFETY STOP: v0.2 training attempted "
            "to write into tokenizer_10m_v0.1."
        )

    if output_directory.exists():

        has_contents = any(
            output_directory.iterdir()
        )

        if has_contents:

            if not overwrite:
                raise FileExistsError(
                    "\nOutput directory already exists "
                    "and is not empty:\n"
                    f"{output_directory}\n\n"
                    "Nothing was overwritten.\n"
                    "If replacement is intentional, "
                    "rerun with --overwrite."
                )

            shutil.rmtree(
                output_directory
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Configuration
# ============================================================

def load_tokenizer_config():
    """
    Load the existing tokenizer design configuration.

    v0.2 keeps the same tokenizer architecture as v0.1;
    only the training corpus and learned merges change.
    """

    if not TOKENIZER_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Tokenizer config not found: "
            f"{TOKENIZER_CONFIG_PATH}"
        )

    with open(
        TOKENIZER_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(
            file
        )

    # Artifact metadata for this experiment.
    config[
        "version"
    ] = "0.2"

    config[
        "artifact_name"
    ] = "tokenizer_50m_v0.2"

    return config


# ============================================================
# Corpus preflight validation
# ============================================================

def load_corpus_report():
    """
    Load the report produced by the validated 50M builder.
    """

    if not CORPUS_REPORT_PATH.exists():
        raise FileNotFoundError(
            "50M corpus report not found: "
            f"{CORPUS_REPORT_PATH}"
        )

    with open(
        CORPUS_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def load_and_validate_corpus():
    """
    Load and independently validate the exact corpus that
    will be passed to BPETrainer.

    Training refuses to start unless:
        - file exists
        - exactly 50,000,000 characters
        - zero U+FFFD replacement characters
        - zero null characters
        - report confirms 35M general / 15M math
    """

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Tokenizer corpus not found: "
            f"{CORPUS_PATH}"
        )

    corpus_report = (
        load_corpus_report()
    )

    print()
    print("=" * 65)
    print(
        "50M CORPUS PREFLIGHT VALIDATION"
    )
    print("=" * 65)

    print()
    print(
        f"Corpus: {CORPUS_PATH}"
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

    character_count = len(
        text
    )

    utf8_bytes = len(
        text.encode(
            "utf-8"
        )
    )

    replacement_count = (
        text.count(
            "\ufffd"
        )
    )

    null_count = (
        text.count(
            "\x00"
        )
    )

    report_characters = (
        corpus_report.get(
            "actual_file_characters"
        )
    )

    report_general = (
        corpus_report.get(
            "general_characters"
        )
    )

    report_math = (
        corpus_report.get(
            "math_characters"
        )
    )

    exact_size_passed = (
        character_count
        == EXPECTED_CORPUS_CHARACTERS
    )

    report_size_passed = (
        report_characters
        == EXPECTED_CORPUS_CHARACTERS
    )

    split_passed = (
        report_general
        == EXPECTED_GENERAL_CHARACTERS
        and
        report_math
        == EXPECTED_MATH_CHARACTERS
    )

    replacement_passed = (
        replacement_count
        == 0
    )

    null_passed = (
        null_count
        == 0
    )

    print()
    print(
        f"Characters: "
        f"{character_count:,}"
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

    print()
    print(
        f"Exact 50M file size: "
        f"{exact_size_passed}"
    )

    print(
        f"Builder report confirms 50M: "
        f"{report_size_passed}"
    )

    print(
        f"35M/15M split confirmed: "
        f"{split_passed}"
    )

    print(
        f"Replacement chars: "
        f"{replacement_count:,}"
    )

    print(
        f"Replacement-char check: "
        f"{replacement_passed}"
    )

    print(
        f"Null chars: "
        f"{null_count:,}"
    )

    print(
        f"Null-char check: "
        f"{null_passed}"
    )

    all_passed = (
        exact_size_passed
        and report_size_passed
        and split_passed
        and replacement_passed
        and null_passed
    )

    if not all_passed:
        raise RuntimeError(
            "SAFETY STOP: 50M tokenizer corpus "
            "failed preflight validation. "
            "BPE training was NOT started."
        )

    print()
    print(
        "SUCCESS: 50M corpus passed "
        "training preflight validation."
    )

    return (
        text,
        utf8_bytes,
        corpus_report,
    )


# ============================================================
# Artifact validation
# ============================================================

def validate_saved_artifacts(
    output_directory,
    num_merges,
):
    """
    Validate the serialized tokenizer after trainer.save().
    """

    merges_path = (
        output_directory
        / "merges.json"
    )

    vocab_path = (
        output_directory
        / "vocab.json"
    )

    config_path = (
        output_directory
        / "tokenizer_config.json"
    )

    for path in (
        merges_path,
        vocab_path,
        config_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                "Expected tokenizer artifact "
                f"was not created: {path}"
            )

    with open(
        merges_path,
        "r",
        encoding="utf-8",
    ) as file:
        saved_merges = json.load(
            file
        )

    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as file:
        saved_vocab = json.load(
            file
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        saved_config = json.load(
            file
        )

    merges_passed = (
        len(saved_merges)
        == num_merges
    )

    expected_serialized_vocab = (
        BYTE_TOKEN_COUNT
        + num_merges
    )

    serialized_vocab_passed = (
        len(saved_vocab)
        == expected_serialized_vocab
    )

    expected_logical_vocab = (
        BASE_VOCAB_SIZE
        + num_merges
    )

    logical_vocab_passed = (
        expected_logical_vocab
        == (
            SPECIAL_TOKEN_COUNT
            + len(saved_vocab)
        )
    )

    config_name_passed = (
        saved_config.get(
            "artifact_name"
        )
        == "tokenizer_50m_v0.2"
    )

    print()
    print("=" * 65)
    print(
        "SAVED ARTIFACT VALIDATION"
    )
    print("=" * 65)

    print()
    print(
        f"Saved merges: "
        f"{len(saved_merges):,}"
    )

    print(
        f"Expected merges: "
        f"{num_merges:,}"
    )

    print(
        f"Merge count passed: "
        f"{merges_passed}"
    )

    print()
    print(
        f"Serialized vocab entries: "
        f"{len(saved_vocab):,}"
    )

    print(
        f"Expected serialized vocab: "
        f"{expected_serialized_vocab:,}"
    )

    print(
        f"Serialized vocab passed: "
        f"{serialized_vocab_passed}"
    )

    print()
    print(
        f"Logical vocab size: "
        f"{SPECIAL_TOKEN_COUNT + len(saved_vocab):,}"
    )

    print(
        f"Logical vocab passed: "
        f"{logical_vocab_passed}"
    )

    print(
        f"Artifact-name metadata passed: "
        f"{config_name_passed}"
    )

    all_passed = (
        merges_passed
        and serialized_vocab_passed
        and logical_vocab_passed
        and config_name_passed
    )

    if not all_passed:
        raise RuntimeError(
            "Saved tokenizer artifacts failed "
            "post-training validation."
        )

    print()
    print(
        "SUCCESS: Saved tokenizer artifacts "
        "passed validation."
    )


# ============================================================
# Training report
# ============================================================

def save_training_report(
    *,
    output_directory,
    num_merges,
    merges_completed,
    final_vocab_size,
    corpus_characters,
    utf8_bytes,
    training_time,
    peak_memory_mb,
):
    """
    Save a compact machine-readable record of the run.
    """

    TRAINING_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "tokenizer": (
            "tokenizer_50m_v0.2"
        ),
        "corpus_path": str(
            CORPUS_PATH
        ),
        "output_directory": str(
            output_directory
        ),
        "corpus_characters": (
            corpus_characters
        ),
        "corpus_utf8_bytes": (
            utf8_bytes
        ),
        "requested_merges": (
            num_merges
        ),
        "completed_merges": (
            merges_completed
        ),
        "base_vocab_size": (
            BASE_VOCAB_SIZE
        ),
        "final_vocab_size": (
            final_vocab_size
        ),
        "training_seconds": (
            training_time
        ),
        "training_minutes": (
            training_time
            / 60
        ),
        "training_hours": (
            training_time
            / 3600
        ),
        "peak_memory_mb": (
            peak_memory_mb
        ),
    }

    with open(
        TRAINING_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Training
# ============================================================

def train_tokenizer(
    num_merges,
    overwrite=False,
):
    """
    Train tokenizer_50m_v0.2 and save its artifacts.
    """

    if num_merges <= 0:
        raise ValueError(
            "Number of merges must be "
            "greater than zero."
        )

    if (
        num_merges
        > FINAL_TARGET_MERGES
    ):
        raise ValueError(
            "This experiment is capped at "
            f"{FINAL_TARGET_MERGES:,} merges."
        )

    output_directory = (
        get_output_directory(
            num_merges
        )
    )

    # Protect outputs BEFORE the expensive training run.
    prepare_output_directory(
        output_directory=(
            output_directory
        ),
        overwrite=overwrite,
    )

    tokenizer_config = (
        load_tokenizer_config()
    )

    (
        text,
        utf8_bytes,
        corpus_report,
    ) = load_and_validate_corpus()

    target_vocab_size = (
        BASE_VOCAB_SIZE
        + num_merges
    )

    print()
    print("=" * 65)
    print(
        "MathGPT tokenizer_50m_v0.2 Training"
    )
    print("=" * 65)

    print()
    print(
        f"Corpus: "
        f"{CORPUS_PATH}"
    )

    print(
        f"Corpus characters: "
        f"{len(text):,}"
    )

    print(
        f"Requested merges: "
        f"{num_merges:,}"
    )

    print(
        f"Base logical vocabulary: "
        f"{BASE_VOCAB_SIZE:,}"
    )

    print(
        f"Expected logical vocabulary: "
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
    # Train
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
    # Final statistics
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
    print("=" * 65)
    print(
        "TRAINING SUMMARY"
    )
    print("=" * 65)

    print()
    print(
        f"Merges requested: "
        f"{num_merges:,}"
    )

    print(
        f"Merges completed: "
        f"{merges_completed:,}"
    )

    print(
        f"Final logical vocabulary: "
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

    if (
        merges_completed
        != num_merges
    ):
        raise RuntimeError(
            "Trainer stopped before completing "
            "all requested merges. Artifacts were "
            "NOT saved as a successful run."
        )

    # --------------------------------------------------------
    # Add v0.2 metadata
    # --------------------------------------------------------

    tokenizer_config[
        "training_corpus"
    ] = {
        "name": (
            "tokenizer_training_corpus_50m_v0.2"
        ),
        "path": str(
            CORPUS_PATH
        ),
        "characters": len(
            text
        ),
        "utf8_bytes": (
            utf8_bytes
        ),
        "general_characters": (
            EXPECTED_GENERAL_CHARACTERS
        ),
        "math_characters": (
            EXPECTED_MATH_CHARACTERS
        ),
        "general_ratio": 0.70,
        "math_ratio": 0.30,
        "source_characters": {
            "fineweb_edu": 29_000_000,
            "wikipedia": 5_400_000,
            "gutenberg": 600_000,
            "openwebmath": 15_000_000,
        },
        "replacement_characters": 0,
        "null_characters": 0,
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
    # Save artifacts
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
    # Validate saved artifacts
    # --------------------------------------------------------

    validate_saved_artifacts(
        output_directory=(
            output_directory
        ),
        num_merges=(
            num_merges
        ),
    )

    # --------------------------------------------------------
    # Save run report
    # --------------------------------------------------------

    save_training_report(
        output_directory=(
            output_directory
        ),
        num_merges=(
            num_merges
        ),
        merges_completed=(
            merges_completed
        ),
        final_vocab_size=(
            final_vocab_size
        ),
        corpus_characters=len(
            text
        ),
        utf8_bytes=(
            utf8_bytes
        ),
        training_time=(
            training_time
        ),
        peak_memory_mb=(
            peak_memory_mb
        ),
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print(
        "TOKENIZER TRAINING COMPLETE"
    )
    print("=" * 65)

    print()
    print(
        "Artifacts saved to:"
    )

    print(
        output_directory
    )

    print()
    print(
        "Training report:"
    )

    print(
        TRAINING_REPORT_PATH
    )

    print()

    if (
        num_merges
        == FINAL_TARGET_MERGES
    ):

        final_vocab_passed = (
            final_vocab_size
            == FINAL_TARGET_VOCAB_SIZE
        )

        print(
            f"Expected final logical vocab: "
            f"{FINAL_TARGET_VOCAB_SIZE:,}"
        )

        print(
            f"Actual final logical vocab: "
            f"{final_vocab_size:,}"
        )

        print(
            f"Final 16K vocab check: "
            f"{final_vocab_passed}"
        )

        if not final_vocab_passed:
            raise RuntimeError(
                "Final vocabulary is not "
                "exactly 16,000."
            )

        print()
        print(
            "SUCCESS: tokenizer_50m_v0.2 "
            "completed all 15,741 merges "
            "and produced exactly 16,000 "
            "logical vocabulary entries."
        )

    else:

        print(
            "SUCCESS: v0.2 tokenizer test "
            f"completed {num_merges:,} merges."
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    train_tokenizer(
        num_merges=(
            args.merges
        ),
        overwrite=(
            args.overwrite
        ),
    )


if __name__ == "__main__":
    main()
