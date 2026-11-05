import hashlib
import json
import sys

from pathlib import Path

import numpy as np


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

from tokenizer.bpe_tokenizer import (
    BPETokenizer,
)


# ============================================================
# Paths
# ============================================================

DATA_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "general_data_v0.2.json"
)


TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.2.json"
)


TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_50m_v0.2"
)


# ============================================================
# Constants
# ============================================================

TOKENIZER_NAME = (
    "tokenizer_50m_v0.2"
)

VOCAB_SIZE = (
    16_000
)

EOS_TOKEN_ID = (
    2
)

EXPECTED_TRAIN_TOKENS = (
    51_776_327
)

EXPECTED_VALIDATION_TOKENS = (
    5_778_078
)

EXPECTED_TRAIN_DOCUMENTS = (
    55_814
)

EXPECTED_VALIDATION_DOCUMENTS = (
    6_242
)

EXPECTED_TRAIN_CHARACTERS = (
    238_924_840
)

EXPECTED_VALIDATION_CHARACTERS = (
    26_567_546
)

TARGET_TRAINING_TOKENS = (
    50_000_000
)


# ============================================================
# Helpers
# ============================================================

def load_json(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def resolve_project_path(
    path,
):

    path = Path(
        path
    )

    if path.is_absolute():

        return path

    return (
        PROJECT_ROOT
        /
        path
    )


def hash_text(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def metadata_path_for_binary(
    binary_path,
):

    return binary_path.with_name(
        binary_path.stem
        +
        "_metadata.json"
    )


# ============================================================
# Tokenizer
# ============================================================

def load_tokenizer():

    config_path = (
        TOKENIZER_DIR
        / "tokenizer_config.json"
    )

    merges_path = (
        TOKENIZER_DIR
        / "merges.json"
    )

    vocab_path = (
        TOKENIZER_DIR
        / "vocab.json"
    )

    tokenizer_config = load_json(
        config_path
    )

    merges = load_json(
        merges_path
    )

    vocab = load_json(
        vocab_path
    )

    artifact_passed = (
        tokenizer_config.get(
            "artifact_name"
        )
        ==
        TOKENIZER_NAME
    )

    merge_count_passed = (
        len(
            merges
        )
        ==
        15_741
    )

    vocab_passed = (
        3
        +
        len(
            vocab
        )
        ==
        VOCAB_SIZE
    )

    print()
    print(
        "Tokenizer checks"
    )

    print(
        "----------------"
    )

    print(
        f"Correct tokenizer artifact: "
        f"{artifact_passed}"
    )

    print(
        f"15,741 merges: "
        f"{merge_count_passed}"
    )

    print(
        f"16,000 vocabulary: "
        f"{vocab_passed}"
    )

    if not (
        artifact_passed
        and
        merge_count_passed
        and
        vocab_passed
    ):

        raise AssertionError(
            "Tokenizer artifact validation failed."
        )

    tokenizer = (
        BPETokenizer.from_files(
            config_path=(
                config_path
            ),

            merges_path=(
                merges_path
            ),

            vocab_path=(
                vocab_path
            ),
        )
    )

    special_ids_passed = (
        tokenizer.pad_token_id
        ==
        0

        and

        tokenizer.bos_token_id
        ==
        1

        and

        tokenizer.eos_token_id
        ==
        EOS_TOKEN_ID
    )

    print(
        f"Special token IDs: "
        f"{special_ids_passed}"
    )

    if not special_ids_passed:

        raise AssertionError(
            "Tokenizer special IDs failed."
        )

    return tokenizer


# ============================================================
# Get first cleaned document
# ============================================================

def get_first_document(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:

                continue

            record = json.loads(
                line
            )

            text = record.get(
                "text",
                "",
            )

            if text:

                return text

    raise RuntimeError(
        f"No usable document in {path}"
    )


# ============================================================
# Binary scan
# ============================================================

def scan_binary(
    binary_path,
):
    """
    Memory-map the binary and scan it in chunks.

    This avoids loading the entire 50M-token
    corpus into RAM.
    """

    tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    total_tokens = len(
        tokens
    )

    if total_tokens == 0:

        raise RuntimeError(
            "Binary contains zero tokens."
        )

    chunk_size = (
        5_000_000
    )

    minimum_token_id = None

    maximum_token_id = None

    eos_count = 0

    for start in range(
        0,
        total_tokens,
        chunk_size,
    ):

        end = min(
            start
            +
            chunk_size,
            total_tokens,
        )

        chunk = tokens[
            start:end
        ]

        chunk_min = int(
            chunk.min()
        )

        chunk_max = int(
            chunk.max()
        )

        chunk_eos = int(
            np.count_nonzero(
                chunk
                ==
                EOS_TOKEN_ID
            )
        )

        if (
            minimum_token_id
            is None
            or
            chunk_min
            <
            minimum_token_id
        ):

            minimum_token_id = (
                chunk_min
            )

        if (
            maximum_token_id
            is None
            or
            chunk_max
            >
            maximum_token_id
        ):

            maximum_token_id = (
                chunk_max
            )

        eos_count += (
            chunk_eos
        )

    return {
        "tokens":
            total_tokens,

        "minimum_token_id":
            minimum_token_id,

        "maximum_token_id":
            maximum_token_id,

        "eos_count":
            eos_count,

        "file_size_bytes":
            binary_path
            .stat()
            .st_size,
    }


# ============================================================
# Validate encoded split
# ============================================================

def validate_encoded_split(
    *,
    split_name,
    binary_path,
    metadata_path,
    first_source_path,
    tokenizer,
    expected_tokens,
    expected_documents,
    expected_characters,
):

    print()
    print(
        "=" * 72
    )

    print(
        f"VALIDATING {split_name.upper()}"
    )

    print(
        "=" * 72
    )

    if not binary_path.exists():

        raise FileNotFoundError(
            binary_path
        )

    metadata = load_json(
        metadata_path
    )

    stats = scan_binary(
        binary_path
    )

    # ========================================================
    # Metadata identity
    # ========================================================

    stage_passed = (
        metadata.get(
            "stage"
        )
        ==
        "12A.6"
    )

    dataset_version_passed = (
        metadata.get(
            "dataset_version"
        )
        ==
        "general_pretraining_v0.2"
    )

    tokenizer_passed = (
        metadata.get(
            "tokenizer"
        )
        ==
        TOKENIZER_NAME
    )

    dtype_passed = (
        metadata.get(
            "dtype"
        )
        ==
        "uint16"
    )

    vocab_passed = (
        metadata.get(
            "vocab_size"
        )
        ==
        VOCAB_SIZE
    )

    eos_metadata_passed = (
        metadata.get(
            "eos_token_id"
        )
        ==
        EOS_TOKEN_ID
    )

    # ========================================================
    # Exact expected corpus statistics
    # ========================================================

    token_count_passed = (
        stats[
            "tokens"
        ]
        ==
        expected_tokens
        ==
        metadata.get(
            "tokens"
        )
    )

    document_count_passed = (
        metadata.get(
            "documents"
        )
        ==
        expected_documents
    )

    character_count_passed = (
        metadata.get(
            "characters"
        )
        ==
        expected_characters
    )

    # ========================================================
    # Binary size
    # ========================================================

    expected_file_size = (
        expected_tokens
        *
        np.dtype(
            np.uint16
        ).itemsize
    )

    file_size_passed = (
        stats[
            "file_size_bytes"
        ]
        ==
        expected_file_size
        ==
        metadata.get(
            "file_size_bytes"
        )
    )

    # ========================================================
    # Token range
    # ========================================================

    token_range_passed = (
        stats[
            "minimum_token_id"
        ]
        >=
        0

        and

        stats[
            "maximum_token_id"
        ]
        <
        VOCAB_SIZE
    )

    metadata_range_passed = (
        stats[
            "minimum_token_id"
        ]
        ==
        metadata.get(
            "minimum_token_id"
        )

        and

        stats[
            "maximum_token_id"
        ]
        ==
        metadata.get(
            "maximum_token_id"
        )
    )

    # ========================================================
    # EOS count
    # ========================================================

    eos_count_passed = (
        stats[
            "eos_count"
        ]
        ==
        expected_documents
    )

    # ========================================================
    # Source accounting
    # ========================================================

    sources = metadata.get(
        "sources",
        {},
    )

    source_documents = sum(
        source.get(
            "documents",
            0,
        )

        for source
        in sources.values()
    )

    source_characters = sum(
        source.get(
            "characters",
            0,
        )

        for source
        in sources.values()
    )

    source_tokens = sum(
        source.get(
            "tokens",
            0,
        )

        for source
        in sources.values()
    )

    source_accounting_passed = (
        source_documents
        ==
        expected_documents

        and

        source_characters
        ==
        expected_characters

        and

        source_tokens
        ==
        expected_tokens
    )

    # ========================================================
    # Re-encode first document
    # ========================================================

    first_text = get_first_document(
        first_source_path
    )

    expected_first_tokens = (
        tokenizer.encode(
            first_text,
            add_bos=False,
            add_eos=True,
        )
    )

    expected_first_array = np.asarray(
        expected_first_tokens,
        dtype=np.uint16,
    )

    binary_tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    actual_first_array = np.asarray(
        binary_tokens[
            :len(
                expected_first_array
            )
        ]
    )

    first_document_passed = (
        np.array_equal(
            actual_first_array,
            expected_first_array,
        )
    )

    # ========================================================
    # Report
    # ========================================================

    checks = {
        "Stage is 12A.6":
            stage_passed,

        "Dataset is General-v0.2":
            dataset_version_passed,

        "Tokenizer is tokenizer_50m_v0.2":
            tokenizer_passed,

        "dtype is uint16":
            dtype_passed,

        "Vocabulary is 16K":
            vocab_passed,

        "EOS metadata is correct":
            eos_metadata_passed,

        "Exact token count":
            token_count_passed,

        "Exact document count":
            document_count_passed,

        "Exact character count":
            character_count_passed,

        "Binary file size":
            file_size_passed,

        "Token IDs in valid range":
            token_range_passed,

        "Metadata min/max matches binary":
            metadata_range_passed,

        "Exactly one EOS per document":
            eos_count_passed,

        "Per-source accounting":
            source_accounting_passed,

        "First document re-encodes exactly":
            first_document_passed,
    }

    print()

    print(
        f"Documents: "
        f"{metadata.get('documents'):,}"
    )

    print(
        f"Characters: "
        f"{metadata.get('characters'):,}"
    )

    print(
        f"Tokens: "
        f"{stats['tokens']:,}"
    )

    print(
        f"EOS tokens: "
        f"{stats['eos_count']:,}"
    )

    print(
        f"Token range: "
        f"{stats['minimum_token_id']} "
        f"to "
        f"{stats['maximum_token_id']}"
    )

    print()

    print(
        "Checks"
    )

    print(
        "------"
    )

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    if not all(
        checks.values()
    ):

        failed = [
            name

            for name, passed
            in checks.items()

            if not passed
        ]

        raise AssertionError(
            f"{split_name} failed checks: "
            f"{failed}"
        )

    print()

    print(
        f"SUCCESS: {split_name} "
        f"encoding is valid."
    )

    return {
        "metadata":
            metadata,

        "stats":
            stats,
    }


# ============================================================
# Leakage audit
# ============================================================

def collect_document_hashes(
    paths,
):

    hashes = {}

    document_count = 0

    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:

                    continue

                record = json.loads(
                    line
                )

                text = record.get(
                    "text",
                    "",
                )

                if not text:

                    continue

                digest = hash_text(
                    text
                )

                document_count += 1

                hashes[
                    digest
                ] = (
                    hashes.get(
                        digest,
                        0,
                    )
                    +
                    1
                )

    return (
        hashes,
        document_count,
    )


def validate_leakage(
    cleaned_dir,
):

    print()
    print(
        "=" * 72
    )

    print(
        "RE-RUNNING EXACT LEAKAGE CHECK"
    )

    print(
        "=" * 72
    )

    train_files = [
        cleaned_dir
        / "fineweb_edu_train.jsonl",

        cleaned_dir
        / "wikipedia_train.jsonl",

        cleaned_dir
        / "gutenberg_train.jsonl",
    ]

    validation_files = [
        cleaned_dir
        / "fineweb_edu_validation.jsonl",

        cleaned_dir
        / "wikipedia_validation.jsonl",

        cleaned_dir
        / "gutenberg_validation.jsonl",
    ]

    (
        train_hashes,
        train_documents,
    ) = collect_document_hashes(
        train_files
    )

    (
        validation_hashes,
        validation_documents,
    ) = collect_document_hashes(
        validation_files
    )

    train_duplicates = sum(
        count
        -
        1

        for count
        in train_hashes.values()

        if count
        >
        1
    )

    validation_duplicates = sum(
        count
        -
        1

        for count
        in validation_hashes.values()

        if count
        >
        1
    )

    overlap = (
        set(
            train_hashes
        )
        &
        set(
            validation_hashes
        )
    )

    checks = {
        "Train documents = 55,814":
            (
                train_documents
                ==
                EXPECTED_TRAIN_DOCUMENTS
            ),

        "Validation documents = 6,242":
            (
                validation_documents
                ==
                EXPECTED_VALIDATION_DOCUMENTS
            ),

        "Train exact duplicates = 0":
            (
                train_duplicates
                ==
                0
            ),

        "Validation exact duplicates = 0":
            (
                validation_duplicates
                ==
                0
            ),

        "Train/validation overlap = 0":
            (
                len(
                    overlap
                )
                ==
                0
            ),
    }

    print()

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    if not all(
        checks.values()
    ):

        raise AssertionError(
            "Leakage validation failed."
        )

    print()

    print(
        "SUCCESS: leakage audit reproduced."
    )


# ============================================================
# Training config validation
# ============================================================

def validate_training_config(
    training_config,
):

    print()
    print(
        "=" * 72
    )

    print(
        "TRAINING CONFIG CONSISTENCY"
    )

    print(
        "=" * 72
    )

    data = (
        training_config[
            "data"
        ]
    )

    training = (
        training_config[
            "training"
        ]
    )

    optimizer = (
        training_config[
            "optimizer"
        ]
    )

    scheduler = (
        training_config[
            "scheduler"
        ]
    )

    batch_size = int(
        training[
            "batch_size"
        ]
    )

    context_length = int(
        training[
            "context_length"
        ]
    )

    configured_tokens_per_step = int(
        training[
            "tokens_per_step"
        ]
    )

    calculated_tokens_per_step = (
        batch_size
        *
        context_length
    )

    max_steps = int(
        training[
            "max_steps"
        ]
    )

    expected_steps = (
        EXPECTED_TRAIN_TOKENS
        +
        calculated_tokens_per_step
        -
        1
    ) // calculated_tokens_per_step

    configured_exposures = int(
        training[
            "target_token_exposures"
        ]
    )

    calculated_exposures = (
        max_steps
        *
        calculated_tokens_per_step
    )

    exposure_overage = (
        calculated_exposures
        -
        EXPECTED_TRAIN_TOKENS
    )

    warmup_steps = int(
        scheduler[
            "warmup_steps"
        ]
    )

    checks = {
        "Training config version = 0.2":
            (
                str(
                    training_config.get(
                        "version"
                    )
                )
                ==
                "0.2"
            ),

        "Tokenizer unchanged":
            (
                data[
                    "tokenizer"
                ]
                ==
                TOKENIZER_NAME
            ),

        "Train tokens match":
            (
                int(
                    data[
                        "training_tokens"
                    ]
                )
                ==
                EXPECTED_TRAIN_TOKENS
            ),

        "Validation tokens match":
            (
                int(
                    data[
                        "validation_tokens"
                    ]
                )
                ==
                EXPECTED_VALIDATION_TOKENS
            ),

        "Batch size = 4":
            (
                batch_size
                ==
                4
            ),

        "Context length = 256":
            (
                context_length
                ==
                256
            ),

        "Tokens/step = batch × context":
            (
                configured_tokens_per_step
                ==
                calculated_tokens_per_step
                ==
                1_024
            ),

        "Max steps = exact corpus-equivalent ceiling":
            (
                max_steps
                ==
                expected_steps
                ==
                50_563
            ),

        "Target exposures are mathematically correct":
            (
                configured_exposures
                ==
                calculated_exposures
                ==
                51_776_512
            ),

        "Exposure overage < one batch":
            (
                0
                <=
                exposure_overage
                <
                calculated_tokens_per_step
            ),

        "Seed = 42":
            (
                int(
                    training[
                        "seed"
                    ]
                )
                ==
                42
            ),

        "AdamW learning rate unchanged":
            (
                float(
                    optimizer[
                        "learning_rate"
                    ]
                )
                ==
                0.0003
            ),

        "AdamW beta1 unchanged":
            (
                float(
                    optimizer[
                        "beta1"
                    ]
                )
                ==
                0.9
            ),

        "AdamW beta2 unchanged":
            (
                float(
                    optimizer[
                        "beta2"
                    ]
                )
                ==
                0.999
            ),

        "Weight decay unchanged":
            (
                float(
                    optimizer[
                        "weight_decay"
                    ]
                )
                ==
                0.01
            ),

        "Warmup = 2,467":
            (
                warmup_steps
                ==
                2_467
            ),

        "Warmup shorter than training":
            (
                0
                <
                warmup_steps
                <
                max_steps
            ),

        "Minimum LR = 3e-5":
            (
                float(
                    scheduler[
                        "min_learning_rate"
                    ]
                )
                ==
                0.00003
            ),
    }

    print()

    print(
        f"Training tokens: "
        f"{EXPECTED_TRAIN_TOKENS:,}"
    )

    print(
        f"Tokens per step: "
        f"{calculated_tokens_per_step:,}"
    )

    print(
        f"Max steps: "
        f"{max_steps:,}"
    )

    print(
        f"Token exposures: "
        f"{calculated_exposures:,}"
    )

    print(
        f"Exposure overage: "
        f"{exposure_overage:,}"
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    if not all(
        checks.values()
    ):

        failed = [
            name

            for name, passed
            in checks.items()

            if not passed
        ]

        raise AssertionError(
            "Training config validation "
            f"failed: {failed}"
        )

    print()

    print(
        "SUCCESS: General-v0.2 training "
        "schedule is internally consistent."
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 12A.9 — GENERAL-v0.2 "
        "FINAL PIPELINE VALIDATION"
    )

    print(
        "=" * 72
    )

    data_config = load_json(
        DATA_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    cleaned_dir = resolve_project_path(
        data_config[
            "paths"
        ][
            "cleaned_directory"
        ]
    )

    # ========================================================
    # Split report
    # ========================================================

    split_report_path = (
        cleaned_dir
        /
        "general_pretraining_split_report.json"
    )

    split_report = load_json(
        split_report_path
    )

    overall = (
        split_report[
            "overall"
        ]
    )

    split_report_checks = {
        "Total documents = 62,056":
            (
                overall[
                    "documents"
                ]
                ==
                62_056
            ),

        "Train documents = 55,814":
            (
                overall[
                    "train_documents"
                ]
                ==
                EXPECTED_TRAIN_DOCUMENTS
            ),

        "Validation documents = 6,242":
            (
                overall[
                    "validation_documents"
                ]
                ==
                EXPECTED_VALIDATION_DOCUMENTS
            ),

        "Train characters exact":
            (
                overall[
                    "train_characters"
                ]
                ==
                EXPECTED_TRAIN_CHARACTERS
            ),

        "Validation characters exact":
            (
                overall[
                    "validation_characters"
                ]
                ==
                EXPECTED_VALIDATION_CHARACTERS
            ),

        "Document accounting passed":
            bool(
                overall[
                    "document_accounting_passed"
                ]
            ),

        "Character accounting passed":
            bool(
                overall[
                    "character_accounting_passed"
                ]
            ),
    }

    print()
    print(
        "Prepared-corpus report"
    )

    print(
        "----------------------"
    )

    for (
        name,
        passed,
    ) in split_report_checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    if not all(
        split_report_checks.values()
    ):

        raise AssertionError(
            "Prepared-corpus report failed."
        )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = load_tokenizer()

    # ========================================================
    # Paths from training config
    # ========================================================

    train_binary = resolve_project_path(
        training_config[
            "data"
        ][
            "train_binary"
        ]
    )

    validation_binary = resolve_project_path(
        training_config[
            "data"
        ][
            "validation_binary"
        ]
    )

    train_metadata = (
        metadata_path_for_binary(
            train_binary
        )
    )

    validation_metadata = (
        metadata_path_for_binary(
            validation_binary
        )
    )

    # ========================================================
    # Validate train binary
    # ========================================================

    validate_encoded_split(
        split_name=(
            "General-v0.2 training"
        ),

        binary_path=(
            train_binary
        ),

        metadata_path=(
            train_metadata
        ),

        first_source_path=(
            cleaned_dir
            /
            "fineweb_edu_train.jsonl"
        ),

        tokenizer=(
            tokenizer
        ),

        expected_tokens=(
            EXPECTED_TRAIN_TOKENS
        ),

        expected_documents=(
            EXPECTED_TRAIN_DOCUMENTS
        ),

        expected_characters=(
            EXPECTED_TRAIN_CHARACTERS
        ),
    )

    # ========================================================
    # Validate validation binary
    # ========================================================

    validate_encoded_split(
        split_name=(
            "General-v0.2 validation"
        ),

        binary_path=(
            validation_binary
        ),

        metadata_path=(
            validation_metadata
        ),

        first_source_path=(
            cleaned_dir
            /
            "fineweb_edu_validation.jsonl"
        ),

        tokenizer=(
            tokenizer
        ),

        expected_tokens=(
            EXPECTED_VALIDATION_TOKENS
        ),

        expected_documents=(
            EXPECTED_VALIDATION_DOCUMENTS
        ),

        expected_characters=(
            EXPECTED_VALIDATION_CHARACTERS
        ),
    )

    # ========================================================
    # Leakage
    # ========================================================

    validate_leakage(
        cleaned_dir
    )

    # ========================================================
    # Training config
    # ========================================================

    validate_training_config(
        training_config
    )

    # ========================================================
    # 50M target
    # ========================================================

    target_passed = (
        EXPECTED_TRAIN_TOKENS
        >=
        TARGET_TRAINING_TOKENS
    )

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL GENERAL-v0.2 ACCEPTANCE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Training tokens: "
        f"{EXPECTED_TRAIN_TOKENS:,}"
    )

    print(
        f"Validation tokens: "
        f"{EXPECTED_VALIDATION_TOKENS:,}"
    )

    print(
        f"Total tokens: "
        f"{EXPECTED_TRAIN_TOKENS + EXPECTED_VALIDATION_TOKENS:,}"
    )

    print()

    print(
        f"50M training-token target reached: "
        f"{target_passed}"
    )

    print(
        f"Optimizer steps: "
        f"{50_563:,}"
    )

    print(
        f"Token exposures: "
        f"{51_776_512:,}"
    )

    print(
        f"Corpus-equivalent overage: "
        f"{185:,} tokens"
    )

    if not target_passed:

        raise AssertionError(
            "50M training-token target "
            "was not reached."
        )

    print()
    print(
        "=" * 72
    )

    print(
        "SUCCESS: STAGE 12A COMPLETE."
    )

    print()

    print(
        "GENERAL-v0.2 DATA PIPELINE "
        "IS READY FOR PRETRAINING."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()