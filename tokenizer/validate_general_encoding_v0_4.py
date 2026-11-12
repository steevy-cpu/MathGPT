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

try:

    from .bpe_tokenizer import (
        BPETokenizer,
    )

except ImportError:

    from bpe_tokenizer import (
        BPETokenizer,
    )


# ============================================================
# Identity
# ============================================================

DATASET_VERSION = (
    "general_pretraining_v0.4"
)

EXPECTED_STAGE = (
    "15D"
)

TOKENIZER_NAME = (
    "tokenizer_50m_v0.2"
)

VOCAB_SIZE = (
    16_000
)

EXPECTED_MERGES = (
    15_741
)

EOS_TOKEN_ID = (
    2
)


# ============================================================
# Paths
# ============================================================

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / TOKENIZER_NAME
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "general_data_v0.4.json"
)


SIZING_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general_v0.4_sized"
    / "general_pretraining_sizing_report.json"
)


CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general_v0.4_sized"
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
)


TRAIN_BINARY = (
    DATA_DIR
    / "general_v0_4_train.bin"
)


TRAIN_METADATA = (
    DATA_DIR
    / "general_v0_4_train_metadata.json"
)


VALIDATION_BINARY = (
    DATA_DIR
    / "general_v0_4_validation.bin"
)


VALIDATION_METADATA = (
    DATA_DIR
    / "general_v0_4_validation_metadata.json"
)


REPORT_PATH = (
    DATA_DIR
    / "general_v0_4_encoding_validation_report.json"
)


SOURCE_NAMES = [
    "fineweb_edu",
    "wikipedia",
    "gutenberg",
]


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


def file_sha256(
    path,
):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024
                *
                1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def get_first_document(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():

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
        f"No usable document in "
        f"{path}"
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

    checks = {
        "artifact_name":
            (
                tokenizer_config.get(
                    "artifact_name"
                )
                ==
                TOKENIZER_NAME
            ),

        "merge_count":
            (
                len(
                    merges
                )
                ==
                EXPECTED_MERGES
            ),

        "vocab_size":
            (
                len(
                    vocab
                )
                +
                3
                ==
                VOCAB_SIZE
            ),
    }

    if not all(
        checks.values()
    ):

        raise RuntimeError(
            "Tokenizer artifact validation "
            "failed."
        )

    tokenizer = (
        BPETokenizer.from_files(
            config_path=config_path,
            merges_path=merges_path,
            vocab_path=vocab_path,
        )
    )

    if (
        tokenizer.pad_token_id
        !=
        0

        or

        tokenizer.bos_token_id
        !=
        1

        or

        tokenizer.eos_token_id
        !=
        EOS_TOKEN_ID
    ):

        raise RuntimeError(
            "Tokenizer special-token IDs "
            "are incorrect."
        )

    return tokenizer


# ============================================================
# Split validation
# ============================================================

def validate_split(
    *,
    split_key,
    split_name,
    binary_path,
    metadata_path,
    tokenizer,
    sizing_report,
    config,
):

    print()
    print(
        "=" * 78
    )

    print(
        f"VALIDATING {split_name.upper()}"
    )

    print(
        "=" * 78
    )

    metadata = load_json(
        metadata_path
    )

    if not binary_path.exists():

        raise FileNotFoundError(
            binary_path
        )

    tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    actual_token_count = int(
        len(
            tokens
        )
    )

    actual_file_size = int(
        binary_path
        .stat()
        .st_size
    )

    actual_minimum = int(
        tokens.min()
    )

    actual_maximum = int(
        tokens.max()
    )

    eos_count = int(
        np.count_nonzero(
            tokens
            ==
            EOS_TOKEN_ID
        )
    )

    source_document_sum = sum(
        int(
            stats[
                "documents"
            ]
        )

        for stats in
        metadata[
            "sources"
        ].values()
    )

    source_character_sum = sum(
        int(
            stats[
                "characters"
            ]
        )

        for stats in
        metadata[
            "sources"
        ].values()
    )

    source_token_sum = sum(
        int(
            stats[
                "tokens"
            ]
        )

        for stats in
        metadata[
            "sources"
        ].values()
    )

    expected_documents = int(
        sizing_report[
            "overall"
        ][
            f"{split_key}_documents"
        ]
    )

    expected_characters = int(
        sizing_report[
            "overall"
        ][
            f"{split_key}_characters"
        ]
    )

    # ========================================================
    # Frozen target window
    # ========================================================

    objective = (
        config[
            "objective"
        ]
    )

    if split_key == "train":

        token_window_min = int(
            objective[
                "training_token_tolerance_min"
            ]
        )

        token_window_max = int(
            objective[
                "training_token_tolerance_max"
            ]
        )

    else:

        token_window_min = int(
            objective[
                "validation_token_tolerance_min"
            ]
        )

        token_window_max = int(
            objective[
                "validation_token_tolerance_max"
            ]
        )

    # ========================================================
    # Source boundary proof
    # ========================================================

    source_boundary_checks = {}

    token_offset = 0

    for source_name in SOURCE_NAMES:

        source_stats = (
            metadata[
                "sources"
            ][
                source_name
            ]
        )

        source_tokens = int(
            source_stats[
                "tokens"
            ]
        )

        source_path = (
            CLEANED_DIR
            /
            (
                f"{source_name}_"
                f"{split_key}.jsonl"
            )
        )

        first_text = get_first_document(
            source_path
        )

        expected_first_tokens = (
            tokenizer.encode(
                first_text,
                add_bos=False,
                add_eos=True,
            )
        )

        expected_array = np.asarray(
            expected_first_tokens,
            dtype=np.uint16,
        )

        actual_array = np.asarray(
            tokens[
                token_offset:
                token_offset
                +
                len(
                    expected_array
                )
            ]
        )

        first_document_matches = (
            np.array_equal(
                expected_array,
                actual_array,
            )
        )

        source_last_position = (
            token_offset
            +
            source_tokens
            -
            1
        )

        source_ends_with_eos = (
            int(
                tokens[
                    source_last_position
                ]
            )
            ==
            EOS_TOKEN_ID
        )

        source_boundary_checks[
            source_name
        ] = {
            "token_offset":
                token_offset,

            "first_document_matches":
                first_document_matches,

            "source_ends_with_eos":
                source_ends_with_eos,
        }

        token_offset += (
            source_tokens
        )

    source_offsets_cover_binary = (
        token_offset
        ==
        actual_token_count
    )

    # ========================================================
    # Checks
    # ========================================================

    checks = {
        "metadata_stage":
            (
                metadata.get(
                    "stage"
                )
                ==
                EXPECTED_STAGE
            ),

        "dataset_version":
            (
                metadata.get(
                    "dataset_version"
                )
                ==
                DATASET_VERSION
            ),

        "split_identity":
            (
                metadata.get(
                    "split"
                )
                ==
                split_name
            ),

        "tokenizer":
            (
                metadata.get(
                    "tokenizer"
                )
                ==
                TOKENIZER_NAME
            ),

        "tokenizer_merges":
            (
                int(
                    metadata.get(
                        "tokenizer_merges"
                    )
                )
                ==
                EXPECTED_MERGES
            ),

        "dtype_uint16":
            (
                metadata.get(
                    "dtype"
                )
                ==
                "uint16"
            ),

        "bytes_per_token":
            (
                int(
                    metadata.get(
                        "bytes_per_token"
                    )
                )
                ==
                2
            ),

        "vocab_size":
            (
                int(
                    metadata.get(
                        "vocab_size"
                    )
                )
                ==
                VOCAB_SIZE
            ),

        "eos_token_id":
            (
                int(
                    metadata.get(
                        "eos_token_id"
                    )
                )
                ==
                EOS_TOKEN_ID
            ),

        "token_count":
            (
                actual_token_count
                ==
                int(
                    metadata[
                        "tokens"
                    ]
                )
            ),

        "file_size":
            (
                actual_file_size
                ==
                actual_token_count
                *
                2

                and

                actual_file_size
                ==
                int(
                    metadata[
                        "file_size_bytes"
                    ]
                )
            ),

        "token_range":
            (
                actual_minimum
                >=
                0

                and

                actual_maximum
                <
                VOCAB_SIZE
            ),

        "metadata_token_range":
            (
                actual_minimum
                ==
                int(
                    metadata[
                        "minimum_token_id"
                    ]
                )

                and

                actual_maximum
                ==
                int(
                    metadata[
                        "maximum_token_id"
                    ]
                )
            ),

        "eos_per_document":
            (
                eos_count
                ==
                int(
                    metadata[
                        "documents"
                    ]
                )
            ),

        "document_count_vs_sizing":
            (
                int(
                    metadata[
                        "documents"
                    ]
                )
                ==
                expected_documents
            ),

        "character_count_vs_sizing":
            (
                int(
                    metadata[
                        "characters"
                    ]
                )
                ==
                expected_characters
            ),

        "source_document_accounting":
            (
                source_document_sum
                ==
                int(
                    metadata[
                        "documents"
                    ]
                )
            ),

        "source_character_accounting":
            (
                source_character_sum
                ==
                int(
                    metadata[
                        "characters"
                    ]
                )
            ),

        "source_token_accounting":
            (
                source_token_sum
                ==
                actual_token_count
            ),

        "token_window":
            (
                token_window_min
                <=
                actual_token_count
                <=
                token_window_max
            ),

        "source_offsets_cover_binary":
            source_offsets_cover_binary,

        "all_source_first_documents_match":
            all(
                item[
                    "first_document_matches"
                ]

                for item in
                source_boundary_checks.values()
            ),

        "all_sources_end_with_eos":
            all(
                item[
                    "source_ends_with_eos"
                ]

                for item in
                source_boundary_checks.values()
            ),
    }

    # ========================================================
    # Binary fingerprint
    # ========================================================

    print()
    print(
        "Calculating binary SHA-256..."
    )

    binary_sha256 = file_sha256(
        binary_path
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        f"Documents: "
        f"{metadata['documents']:,}"
    )

    print(
        f"Characters: "
        f"{metadata['characters']:,}"
    )

    print(
        f"Tokens: "
        f"{actual_token_count:,}"
    )

    print(
        f"EOS tokens: "
        f"{eos_count:,}"
    )

    print(
        f"Token range: "
        f"{actual_minimum} – "
        f"{actual_maximum}"
    )

    print(
        f"Binary size: "
        f"{actual_file_size / (1024 ** 2):,.2f} MiB"
    )

    print(
        f"SHA-256: "
        f"{binary_sha256}"
    )

    print()

    print(
        "Source boundary checks"
    )

    print(
        "----------------------"
    )

    for (
        source_name,
        result,
    ) in source_boundary_checks.items():

        print(
            f"{source_name:<16} "
            f"first_match="
            f"{result['first_document_matches']} "
            f"ends_eos="
            f"{result['source_ends_with_eos']}"
        )

    print()

    print(
        "Checks"
    )

    print(
        "------"
    )

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name:<40} "
            f"{passed}"
        )

    all_passed = all(
        checks.values()
    )

    if not all_passed:

        raise RuntimeError(
            f"{split_name} encoding "
            "validation failed."
        )

    print()
    print(
        f"SUCCESS: {split_name} "
        f"ENCODING VALIDATED."
    )

    result = {
        "split":
            split_name,

        "binary":
            str(
                binary_path
            ),

        "metadata":
            str(
                metadata_path
            ),

        "binary_sha256":
            binary_sha256,

        "documents":
            int(
                metadata[
                    "documents"
                ]
            ),

        "characters":
            int(
                metadata[
                    "characters"
                ]
            ),

        "tokens":
            actual_token_count,

        "eos_count":
            eos_count,

        "minimum_token_id":
            actual_minimum,

        "maximum_token_id":
            actual_maximum,

        "file_size_bytes":
            actual_file_size,

        "source_boundary_checks":
            source_boundary_checks,

        "checks":
            checks,

        "all_passed":
            all_passed,
    }

    del tokens

    return result


# ============================================================
# Main
# ============================================================

def main():

    config = load_json(
        CONFIG_PATH
    )

    sizing_report = load_json(
        SIZING_REPORT_PATH
    )

    tokenizer = load_tokenizer()

    print()
    print(
        "=" * 78
    )

    print(
        "STAGE 15E.1 — GENERAL-v0.4 "
        "ENCODING INTEGRITY AUDIT"
    )

    print(
        "=" * 78
    )

    train_result = validate_split(
        split_key="train",

        split_name=(
            "general_v0_4_train"
        ),

        binary_path=TRAIN_BINARY,

        metadata_path=TRAIN_METADATA,

        tokenizer=tokenizer,

        sizing_report=sizing_report,

        config=config,
    )

    validation_result = validate_split(
        split_key="validation",

        split_name=(
            "general_v0_4_validation"
        ),

        binary_path=VALIDATION_BINARY,

        metadata_path=(
            VALIDATION_METADATA
        ),

        tokenizer=tokenizer,

        sizing_report=sizing_report,

        config=config,
    )

    total_tokens = (
        train_result[
            "tokens"
        ]
        +
        validation_result[
            "tokens"
        ]
    )

    validation_ratio = (
        validation_result[
            "tokens"
        ]
        /
        total_tokens
    )

    report = {
        "stage":
            "15E.1",

        "dataset_version":
            DATASET_VERSION,

        "tokenizer":
            TOKENIZER_NAME,

        "train":
            train_result,

        "validation":
            validation_result,

        "total_tokens":
            total_tokens,

        "validation_token_ratio":
            validation_ratio,

        "all_passed":
            (
                train_result[
                    "all_passed"
                ]
                and
                validation_result[
                    "all_passed"
                ]
            ),
    }

    temporary = REPORT_PATH.with_name(
        REPORT_PATH.name
        +
        ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary.replace(
        REPORT_PATH
    )

    print()
    print(
        "=" * 78
    )

    print(
        "GENERAL-v0.4 ENCODING VALIDATION SUMMARY"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"Train tokens: "
        f"{train_result['tokens']:,}"
    )

    print(
        f"Validation tokens: "
        f"{validation_result['tokens']:,}"
    )

    print(
        f"Total tokens: "
        f"{total_tokens:,}"
    )

    print(
        f"Validation ratio: "
        f"{validation_ratio:.4%}"
    )

    print()

    print(
        "Report:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "=" * 78
    )

    if report[
        "all_passed"
    ]:

        print(
            "SUCCESS: GENERAL-v0.4 "
            "TOKENIZED ARTIFACTS ARE VALID."
        )

    else:

        raise RuntimeError(
            "GENERAL-v0.4 encoding "
            "integrity audit failed."
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()