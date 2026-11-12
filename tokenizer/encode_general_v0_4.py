import argparse
import json
import os
import sys
import time

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

STAGE = (
    "15D"
)


# ============================================================
# Tokenizer
# ============================================================

TOKENIZER_NAME = (
    "tokenizer_50m_v0.2"
)

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / TOKENIZER_NAME
)

VOCAB_SIZE = 16_000

EXPECTED_MERGES = 15_741

SPECIAL_TOKEN_COUNT = 3

EOS_TOKEN_ID = 2


# ============================================================
# Configuration
# ============================================================

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


# ============================================================
# Input / output
# ============================================================

CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general_v0.4_sized"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
)


TRAIN_FILES = [
    (
        "fineweb_edu",
        CLEANED_DIR
        / "fineweb_edu_train.jsonl",
    ),

    (
        "wikipedia",
        CLEANED_DIR
        / "wikipedia_train.jsonl",
    ),

    (
        "gutenberg",
        CLEANED_DIR
        / "gutenberg_train.jsonl",
    ),
]


VALIDATION_FILES = [
    (
        "fineweb_edu",
        CLEANED_DIR
        / "fineweb_edu_validation.jsonl",
    ),

    (
        "wikipedia",
        CLEANED_DIR
        / "wikipedia_validation.jsonl",
    ),

    (
        "gutenberg",
        CLEANED_DIR
        / "gutenberg_validation.jsonl",
    ),
]


TRAIN_BINARY = (
    OUTPUT_DIR
    / "general_v0_4_train.bin"
)

TRAIN_METADATA = (
    OUTPUT_DIR
    / "general_v0_4_train_metadata.json"
)

VALIDATION_BINARY = (
    OUTPUT_DIR
    / "general_v0_4_validation.bin"
)

VALIDATION_METADATA = (
    OUTPUT_DIR
    / "general_v0_4_validation_metadata.json"
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


# ============================================================
# Tokenizer preflight
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

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 TOKENIZER PREFLIGHT"
    )

    print(
        "=" * 76
    )

    for path in (
        config_path,
        merges_path,
        vocab_path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Tokenizer artifact missing: "
                f"{path}"
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

    artifact_name = (
        tokenizer_config.get(
            "artifact_name"
        )
    )

    merge_count = len(
        merges
    )

    logical_vocab_size = (
        SPECIAL_TOKEN_COUNT
        +
        len(
            vocab
        )
    )

    artifact_passed = (
        artifact_name
        ==
        TOKENIZER_NAME
    )

    merges_passed = (
        merge_count
        ==
        EXPECTED_MERGES
    )

    vocabulary_passed = (
        logical_vocab_size
        ==
        VOCAB_SIZE
    )

    print()
    print(
        f"Tokenizer artifact: "
        f"{artifact_name}"
    )

    print(
        f"Merges: "
        f"{merge_count:,}"
    )

    print(
        f"Vocabulary: "
        f"{logical_vocab_size:,}"
    )

    print()

    print(
        f"Correct tokenizer: "
        f"{artifact_passed}"
    )

    print(
        f"15,741 merges: "
        f"{merges_passed}"
    )

    print(
        f"16K vocabulary: "
        f"{vocabulary_passed}"
    )

    if not (
        artifact_passed
        and
        merges_passed
        and
        vocabulary_passed
    ):

        raise RuntimeError(
            "Tokenizer preflight failed."
        )

    tokenizer = (
        BPETokenizer.from_files(
            config_path=config_path,
            merges_path=merges_path,
            vocab_path=vocab_path,
        )
    )

    special_tokens_passed = (
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
        f"Special token IDs correct: "
        f"{special_tokens_passed}"
    )

    if not special_tokens_passed:

        raise RuntimeError(
            "Unexpected special token IDs."
        )

    print()
    print(
        "SUCCESS: tokenizer_50m_v0.2 "
        "passed preflight."
    )

    return tokenizer


# ============================================================
# Corpus preflight
# ============================================================

def corpus_preflight(
    config,
):

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 SIZED CORPUS PREFLIGHT"
    )

    print(
        "=" * 76
    )

    report = load_json(
        SIZING_REPORT_PATH
    )

    checks = {
        "Dataset identity":
            (
                report.get(
                    "dataset_version"
                )
                ==
                DATASET_VERSION
            ),

        "Sizing stage":
            (
                report.get(
                    "stage"
                )
                ==
                "15C.1"
            ),

        "Training projection passed":
            bool(
                report[
                    "overall"
                ][
                    "training_projection_within_tolerance"
                ]
            ),

        "Validation projection passed":
            bool(
                report[
                    "overall"
                ][
                    "validation_projection_within_tolerance"
                ]
            ),

        "Projected train near 250M":
            (
                245_000_000
                <=
                int(
                    report[
                        "overall"
                    ][
                        "projected_train_tokens"
                    ]
                )
                <=
                255_000_000
            ),

        "Projected validation near 27.8M":
            (
                27_000_000
                <=
                int(
                    report[
                        "overall"
                    ][
                        "projected_validation_tokens"
                    ]
                )
                <=
                29_000_000
            ),

        "Tokenizer frozen":
            (
                config[
                    "objective"
                ][
                    "tokenizer"
                ]
                ==
                TOKENIZER_NAME
            ),

        "Vocab frozen":
            (
                int(
                    config[
                        "objective"
                    ][
                        "vocab_size"
                    ]
                )
                ==
                VOCAB_SIZE
            ),
    }

    for (
        source_name,
        path,
    ) in (
        TRAIN_FILES
        +
        VALIDATION_FILES
    ):

        checks[
            f"{path.name} exists"
        ] = path.exists()

    print()

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name:<48} "
            f"{passed}"
        )

    all_passed = all(
        checks.values()
    )

    if not all_passed:

        raise RuntimeError(
            "GENERAL-v0.4 corpus "
            "preflight failed."
        )

    print()
    print(
        f"Projected train tokens: "
        f"{report['overall']['projected_train_tokens']:,}"
    )

    print(
        f"Projected validation tokens: "
        f"{report['overall']['projected_validation_tokens']:,}"
    )

    print()

    print(
        "SUCCESS: GENERAL-v0.4 "
        "sized corpus passed preflight."
    )

    return report


# ============================================================
# Output protection
# ============================================================

def protect_outputs(
    overwrite,
):

    outputs = [
        TRAIN_BINARY,
        TRAIN_METADATA,
        VALIDATION_BINARY,
        VALIDATION_METADATA,
    ]

    existing = [
        path
        for path
        in outputs
        if path.exists()
    ]

    if (
        existing
        and
        not overwrite
    ):

        print()
        print(
            "GENERAL-v0.4 tokenized outputs "
            "already exist:"
        )

        for path in existing:

            print(
                f"  {path}"
            )

        print()
        print(
            "Use --overwrite only if you "
            "intend to rebuild them."
        )

        raise SystemExit(
            0
        )

    if overwrite:

        for path in existing:

            path.unlink()


# ============================================================
# Encode one split
# ============================================================

def encode_split(
    *,
    tokenizer,
    input_files,
    binary_path,
    metadata_path,
    split_name,
):

    print()
    print(
        "=" * 76
    )

    print(
        f"ENCODING {split_name.upper()}"
    )

    print(
        "=" * 76
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_binary = (
        binary_path.with_name(
            binary_path.name
            +
            ".tmp"
        )
    )

    temporary_metadata = (
        metadata_path.with_name(
            metadata_path.name
            +
            ".tmp"
        )
    )

    for path in (
        temporary_binary,
        temporary_metadata,
    ):

        if path.exists():

            path.unlink()

    total_documents = 0

    total_characters = 0

    total_tokens = 0

    minimum_token_id = None

    maximum_token_id = None

    source_stats = {}

    start_time = (
        time.perf_counter()
    )

    try:

        with open(
            temporary_binary,
            "wb",
        ) as binary_file:

            for (
                source_name,
                input_path,
            ) in input_files:

                if not input_path.exists():

                    raise FileNotFoundError(
                        input_path
                    )

                print()
                print(
                    f"Source: "
                    f"{source_name}"
                )

                print(
                    f"Input: "
                    f"{input_path}"
                )

                source_documents = 0

                source_characters = 0

                source_tokens = 0

                with open(
                    input_path,
                    "r",
                    encoding="utf-8",
                ) as input_file:

                    for line in input_file:

                        line = line.strip()

                        if not line:

                            continue

                        document = json.loads(
                            line
                        )

                        text = document.get(
                            "text",
                            "",
                        )

                        if not isinstance(
                            text,
                            str,
                        ):

                            raise RuntimeError(
                                "Non-string text found "
                                "during encoding."
                            )

                        if not text:

                            continue

                        token_ids = (
                            tokenizer.encode(
                                text,
                                add_bos=False,
                                add_eos=True,
                            )
                        )

                        if not token_ids:

                            raise RuntimeError(
                                "Tokenizer produced "
                                "empty output."
                            )

                        if (
                            token_ids[
                                -1
                            ]
                            !=
                            EOS_TOKEN_ID
                        ):

                            raise RuntimeError(
                                "Document did not end "
                                "with EOS."
                            )

                        document_minimum = min(
                            token_ids
                        )

                        document_maximum = max(
                            token_ids
                        )

                        if (
                            document_minimum
                            <
                            0
                        ):

                            raise RuntimeError(
                                "Negative token ID found."
                            )

                        if (
                            document_maximum
                            >=
                            VOCAB_SIZE
                        ):

                            raise RuntimeError(
                                "Token ID exceeds "
                                "16K vocabulary."
                            )

                        token_array = np.asarray(
                            token_ids,
                            dtype=np.uint16,
                        )

                        token_array.tofile(
                            binary_file
                        )

                        document_tokens = len(
                            token_ids
                        )

                        document_characters = len(
                            text
                        )

                        total_documents += 1

                        total_characters += (
                            document_characters
                        )

                        total_tokens += (
                            document_tokens
                        )

                        source_documents += 1

                        source_characters += (
                            document_characters
                        )

                        source_tokens += (
                            document_tokens
                        )

                        if (
                            minimum_token_id
                            is None
                            or
                            document_minimum
                            <
                            minimum_token_id
                        ):

                            minimum_token_id = (
                                document_minimum
                            )

                        if (
                            maximum_token_id
                            is None
                            or
                            document_maximum
                            >
                            maximum_token_id
                        ):

                            maximum_token_id = (
                                document_maximum
                            )

                        if (
                            total_documents
                            %
                            1_000
                            ==
                            0
                        ):

                            elapsed = (
                                time.perf_counter()
                                -
                                start_time
                            )

                            throughput = (
                                total_tokens
                                /
                                elapsed
                                if elapsed > 0
                                else 0.0
                            )

                            print(
                                f"\rDocuments "
                                f"{total_documents:,} | "
                                f"Tokens "
                                f"{total_tokens:,} | "
                                f"{throughput:,.0f} "
                                f"tok/s",
                                end="",
                            )

                print()
                print(
                    f"{source_name} complete"
                )

                print(
                    f"  Documents: "
                    f"{source_documents:,}"
                )

                print(
                    f"  Characters: "
                    f"{source_characters:,}"
                )

                print(
                    f"  Tokens: "
                    f"{source_tokens:,}"
                )

                source_stats[
                    source_name
                ] = {
                    "documents":
                        source_documents,

                    "characters":
                        source_characters,

                    "tokens":
                        source_tokens,

                    "tokens_per_character":
                        (
                            source_tokens
                            /
                            source_characters
                            if source_characters > 0
                            else 0.0
                        ),
                }

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        file_size = (
            temporary_binary
            .stat()
            .st_size
        )

        expected_file_size = (
            total_tokens
            *
            np.dtype(
                np.uint16
            ).itemsize
        )

        if (
            file_size
            !=
            expected_file_size
        ):

            raise RuntimeError(
                "Binary file-size mismatch."
            )

        metadata = {
            "stage":
                STAGE,

            "dataset_version":
                DATASET_VERSION,

            "split":
                split_name,

            "dtype":
                "uint16",

            "bytes_per_token":
                2,

            "tokenizer":
                TOKENIZER_NAME,

            "tokenizer_merges":
                EXPECTED_MERGES,

            "vocab_size":
                VOCAB_SIZE,

            "eos_token_id":
                EOS_TOKEN_ID,

            "documents":
                total_documents,

            "characters":
                total_characters,

            "tokens":
                total_tokens,

            "minimum_token_id":
                minimum_token_id,

            "maximum_token_id":
                maximum_token_id,

            "file_size_bytes":
                file_size,

            "encoding_seconds":
                elapsed,

            "tokens_per_second":
                (
                    total_tokens
                    /
                    elapsed
                    if elapsed > 0
                    else 0.0
                ),

            "sizing_report":
                str(
                    SIZING_REPORT_PATH
                ),

            "sources":
                source_stats,
        }

        with open(
            temporary_metadata,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        os.replace(
            temporary_binary,
            binary_path,
        )

        os.replace(
            temporary_metadata,
            metadata_path,
        )

    finally:

        for path in (
            temporary_binary,
            temporary_metadata,
        ):

            if path.exists():

                path.unlink()

    print()
    print(
        f"{split_name} complete"
    )

    print(
        "-" * 44
    )

    print(
        f"Documents: "
        f"{total_documents:,}"
    )

    print(
        f"Characters: "
        f"{total_characters:,}"
    )

    print(
        f"Tokens: "
        f"{total_tokens:,}"
    )

    print(
        f"Minimum token ID: "
        f"{minimum_token_id}"
    )

    print(
        f"Maximum token ID: "
        f"{maximum_token_id}"
    )

    print(
        f"Binary size: "
        f"{file_size / (1024 ** 2):,.2f} MiB"
    )

    print(
        f"Encoding time: "
        f"{elapsed:,.2f}s"
    )

    print(
        f"Encoding throughput: "
        f"{metadata['tokens_per_second']:,.0f} "
        f"tokens/s"
    )

    return metadata


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Encode sized GENERAL-v0.4 "
            "corpus with tokenizer_50m_v0.2."
        )
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    config = load_json(
        CONFIG_PATH
    )

    if (
        config.get(
            "version"
        )
        !=
        "0.4"
    ):

        raise RuntimeError(
            "Expected GENERAL-v0.4 config."
        )

    sizing_report = corpus_preflight(
        config
    )

    tokenizer = load_tokenizer()

    if args.preflight_only:

        print()
        print(
            "=" * 76
        )

        print(
            "SUCCESS: GENERAL-v0.4 "
            "ENCODING PREFLIGHT PASSED."
        )

        print(
            "=" * 76
        )

        return

    protect_outputs(
        args.overwrite
    )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE 15D — GENERAL-v0.4 ENCODING"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Input corpus:"
    )

    print(
        CLEANED_DIR
    )

    print()

    print(
        "Historical v0.1/v0.2/v0.3 "
        "binaries will NOT be overwritten."
    )

    print()

    train_metadata = encode_split(
        tokenizer=tokenizer,

        input_files=TRAIN_FILES,

        binary_path=TRAIN_BINARY,

        metadata_path=TRAIN_METADATA,

        split_name="general_v0_4_train",
    )

    validation_metadata = encode_split(
        tokenizer=tokenizer,

        input_files=VALIDATION_FILES,

        binary_path=VALIDATION_BINARY,

        metadata_path=(
            VALIDATION_METADATA
        ),

        split_name=(
            "general_v0_4_validation"
        ),
    )

    train_tokens = int(
        train_metadata[
            "tokens"
        ]
    )

    validation_tokens = int(
        validation_metadata[
            "tokens"
        ]
    )

    total_tokens = (
        train_tokens
        +
        validation_tokens
    )

    objective = (
        config[
            "objective"
        ]
    )

    target_train = int(
        objective[
            "target_training_tokens"
        ]
    )

    target_validation = int(
        objective[
            "target_validation_tokens"
        ]
    )

    train_min = int(
        objective[
            "training_token_tolerance_min"
        ]
    )

    train_max = int(
        objective[
            "training_token_tolerance_max"
        ]
    )

    validation_min = int(
        objective[
            "validation_token_tolerance_min"
        ]
    )

    validation_max = int(
        objective[
            "validation_token_tolerance_max"
        ]
    )

    train_window_passed = (
        train_min
        <=
        train_tokens
        <=
        train_max
    )

    validation_window_passed = (
        validation_min
        <=
        validation_tokens
        <=
        validation_max
    )

    train_difference = (
        train_tokens
        -
        target_train
    )

    validation_difference = (
        validation_tokens
        -
        target_validation
    )

    actual_validation_ratio = (
        validation_tokens
        /
        total_tokens
    )

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 TOKENIZATION SUMMARY"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Training tokens: "
        f"{train_tokens:,}"
    )

    print(
        f"Validation tokens: "
        f"{validation_tokens:,}"
    )

    print(
        f"Total tokens: "
        f"{total_tokens:,}"
    )

    print(
        f"Token validation ratio: "
        f"{actual_validation_ratio:.4%}"
    )

    print()

    print(
        f"Training target: "
        f"{target_train:,}"
    )

    print(
        f"Difference: "
        f"{train_difference:+,}"
    )

    print(
        f"Allowed window: "
        f"{train_min:,} – "
        f"{train_max:,}"
    )

    print(
        f"Training window passed: "
        f"{train_window_passed}"
    )

    print()

    print(
        f"Validation target: "
        f"{target_validation:,}"
    )

    print(
        f"Difference: "
        f"{validation_difference:+,}"
    )

    print(
        f"Allowed window: "
        f"{validation_min:,} – "
        f"{validation_max:,}"
    )

    print(
        f"Validation window passed: "
        f"{validation_window_passed}"
    )

    print()

    print(
        "Source token counts — training"
    )

    print(
        "------------------------------"
    )

    for source_name, stats in (
        train_metadata[
            "sources"
        ].items()
    ):

        print(
            f"{source_name:<16} "
            f"{stats['tokens']:>14,}"
        )

    print()

    print(
        "Source token counts — validation"
    )

    print(
        "--------------------------------"
    )

    for source_name, stats in (
        validation_metadata[
            "sources"
        ].items()
    ):

        print(
            f"{source_name:<16} "
            f"{stats['tokens']:>14,}"
        )

    print()

    projected_train = int(
        sizing_report[
            "overall"
        ][
            "projected_train_tokens"
        ]
    )

    projected_validation = int(
        sizing_report[
            "overall"
        ][
            "projected_validation_tokens"
        ]
    )

    print(
        "Projection accuracy"
    )

    print(
        "-------------------"
    )

    print(
        f"Projected train: "
        f"{projected_train:,}"
    )

    print(
        f"Actual train:    "
        f"{train_tokens:,}"
    )

    print(
        f"Projection error: "
        f"{train_tokens - projected_train:+,}"
    )

    print()

    print(
        f"Projected validation: "
        f"{projected_validation:,}"
    )

    print(
        f"Actual validation:    "
        f"{validation_tokens:,}"
    )

    print(
        f"Projection error: "
        f"{validation_tokens - projected_validation:+,}"
    )

    print()

    print(
        "Artifacts"
    )

    print(
        "---------"
    )

    print(
        TRAIN_BINARY
    )

    print(
        TRAIN_METADATA
    )

    print(
        VALIDATION_BINARY
    )

    print(
        VALIDATION_METADATA
    )

    print()
    print(
        "=" * 76
    )

    if (
        train_window_passed
        and
        validation_window_passed
    ):

        print(
            "SUCCESS: GENERAL-v0.4 "
            "TOKENIZATION PASSED TARGET WINDOWS."
        )

    else:

        raise RuntimeError(
            "GENERAL-v0.4 actual token "
            "counts fell outside the frozen "
            "experimental windows."
        )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()