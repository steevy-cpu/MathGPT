import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    from bpe_tokenizer import BPETokenizer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


TOKENIZER_NAME = (
    "tokenizer_50m_v0.2"
)


TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / TOKENIZER_NAME
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
)


# ============================================================
# Tokenizer specification
# ============================================================

VOCAB_SIZE = 16_000

EXPECTED_MERGES = 15_741

SPECIAL_TOKEN_COUNT = 3

EOS_TOKEN_ID = 2


# ============================================================
# General-corpus source files
# ============================================================

TRAIN_FILES = [
    (
        "fineweb_edu",
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "fineweb_edu_train.jsonl",
    ),

    (
        "wikipedia",
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "wikipedia_train.jsonl",
    ),

    (
        "gutenberg",
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "gutenberg_train.jsonl",
    ),
]


VALIDATION_FILES = [
    (
        "fineweb_edu",
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "fineweb_edu_validation.jsonl",
    ),

    (
        "wikipedia",
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "wikipedia_validation.jsonl",
    ),

    (
        "gutenberg",
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "gutenberg_validation.jsonl",
    ),
]


# ============================================================
# Tokenizer preflight validation
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
    print("=" * 65)
    print(
        "TOKENIZER PREFLIGHT"
    )
    print("=" * 65)

    for path in (
        config_path,
        merges_path,
        vocab_path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Tokenizer artifact "
                f"missing: {path}"
            )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    with open(
        merges_path,
        "r",
        encoding="utf-8",
    ) as file:

        merges = json.load(
            file
        )

    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as file:

        vocab = json.load(
            file
        )

    artifact_name = config.get(
        "artifact_name"
    )

    merge_count = len(
        merges
    )

    logical_vocab_size = (
        SPECIAL_TOKEN_COUNT
        + len(
            vocab
        )
    )

    artifact_passed = (
        artifact_name
        == TOKENIZER_NAME
    )

    merges_passed = (
        merge_count
        == EXPECTED_MERGES
    )

    vocab_passed = (
        logical_vocab_size
        == VOCAB_SIZE
    )

    print()

    print(
        f"Artifact: "
        f"{artifact_name}"
    )

    print(
        f"Expected artifact: "
        f"{TOKENIZER_NAME}"
    )

    print()

    print(
        f"Merges: "
        f"{merge_count:,}"
    )

    print(
        f"Logical vocabulary: "
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
        f"16,000 vocabulary: "
        f"{vocab_passed}"
    )

    if not (
        artifact_passed
        and
        merges_passed
        and
        vocab_passed
    ):

        raise RuntimeError(
            "SAFETY STOP: tokenizer_50m_v0.2 "
            "failed preflight validation."
        )

    tokenizer = (
        BPETokenizer.from_files(
            config_path=config_path,
            merges_path=merges_path,
            vocab_path=vocab_path,
        )
    )

    special_ids_passed = (
        tokenizer.pad_token_id
        == 0
        and
        tokenizer.bos_token_id
        == 1
        and
        tokenizer.eos_token_id
        == EOS_TOKEN_ID
    )

    print(
        f"Special token IDs passed: "
        f"{special_ids_passed}"
    )

    if not special_ids_passed:

        raise RuntimeError(
            "Unexpected special token IDs."
        )

    print()

    print(
        "SUCCESS: tokenizer_50m_v0.2 "
        "passed encoding preflight."
    )

    return tokenizer


# ============================================================
# Encode one split
# ============================================================

def encode_split(
    tokenizer,
    input_files,
    output_bin,
    metadata_path,
    split_name,
):
    """
    Encode several JSONL source files into
    one contiguous uint16 token stream.

    Every document receives one EOS token.

    Output is written to a temporary file first.
    The old binary is replaced only after the
    new encoding completes successfully.
    """

    print()
    print("=" * 65)

    print(
        f"Encoding {split_name} "
        f"with {TOKENIZER_NAME}"
    )

    print("=" * 65)

    output_bin.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_bin = (
        output_bin.with_name(
            output_bin.name
            + ".tmp"
        )
    )

    temporary_metadata = (
        metadata_path.with_name(
            metadata_path.name
            + ".tmp"
        )
    )

    if temporary_bin.exists():

        temporary_bin.unlink()

    if temporary_metadata.exists():

        temporary_metadata.unlink()

    # ========================================================
    # Statistics
    # ========================================================

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

        # ====================================================
        # Write temporary binary
        # ====================================================

        with open(
            temporary_bin,
            "wb",
        ) as binary_file:

            for (
                source_name,
                input_path,
            ) in input_files:

                print()

                print(
                    f"Source: "
                    f"{source_name}"
                )

                print(
                    f"Input: "
                    f"{input_path}"
                )

                if not input_path.exists():

                    raise FileNotFoundError(
                        f"Input file not found: "
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

                        line = (
                            line.strip()
                        )

                        if not line:

                            continue

                        document = json.loads(
                            line
                        )

                        text = document.get(
                            "text",
                            "",
                        )

                        if not text:

                            continue

                        # ------------------------------------
                        # FINAL v0.2 ENCODING
                        # ------------------------------------

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
                                "an empty token list."
                            )

                        # Every document MUST end in EOS.
                        if (
                            token_ids[-1]
                            != EOS_TOKEN_ID
                        ):

                            raise RuntimeError(
                                "Document is missing EOS."
                            )

                        document_min = min(
                            token_ids
                        )

                        document_max = max(
                            token_ids
                        )

                        if document_min < 0:

                            raise ValueError(
                                "Negative token ID found."
                            )

                        if (
                            document_max
                            >= VOCAB_SIZE
                        ):

                            raise ValueError(
                                "Token ID outside "
                                "16K vocabulary: "
                                f"{document_max}"
                            )

                        token_array = (
                            np.asarray(
                                token_ids,
                                dtype=np.uint16,
                            )
                        )

                        token_array.tofile(
                            binary_file
                        )

                        # ------------------------------------
                        # Statistics
                        # ------------------------------------

                        document_tokens = len(
                            token_ids
                        )

                        character_count = len(
                            text
                        )

                        total_documents += 1

                        total_characters += (
                            character_count
                        )

                        total_tokens += (
                            document_tokens
                        )

                        source_documents += 1

                        source_characters += (
                            character_count
                        )

                        source_tokens += (
                            document_tokens
                        )

                        if (
                            minimum_token_id
                            is None
                            or
                            document_min
                            < minimum_token_id
                        ):

                            minimum_token_id = (
                                document_min
                            )

                        if (
                            maximum_token_id
                            is None
                            or
                            document_max
                            > maximum_token_id
                        ):

                            maximum_token_id = (
                                document_max
                            )

                        # ------------------------------------
                        # Progress
                        # ------------------------------------

                        if (
                            total_documents
                            % 500
                            == 0
                        ):

                            elapsed = (
                                time.perf_counter()
                                - start_time
                            )

                            print(
                                f"\rDocuments: "
                                f"{total_documents:,} | "
                                f"Tokens: "
                                f"{total_tokens:,} | "
                                f"Elapsed: "
                                f"{elapsed:,.1f}s",
                                end="",
                            )

                print()

                print(
                    f"{source_name} complete:"
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
                }

        # ====================================================
        # Validate temporary binary size
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start_time
        )

        file_size_bytes = (
            temporary_bin
            .stat()
            .st_size
        )

        expected_file_size = (
            total_tokens
            * np.dtype(
                np.uint16
            ).itemsize
        )

        if (
            file_size_bytes
            != expected_file_size
        ):

            raise RuntimeError(
                "Binary size mismatch: "
                f"{file_size_bytes:,} actual "
                f"vs "
                f"{expected_file_size:,} expected."
            )

        # ====================================================
        # Metadata
        # ====================================================

        metadata = {
            "stage": "5M.10",

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
                tokenizer.eos_token_id,

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
                file_size_bytes,

            "encoding_seconds":
                elapsed,

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

        # ====================================================
        # Replace obsolete v0.1 output only after success.
        # ====================================================

        os.replace(
            temporary_bin,
            output_bin,
        )

        os.replace(
            temporary_metadata,
            metadata_path,
        )

    finally:

        if temporary_bin.exists():

            temporary_bin.unlink()

        if temporary_metadata.exists():

            temporary_metadata.unlink()

    # ========================================================
    # Report
    # ========================================================

    print()

    print(
        f"{split_name} encoding complete"
    )

    print(
        "-" * 40
    )

    print(
        f"Tokenizer: "
        f"{TOKENIZER_NAME}"
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
        f"{file_size_bytes / (1024 ** 2):,.2f} MB"
    )

    print(
        f"Encoding time: "
        f"{elapsed:,.2f} seconds"
    )

    if elapsed > 0:

        print(
            f"Tokens/second: "
            f"{total_tokens / elapsed:,.0f}"
        )

    print(
        f"Binary output: "
        f"{output_bin}"
    )

    print(
        f"Metadata output: "
        f"{metadata_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    tokenizer = load_tokenizer()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Train
    # ========================================================

    encode_split(
        tokenizer=tokenizer,

        input_files=TRAIN_FILES,

        output_bin=(
            OUTPUT_DIR
            / "general_train.bin"
        ),

        metadata_path=(
            OUTPUT_DIR
            / "general_train_metadata.json"
        ),

        split_name=(
            "general_train"
        ),
    )

    # ========================================================
    # Validation
    # ========================================================

    encode_split(
        tokenizer=tokenizer,

        input_files=VALIDATION_FILES,

        output_bin=(
            OUTPUT_DIR
            / "general_validation.bin"
        ),

        metadata_path=(
            OUTPUT_DIR
            / "general_validation_metadata.json"
        ),

        split_name=(
            "general_validation"
        ),
    )

    print()
    print("=" * 65)

    print(
        "SUCCESS: GENERAL CORPUS "
        "RE-ENCODED WITH "
        "tokenizer_50m_v0.2."
    )

    print("=" * 65)


if __name__ == "__main__":
    main()