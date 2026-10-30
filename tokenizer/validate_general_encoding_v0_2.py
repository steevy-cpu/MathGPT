import json
from pathlib import Path

import numpy as np


try:
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    from bpe_tokenizer import BPETokenizer


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


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
)


VOCAB_SIZE = 16_000

EOS_TOKEN_ID = 2


TRAIN_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general"
    / "fineweb_edu_train.jsonl"
)


VALIDATION_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general"
    / "fineweb_edu_validation.jsonl"
)


def load_tokenizer():

    return BPETokenizer.from_files(
        config_path=(
            TOKENIZER_DIR
            / "tokenizer_config.json"
        ),

        merges_path=(
            TOKENIZER_DIR
            / "merges.json"
        ),

        vocab_path=(
            TOKENIZER_DIR
            / "vocab.json"
        ),
    )


def get_first_document(
    source_path,
):

    with open(
        source_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

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

            if text:

                return text

    raise RuntimeError(
        f"No usable document in "
        f"{source_path}"
    )


def validate_split(
    *,
    split_name,
    binary_path,
    metadata_path,
    source_path,
    tokenizer,
):

    print()
    print("=" * 65)

    print(
        f"Validating {split_name}"
    )

    print("=" * 65)

    if not binary_path.exists():

        raise FileNotFoundError(
            binary_path
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            metadata_path
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    # ========================================================
    # Metadata
    # ========================================================

    tokenizer_passed = (
        metadata.get(
            "tokenizer"
        )
        == TOKENIZER_NAME
    )

    stage_passed = (
        metadata.get(
            "stage"
        )
        == "5M.10"
    )

    dtype_passed = (
        metadata.get(
            "dtype"
        )
        == "uint16"
    )

    vocab_passed = (
        metadata.get(
            "vocab_size"
        )
        == VOCAB_SIZE
    )

    eos_metadata_passed = (
        metadata.get(
            "eos_token_id"
        )
        == EOS_TOKEN_ID
    )

    # ========================================================
    # Load binary using memory map
    # ========================================================

    tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    actual_token_count = len(
        tokens
    )

    expected_token_count = (
        metadata[
            "tokens"
        ]
    )

    token_count_passed = (
        actual_token_count
        == expected_token_count
    )

    # ========================================================
    # File size
    # ========================================================

    actual_file_size = (
        binary_path
        .stat()
        .st_size
    )

    expected_file_size = (
        actual_token_count
        * 2
    )

    file_size_passed = (
        actual_file_size
        == expected_file_size
        == metadata[
            "file_size_bytes"
        ]
    )

    # ========================================================
    # Token range
    # ========================================================

    actual_minimum = int(
        tokens.min()
    )

    actual_maximum = int(
        tokens.max()
    )

    range_passed = (
        actual_minimum >= 0
        and
        actual_maximum < VOCAB_SIZE
    )

    metadata_range_passed = (
        actual_minimum
        == metadata[
            "minimum_token_id"
        ]
        and
        actual_maximum
        == metadata[
            "maximum_token_id"
        ]
    )

    # ========================================================
    # EOS count
    #
    # Token ID 2 is reserved exclusively for EOS.
    #
    # Every encoded document receives exactly one EOS.
    # ========================================================

    eos_count = int(
        np.count_nonzero(
            tokens
            == EOS_TOKEN_ID
        )
    )

    document_count = (
        metadata[
            "documents"
        ]
    )

    eos_count_passed = (
        eos_count
        == document_count
    )

    # ========================================================
    # Strong tokenizer proof
    #
    # Re-encode the first source document with v0.2
    # and compare it directly to the beginning of
    # the binary file.
    # ========================================================

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

    expected_first_array = np.asarray(
        expected_first_tokens,
        dtype=np.uint16,
    )

    actual_first_array = np.asarray(
        tokens[
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
    # Enough tokens for our 256-token context
    # ========================================================

    context_passed = (
        actual_token_count
        >= 257
    )

    # ========================================================
    # Report
    # ========================================================

    print()

    print(
        f"Tokenizer metadata: "
        f"{metadata.get('tokenizer')}"
    )

    print(
        f"Documents: "
        f"{document_count:,}"
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
        f"Minimum token ID: "
        f"{actual_minimum}"
    )

    print(
        f"Maximum token ID: "
        f"{actual_maximum}"
    )

    print(
        f"Binary size: "
        f"{actual_file_size / (1024 ** 2):,.2f} MB"
    )

    print()

    print(
        "Checks"
    )

    print(
        "------"
    )

    print(
        f"Stage 5M.10 metadata: "
        f"{stage_passed}"
    )

    print(
        f"Tokenizer is v0.2: "
        f"{tokenizer_passed}"
    )

    print(
        f"uint16 metadata: "
        f"{dtype_passed}"
    )

    print(
        f"16K vocabulary metadata: "
        f"{vocab_passed}"
    )

    print(
        f"EOS metadata passed: "
        f"{eos_metadata_passed}"
    )

    print(
        f"Token count passed: "
        f"{token_count_passed}"
    )

    print(
        f"Binary file size passed: "
        f"{file_size_passed}"
    )

    print(
        f"Token range passed: "
        f"{range_passed}"
    )

    print(
        f"Metadata token range passed: "
        f"{metadata_range_passed}"
    )

    print(
        f"One EOS per document: "
        f"{eos_count_passed}"
    )

    print(
        f"First document matches v0.2: "
        f"{first_document_passed}"
    )

    print(
        f"Enough tokens for context: "
        f"{context_passed}"
    )

    all_passed = (
        stage_passed
        and
        tokenizer_passed
        and
        dtype_passed
        and
        vocab_passed
        and
        eos_metadata_passed
        and
        token_count_passed
        and
        file_size_passed
        and
        range_passed
        and
        metadata_range_passed
        and
        eos_count_passed
        and
        first_document_passed
        and
        context_passed
    )

    if not all_passed:

        raise AssertionError(
            f"{split_name} validation failed."
        )

    print()

    print(
        f"SUCCESS: {split_name} "
        f"v0.2 encoding validated."
    )

    return metadata


def main():

    tokenizer = load_tokenizer()

    train_metadata = validate_split(
        split_name="general_train",

        binary_path=(
            DATA_DIR
            / "general_train.bin"
        ),

        metadata_path=(
            DATA_DIR
            / "general_train_metadata.json"
        ),

        source_path=TRAIN_SOURCE,

        tokenizer=tokenizer,
    )

    validation_metadata = validate_split(
        split_name="general_validation",

        binary_path=(
            DATA_DIR
            / "general_validation.bin"
        ),

        metadata_path=(
            DATA_DIR
            / "general_validation_metadata.json"
        ),

        source_path=VALIDATION_SOURCE,

        tokenizer=tokenizer,
    )

    print()
    print("=" * 65)

    print(
        "STAGE 5M.10 VALIDATION SUMMARY"
    )

    print("=" * 65)

    print()

    print(
        f"Train tokens: "
        f"{train_metadata['tokens']:,}"
    )

    print(
        f"Validation tokens: "
        f"{validation_metadata['tokens']:,}"
    )

    print(
        f"Total tokens: "
        f"{train_metadata['tokens'] + validation_metadata['tokens']:,}"
    )

    print()

    print(
        "SUCCESS: GENERAL TRAIN AND "
        "VALIDATION BINARIES ARE BOTH "
        "ENCODED WITH tokenizer_50m_v0.2."
    )


if __name__ == "__main__":
    main()