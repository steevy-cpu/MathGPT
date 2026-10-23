import json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_10m_v0.1"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
)


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
# Load tokenizer
# ============================================================

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
    one contiguous uint16 binary token stream.

    Each document receives one EOS token.
    """

    print()
    print("=" * 60)
    print(
        f"Encoding {split_name}"
    )
    print("=" * 60)

    output_bin.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_documents = 0
    total_characters = 0
    total_tokens = 0

    minimum_token_id = None
    maximum_token_id = None

    source_stats = {}

    start_time = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Open binary output once
    # --------------------------------------------------------

    with open(
        output_bin,
        "wb",
    ) as binary_file:

        # ----------------------------------------------------
        # Process each source
        # ----------------------------------------------------

        for (
            source_name,
            input_path,
        ) in input_files:

            print()
            print(
                f"Source: {source_name}"
            )

            print(
                f"Input: {input_path}"
            )

            if not input_path.exists():

                raise FileNotFoundError(
                    f"Input file not found: "
                    f"{input_path}"
                )

            source_documents = 0
            source_characters = 0
            source_tokens = 0

            # ------------------------------------------------
            # Read JSONL one document at a time
            # ------------------------------------------------

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

                    if not text:
                        continue

                    # ----------------------------------------
                    # Encode one document.
                    #
                    # add_eos=True means token ID 2 is placed
                    # after every document.
                    # ----------------------------------------

                    token_ids = (
                        tokenizer.encode(
                            text,
                            add_bos=False,
                            add_eos=True,
                        )
                    )

                    # ----------------------------------------
                    # Safety check:
                    # all IDs must fit our 16K vocabulary.
                    # ----------------------------------------

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

                    if document_max >= 16_000:

                        raise ValueError(
                            f"Token ID outside "
                            f"16K vocabulary: "
                            f"{document_max}"
                        )

                    # ----------------------------------------
                    # Convert Python ints → uint16
                    # ----------------------------------------

                    token_array = np.asarray(
                        token_ids,
                        dtype=np.uint16,
                    )

                    # ----------------------------------------
                    # Append directly to binary file.
                    # ----------------------------------------

                    token_array.tofile(
                        binary_file
                    )

                    # ----------------------------------------
                    # Update statistics
                    # ----------------------------------------

                    document_tokens = len(
                        token_ids
                    )

                    characters = len(
                        text
                    )

                    total_documents += 1
                    total_characters += (
                        characters
                    )
                    total_tokens += (
                        document_tokens
                    )

                    source_documents += 1

                    source_characters += (
                        characters
                    )

                    source_tokens += (
                        document_tokens
                    )

                    if (
                        minimum_token_id
                        is None
                        or document_min
                        < minimum_token_id
                    ):

                        minimum_token_id = (
                            document_min
                        )

                    if (
                        maximum_token_id
                        is None
                        or document_max
                        > maximum_token_id
                    ):

                        maximum_token_id = (
                            document_max
                        )

                    # ----------------------------------------
                    # Progress
                    # ----------------------------------------

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
                "documents": (
                    source_documents
                ),

                "characters": (
                    source_characters
                ),

                "tokens": (
                    source_tokens
                ),
            }

    # ========================================================
    # Final statistics
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    file_size_bytes = (
        output_bin.stat().st_size
    )

    expected_file_size = (
        total_tokens
        * 2
    )

    if (
        file_size_bytes
        != expected_file_size
    ):

        raise ValueError(
            "Binary file size does not "
            "match uint16 token count."
        )

    metadata = {
        "split": split_name,

        "dtype": "uint16",

        "bytes_per_token": 2,

        "tokenizer": (
            "tokenizer_10m_v0.1"
        ),

        "vocab_size": 16_000,

        "eos_token_id": (
            tokenizer.eos_token_id
        ),

        "documents": (
            total_documents
        ),

        "characters": (
            total_characters
        ),

        "tokens": (
            total_tokens
        ),

        "minimum_token_id": (
            minimum_token_id
        ),

        "maximum_token_id": (
            maximum_token_id
        ),

        "file_size_bytes": (
            file_size_bytes
        ),

        "encoding_seconds": (
            elapsed
        ),

        "sources": (
            source_stats
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        f"{split_name} encoding complete"
    )

    print(
        "-" * (
            len(split_name)
            + 18
        )
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

    tokenizer = (
        load_tokenizer()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training split
    # --------------------------------------------------------

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

        split_name="general_train",
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    encode_split(
        tokenizer=tokenizer,

        input_files=(
            VALIDATION_FILES
        ),

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


if __name__ == "__main__":
    main()