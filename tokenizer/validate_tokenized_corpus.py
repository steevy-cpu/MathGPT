import json
import random
from pathlib import Path

import numpy as np

try:
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    from bpe_tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_10m_v0.1"
)


TOKENIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
)


FILES = {
    "general_train": {
        "bin": (
            TOKENIZED_DIR
            / "general_train.bin"
        ),
        "metadata": (
            TOKENIZED_DIR
            / "general_train_metadata.json"
        ),
    },

    "general_validation": {
        "bin": (
            TOKENIZED_DIR
            / "general_validation.bin"
        ),
        "metadata": (
            TOKENIZED_DIR
            / "general_validation_metadata.json"
        ),
    },
}


RANDOM_SEED = 42

SAMPLE_TOKEN_COUNT = 256

NUMBER_OF_RANDOM_SAMPLES = 5


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


def load_metadata(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def validate_file(
    name,
    paths,
    tokenizer,
    rng,
):

    print()
    print("=" * 60)
    print(f"Validating {name}")
    print("=" * 60)

    binary_path = paths[
        "bin"
    ]

    metadata_path = paths[
        "metadata"
    ]

    if not binary_path.exists():

        raise FileNotFoundError(
            binary_path
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            metadata_path
        )

    metadata = load_metadata(
        metadata_path
    )

    # --------------------------------------------------------
    # Memory-map instead of loading everything into RAM.
    # --------------------------------------------------------

    tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    token_count = len(
        tokens
    )

    minimum_token = int(
        tokens.min()
    )

    maximum_token = int(
        tokens.max()
    )

    file_size = (
        binary_path.stat().st_size
    )

    expected_size = (
        token_count
        * 2
    )

    print()
    print("Binary statistics")
    print("-----------------")

    print(
        f"Tokens on disk: "
        f"{token_count:,}"
    )

    print(
        f"Metadata tokens: "
        f"{metadata['tokens']:,}"
    )

    print(
        f"Minimum token ID: "
        f"{minimum_token}"
    )

    print(
        f"Maximum token ID: "
        f"{maximum_token}"
    )

    print(
        f"File size: "
        f"{file_size:,} bytes"
    )

    print(
        f"Expected uint16 size: "
        f"{expected_size:,} bytes"
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    token_count_match = (
        token_count
        == metadata["tokens"]
    )

    file_size_match = (
        file_size
        == expected_size
    )

    valid_range = (
        minimum_token >= 0
        and maximum_token < 16_000
    )

    print()
    print("Integrity checks")
    print("----------------")

    print(
        f"Token count matches metadata: "
        f"{token_count_match}"
    )

    print(
        f"Binary size matches uint16: "
        f"{file_size_match}"
    )

    print(
        f"Token IDs inside vocabulary: "
        f"{valid_range}"
    )

    # --------------------------------------------------------
    # EOS check
    # --------------------------------------------------------

    eos_count = int(
        np.count_nonzero(
            tokens
            == tokenizer.eos_token_id
        )
    )

    print(
        f"EOS tokens found: "
        f"{eos_count:,}"
    )

    print(
        f"Expected documents/EOS: "
        f"{metadata['documents']:,}"
    )

    eos_match = (
        eos_count
        == metadata[
            "documents"
        ]
    )

    print(
        f"EOS count matches documents: "
        f"{eos_match}"
    )

    # --------------------------------------------------------
    # Random decode windows
    # --------------------------------------------------------

    print()
    print("Random decoded windows")
    print("----------------------")

    for sample_number in range(
        1,
        NUMBER_OF_RANDOM_SAMPLES + 1,
    ):

        maximum_start = (
            token_count
            - SAMPLE_TOKEN_COUNT
        )

        start = rng.randint(
            0,
            maximum_start
        )

        window = tokens[
            start:
            start
            + SAMPLE_TOKEN_COUNT
        ]

        token_ids = [
            int(token_id)
            for token_id in window
        ]

        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

        print()
        print(
            f"Sample {sample_number}"
        )

        print(
            f"Token range: "
            f"{start:,} → "
            f"{start + SAMPLE_TOKEN_COUNT:,}"
        )

        print()

        # Keep terminal output manageable.
        print(
            decoded[:1000]
        )

        print()
        print("-" * 60)

    passed = (
        token_count_match
        and file_size_match
        and valid_range
        and eos_match
    )

    return {
        "passed": passed,
        "token_count": token_count,
        "minimum_token_id": (
            minimum_token
        ),
        "maximum_token_id": (
            maximum_token
        ),
        "eos_count": eos_count,
    }


def main():

    tokenizer = (
        load_tokenizer()
    )

    rng = random.Random(
        RANDOM_SEED
    )

    results = {}

    for name, paths in (
        FILES.items()
    ):

        results[name] = (
            validate_file(
                name=name,
                paths=paths,
                tokenizer=tokenizer,
                rng=rng,
            )
        )

    all_passed = all(
        result["passed"]
        for result
        in results.values()
    )

    print()
    print("=" * 60)
    print("FINAL BINARY CORPUS VALIDATION")
    print("=" * 60)

    if all_passed:

        print(
            "SUCCESS: All tokenized "
            "binary files passed validation."
        )

    else:

        print(
            "FAILURE: One or more binary "
            "files failed validation."
        )


if __name__ == "__main__":
    main()