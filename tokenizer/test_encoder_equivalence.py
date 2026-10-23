import json
import random
from pathlib import Path

from bpe_tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


ARTIFACT_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_10m_v0.1"
)


CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "tokenizer_training_corpus.txt"
)


RANDOM_SEED = 42

NUMBER_OF_RANDOM_TESTS = 20

RANDOM_SAMPLE_LENGTH = 500


# ============================================================
# Load tokenizer
# ============================================================

def load_tokenizer():

    return BPETokenizer.from_files(
        config_path=(
            ARTIFACT_DIR
            / "tokenizer_config.json"
        ),

        merges_path=(
            ARTIFACT_DIR
            / "merges.json"
        ),

        vocab_path=(
            ARTIFACT_DIR
            / "vocab.json"
        ),
    )


# ============================================================
# Old slow encoder
# ============================================================

def slow_encode(
    tokenizer,
    text,
):
    """
    Reproduce the original BPE encoding algorithm.

    This applies every learned merge rule
    sequentially across the entire sequence.

    It is slow, but useful as a reference
    implementation for correctness testing.
    """

    token_ids = (
        tokenizer.byte_tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )
    )

    for merge in tokenizer.merges:

        left_id = merge[
            "left"
        ]

        right_id = merge[
            "right"
        ]

        new_id = merge[
            "new_id"
        ]

        merged = []

        index = 0

        while index < len(
            token_ids
        ):

            if (
                index
                < len(token_ids) - 1
                and token_ids[index]
                == left_id
                and token_ids[index + 1]
                == right_id
            ):

                merged.append(
                    new_id
                )

                index += 2

            else:

                merged.append(
                    token_ids[index]
                )

                index += 1

        token_ids = merged

    return token_ids


# ============================================================
# Test one sample
# ============================================================

def compare_encoders(
    tokenizer,
    text,
    label,
):

    slow_ids = slow_encode(
        tokenizer,
        text,
    )

    fast_ids = tokenizer.encode(
        text,
        add_bos=False,
        add_eos=False,
    )

    same = (
        slow_ids
        == fast_ids
    )

    decoded = tokenizer.decode(
        fast_ids
    )

    round_trip = (
        decoded
        == text
    )

    print()
    print(
        f"{label}"
    )

    print(
        "-" * len(label)
    )

    print(
        f"Characters: "
        f"{len(text):,}"
    )

    print(
        f"Slow token count: "
        f"{len(slow_ids):,}"
    )

    print(
        f"Fast token count: "
        f"{len(fast_ids):,}"
    )

    print(
        f"Exact token ID match: "
        f"{same}"
    )

    print(
        f"Round-trip passed: "
        f"{round_trip}"
    )

    if not same:

        print()
        print(
            "First mismatch"
        )

        print(
            "--------------"
        )

        max_length = max(
            len(slow_ids),
            len(fast_ids),
        )

        for index in range(
            max_length
        ):

            slow_value = (
                slow_ids[index]
                if index < len(slow_ids)
                else None
            )

            fast_value = (
                fast_ids[index]
                if index < len(fast_ids)
                else None
            )

            if (
                slow_value
                != fast_value
            ):

                print(
                    f"Index: "
                    f"{index}"
                )

                print(
                    f"Slow: "
                    f"{slow_value}"
                )

                print(
                    f"Fast: "
                    f"{fast_value}"
                )

                break

    return (
        same
        and round_trip
    )


# ============================================================
# Main
# ============================================================

def main():

    tokenizer = (
        load_tokenizer()
    )

    print()
    print(
        "MathGPT Encoder Equivalence Test"
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # Fixed tests
    # --------------------------------------------------------

    fixed_tests = [
        (
            "General English",
            (
                "Artificial intelligence "
                "learns patterns from data."
            ),
        ),

        (
            "Math Prose",
            (
                "The derivative of x squared "
                "is equal to 2x."
            ),
        ),

        (
            "LaTeX Fraction",
            r"\frac{x^2}{2}",
        ),

        (
            "LaTeX Integral",
            (
                r"\int_0^1 x^2 \, dx "
                r"= \frac{1}{3}"
            ),
        ),

        (
            "Unicode Math",
            "π ≤ 3.14159 and α + β = γ",
        ),

        (
            "Mixed Technical",
            (
                "The matrix A^T A is symmetric "
                "when A is a real matrix."
            ),
        ),
    ]

    all_passed = True

    for label, text in (
        fixed_tests
    ):

        passed = compare_encoders(
            tokenizer,
            text,
            label,
        )

        if not passed:
            all_passed = False

    # --------------------------------------------------------
    # Random corpus samples
    # --------------------------------------------------------

    print()
    print(
        "Random corpus samples"
    )

    print(
        "====================="
    )

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        corpus = file.read()

    rng = random.Random(
        RANDOM_SEED
    )

    maximum_start = (
        len(corpus)
        - RANDOM_SAMPLE_LENGTH
    )

    for test_number in range(
        1,
        NUMBER_OF_RANDOM_TESTS + 1,
    ):

        start = rng.randint(
            0,
            maximum_start
        )

        text = corpus[
            start:
            start
            + RANDOM_SAMPLE_LENGTH
        ]

        label = (
            f"Random sample "
            f"{test_number}"
        )

        passed = compare_encoders(
            tokenizer,
            text,
            label,
        )

        if not passed:
            all_passed = False

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 60
    )

    if all_passed:

        print(
            "SUCCESS: Fast heap encoder "
            "matches the original BPE encoder exactly."
        )

    else:

        print(
            "FAILURE: Fast encoder produced "
            "different token IDs."
        )


if __name__ == "__main__":
    main()