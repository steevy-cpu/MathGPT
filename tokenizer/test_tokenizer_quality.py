from pathlib import Path

from bpe_tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compression_ratio(byte_count, bpe_count):
    """
    Return how much smaller the BPE representation is
    compared with the raw byte-token representation.
    """

    if byte_count == 0:
        return 0.0

    return 1 - (bpe_count / byte_count)


def main():

    artifact_directory = (
        PROJECT_ROOT
        / "tokenizer"
        / "artifacts"
    )

    tokenizer = BPETokenizer.from_files(
        config_path=(
            artifact_directory
            / "tokenizer_config.json"
        ),
        merges_path=(
            artifact_directory
            / "merges.json"
        ),
        vocab_path=(
            artifact_directory
            / "vocab.json"
        ),
    )

    test_cases = [
        # General English
        "The derivative of a function measures its rate of change.",

        # Superscripts
        "x² + y² = z²",

        # Calculus
        "∫ x² dx",
        "∂f/∂x",
        "lim x→0",

        # Greek symbols
        "α + β = γ",
        "π ≈ 3.14159",

        # Comparisons
        "x ≤ y",
        "x ≥ y",
        "x ≠ y",

        # Roots and infinity
        "√x",
        "x → ∞",

        # Summation
        "Σ i = 1 to n",

        # Linear algebra
        "AᵀA",
        "A⁻¹",
        "Ax = b",

        # LaTeX
        r"\frac{x^2 + 1}{x - 3}",
        r"\sum_{i=1}^{n} i",
        r"\sqrt{x^2 + y^2}",

        # Mixed math + English
        (
            "The derivative of f(x) = x² "
            "is f'(x) = 2x."
        ),
    ]

    total_byte_tokens = 0
    total_bpe_tokens = 0

    print()
    print("MathGPT Tokenizer Quality Tests")
    print("===============================")

    for text in test_cases:

        byte_ids = (
            tokenizer.byte_tokenizer.encode(
                text
            )
        )

        bpe_ids = tokenizer.encode(
            text
        )

        decoded = tokenizer.decode(
            bpe_ids
        )

        passed = (
            decoded == text
        )

        byte_count = len(
            byte_ids
        )

        bpe_count = len(
            bpe_ids
        )

        ratio = compression_ratio(
            byte_count,
            bpe_count,
        )

        total_byte_tokens += (
            byte_count
        )

        total_bpe_tokens += (
            bpe_count
        )

        print()
        print(f"Text: {text}")

        print(
            f"Byte tokens: "
            f"{byte_count}"
        )

        print(
            f"BPE tokens:  "
            f"{bpe_count}"
        )

        print(
            f"Compression: "
            f"{ratio:.2%}"
        )

        print(
            f"Round-trip: "
            f"{passed}"
        )

        if not passed:
            raise AssertionError(
                f"Tokenizer changed text: {text}"
            )

    print()
    print("Overall")
    print("-------")

    print(
        f"Total byte tokens: "
        f"{total_byte_tokens}"
    )

    print(
        f"Total BPE tokens: "
        f"{total_bpe_tokens}"
    )

    overall_ratio = compression_ratio(
        total_byte_tokens,
        total_bpe_tokens,
    )

    print(
        f"Overall compression: "
        f"{overall_ratio:.2%}"
    )

    print()
    print(
        "All math-symbol preservation tests passed."
    )


if __name__ == "__main__":
    main()