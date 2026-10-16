from pathlib import Path

from byte_tokenizer import ByteTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():

    config_path = (
        PROJECT_ROOT
        / "config"
        / "tokenizer_config_v0.1.json"
    )

    tokenizer = ByteTokenizer(
        config_path
    )

    test_texts = [
        "Hello world!",
        "The derivative of x² is 2x.",
        "∫ x² dx",
        "π ≤ 3.14159",
        r"\frac{x^2}{2}",
        "Café",
        "α + β = γ",
    ]

    print()
    print("Byte Tokenizer Tests")
    print("====================")

    for text in test_texts:

        token_ids = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        decoded = tokenizer.decode(
            token_ids
        )

        passed = (
            decoded == text
        )

        print()
        print(f"Original: {text}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded:  {decoded}")
        print(f"Round-trip passed: {passed}")

        if not passed:
            raise AssertionError(
                f"Round-trip failed for: {text}"
            )

    print()
    print("All byte tokenizer tests passed.")


if __name__ == "__main__":
    main()