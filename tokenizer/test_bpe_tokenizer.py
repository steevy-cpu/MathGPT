from pathlib import Path

from bpe_trainer import BPETrainer
from bpe_tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():

    config_path = (
        PROJECT_ROOT
        / "config"
        / "tokenizer_config_v0.1.json"
    )

    training_texts = [
        "the derivative of x is useful",
        "the derivative of y is useful",
        "the derivative helps with calculus",
        "calculus uses the derivative",
        "the matrix is useful",
        "the matrix and the vector",
        "linear algebra uses a matrix",
        "the derivative and the matrix",
    ]

    trainer = BPETrainer(
        num_merges=20
    )

    trainer.train(
        training_texts
    )

    tokenizer = BPETokenizer(
        config_path=config_path,
        merges=trainer.merges,
        token_bytes=trainer.token_bytes,
    )

    test_texts = [
        "the derivative of x is useful",
        "the matrix is useful",
        "calculus uses the derivative",
        "∫ x² dx",
        "π ≤ 3.14159",
        r"\frac{x^2}{2}",
        "Café",
        "α + β = γ",
    ]

    print()
    print("BPE Tokenizer Tests")
    print("===================")

    for text in test_texts:

        byte_ids = (
            tokenizer.byte_tokenizer.encode(
                text
            )
        )

        bpe_ids = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        decoded = tokenizer.decode(
            bpe_ids
        )

        passed = (
            decoded == text
        )

        print()
        print(f"Original: {text}")

        print(
            f"Byte token count: "
            f"{len(byte_ids)}"
        )

        print(
            f"BPE token count:  "
            f"{len(bpe_ids) - 2}"
        )

        print(
            f"BPE token IDs: "
            f"{bpe_ids}"
        )

        print(
            f"Decoded: {decoded}"
        )

        print(
            f"Round-trip passed: "
            f"{passed}"
        )

        if not passed:
            raise AssertionError(
                f"BPE round-trip failed for: "
                f"{text}"
            )

    print()
    print(
        "All BPE tokenizer tests passed."
    )


if __name__ == "__main__":
    main()