import json
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

    artifact_directory = (
        PROJECT_ROOT
        / "tokenizer"
        / "artifacts"
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

    # Load tokenizer config.
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        tokenizer_config = json.load(
            file
        )

    # -----------------------------
    # Train tokenizer
    # -----------------------------

    trainer = BPETrainer(
        num_merges=20
    )

    trainer.train(
        training_texts
    )

    # -----------------------------
    # Save tokenizer
    # -----------------------------

    trainer.save(
        output_directory=artifact_directory,
        tokenizer_config=tokenizer_config,
    )

    # -----------------------------
    # Create tokenizer before load
    # -----------------------------

    original_tokenizer = BPETokenizer(
        config_path=config_path,
        merges=trainer.merges,
        token_bytes=trainer.token_bytes,
    )

    # -----------------------------
    # Load tokenizer from disk
    # -----------------------------

    loaded_tokenizer = (
        BPETokenizer.from_files(
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
    )

    test_texts = [
        "the derivative of x is useful",
        "the matrix is useful",
        "∫ x² dx",
        r"\frac{x^2}{2}",
        "α + β = γ",
    ]

    print()
    print("Tokenizer Save/Load Tests")
    print("=========================")

    for text in test_texts:

        original_ids = (
            original_tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
            )
        )

        loaded_ids = (
            loaded_tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
            )
        )

        decoded = (
            loaded_tokenizer.decode(
                loaded_ids
            )
        )

        same_encoding = (
            original_ids
            == loaded_ids
        )

        correct_decoding = (
            decoded == text
        )

        print()
        print(f"Text: {text}")
        print(
            f"Same encoding: "
            f"{same_encoding}"
        )

        print(
            f"Correct decoding: "
            f"{correct_decoding}"
        )

        if not same_encoding:
            raise AssertionError(
                "Loaded tokenizer produced "
                "different token IDs."
            )

        if not correct_decoding:
            raise AssertionError(
                "Loaded tokenizer failed "
                "round-trip test."
            )

    print()
    print(
        "All tokenizer save/load tests passed."
    )


if __name__ == "__main__":
    main()