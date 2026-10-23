import time
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


def benchmark(
    tokenizer,
    text,
    label,
):

    byte_count = len(
        text.encode("utf-8")
    )

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    print(
        f"Characters: "
        f"{len(text):,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{byte_count:,}"
    )

    start = time.perf_counter()

    token_ids = tokenizer.encode(
        text,
        add_bos=False,
        add_eos=True,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Tokens produced: "
        f"{len(token_ids):,}"
    )

    print(
        f"Encoding time: "
        f"{elapsed:.4f} seconds"
    )

    if elapsed > 0:

        print(
            f"Characters/second: "
            f"{len(text) / elapsed:,.0f}"
        )

        print(
            f"Tokens/second: "
            f"{len(token_ids) / elapsed:,.0f}"
        )

    if byte_count > 0:

        reduction = (
            1
            - (
                len(token_ids)
                / byte_count
            )
        )

        print(
            f"Byte → BPE reduction: "
            f"{reduction:.2%}"
        )


def main():

    tokenizer = load_tokenizer()

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read(
            100_000
        )

    benchmark(
        tokenizer,
        text[:1_000],
        "1,000 Character Test",
    )

    benchmark(
        tokenizer,
        text[:10_000],
        "10,000 Character Test",
    )

    benchmark(
        tokenizer,
        text[:100_000],
        "100,000 Character Test",
    )


if __name__ == "__main__":
    main()