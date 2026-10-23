import argparse
import resource
import sys
import time
from pathlib import Path

from bpe_trainer import BPETrainer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "tokenizer_training_corpus.txt"
)


def get_peak_memory_mb():
    """
    Return peak process memory usage in MB.

    macOS reports ru_maxrss in bytes.
    Linux reports ru_maxrss in kilobytes.
    """

    usage = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if sys.platform == "darwin":
        return usage / (1024 * 1024)

    return usage / 1024


def load_corpus(
    path,
    character_limit,
):
    """
    Load up to character_limit characters
    from the tokenizer training corpus.
    """

    print()
    print("Loading tokenizer corpus...")
    print(
        f"Character limit: "
        f"{character_limit:,}"
    )

    start_time = time.perf_counter()

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read(
            character_limit
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    utf8_bytes = len(
        text.encode("utf-8")
    )

    print(
        f"Characters loaded: "
        f"{len(text):,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{utf8_bytes:,}"
    )

    print(
        f"Load time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Peak memory after load: "
        f"{get_peak_memory_mb():,.2f} MB"
    )

    return text


def run_benchmark(
    text,
    num_merges,
):
    """
    Run the current BPE trainer and measure
    total runtime and peak memory.
    """

    print()
    print("=" * 60)
    print("BPE BENCHMARK")
    print("=" * 60)

    print(
        f"Characters: "
        f"{len(text):,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{len(text.encode('utf-8')):,}"
    )

    print(
        f"Requested merges: "
        f"{num_merges:,}"
    )

    print(
        f"Memory before training: "
        f"{get_peak_memory_mb():,.2f} MB"
    )

    trainer = BPETrainer(
        num_merges=num_merges
    )

    start_time = time.perf_counter()

    trainer.train(
        [text]
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    peak_memory = (
        get_peak_memory_mb()
    )

    merges_completed = len(
        trainer.merges
    )

    vocabulary_size = (
        trainer.base_vocab_size
        + merges_completed
    )

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(
        f"Merges completed: "
        f"{merges_completed:,}"
    )

    print(
        f"Vocabulary size: "
        f"{vocabulary_size:,}"
    )

    print(
        f"Total training time: "
        f"{elapsed:,.2f} seconds"
    )

    if merges_completed > 0:

        seconds_per_merge = (
            elapsed
            / merges_completed
        )

        print(
            f"Average seconds/merge: "
            f"{seconds_per_merge:.3f}"
        )

        merges_per_second = (
            merges_completed
            / elapsed
        )

        print(
            f"Average merges/second: "
            f"{merges_per_second:.4f}"
        )

        projected_1000 = (
            seconds_per_merge
            * 1000
        )

        projected_full = (
            seconds_per_merge
            * 15741
        )

        print()
        print("Naive linear projections")
        print("------------------------")

        print(
            f"1,000 merges: "
            f"{projected_1000 / 60:,.2f} minutes"
        )

        print(
            f"15,741 merges: "
            f"{projected_full / 3600:,.2f} hours"
        )

        print()
        print(
            "NOTE: These projections are only "
            "rough estimates."
        )

        print(
            "Merge cost changes as the token "
            "sequence gets shorter."
        )

    print()

    print(
        f"Peak process memory: "
        f"{peak_memory:,.2f} MB"
    )

    print()


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MathGPT's current "
            "from-scratch BPE trainer."
        )
    )

    parser.add_argument(
        "--characters",
        type=int,
        default=1_000_000,
        help=(
            "Number of corpus characters "
            "to benchmark."
        ),
    )

    parser.add_argument(
        "--merges",
        type=int,
        default=20,
        help=(
            "Number of BPE merges "
            "to perform."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"Corpus not found: "
            f"{CORPUS_PATH}"
        )

    text = load_corpus(
        CORPUS_PATH,
        args.characters,
    )

    run_benchmark(
        text,
        args.merges,
    )


if __name__ == "__main__":
    main()