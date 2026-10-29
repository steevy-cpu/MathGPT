import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "math"
    / "openwebmath_50m_source.jsonl"
)


DATASET_NAME = "open-web-math/open-web-math"

MIN_MATH_SCORE = 0.70

MAX_USABLE_CHARS_PER_DOCUMENT = 20_000

# We only need 15M for the final tokenizer corpus,
# but collect 18M to leave cleaning headroom.
TARGET_USABLE_CHARS = 18_000_000

SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 5_000


def get_math_score(metadata):
    """
    Extract math_score from the OpenWebMath
    metadata JSON.
    """

    if metadata is None:
        return None

    if isinstance(metadata, str):

        try:
            metadata = json.loads(
                metadata
            )

        except json.JSONDecodeError:
            return None

    if not isinstance(
        metadata,
        dict,
    ):
        return None

    extraction_info = metadata.get(
        "extraction_info",
        {},
    )

    if isinstance(
        extraction_info,
        dict,
    ):

        score = extraction_info.get(
            "math_score"
        )

        if score is not None:

            try:
                return float(score)

            except (
                TypeError,
                ValueError,
            ):
                return None

    # Fallback in case the dataset format
    # places math_score at the top level.
    score = metadata.get(
        "math_score"
    )

    if score is not None:

        try:
            return float(score)

        except (
            TypeError,
            ValueError,
        ):
            return None

    return None


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 65)
    print(
        "OpenWebMath — 50M Tokenizer Source Expansion"
    )
    print("=" * 65)

    print()
    print(
        f"Minimum math score: "
        f"{MIN_MATH_SCORE}"
    )

    print(
        f"Per-document usable cap: "
        f"{MAX_USABLE_CHARS_PER_DOCUMENT:,}"
    )

    print(
        f"Target usable characters: "
        f"{TARGET_USABLE_CHARS:,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Deterministic buffered shuffle.
    #
    # This prevents us from relying only on
    # whatever happens to be at the beginning
    # of the dataset shards.
    dataset = dataset.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER,
    )

    scanned_documents = 0
    accepted_documents = 0

    empty_documents = 0
    missing_score_documents = 0
    low_score_documents = 0

    total_full_chars = 0
    total_usable_chars = 0

    minimum_accepted_score = None
    maximum_accepted_score = None

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for document in dataset:

            scanned_documents += 1

            text = document.get(
                "text",
                ""
            )

            if not isinstance(
                text,
                str,
            ):

                empty_documents += 1
                continue

            if not text.strip():

                empty_documents += 1
                continue

            metadata = document.get(
                "metadata"
            )

            math_score = get_math_score(
                metadata
            )

            if math_score is None:

                missing_score_documents += 1
                continue

            if (
                math_score
                < MIN_MATH_SCORE
            ):

                low_score_documents += 1
                continue

            # ---------------------------------------------
            # Accepted document.
            # ---------------------------------------------

            accepted_documents += 1

            full_chars = len(
                text
            )

            usable_chars = min(
                full_chars,
                MAX_USABLE_CHARS_PER_DOCUMENT,
            )

            total_full_chars += (
                full_chars
            )

            total_usable_chars += (
                usable_chars
            )

            if (
                minimum_accepted_score
                is None
                or math_score
                < minimum_accepted_score
            ):

                minimum_accepted_score = (
                    math_score
                )

            if (
                maximum_accepted_score
                is None
                or math_score
                > maximum_accepted_score
            ):

                maximum_accepted_score = (
                    math_score
                )

            # Preserve the original OpenWebMath fields
            # so our existing corpus-processing code
            # can continue to work with this file.
            output_record = {
                "url": document.get(
                    "url"
                ),
                "text": text,
                "date": document.get(
                    "date"
                ),
                "metadata": metadata,
            }

            output_file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
            )

            output_file.write(
                "\n"
            )

            # ---------------------------------------------
            # Progress report.
            # ---------------------------------------------

            if (
                accepted_documents
                % 500
                == 0
            ):

                percentage = (
                    total_usable_chars
                    / TARGET_USABLE_CHARS
                    * 100
                )

                print()
                print(
                    f"Accepted: "
                    f"{accepted_documents:,}"
                )

                print(
                    f"Scanned: "
                    f"{scanned_documents:,}"
                )

                print(
                    f"Usable chars: "
                    f"{total_usable_chars:,} "
                    f"({percentage:.2f}%)"
                )

            # ---------------------------------------------
            # Stop once enough high-quality math has
            # been collected.
            # ---------------------------------------------

            if (
                total_usable_chars
                >= TARGET_USABLE_CHARS
            ):
                break

    print()
    print("=" * 65)
    print(
        "DOWNLOAD COMPLETE"
    )
    print("=" * 65)

    print()

    print(
        f"Documents scanned: "
        f"{scanned_documents:,}"
    )

    print(
        f"Documents accepted: "
        f"{accepted_documents:,}"
    )

    print(
        f"Empty documents skipped: "
        f"{empty_documents:,}"
    )

    print(
        f"Missing score skipped: "
        f"{missing_score_documents:,}"
    )

    print(
        f"Low score skipped: "
        f"{low_score_documents:,}"
    )

    print()

    print(
        f"Full accepted characters: "
        f"{total_full_chars:,}"
    )

    print(
        f"Usable characters after "
        f"20K/doc cap: "
        f"{total_usable_chars:,}"
    )

    print()

    print(
        f"Minimum accepted math score: "
        f"{minimum_accepted_score}"
    )

    print(
        f"Maximum accepted math score: "
        f"{maximum_accepted_score}"
    )

    print()

    print(
        f"Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    if (
        total_usable_chars
        >= TARGET_USABLE_CHARS
    ):

        print()
        print(
            "SUCCESS: Enough high-quality "
            "OpenWebMath was collected for "
            "the 50M tokenizer experiment."
        )

    else:

        raise RuntimeError(
            "Dataset ended before the target "
            "usable character count was reached."
        )


if __name__ == "__main__":
    main()