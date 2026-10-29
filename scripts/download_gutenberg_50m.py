import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "general"
    / "gutenberg_50m_source.jsonl"
)


# Final tokenizer corpus only needs 600K.
# We collect 1.2M for diversity and cleaning headroom.
TARGET_CHARS = 1_200_000

# Prevent one enormous book from dominating
# the collected source.
MAX_CHARS_PER_BOOK = 100_000

SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 1_000


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 65)
    print(
        "Project Gutenberg — 50M Tokenizer Source Expansion"
    )
    print("=" * 65)

    print()
    print(
        f"Target characters: "
        f"{TARGET_CHARS:,}"
    )

    print(
        f"Maximum chars/book: "
        f"{MAX_CHARS_PER_BOOK:,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("Loading streaming dataset...")

    dataset = load_dataset(
        "common-pile/project_gutenberg",
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER,
    )

    scanned = 0
    accepted = 0
    total_chars = 0

    empty_skipped = 0
    non_english_skipped = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for document in dataset:

            scanned += 1

            text = document.get(
                "text",
                ""
            )

            if (
                not isinstance(text, str)
                or not text.strip()
            ):
                empty_skipped += 1
                continue

            metadata = document.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            language = metadata.get(
                "language"
            )

            # Dataset is intended to be English,
            # but keep the guard anyway.
            if (
                language is not None
                and language != "en"
            ):
                non_english_skipped += 1
                continue

            # Limit contribution from one book.
            contribution = text[
                :MAX_CHARS_PER_BOOK
            ]

            remaining = (
                TARGET_CHARS
                - total_chars
            )

            if len(contribution) > remaining:
                contribution = contribution[
                    :remaining
                ]

            output_record = {
                "id": document.get("id"),
                "source": document.get(
                    "source"
                ),
                "text": contribution,
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

            accepted += 1
            total_chars += len(
                contribution
            )

            if accepted % 5 == 0:

                print(
                    f"Books: "
                    f"{accepted:,} | "
                    f"Characters: "
                    f"{total_chars:,}"
                )

            if (
                total_chars
                >= TARGET_CHARS
            ):
                break

    print()
    print("=" * 65)
    print("DOWNLOAD COMPLETE")
    print("=" * 65)

    print(
        f"Documents scanned: "
        f"{scanned:,}"
    )

    print(
        f"Books accepted: "
        f"{accepted:,}"
    )

    print(
        f"Empty skipped: "
        f"{empty_skipped:,}"
    )

    print(
        f"Non-English skipped: "
        f"{non_english_skipped:,}"
    )

    print(
        f"Characters saved: "
        f"{total_chars:,}"
    )

    print(
        f"Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    if (
        total_chars
        >= TARGET_CHARS
    ):

        print()
        print(
            "SUCCESS: Enough diverse "
            "Gutenberg text was collected."
        )

    else:

        raise RuntimeError(
            "Gutenberg source ended before "
            "target character count."
        )


if __name__ == "__main__":
    main()