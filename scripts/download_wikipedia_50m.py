import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "general"
    / "wikipedia_50m_source.jsonl"
)

# Final corpus needs 5.4M chars.
# Collect extra room for cleaning.
TARGET_CHARS = 7_000_000

SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 2_000


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 65)
    print("Wikipedia — 50M Tokenizer Source Expansion")
    print("=" * 65)

    print()
    print(
        f"Target raw characters: "
        f"{TARGET_CHARS:,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("Loading streaming dataset...")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER,
    )

    documents = 0
    total_chars = 0
    empty_documents = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for document in dataset:

            text = document.get(
                "text",
                ""
            )

            if (
                not isinstance(text, str)
                or not text.strip()
            ):
                empty_documents += 1
                continue

            output_record = {
                "text": text,
                "id": document.get("id"),
                "url": document.get("url"),
                "title": document.get("title"),
            }

            output_file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
            )

            output_file.write("\n")

            documents += 1
            total_chars += len(text)

            if documents % 100 == 0:

                percentage = (
                    total_chars
                    / TARGET_CHARS
                    * 100
                )

                print(
                    f"Documents: "
                    f"{documents:,} | "
                    f"Characters: "
                    f"{total_chars:,} "
                    f"({percentage:.2f}%)"
                )

            if total_chars >= TARGET_CHARS:
                break

    print()
    print("=" * 65)
    print("DOWNLOAD COMPLETE")
    print("=" * 65)

    print(
        f"Documents saved: "
        f"{documents:,}"
    )

    print(
        f"Empty skipped: "
        f"{empty_documents:,}"
    )

    print(
        f"Characters saved: "
        f"{total_chars:,}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    if total_chars >= TARGET_CHARS:

        print()
        print(
            "SUCCESS: Enough Wikipedia "
            "was collected."
        )

    else:
        raise RuntimeError(
            "Wikipedia ended before "
            "target size was reached."
        )


if __name__ == "__main__":
    main()