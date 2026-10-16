import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def inspect_dataset(file_path):

    total_documents = 0
    total_tokens = 0
    total_characters = 0

    min_tokens = None
    max_tokens = 0

    quality_scores = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            document = json.loads(line)

            total_documents += 1

            token_count = document.get(
                "token_count_source",
                0,
            )

            total_tokens += token_count

            total_characters += len(
                document.get(
                    "text",
                    "",
                )
            )

            if min_tokens is None:
                min_tokens = token_count
            else:
                min_tokens = min(
                    min_tokens,
                    token_count,
                )

            max_tokens = max(
                max_tokens,
                token_count,
            )

            score = document.get(
                "quality_score"
            )

            if score is not None:
                quality_scores.append(
                    score
                )

    if total_documents == 0:

        print("Dataset is empty.")
        return

    average_tokens = (
        total_tokens / total_documents
    )

    average_characters = (
        total_characters
        / total_documents
    )

    print()
    print("Dataset Statistics")
    print("------------------")

    print(
        f"Documents: {total_documents:,}"
    )

    print(
        f"Estimated tokens: {total_tokens:,}"
    )

    print(
        f"Average tokens/document: "
        f"{average_tokens:.2f}"
    )

    print(
        f"Smallest document: "
        f"{min_tokens:,} tokens"
    )

    print(
        f"Largest document: "
        f"{max_tokens:,} tokens"
    )

    print(
        f"Average characters/document: "
        f"{average_characters:.2f}"
    )

    if quality_scores:

        average_quality = (
            sum(quality_scores)
            / len(quality_scores)
        )

        print(
            f"Average quality score: "
            f"{average_quality:.3f}"
        )

    print()


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/inspect_dataset.py "
            "<dataset.jsonl>"
        )
        return

    file_path = Path(sys.argv[1])

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        print(f"Dataset not found: {file_path}")
        return

    print(f"\nInspecting: {file_path.name}")

    inspect_dataset(file_path)


if __name__ == "__main__":
    main()