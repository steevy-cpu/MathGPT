import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_documents(path):
    documents = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                documents.append(json.loads(line))

    return documents


def print_document(document, label):
    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    print(
        f"ID: {document.get('id')}"
    )

    print(
        f"Math score: "
        f"{document.get('math_score')}"
    )

    print(
        f"Perplexity: "
        f"{document.get('perplexity_source')}"
    )

    print(
        f"Characters: "
        f"{document.get('character_count_source'):,}"
    )

    print(
        f"URL: "
        f"{document.get('url')}"
    )

    print()
    print("TEXT PREVIEW")
    print("-" * 80)

    text = document.get(
        "text",
        ""
    )

    print(text[:5000])


def main():

    input_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "math"
        / "openwebmath_sample.jsonl"
    )

    documents = load_documents(
        input_path
    )

    sorted_documents = sorted(
        documents,
        key=lambda document: (
            document.get(
                "math_score",
                0,
            )
        ),
    )

    lowest = sorted_documents[0]
    highest = sorted_documents[-1]

    print_document(
        lowest,
        "LOWEST MATH SCORE DOCUMENT",
    )

    print_document(
        highest,
        "HIGHEST MATH SCORE DOCUMENT",
    )


if __name__ == "__main__":
    main()