import json
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_documents(path):
    documents = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            documents.append(
                json.loads(line)
            )

    return documents


def print_math_score_distribution(scores):
    buckets = {
        "0.50 - <0.60": 0,
        "0.60 - <0.70": 0,
        "0.70 - <0.80": 0,
        "0.80 - <0.90": 0,
        "0.90 - 1.00": 0,
    }

    for score in scores:

        if 0.50 <= score < 0.60:
            buckets["0.50 - <0.60"] += 1

        elif 0.60 <= score < 0.70:
            buckets["0.60 - <0.70"] += 1

        elif 0.70 <= score < 0.80:
            buckets["0.70 - <0.80"] += 1

        elif 0.80 <= score < 0.90:
            buckets["0.80 - <0.90"] += 1

        elif 0.90 <= score <= 1.00:
            buckets["0.90 - 1.00"] += 1

    print()
    print("Math score distribution")
    print("-----------------------")

    for bucket, count in buckets.items():
        print(
            f"{bucket}: {count}"
        )


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

    math_scores = []

    perplexities = []

    character_counts = []

    math_symbol_documents = 0

    latex_documents = 0

    for document in documents:

        score = document.get(
            "math_score"
        )

        perplexity = document.get(
            "perplexity_source"
        )

        character_count = document.get(
            "character_count_source",
            0,
        )

        text = document.get(
            "text",
            "",
        )

        if score is not None:
            math_scores.append(score)

        if perplexity is not None:
            perplexities.append(perplexity)

        character_counts.append(
            character_count
        )

        # Very simple first-pass checks.
        math_symbols = [
            "=",
            "+",
            "-",
            "^",
            "∫",
            "∑",
            "√",
            "≤",
            "≥",
        ]

        if any(
            symbol in text
            for symbol in math_symbols
        ):
            math_symbol_documents += 1

        if "$" in text or "\\" in text:
            latex_documents += 1

    print()
    print("OpenWebMath Quality Report")
    print("==========================")

    print()
    print(f"Documents: {len(documents):,}")

    print(
        f"Total characters: "
        f"{sum(character_counts):,}"
    )

    print(
        f"Average characters/document: "
        f"{mean(character_counts):,.2f}"
    )

    print(
        f"Median characters/document: "
        f"{median(character_counts):,.2f}"
    )

    print(
        f"Smallest document: "
        f"{min(character_counts):,}"
    )

    print(
        f"Largest document: "
        f"{max(character_counts):,}"
    )

    if math_scores:

        print()
        print("Math score")
        print("----------")

        print(
            f"Average: "
            f"{mean(math_scores):.4f}"
        )

        print(
            f"Median: "
            f"{median(math_scores):.4f}"
        )

        print(
            f"Minimum: "
            f"{min(math_scores):.4f}"
        )

        print(
            f"Maximum: "
            f"{max(math_scores):.4f}"
        )

    if perplexities:

        print()
        print("Source perplexity")
        print("-----------------")

        print(
            f"Average: "
            f"{mean(perplexities):,.2f}"
        )

        print(
            f"Median: "
            f"{median(perplexities):,.2f}"
        )

        print(
            f"Minimum: "
            f"{min(perplexities):,.2f}"
        )

        print(
            f"Maximum: "
            f"{max(perplexities):,.2f}"
        )

    print()
    print("Math-content indicators")
    print("-----------------------")

    print(
        f"Documents with common math symbols: "
        f"{math_symbol_documents}/{len(documents)}"
    )

    print(
        f"Documents with $ or \\ notation: "
        f"{latex_documents}/{len(documents)}"
    )

    print_math_score_distribution(
        math_scores
    )


if __name__ == "__main__":
    main()