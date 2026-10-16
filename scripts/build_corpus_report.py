import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def analyze_dataset(file_path):
    """
    Read one JSONL dataset and return basic statistics.
    """

    total_documents = 0
    total_tokens = 0

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:
            document = json.loads(line)

            total_documents += 1

            total_tokens += document.get(
                "token_count_source",
                0,
            )

    if total_documents == 0:
        average_tokens = 0
    else:
        average_tokens = (
            total_tokens / total_documents
        )

    return {
        "documents": total_documents,
        "estimated_tokens": total_tokens,
        "average_tokens_per_document": average_tokens,
    }


def build_report():
    """
    Build a combined report for all general-data sources.
    """

    datasets = {
        "FineWeb-Edu": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "general"
            / "fineweb_edu_sample.jsonl"
        ),

        "Wikipedia": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "general"
            / "wikipedia_sample.jsonl"
        ),

        "Project Gutenberg": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "general"
            / "gutenberg_sample.jsonl"
        ),
    }

    report = {}

    total_documents = 0
    total_tokens = 0

    for source_name, file_path in datasets.items():

        if not file_path.exists():
            print(
                f"Warning: dataset not found: "
                f"{file_path}"
            )

            continue

        stats = analyze_dataset(
            file_path
        )

        report[source_name] = stats

        total_documents += stats["documents"]

        total_tokens += stats[
            "estimated_tokens"
        ]

    # Add percentage of combined tokens.
    for source_name, stats in report.items():

        if total_tokens > 0:

            percentage = (
                stats["estimated_tokens"]
                / total_tokens
                * 100
            )

        else:
            percentage = 0

        stats["percentage_of_total_tokens"] = (
            percentage
        )

    report["total"] = {
        "documents": total_documents,
        "estimated_tokens": total_tokens,
    }

    return report


def print_report(report):
    """
    Print the corpus report in a readable format.
    """

    print()
    print("MathGPT General Corpus Build Report")
    print("===================================")
    print()

    for source_name, stats in report.items():

        if source_name == "total":
            continue

        print(source_name)
        print("-" * len(source_name))

        print(
            f"Documents: "
            f"{stats['documents']:,}"
        )

        print(
            f"Estimated tokens: "
            f"{stats['estimated_tokens']:,}"
        )

        print(
            f"Average tokens/document: "
            f"{stats['average_tokens_per_document']:.2f}"
        )

        print(
            f"Percentage of combined sample: "
            f"{stats['percentage_of_total_tokens']:.2f}%"
        )

        print()

    print("Combined Total")
    print("--------------")

    print(
        f"Documents: "
        f"{report['total']['documents']:,}"
    )

    print(
        f"Estimated tokens: "
        f"{report['total']['estimated_tokens']:,}"
    )

    print()


def save_report(report):
    """
    Save the report as JSON.
    """

    output_path = (
        PROJECT_ROOT
        / "logs"
        / "general_corpus_build_report.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"Report saved to: "
        f"{output_path}"
    )


def main():

    report = build_report()

    print_report(report)

    save_report(report)


if __name__ == "__main__":
    main()