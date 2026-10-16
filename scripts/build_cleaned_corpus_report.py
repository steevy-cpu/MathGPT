import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def analyze_dataset(file_path):
    """
    Analyze one JSONL dataset file.
    """

    total_documents = 0
    total_tokens = 0

    if not file_path.exists():
        return {
            "documents": 0,
            "estimated_tokens": 0,
        }

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            document = json.loads(line)

            total_documents += 1

            total_tokens += document.get(
                "token_count_source",
                0,
            )

    return {
        "documents": total_documents,
        "estimated_tokens": total_tokens,
    }


def build_report():
    """
    Build the final cleaned general-corpus report.
    """

    sources = {
        "FineWeb-Edu": {
            "train": (
                PROJECT_ROOT
                / "data"
                / "cleaned"
                / "general"
                / "fineweb_edu_train.jsonl"
            ),
            "validation": (
                PROJECT_ROOT
                / "data"
                / "cleaned"
                / "general"
                / "fineweb_edu_validation.jsonl"
            ),
        },

        "Wikipedia": {
            "train": (
                PROJECT_ROOT
                / "data"
                / "cleaned"
                / "general"
                / "wikipedia_train.jsonl"
            ),
            "validation": (
                PROJECT_ROOT
                / "data"
                / "cleaned"
                / "general"
                / "wikipedia_validation.jsonl"
            ),
        },

        "Project Gutenberg": {
            "train": (
                PROJECT_ROOT
                / "data"
                / "cleaned"
                / "general"
                / "gutenberg_train.jsonl"
            ),
            "validation": (
                PROJECT_ROOT
                / "data"
                / "cleaned"
                / "general"
                / "gutenberg_validation.jsonl"
            ),
        },
    }

    report = {}

    total_train_documents = 0
    total_validation_documents = 0

    total_train_tokens = 0
    total_validation_tokens = 0

    for source_name, paths in sources.items():

        train_stats = analyze_dataset(
            paths["train"]
        )

        validation_stats = analyze_dataset(
            paths["validation"]
        )

        report[source_name] = {
            "train": train_stats,
            "validation": validation_stats,
        }

        total_train_documents += (
            train_stats["documents"]
        )

        total_validation_documents += (
            validation_stats["documents"]
        )

        total_train_tokens += (
            train_stats["estimated_tokens"]
        )

        total_validation_tokens += (
            validation_stats["estimated_tokens"]
        )

    report["total"] = {
        "train_documents": total_train_documents,
        "validation_documents": total_validation_documents,
        "train_estimated_tokens": total_train_tokens,
        "validation_estimated_tokens": total_validation_tokens,
        "all_estimated_tokens": (
            total_train_tokens
            + total_validation_tokens
        ),
    }

    return report


def print_report(report):
    """
    Print a readable cleaned-corpus report.
    """

    print()
    print("MathGPT Cleaned General Corpus Report")
    print("=====================================")
    print()

    for source_name, stats in report.items():

        if source_name == "total":
            continue

        print(source_name)
        print("-" * len(source_name))

        print(
            f"Training documents: "
            f"{stats['train']['documents']:,}"
        )

        print(
            f"Validation documents: "
            f"{stats['validation']['documents']:,}"
        )

        print(
            f"Training estimated tokens: "
            f"{stats['train']['estimated_tokens']:,}"
        )

        print(
            f"Validation estimated tokens: "
            f"{stats['validation']['estimated_tokens']:,}"
        )

        total_source_tokens = (
            stats["train"]["estimated_tokens"]
            + stats["validation"]["estimated_tokens"]
        )

        if total_source_tokens > 0:

            validation_ratio = (
                stats["validation"]["estimated_tokens"]
                / total_source_tokens
            )

            print(
                f"Validation token ratio: "
                f"{validation_ratio:.2%}"
            )

        print()

    total = report["total"]

    print("Combined Total")
    print("--------------")

    print(
        f"Training documents: "
        f"{total['train_documents']:,}"
    )

    print(
        f"Validation documents: "
        f"{total['validation_documents']:,}"
    )

    print(
        f"Training estimated tokens: "
        f"{total['train_estimated_tokens']:,}"
    )

    print(
        f"Validation estimated tokens: "
        f"{total['validation_estimated_tokens']:,}"
    )

    print(
        f"All estimated tokens: "
        f"{total['all_estimated_tokens']:,}"
    )

    if total["all_estimated_tokens"] > 0:

        combined_validation_ratio = (
            total["validation_estimated_tokens"]
            / total["all_estimated_tokens"]
        )

        print(
            f"Combined validation token ratio: "
            f"{combined_validation_ratio:.2%}"
        )

    print()


def save_report(report):
    """
    Save the report to the logs folder.
    """

    output_path = (
        PROJECT_ROOT
        / "logs"
        / "cleaned_general_corpus_report.json"
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