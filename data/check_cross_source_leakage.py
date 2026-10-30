import hashlib
import json
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


GENERAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general"
)


TRAIN_FILES = [
    GENERAL_DIR
    / "fineweb_edu_train.jsonl",

    GENERAL_DIR
    / "wikipedia_train.jsonl",

    GENERAL_DIR
    / "gutenberg_train.jsonl",
]


VALIDATION_FILES = [
    GENERAL_DIR
    / "fineweb_edu_validation.jsonl",

    GENERAL_DIR
    / "wikipedia_validation.jsonl",

    GENERAL_DIR
    / "gutenberg_validation.jsonl",
]


def hash_text(
    text,
):
    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def collect_hashes(
    paths,
):
    hashes = {}

    document_count = 0

    for path in paths:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                record = json.loads(
                    line
                )

                text = record.get(
                    "text",
                    "",
                )

                if not text:
                    continue

                digest = hash_text(
                    text
                )

                document_count += 1

                hashes.setdefault(
                    digest,
                    []
                ).append(
                    path.name
                )

    return (
        hashes,
        document_count,
    )


def main():

    print()
    print(
        "Cross-Source Leakage Audit"
    )

    print(
        "=========================="
    )

    (
        train_hashes,
        train_documents,
    ) = collect_hashes(
        TRAIN_FILES
    )

    (
        validation_hashes,
        validation_documents,
    ) = collect_hashes(
        VALIDATION_FILES
    )

    overlap = (
        set(
            train_hashes
        )
        &
        set(
            validation_hashes
        )
    )

    print()

    print(
        f"Train documents: "
        f"{train_documents:,}"
    )

    print(
        f"Validation documents: "
        f"{validation_documents:,}"
    )

    print(
        f"Exact train/validation overlaps: "
        f"{len(overlap):,}"
    )

    if overlap:

        print()
        print(
            "Example overlaps:"
        )

        for digest in list(
            overlap
        )[:10]:

            print()

            print(
                f"Hash: "
                f"{digest}"
            )

            print(
                f"Train sources: "
                f"{train_hashes[digest]}"
            )

            print(
                f"Validation sources: "
                f"{validation_hashes[digest]}"
            )

        raise AssertionError(
            "Exact text leakage exists "
            "between train and validation."
        )

    print()

    print(
        "Train/validation exact-text "
        "overlap: False"
    )

    print()

    print(
        "=" * 65
    )

    print(
        "SUCCESS: NO EXACT CROSS-SOURCE "
        "TRAIN/VALIDATION LEAKAGE."
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()