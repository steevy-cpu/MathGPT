import hashlib
import json

from pathlib import Path


# ============================================================
# Project root
# ============================================================

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
    / "general_v0.2"
)


# ============================================================
# Files
# ============================================================

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


# ============================================================
# Hash
# ============================================================

def hash_text(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Collect document hashes
# ============================================================

def collect_hashes(
    paths,
):

    hashes = {}

    document_count = 0

    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

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
                    {
                        "file":
                            path.name,

                        "line":
                            line_number,
                    }
                )

    return (
        hashes,
        document_count,
    )


# ============================================================
# Duplicate check inside one split
# ============================================================

def internal_duplicates(
    hashes,
):

    return {
        digest:
            locations

        for (
            digest,
            locations,
        ) in hashes.items()

        if len(
            locations
        )
        >
        1
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 12A.5 — GENERAL-v0.2 "
        "LEAKAGE AUDIT"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # Collect train
    # ========================================================

    (
        train_hashes,
        train_documents,
    ) = collect_hashes(
        TRAIN_FILES
    )

    # ========================================================
    # Collect validation
    # ========================================================

    (
        validation_hashes,
        validation_documents,
    ) = collect_hashes(
        VALIDATION_FILES
    )

    # ========================================================
    # Exact train/validation overlap
    # ========================================================

    overlap = (
        set(
            train_hashes
        )
        &
        set(
            validation_hashes
        )
    )

    # ========================================================
    # Internal duplicate checks
    # ========================================================

    train_duplicates = (
        internal_duplicates(
            train_hashes
        )
    )

    validation_duplicates = (
        internal_duplicates(
            validation_hashes
        )
    )

    # ========================================================
    # Accounting
    # ========================================================

    expected_train_documents = (
        55_814
    )

    expected_validation_documents = (
        6_242
    )

    train_count_passed = (
        train_documents
        ==
        expected_train_documents
    )

    validation_count_passed = (
        validation_documents
        ==
        expected_validation_documents
    )

    no_train_duplicates = (
        len(
            train_duplicates
        )
        ==
        0
    )

    no_validation_duplicates = (
        len(
            validation_duplicates
        )
        ==
        0
    )

    no_train_validation_overlap = (
        len(
            overlap
        )
        ==
        0
    )

    # ========================================================
    # Report
    # ========================================================

    print()

    print(
        f"Train documents: "
        f"{train_documents:,}"
    )

    print(
        f"Validation documents: "
        f"{validation_documents:,}"
    )

    print()

    print(
        f"Unique train hashes: "
        f"{len(train_hashes):,}"
    )

    print(
        f"Unique validation hashes: "
        f"{len(validation_hashes):,}"
    )

    print()

    print(
        f"Duplicate hashes inside train: "
        f"{len(train_duplicates):,}"
    )

    print(
        f"Duplicate hashes inside validation: "
        f"{len(validation_duplicates):,}"
    )

    print(
        f"Exact train/validation overlaps: "
        f"{len(overlap):,}"
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    if overlap:

        print()
        print(
            "Example train/validation overlaps:"
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
                f"Train:"
            )

            print(
                train_hashes[
                    digest
                ]
            )

            print(
                f"Validation:"
            )

            print(
                validation_hashes[
                    digest
                ]
            )

    if train_duplicates:

        print()
        print(
            "Example train duplicates:"
        )

        for (
            digest,
            locations,
        ) in list(
            train_duplicates.items()
        )[:10]:

            print(
                digest,
                locations,
            )

    if validation_duplicates:

        print()
        print(
            "Example validation duplicates:"
        )

        for (
            digest,
            locations,
        ) in list(
            validation_duplicates.items()
        )[:10]:

            print(
                digest,
                locations,
            )

    # ========================================================
    # Final checks
    # ========================================================

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Train count = 55,814: "
        f"{train_count_passed}"
    )

    print(
        f"Validation count = 6,242: "
        f"{validation_count_passed}"
    )

    print(
        f"No duplicate documents in train: "
        f"{no_train_duplicates}"
    )

    print(
        f"No duplicate documents in validation: "
        f"{no_validation_duplicates}"
    )

    print(
        f"No train/validation leakage: "
        f"{no_train_validation_overlap}"
    )

    all_passed = (
        train_count_passed
        and
        validation_count_passed
        and
        no_train_duplicates
        and
        no_validation_duplicates
        and
        no_train_validation_overlap
    )

    print()
    print(
        "=" * 72
    )

    if all_passed:

        print(
            "SUCCESS: GENERAL-v0.2 HAS NO "
            "EXACT TRAIN/VALIDATION LEAKAGE."
        )

    else:

        raise AssertionError(
            "General-v0.2 leakage audit failed."
        )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()