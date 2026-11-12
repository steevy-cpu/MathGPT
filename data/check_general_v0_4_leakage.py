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


# ============================================================
# Paths
# ============================================================

GENERAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general_v0.4_sized"
)


SIZING_REPORT_PATH = (
    GENERAL_DIR
    / "general_pretraining_sizing_report.json"
)


OUTPUT_REPORT = (
    GENERAL_DIR
    / "general_pretraining_leakage_report.json"
)


SOURCE_NAMES = [
    "fineweb_edu",
    "wikipedia",
    "gutenberg",
]


TRAIN_FILES = [
    (
        source,
        GENERAL_DIR
        /
        f"{source}_train.jsonl"
    )

    for source in SOURCE_NAMES
]


VALIDATION_FILES = [
    (
        source,
        GENERAL_DIR
        /
        f"{source}_validation.jsonl"
    )

    for source in SOURCE_NAMES
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


def load_json(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Collect one split
# ============================================================

def collect_split(
    files,
):

    hashes = set()

    source_counts = {}

    documents = 0

    duplicate_occurrences = 0

    duplicate_examples = []

    for (
        source_name,
        path,
    ) in files:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        source_documents = 0

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                if not line.strip():

                    continue

                record = json.loads(
                    line
                )

                text = record.get(
                    "text",
                    "",
                )

                if not isinstance(
                    text,
                    str,
                ):

                    raise RuntimeError(
                        f"Non-string text in "
                        f"{path.name}."
                    )

                if not text:

                    raise RuntimeError(
                        f"Empty text in "
                        f"{path.name}."
                    )

                digest = hash_text(
                    text
                )

                if digest in hashes:

                    duplicate_occurrences += 1

                    if (
                        len(
                            duplicate_examples
                        )
                        <
                        10
                    ):

                        duplicate_examples.append(
                            {
                                "hash":
                                    digest,

                                "file":
                                    path.name,

                                "line":
                                    line_number,
                            }
                        )

                else:

                    hashes.add(
                        digest
                    )

                documents += 1

                source_documents += 1

        source_counts[
            source_name
        ] = source_documents

    return {
        "hashes":
            hashes,

        "documents":
            documents,

        "unique_hashes":
            len(
                hashes
            ),

        "duplicate_occurrences":
            duplicate_occurrences,

        "duplicate_examples":
            duplicate_examples,

        "source_counts":
            source_counts,
    }


# ============================================================
# Main
# ============================================================

def main():

    sizing_report = load_json(
        SIZING_REPORT_PATH
    )

    print()
    print(
        "=" * 78
    )

    print(
        "STAGE 15E.2 — GENERAL-v0.4 "
        "EXACT LEAKAGE AUDIT"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Hashing training documents..."
    )

    train = collect_split(
        TRAIN_FILES
    )

    print()
    print(
        "Hashing validation documents..."
    )

    validation = collect_split(
        VALIDATION_FILES
    )

    overlap = (
        train[
            "hashes"
        ]
        &
        validation[
            "hashes"
        ]
    )

    expected_train_documents = int(
        sizing_report[
            "overall"
        ][
            "train_documents"
        ]
    )

    expected_validation_documents = int(
        sizing_report[
            "overall"
        ][
            "validation_documents"
        ]
    )

    source_count_checks = {}

    for source_name in SOURCE_NAMES:

        expected_train = int(
            sizing_report[
                "sources"
            ][
                source_name
            ][
                "train"
            ][
                "selected_documents"
            ]
        )

        expected_validation = int(
            sizing_report[
                "sources"
            ][
                source_name
            ][
                "validation"
            ][
                "selected_documents"
            ]
        )

        source_count_checks[
            source_name
        ] = {
            "train_expected":
                expected_train,

            "train_actual":
                train[
                    "source_counts"
                ][
                    source_name
                ],

            "train_matches":
                (
                    train[
                        "source_counts"
                    ][
                        source_name
                    ]
                    ==
                    expected_train
                ),

            "validation_expected":
                expected_validation,

            "validation_actual":
                validation[
                    "source_counts"
                ][
                    source_name
                ],

            "validation_matches":
                (
                    validation[
                        "source_counts"
                    ][
                        source_name
                    ]
                    ==
                    expected_validation
                ),
        }

    checks = {
        "train_document_count":
            (
                train[
                    "documents"
                ]
                ==
                expected_train_documents
            ),

        "validation_document_count":
            (
                validation[
                    "documents"
                ]
                ==
                expected_validation_documents
            ),

        "no_train_duplicates":
            (
                train[
                    "duplicate_occurrences"
                ]
                ==
                0
            ),

        "no_validation_duplicates":
            (
                validation[
                    "duplicate_occurrences"
                ]
                ==
                0
            ),

        "no_train_validation_overlap":
            (
                len(
                    overlap
                )
                ==
                0
            ),

        "train_unique_hash_accounting":
            (
                train[
                    "unique_hashes"
                ]
                ==
                train[
                    "documents"
                ]
            ),

        "validation_unique_hash_accounting":
            (
                validation[
                    "unique_hashes"
                ]
                ==
                validation[
                    "documents"
                ]
            ),

        "source_document_counts":
            all(
                result[
                    "train_matches"
                ]
                and
                result[
                    "validation_matches"
                ]

                for result in
                source_count_checks.values()
            ),
    }

    print()
    print(
        "Corpus"
    )

    print(
        "------"
    )

    print(
        f"Train documents: "
        f"{train['documents']:,}"
    )

    print(
        f"Unique train hashes: "
        f"{train['unique_hashes']:,}"
    )

    print(
        f"Train duplicates: "
        f"{train['duplicate_occurrences']:,}"
    )

    print()

    print(
        f"Validation documents: "
        f"{validation['documents']:,}"
    )

    print(
        f"Unique validation hashes: "
        f"{validation['unique_hashes']:,}"
    )

    print(
        f"Validation duplicates: "
        f"{validation['duplicate_occurrences']:,}"
    )

    print()

    print(
        f"Exact train/validation overlap: "
        f"{len(overlap):,}"
    )

    print()

    print(
        "Source document accounting"
    )

    print(
        "--------------------------"
    )

    for (
        source_name,
        result,
    ) in source_count_checks.items():

        print(
            f"{source_name:<16} "
            f"train="
            f"{result['train_actual']:,}/"
            f"{result['train_expected']:,} "
            f"validation="
            f"{result['validation_actual']:,}/"
            f"{result['validation_expected']:,}"
        )

    print()

    print(
        "Checks"
    )

    print(
        "------"
    )

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name:<40} "
            f"{passed}"
        )

    if overlap:

        print()
        print(
            "Example leakage hashes:"
        )

        for digest in list(
            overlap
        )[:10]:

            print(
                digest
            )

    if (
        train[
            "duplicate_examples"
        ]
    ):

        print()
        print(
            "Example train duplicates:"
        )

        for example in (
            train[
                "duplicate_examples"
            ]
        ):

            print(
                example
            )

    if (
        validation[
            "duplicate_examples"
        ]
    ):

        print()
        print(
            "Example validation duplicates:"
        )

        for example in (
            validation[
                "duplicate_examples"
            ]
        ):

            print(
                example
            )

    all_passed = all(
        checks.values()
    )

    report = {
        "stage":
            "15E.2",

        "dataset_version":
            "general_pretraining_v0.4",

        "train": {
            key:
                value

            for key, value in
            train.items()

            if key
            !=
            "hashes"
        },

        "validation": {
            key:
                value

            for key, value in
            validation.items()

            if key
            !=
            "hashes"
        },

        "exact_train_validation_overlap":
            len(
                overlap
            ),

        "source_count_checks":
            source_count_checks,

        "checks":
            checks,

        "all_passed":
            all_passed,
    }

    temporary = OUTPUT_REPORT.with_name(
        OUTPUT_REPORT.name
        +
        ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary.replace(
        OUTPUT_REPORT
    )

    print()
    print(
        "Report:"
    )

    print(
        OUTPUT_REPORT
    )

    print()
    print(
        "=" * 78
    )

    if all_passed:

        print(
            "SUCCESS: GENERAL-v0.4 HAS "
            "ZERO EXACT TRAIN/VALIDATION "
            "LEAKAGE OR DUPLICATION."
        )

    else:

        raise RuntimeError(
            "GENERAL-v0.4 exact leakage "
            "audit failed."
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()