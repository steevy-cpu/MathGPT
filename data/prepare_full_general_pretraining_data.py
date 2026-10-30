import hashlib
import json
import random
import sys
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from tokenizer.build_tokenizer_corpus_50m import (
    clean_text,
)


RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "general"
)


CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general"
)


# ============================================================
# Final general-pretraining sources
# ============================================================

SOURCES = {
    "fineweb_edu": (
        RAW_DIR
        / "fineweb_edu_50m_source.jsonl"
    ),

    "wikipedia": (
        RAW_DIR
        / "wikipedia_50m_source.jsonl"
    ),

    "gutenberg": (
        RAW_DIR
        / "gutenberg_50m_source.jsonl"
    ),
}


VALIDATION_RATIO = 0.10

SEED = 42


# Safety check:
#
# A final general corpus should be many millions
# of characters, not the ~1.38M-character sample
# we accidentally used earlier.
MINIMUM_TOTAL_CLEAN_CHARACTERS = (
    30_000_000
)


# ============================================================
# Exact text hash
# ============================================================

def text_hash(
    text,
):
    """
    Exact-document hash after cleaning.

    If two cleaned documents are identical,
    only the first one is retained.
    """

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Load, clean, exact-deduplicate
# ============================================================

def build_documents(
    source_name,
    source_path,
):
    """
    Read raw JSONL.

    Apply the SAME conservative cleaner used
    for tokenizer_50m_v0.2.

    Then exact-deduplicate cleaned documents.
    """

    print()
    print("=" * 65)

    print(
        f"Preparing source: "
        f"{source_name}"
    )

    print("=" * 65)

    print(
        f"Input: "
        f"{source_path}"
    )

    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )

    documents = []

    seen_hashes = set()

    records_read = 0

    empty_after_cleaning = 0

    duplicates_removed = 0

    raw_characters = 0

    clean_characters = 0

    replacement_characters_upstream = 0

    with open(
        source_path,
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

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue

            records_read += 1

            text = record.get(
                "text",
                "",
            )

            if not isinstance(
                text,
                str,
            ):

                continue

            raw_characters += len(
                text
            )

            replacement_characters_upstream += (
                text.count(
                    "\ufffd"
                )
            )

            cleaned = clean_text(
                text
            )

            if not cleaned:

                empty_after_cleaning += 1

                continue

            digest = text_hash(
                cleaned
            )

            if digest in seen_hashes:

                duplicates_removed += 1

                continue

            seen_hashes.add(
                digest
            )

            clean_characters += len(
                cleaned
            )

            # Keep original metadata, but replace
            # text with the final cleaned version.
            output_record = dict(
                record
            )

            output_record[
                "text"
            ] = cleaned

            documents.append(
                {
                    "record":
                        output_record,

                    "characters":
                        len(
                            cleaned
                        ),

                    "original_index":
                        line_number,
                }
            )

    print()

    print(
        f"Records read: "
        f"{records_read:,}"
    )

    print(
        f"Documents retained: "
        f"{len(documents):,}"
    )

    print(
        f"Exact duplicates removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Empty after cleaning: "
        f"{empty_after_cleaning:,}"
    )

    print()

    print(
        f"Raw characters: "
        f"{raw_characters:,}"
    )

    print(
        f"Clean characters: "
        f"{clean_characters:,}"
    )

    print(
        "Replacement characters upstream: "
        f"{replacement_characters_upstream:,}"
    )

    if len(
        documents
    ) < 2:

        raise RuntimeError(
            f"{source_name} does not have "
            "enough usable documents."
        )

    return (
        documents,
        {
            "records_read":
                records_read,

            "documents_retained":
                len(
                    documents
                ),

            "duplicates_removed":
                duplicates_removed,

            "empty_after_cleaning":
                empty_after_cleaning,

            "raw_characters":
                raw_characters,

            "clean_characters":
                clean_characters,

            "replacement_characters_upstream":
                replacement_characters_upstream,
        },
    )


# ============================================================
# Deterministic character-balanced split
# ============================================================

def split_documents(
    documents,
    validation_ratio,
    seed,
):
    """
    Split whole documents.

    Documents are never cut between train
    and validation.

    A deterministic shuffle prevents source-order
    bias.

    We greedily approach the target validation
    character count.
    """

    rng = random.Random(
        seed
    )

    candidates = list(
        documents
    )

    rng.shuffle(
        candidates
    )

    total_characters = sum(
        document[
            "characters"
        ]
        for document
        in candidates
    )

    target_validation_characters = round(
        total_characters
        * validation_ratio
    )

    validation = []

    validation_characters = 0

    # ========================================================
    # Fill until target is reached/passed.
    # ========================================================

    for document in candidates:

        if (
            validation_characters
            >= target_validation_characters
        ):
            break

        validation.append(
            document
        )

        validation_characters += (
            document[
                "characters"
            ]
        )

    # ========================================================
    # Check whether removing the last document
    # would be closer to the target.
    # ========================================================

    if validation:

        last_document = (
            validation[
                -1
            ]
        )

        with_last_distance = abs(
            validation_characters
            -
            target_validation_characters
        )

        without_last_characters = (
            validation_characters
            -
            last_document[
                "characters"
            ]
        )

        without_last_distance = abs(
            without_last_characters
            -
            target_validation_characters
        )

        # Keep at least one validation document.
        if (
            len(
                validation
            ) > 1
            and
            without_last_distance
            <
            with_last_distance
        ):

            validation.pop()

            validation_characters = (
                without_last_characters
            )

    validation_ids = {
        document[
            "original_index"
        ]
        for document
        in validation
    }

    train = [
        document

        for document
        in documents

        if document[
            "original_index"
        ]
        not in validation_ids
    ]

    if not train:

        raise RuntimeError(
            "Validation split consumed "
            "the entire source."
        )

    train.sort(
        key=lambda document:
            document[
                "original_index"
            ]
    )

    validation.sort(
        key=lambda document:
            document[
                "original_index"
            ]
    )

    return (
        train,
        validation,
        target_validation_characters,
    )


# ============================================================
# JSONL writing
# ============================================================

def write_jsonl_atomic(
    documents,
    output_path,
):
    """
    Write through a temporary file so an
    interrupted run does not replace a
    known-good split.
    """

    temporary_path = (
        output_path.with_name(
            output_path.name
            + ".tmp"
        )
    )

    if temporary_path.exists():

        temporary_path.unlink()

    try:

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:

            for document in documents:

                json.dump(
                    document[
                        "record"
                    ],
                    file,
                    ensure_ascii=False,
                )

                file.write(
                    "\n"
                )

        temporary_path.replace(
            output_path
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink()


def document_characters(
    documents,
):
    return sum(
        document[
            "characters"
        ]
        for document
        in documents
    )


# ============================================================
# Main
# ============================================================

def main():

    CLEANED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 65)

    print(
        "FULL GENERAL PRETRAINING DATA PREPARATION"
    )

    print("=" * 65)

    print()

    print(
        f"Validation target: "
        f"{VALIDATION_RATIO:.0%}"
    )

    print(
        f"Seed: "
        f"{SEED}"
    )

    report = {
        "version":
            "general_pretraining_v0.2",

        "seed":
            SEED,

        "target_validation_ratio":
            VALIDATION_RATIO,

        "sources":
            {},
    }

    total_documents = 0

    total_train_documents = 0

    total_validation_documents = 0

    total_clean_characters = 0

    total_train_characters = 0

    total_validation_characters = 0

    for source_offset, (
        source_name,
        source_path,
    ) in enumerate(
        SOURCES.items()
    ):

        (
            documents,
            cleaning_report,
        ) = build_documents(
            source_name,
            source_path,
        )

        (
            train_documents,
            validation_documents,
            target_validation_characters,
        ) = split_documents(
            documents=documents,
            validation_ratio=(
                VALIDATION_RATIO
            ),
            seed=(
                SEED
                +
                source_offset
            ),
        )

        train_characters = (
            document_characters(
                train_documents
            )
        )

        validation_characters = (
            document_characters(
                validation_documents
            )
        )

        clean_characters = (
            train_characters
            +
            validation_characters
        )

        actual_validation_ratio = (
            validation_characters
            /
            clean_characters
        )

        train_path = (
            CLEANED_DIR
            /
            f"{source_name}_train.jsonl"
        )

        validation_path = (
            CLEANED_DIR
            /
            f"{source_name}_validation.jsonl"
        )

        # ====================================================
        # Leakage safety
        # ====================================================

        train_hashes = {
            text_hash(
                document[
                    "record"
                ][
                    "text"
                ]
            )
            for document
            in train_documents
        }

        validation_hashes = {
            text_hash(
                document[
                    "record"
                ][
                    "text"
                ]
            )
            for document
            in validation_documents
        }

        no_leakage = (
            train_hashes.isdisjoint(
                validation_hashes
            )
        )

        if not no_leakage:

            raise RuntimeError(
                f"Train/validation leakage "
                f"in {source_name}."
            )

        write_jsonl_atomic(
            train_documents,
            train_path,
        )

        write_jsonl_atomic(
            validation_documents,
            validation_path,
        )

        print()

        print(
            f"{source_name} split"
        )

        print(
            "-" * 40
        )

        print(
            f"Documents: "
            f"{len(documents):,}"
        )

        print(
            f"Train documents: "
            f"{len(train_documents):,}"
        )

        print(
            f"Validation documents: "
            f"{len(validation_documents):,}"
        )

        print()

        print(
            f"Clean characters: "
            f"{clean_characters:,}"
        )

        print(
            f"Train characters: "
            f"{train_characters:,}"
        )

        print(
            f"Validation characters: "
            f"{validation_characters:,}"
        )

        print(
            f"Validation target: "
            f"{target_validation_characters:,}"
        )

        print(
            f"Validation ratio: "
            f"{actual_validation_ratio:.2%}"
        )

        print(
            f"No exact-text leakage: "
            f"{no_leakage}"
        )

        report[
            "sources"
        ][
            source_name
        ] = {
            **cleaning_report,

            "train_documents":
                len(
                    train_documents
                ),

            "validation_documents":
                len(
                    validation_documents
                ),

            "train_characters":
                train_characters,

            "validation_characters":
                validation_characters,

            "actual_validation_ratio":
                actual_validation_ratio,
        }

        total_documents += len(
            documents
        )

        total_train_documents += len(
            train_documents
        )

        total_validation_documents += len(
            validation_documents
        )

        total_clean_characters += (
            clean_characters
        )

        total_train_characters += (
            train_characters
        )

        total_validation_characters += (
            validation_characters
        )

    # ========================================================
    # Critical corpus-scale safety check
    # ========================================================

    scale_passed = (
        total_clean_characters
        >=
        MINIMUM_TOTAL_CLEAN_CHARACTERS
    )

    overall_validation_ratio = (
        total_validation_characters
        /
        total_clean_characters
    )

    report[
        "overall"
    ] = {
        "documents":
            total_documents,

        "train_documents":
            total_train_documents,

        "validation_documents":
            total_validation_documents,

        "clean_characters":
            total_clean_characters,

        "train_characters":
            total_train_characters,

        "validation_characters":
            total_validation_characters,

        "actual_validation_ratio":
            overall_validation_ratio,

        "minimum_required_characters":
            MINIMUM_TOTAL_CLEAN_CHARACTERS,

        "scale_check_passed":
            scale_passed,
    }

    report_path = (
        CLEANED_DIR
        /
        "general_pretraining_split_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()
    print("=" * 65)

    print(
        "OVERALL GENERAL CORPUS"
    )

    print("=" * 65)

    print()

    print(
        f"Documents: "
        f"{total_documents:,}"
    )

    print(
        f"Train documents: "
        f"{total_train_documents:,}"
    )

    print(
        f"Validation documents: "
        f"{total_validation_documents:,}"
    )

    print()

    print(
        f"Clean characters: "
        f"{total_clean_characters:,}"
    )

    print(
        f"Train characters: "
        f"{total_train_characters:,}"
    )

    print(
        f"Validation characters: "
        f"{total_validation_characters:,}"
    )

    print(
        f"Validation ratio: "
        f"{overall_validation_ratio:.2%}"
    )

    print()

    print(
        f"At least "
        f"{MINIMUM_TOTAL_CLEAN_CHARACTERS:,} "
        f"clean chars: "
        f"{scale_passed}"
    )

    print()

    print(
        f"Report:"
    )

    print(
        report_path
    )

    if not scale_passed:

        raise RuntimeError(
            "SAFETY STOP: final general corpus "
            "is unexpectedly small."
        )

    print()
    print("=" * 65)

    print(
        "SUCCESS: FULL GENERAL PRETRAINING "
        "SPLITS CREATED."
    )

    print("=" * 65)


if __name__ == "__main__":
    main()