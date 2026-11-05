import argparse
import hashlib
import json
import random
import sys

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

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports
# ============================================================

from tokenizer.build_tokenizer_corpus_50m import (
    clean_text,
)


# ============================================================
# Paths
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "general_data_v0.2.json"
)


# ============================================================
# Helpers
# ============================================================

def load_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def text_hash(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def character_count(
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
# Atomic JSONL writer
# ============================================================

def write_jsonl_atomic(
    documents,
    output_path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_name(
            output_path.name
            +
            ".tmp"
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


# ============================================================
# Build cleaned documents
# ============================================================

def build_documents(
    *,
    source_name,
    source_path,
    global_hashes,
):
    """
    Read one raw source.

    Steps:

        raw JSONL
            ↓
        clean_text()
            ↓
        exact SHA-256 hash
            ↓
        local + cross-source exact dedup
            ↓
        retained documents
    """

    print()
    print(
        "=" * 72
    )

    print(
        f"PREPARING SOURCE: "
        f"{source_name}"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Input:"
    )

    print(
        source_path
    )

    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )

    documents = []

    local_hashes = set()

    records_read = 0

    raw_characters = 0

    clean_characters = 0

    empty_after_cleaning = 0

    local_duplicates = 0

    cross_source_duplicates = 0

    malformed_json = 0

    non_string_text = 0

    replacement_characters = 0

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

                malformed_json += 1

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

                non_string_text += 1

                continue

            raw_characters += len(
                text
            )

            replacement_characters += (
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

            # ------------------------------------------------
            # Duplicate within same source
            # ------------------------------------------------

            if digest in local_hashes:

                local_duplicates += 1

                continue

            # ------------------------------------------------
            # Duplicate already retained from another source
            # ------------------------------------------------

            if digest in global_hashes:

                cross_source_duplicates += 1

                continue

            local_hashes.add(
                digest
            )

            global_hashes.add(
                digest
            )

            output_record = dict(
                record
            )

            output_record[
                "text"
            ] = cleaned

            output_record[
                "_mathgpt_source"
            ] = source_name

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

                    "hash":
                        digest,
                }
            )

            clean_characters += len(
                cleaned
            )

    report = {
        "records_read":
            records_read,

        "documents_retained":
            len(
                documents
            ),

        "raw_characters":
            raw_characters,

        "clean_characters":
            clean_characters,

        "empty_after_cleaning":
            empty_after_cleaning,

        "local_duplicates_removed":
            local_duplicates,

        "cross_source_duplicates_removed":
            cross_source_duplicates,

        "malformed_json":
            malformed_json,

        "non_string_text":
            non_string_text,

        "replacement_characters_upstream":
            replacement_characters,
    }

    print()

    print(
        f"Records read: "
        f"{records_read:,}"
    )

    print(
        f"Documents retained: "
        f"{len(documents):,}"
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

    print()

    print(
        f"Local duplicates removed: "
        f"{local_duplicates:,}"
    )

    print(
        f"Cross-source duplicates removed: "
        f"{cross_source_duplicates:,}"
    )

    print(
        f"Empty after cleaning: "
        f"{empty_after_cleaning:,}"
    )

    print(
        f"Malformed JSON: "
        f"{malformed_json:,}"
    )

    if len(
        documents
    ) < 2:

        raise RuntimeError(
            f"{source_name} has fewer than "
            "two usable documents."
        )

    return (
        documents,
        report,
    )


# ============================================================
# Character-balanced split
# ============================================================

def split_documents(
    *,
    documents,
    validation_ratio,
    seed,
):
    """
    Whole-document deterministic split.

    We shuffle document references using a fixed
    seed and select validation documents until
    approximately validation_ratio of characters
    are represented.

    Documents themselves are never cut.
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

    total_characters = (
        character_count(
            candidates
        )
    )

    target_validation_characters = round(
        total_characters
        *
        validation_ratio
    )

    validation = []

    validation_characters = 0

    for document in candidates:

        if (
            validation_characters
            >=
            target_validation_characters
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
    # Check if dropping the final selected document
    # gets us closer to the target.
    # ========================================================

    if (
        len(
            validation
        )
        >
        1
    ):

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

        if (
            without_last_distance
            <
            with_last_distance
        ):

            validation.pop()

            validation_characters = (
                without_last_characters
            )

    validation_hashes = {
        document[
            "hash"
        ]

        for document
        in validation
    }

    train = [
        document

        for document
        in documents

        if document[
            "hash"
        ]
        not in validation_hashes
    ]

    if not train:

        raise RuntimeError(
            "Validation split consumed "
            "all documents."
        )

    # Restore source order for deterministic writing.
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

    train_characters = (
        character_count(
            train
        )
    )

    validation_characters = (
        character_count(
            validation
        )
    )

    actual_validation_ratio = (
        validation_characters
        /
        total_characters
    )

    return (
        train,
        validation,
        {
            "total_characters":
                total_characters,

            "target_validation_characters":
                target_validation_characters,

            "train_characters":
                train_characters,

            "validation_characters":
                validation_characters,

            "actual_validation_ratio":
                actual_validation_ratio,
        },
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare General-v0.2 "
            "train/validation corpus."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing General-v0.2 "
            "cleaned split files."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    config = load_json(
        CONFIG_PATH
    )

    seed = int(
        config[
            "reproducibility"
        ][
            "seed"
        ]
    )

    validation_ratio = float(
        config[
            "objective"
        ][
            "validation_ratio"
        ]
    )

    raw_dir = (
        PROJECT_ROOT
        /
        config[
            "paths"
        ][
            "raw_directory"
        ]
    )

    cleaned_dir = (
        PROJECT_ROOT
        /
        config[
            "paths"
        ][
            "cleaned_directory"
        ]
    )

    cleaned_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Source map
    # ========================================================

    source_names = [
        "fineweb_edu",
        "wikipedia",
        "gutenberg",
    ]

    output_files = []

    for source_name in source_names:

        output_files.extend(
            [
                (
                    cleaned_dir
                    /
                    f"{source_name}_train.jsonl"
                ),
                (
                    cleaned_dir
                    /
                    f"{source_name}_validation.jsonl"
                ),
            ]
        )

    report_path = (
        cleaned_dir
        /
        "general_pretraining_split_report.json"
    )

    output_files.append(
        report_path
    )

    # ========================================================
    # Protect completed output
    # ========================================================

    existing_outputs = [
        path

        for path
        in output_files

        if path.exists()
    ]

    if (
        existing_outputs
        and
        not args.overwrite
    ):

        print()
        print(
            "General-v0.2 prepared outputs "
            "already exist:"
        )

        for path in existing_outputs:

            print(
                f"  {path}"
            )

        print()
        print(
            "Use --overwrite only if you "
            "intend to rebuild them."
        )

        raise SystemExit(
            0
        )

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 12A.3/12A.4 — "
        "GENERAL-v0.2 DATA PREPARATION"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Seed: "
        f"{seed}"
    )

    print(
        f"Validation target: "
        f"{validation_ratio:.2%}"
    )

    print(
        f"Tokenizer remains: "
        f"{config['objective']['tokenizer']}"
    )

    print(
        f"Target training tokens: "
        f"{config['objective']['target_training_tokens']:,}"
    )

    # ========================================================
    # Global dedup set
    # ========================================================

    global_hashes = set()

    report = {
        "version":
            "general_pretraining_v0.2",

        "experiment":
            config[
                "experiment"
            ],

        "seed":
            seed,

        "validation_ratio_target":
            validation_ratio,

        "tokenizer":
            config[
                "objective"
            ][
                "tokenizer"
            ],

        "target_training_tokens":
            config[
                "objective"
            ][
                "target_training_tokens"
            ],

        "sources":
            {},
    }

    total_documents = 0

    total_train_documents = 0

    total_validation_documents = 0

    total_clean_characters = 0

    total_train_characters = 0

    total_validation_characters = 0

    # ========================================================
    # Process each source
    # ========================================================

    for source_offset, source_name in enumerate(
        source_names
    ):

        source_config = (
            config[
                "sources"
            ][
                source_name
            ]
        )

        source_path = (
            raw_dir
            /
            source_config[
                "output_file"
            ]
        )

        (
            documents,
            cleaning_report,
        ) = build_documents(
            source_name=source_name,
            source_path=source_path,
            global_hashes=global_hashes,
        )

        (
            train_documents,
            validation_documents,
            split_report,
        ) = split_documents(
            documents=documents,
            validation_ratio=validation_ratio,
            seed=(
                seed
                +
                source_offset
            ),
        )

        train_path = (
            cleaned_dir
            /
            f"{source_name}_train.jsonl"
        )

        validation_path = (
            cleaned_dir
            /
            f"{source_name}_validation.jsonl"
        )

        write_jsonl_atomic(
            train_documents,
            train_path,
        )

        write_jsonl_atomic(
            validation_documents,
            validation_path,
        )

        source_total_documents = len(
            documents
        )

        source_train_documents = len(
            train_documents
        )

        source_validation_documents = len(
            validation_documents
        )

        source_clean_characters = (
            split_report[
                "total_characters"
            ]
        )

        source_train_characters = (
            split_report[
                "train_characters"
            ]
        )

        source_validation_characters = (
            split_report[
                "validation_characters"
            ]
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
            f"{source_total_documents:,}"
        )

        print(
            f"Train documents: "
            f"{source_train_documents:,}"
        )

        print(
            f"Validation documents: "
            f"{source_validation_documents:,}"
        )

        print()

        print(
            f"Clean characters: "
            f"{source_clean_characters:,}"
        )

        print(
            f"Train characters: "
            f"{source_train_characters:,}"
        )

        print(
            f"Validation characters: "
            f"{source_validation_characters:,}"
        )

        print(
            f"Validation ratio: "
            f"{split_report['actual_validation_ratio']:.2%}"
        )

        report[
            "sources"
        ][
            source_name
        ] = {
            "raw_path":
                str(
                    source_path
                ),

            "train_path":
                str(
                    train_path
                ),

            "validation_path":
                str(
                    validation_path
                ),

            "cleaning":
                cleaning_report,

            "split": {
                "documents":
                    source_total_documents,

                "train_documents":
                    source_train_documents,

                "validation_documents":
                    source_validation_documents,

                **split_report,
            },
        }

        total_documents += (
            source_total_documents
        )

        total_train_documents += (
            source_train_documents
        )

        total_validation_documents += (
            source_validation_documents
        )

        total_clean_characters += (
            source_clean_characters
        )

        total_train_characters += (
            source_train_characters
        )

        total_validation_characters += (
            source_validation_characters
        )

    # ========================================================
    # Overall safety checks
    # ========================================================

    overall_validation_ratio = (
        total_validation_characters
        /
        total_clean_characters
    )

    minimum_clean_characters = (
        240_000_000
    )

    enough_clean_text = (
        total_clean_characters
        >=
        minimum_clean_characters
    )

    document_accounting_passed = (
        total_documents
        ==
        total_train_documents
        +
        total_validation_documents
    )

    character_accounting_passed = (
        total_clean_characters
        ==
        total_train_characters
        +
        total_validation_characters
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

        "validation_ratio":
            overall_validation_ratio,

        "unique_document_hashes":
            len(
                global_hashes
            ),

        "minimum_clean_characters":
            minimum_clean_characters,

        "enough_clean_text":
            enough_clean_text,

        "document_accounting_passed":
            document_accounting_passed,

        "character_accounting_passed":
            character_accounting_passed,
    }

    # ========================================================
    # Save report atomically
    # ========================================================

    temporary_report = (
        report_path.with_name(
            report_path.name
            +
            ".tmp"
        )
    )

    with open(
        temporary_report,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    temporary_report.replace(
        report_path
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "GENERAL-v0.2 PREPARED CORPUS"
    )

    print(
        "=" * 72
    )

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
        f"Unique document hashes: "
        f"{len(global_hashes):,}"
    )

    print()

    print(
        f"At least "
        f"{minimum_clean_characters:,} "
        f"clean characters: "
        f"{enough_clean_text}"
    )

    print(
        f"Document accounting passed: "
        f"{document_accounting_passed}"
    )

    print(
        f"Character accounting passed: "
        f"{character_accounting_passed}"
    )

    print()

    print(
        f"Report:"
    )

    print(
        report_path
    )

    all_passed = (
        enough_clean_text
        and
        document_accounting_passed
        and
        character_accounting_passed
    )

    print()
    print(
        "=" * 72
    )

    if all_passed:

        print(
            "SUCCESS: GENERAL-v0.2 "
            "CLEANED AND SPLIT."
        )

    else:

        raise RuntimeError(
            "General-v0.2 preparation "
            "failed safety checks."
        )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()