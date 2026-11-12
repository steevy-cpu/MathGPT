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
    / "general_data_v0.4.json"
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


def file_sha256(
    path,
):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024
                *
                1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


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
# Raw manifest verification
# ============================================================

def verify_raw_sources(
    *,
    config,
    raw_dir,
):

    manifest_path = (
        raw_dir
        /
        "download_manifest.json"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Download manifest missing: "
            f"{manifest_path}"
        )

    manifest = load_json(
        manifest_path
    )

    if (
        manifest.get(
            "dataset_version"
        )
        !=
        "general_pretraining_v0.4"
    ):

        raise RuntimeError(
            "Raw manifest dataset identity "
            "does not match GENERAL-v0.4."
        )

    source_names = [
        "fineweb_edu",
        "wikipedia",
        "gutenberg",
    ]

    results = {}

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 RAW SOURCE VERIFICATION"
    )

    print(
        "=" * 76
    )

    print()

    for source_name in source_names:

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

        if not source_path.exists():

            raise FileNotFoundError(
                source_path
            )

        manifest_source = (
            manifest[
                "sources"
            ].get(
                source_name
            )
        )

        if manifest_source is None:

            raise RuntimeError(
                f"{source_name} missing "
                "from raw manifest."
            )

        expected_characters = int(
            manifest_source[
                "characters"
            ]
        )

        expected_sha256 = (
            manifest_source[
                "sha256"
            ]
        )

        expected_target = int(
            source_config[
                "target_raw_characters"
            ]
        )

        character_target_matches = (
            expected_characters
            ==
            expected_target
        )

        print(
            f"{source_name}"
        )

        print(
            "-" * 44
        )

        print(
            f"Manifest characters: "
            f"{expected_characters:,}"
        )

        print(
            f"Configured target:   "
            f"{expected_target:,}"
        )

        print(
            f"Character target:    "
            f"{character_target_matches}"
        )

        print(
            "Calculating SHA-256..."
        )

        actual_sha256 = file_sha256(
            source_path
        )

        sha_matches = (
            actual_sha256
            ==
            expected_sha256
        )

        print(
            f"SHA matches:         "
            f"{sha_matches}"
        )

        print()

        if not character_target_matches:

            raise RuntimeError(
                f"{source_name} raw character "
                "target mismatch."
            )

        if not sha_matches:

            raise RuntimeError(
                f"{source_name} raw SHA-256 "
                "does not match manifest."
            )

        results[
            source_name
        ] = {
            "path":
                str(
                    source_path
                ),

            "expected_characters":
                expected_characters,

            "expected_sha256":
                expected_sha256,

            "actual_sha256":
                actual_sha256,

            "sha256_verified":
                sha_matches,
        }

    print(
        "SUCCESS: all raw source "
        "fingerprints verified."
    )

    return (
        manifest_path,
        results,
    )


# ============================================================
# Prepare temporary cleaned source
# ============================================================

def build_cleaned_source(
    *,
    source_name,
    source_path,
    temporary_clean_path,
    global_hashes,
    expected_raw_characters,
):
    """
    Streaming cleaning + exact dedup.

    Text remains on disk.

    Only lightweight metadata is kept in RAM
    for deterministic whole-document splitting.
    """

    print()
    print(
        "=" * 76
    )

    print(
        f"CLEANING SOURCE: "
        f"{source_name}"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Input:"
    )

    print(
        source_path
    )

    print()

    if temporary_clean_path.exists():

        temporary_clean_path.unlink()

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

    try:

        with open(
            source_path,
            "r",
            encoding="utf-8",
        ) as input_file:

            with open(
                temporary_clean_path,
                "w",
                encoding="utf-8",
            ) as output_file:

                for line_number, line in enumerate(
                    input_file,
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

                    # ----------------------------------------
                    # Exact duplicate within source
                    # ----------------------------------------

                    if digest in local_hashes:

                        local_duplicates += 1

                        continue

                    # ----------------------------------------
                    # Exact duplicate retained from an
                    # earlier source
                    # ----------------------------------------

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

                    json.dump(
                        output_record,
                        output_file,
                        ensure_ascii=False,
                    )

                    output_file.write(
                        "\n"
                    )

                    characters = len(
                        cleaned
                    )

                    documents.append(
                        {
                            "hash":
                                digest,

                            "characters":
                                characters,

                            "original_index":
                                len(
                                    documents
                                ),
                        }
                    )

                    clean_characters += (
                        characters
                    )

                    if (
                        records_read
                        %
                        25_000
                        ==
                        0
                    ):

                        print(
                            f"Records: "
                            f"{records_read:,} | "
                            f"Retained: "
                            f"{len(documents):,} | "
                            f"Clean chars: "
                            f"{clean_characters:,}"
                        )

    except Exception:

        if temporary_clean_path.exists():

            temporary_clean_path.unlink()

        raise

    raw_character_accounting = (
        raw_characters
        ==
        expected_raw_characters
    )

    if not raw_character_accounting:

        raise RuntimeError(
            f"{source_name} raw characters "
            f"do not match manifest: "
            f"{raw_characters:,} != "
            f"{expected_raw_characters:,}"
        )

    if len(
        documents
    ) < 2:

        raise RuntimeError(
            f"{source_name} has fewer than "
            "two usable documents."
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

        "character_retention_ratio":
            (
                clean_characters
                /
                raw_characters
            ),

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

        "raw_character_accounting_passed":
            raw_character_accounting,
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

    print(
        f"Character retention: "
        f"{report['character_retention_ratio']:.4%}"
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
    Same split logic used by GENERAL-v0.2:

    1. deterministic shuffle
    2. choose whole validation documents until
       ~10% of characters are represented
    3. never cut a document
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
    # Remove final selected document if doing so is
    # closer to the character target.
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

    validation_characters = sum(
        document[
            "characters"
        ]

        for document
        in validation
    )

    train_documents = (
        len(
            documents
        )
        -
        len(
            validation
        )
    )

    train_characters = (
        total_characters
        -
        validation_characters
    )

    if train_documents <= 0:

        raise RuntimeError(
            "Validation split consumed "
            "all documents."
        )

    actual_validation_ratio = (
        validation_characters
        /
        total_characters
    )

    return {
        "validation_hashes":
            validation_hashes,

        "documents":
            len(
                documents
            ),

        "train_documents":
            train_documents,

        "validation_documents":
            len(
                validation
            ),

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
    }


# ============================================================
# Atomic split writer
# ============================================================

def write_split_files(
    *,
    temporary_clean_path,
    train_path,
    validation_path,
    validation_hashes,
):

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_tmp = train_path.with_name(
        train_path.name
        +
        ".tmp"
    )

    validation_tmp = (
        validation_path.with_name(
            validation_path.name
            +
            ".tmp"
        )
    )

    for path in (
        train_tmp,
        validation_tmp,
    ):

        if path.exists():

            path.unlink()

    train_documents = 0

    validation_documents = 0

    train_characters = 0

    validation_characters = 0

    try:

        with open(
            temporary_clean_path,
            "r",
            encoding="utf-8",
        ) as input_file:

            with open(
                train_tmp,
                "w",
                encoding="utf-8",
            ) as train_file:

                with open(
                    validation_tmp,
                    "w",
                    encoding="utf-8",
                ) as validation_file:

                    for line in input_file:

                        if not line.strip():

                            continue

                        record = json.loads(
                            line
                        )

                        text = record[
                            "text"
                        ]

                        digest = text_hash(
                            text
                        )

                        if digest in validation_hashes:

                            target_file = (
                                validation_file
                            )

                            validation_documents += 1

                            validation_characters += (
                                len(
                                    text
                                )
                            )

                        else:

                            target_file = (
                                train_file
                            )

                            train_documents += 1

                            train_characters += (
                                len(
                                    text
                                )
                            )

                        json.dump(
                            record,
                            target_file,
                            ensure_ascii=False,
                        )

                        target_file.write(
                            "\n"
                        )

        train_tmp.replace(
            train_path
        )

        validation_tmp.replace(
            validation_path
        )

    finally:

        for path in (
            train_tmp,
            validation_tmp,
        ):

            if path.exists():

                path.unlink()

    return {
        "train_documents":
            train_documents,

        "validation_documents":
            validation_documents,

        "train_characters":
            train_characters,

        "validation_characters":
            validation_characters,
    }


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Clean, exact-deduplicate, and split "
            "GENERAL-v0.4."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing GENERAL-v0.4 "
            "cleaned split files."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate inputs and fingerprints "
            "without cleaning."
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

    if (
        config.get(
            "version"
        )
        !=
        "0.4"
    ):

        raise RuntimeError(
            "Expected GENERAL-v0.4 config."
        )

    if (
        config.get(
            "dataset_version"
        )
        !=
        "general_pretraining_v0.4"
    ):

        raise RuntimeError(
            "Unexpected dataset identity."
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

    source_names = [
        "fineweb_edu",
        "wikipedia",
        "gutenberg",
    ]

    # ========================================================
    # Output protection
    # ========================================================

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
        and
        not args.preflight_only
    ):

        print()
        print(
            "GENERAL-v0.4 prepared outputs "
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
    # Verify raw source identities first
    # ========================================================

    (
        manifest_path,
        raw_verification,
    ) = verify_raw_sources(
        config=config,
        raw_dir=raw_dir,
    )

    if args.preflight_only:

        print()
        print(
            "=" * 76
        )

        print(
            "SUCCESS: GENERAL-v0.4 "
            "PREPARATION PREFLIGHT PASSED."
        )

        print(
            "=" * 76
        )

        return

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE 15C — GENERAL-v0.4 "
        "CLEAN / DEDUP / SPLIT"
    )

    print(
        "=" * 76
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

    print()

    # ========================================================
    # Global exact dedup set
    # ========================================================

    global_hashes = set()

    report = {
        "version":
            "general_pretraining_v0.4",

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

        "target_validation_tokens":
            config[
                "objective"
            ][
                "target_validation_tokens"
            ],

        "target_total_tokens":
            config[
                "objective"
            ][
                "target_total_tokens"
            ],

        "raw_manifest":
            str(
                manifest_path
            ),

        "raw_verification":
            raw_verification,

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
    # Process sources in frozen order
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

        temporary_clean_path = (
            cleaned_dir
            /
            (
                f".{source_name}"
                f"_general_v0.4_clean.tmp.jsonl"
            )
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

        (
            documents,
            cleaning_report,
        ) = build_cleaned_source(
            source_name=source_name,
            source_path=source_path,
            temporary_clean_path=temporary_clean_path,
            global_hashes=global_hashes,
            expected_raw_characters=(
                raw_verification[
                    source_name
                ][
                    "expected_characters"
                ]
            ),
        )

        split_report = split_documents(
            documents=documents,
            validation_ratio=validation_ratio,
            seed=(
                seed
                +
                source_offset
            ),
        )

        write_report = write_split_files(
            temporary_clean_path=temporary_clean_path,
            train_path=train_path,
            validation_path=validation_path,
            validation_hashes=(
                split_report[
                    "validation_hashes"
                ]
            ),
        )

        temporary_clean_path.unlink(
            missing_ok=True
        )

        # validation_hashes is useful during writing,
        # but should not be dumped into the JSON report.
        split_report.pop(
            "validation_hashes"
        )

        # ====================================================
        # Cross-check the streamed writer
        # ====================================================

        writer_accounting_passed = (
            write_report[
                "train_documents"
            ]
            ==
            split_report[
                "train_documents"
            ]
            and
            write_report[
                "validation_documents"
            ]
            ==
            split_report[
                "validation_documents"
            ]
            and
            write_report[
                "train_characters"
            ]
            ==
            split_report[
                "train_characters"
            ]
            and
            write_report[
                "validation_characters"
            ]
            ==
            split_report[
                "validation_characters"
            ]
        )

        if not writer_accounting_passed:

            raise RuntimeError(
                f"{source_name} split writer "
                "accounting mismatch."
            )

        print()
        print(
            f"{source_name} split"
        )

        print(
            "-" * 44
        )

        print(
            f"Documents: "
            f"{split_report['documents']:,}"
        )

        print(
            f"Train documents: "
            f"{split_report['train_documents']:,}"
        )

        print(
            f"Validation documents: "
            f"{split_report['validation_documents']:,}"
        )

        print()

        print(
            f"Clean characters: "
            f"{split_report['total_characters']:,}"
        )

        print(
            f"Train characters: "
            f"{split_report['train_characters']:,}"
        )

        print(
            f"Validation characters: "
            f"{split_report['validation_characters']:,}"
        )

        print(
            f"Validation ratio: "
            f"{split_report['actual_validation_ratio']:.4%}"
        )

        print(
            f"Writer accounting: "
            f"{writer_accounting_passed}"
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

            "split":
                split_report,

            "writer_accounting_passed":
                writer_accounting_passed,
        }

        total_documents += (
            split_report[
                "documents"
            ]
        )

        total_train_documents += (
            split_report[
                "train_documents"
            ]
        )

        total_validation_documents += (
            split_report[
                "validation_documents"
            ]
        )

        total_clean_characters += (
            split_report[
                "total_characters"
            ]
        )

        total_train_characters += (
            split_report[
                "train_characters"
            ]
        )

        total_validation_characters += (
            split_report[
                "validation_characters"
            ]
        )

    # ========================================================
    # Overall checks
    # ========================================================

    overall_validation_ratio = (
        total_validation_characters
        /
        total_clean_characters
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

    validation_ratio_passed = (
        abs(
            overall_validation_ratio
            -
            validation_ratio
        )
        <
        0.001
    )

    # --------------------------------------------------------
    # This is only a pre-tokenization plausibility check.
    # Final token sufficiency is determined in 15D/15E.
    # --------------------------------------------------------

    recommended_clean_characters = (
        1_275_000_000
    )

    hard_minimum_clean_characters = (
        1_200_000_000
    )

    recommended_clean_size_reached = (
        total_clean_characters
        >=
        recommended_clean_characters
    )

    hard_minimum_clean_size_reached = (
        total_clean_characters
        >=
        hard_minimum_clean_characters
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

        "recommended_clean_characters":
            recommended_clean_characters,

        "hard_minimum_clean_characters":
            hard_minimum_clean_characters,

        "recommended_clean_size_reached":
            recommended_clean_size_reached,

        "hard_minimum_clean_size_reached":
            hard_minimum_clean_size_reached,

        "document_accounting_passed":
            document_accounting_passed,

        "character_accounting_passed":
            character_accounting_passed,

        "validation_ratio_passed":
            validation_ratio_passed,
    }

    # ========================================================
    # Save report atomically
    # ========================================================

    temporary_report = report_path.with_name(
        report_path.name
        +
        ".tmp"
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
        "=" * 76
    )

    print(
        "GENERAL-v0.4 PREPARED CORPUS"
    )

    print(
        "=" * 76
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
        f"{overall_validation_ratio:.4%}"
    )

    print()

    print(
        f"Unique document hashes: "
        f"{len(global_hashes):,}"
    )

    print()

    print(
        f"Recommended clean size "
        f"({recommended_clean_characters:,}): "
        f"{recommended_clean_size_reached}"
    )

    print(
        f"Hard minimum clean size "
        f"({hard_minimum_clean_characters:,}): "
        f"{hard_minimum_clean_size_reached}"
    )

    print(
        f"Document accounting: "
        f"{document_accounting_passed}"
    )

    print(
        f"Character accounting: "
        f"{character_accounting_passed}"
    )

    print(
        f"Validation ratio check: "
        f"{validation_ratio_passed}"
    )

    print()

    print(
        "Report:"
    )

    print(
        report_path
    )

    all_passed = (
        hard_minimum_clean_size_reached
        and
        document_accounting_passed
        and
        character_accounting_passed
        and
        validation_ratio_passed
    )

    print()
    print(
        "=" * 76
    )

    if all_passed:

        print(
            "SUCCESS: GENERAL-v0.4 "
            "CLEANED, DEDUPLICATED, AND SPLIT."
        )

        if not recommended_clean_size_reached:

            print()
            print(
                "WARNING: clean corpus is below "
                "the recommended pre-tokenization "
                "size. Token encoding must determine "
                "whether the 250M training-token "
                "target is still achievable."
            )

    else:

        raise RuntimeError(
            "GENERAL-v0.4 preparation "
            "failed safety checks."
        )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()