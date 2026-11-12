import argparse
import hashlib
import json

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

import datasets as hf_datasets

from datasets import (
    load_dataset,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "general_data_v0.4.json"
)


# ============================================================
# Helpers
# ============================================================

def load_config():

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Config missing: "
            f"{CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def raw_directory(
    config,
):

    path = (
        PROJECT_ROOT
        /
        config[
            "paths"
        ][
            "raw_directory"
        ]
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def temporary_path(
    output_path,
):

    return output_path.with_name(
        output_path.name
        +
        ".tmp"
    )


def prepare_output(
    output_path,
    overwrite,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = temporary_path(
        output_path
    )

    if temp_path.exists():

        temp_path.unlink()

    if output_path.exists():

        if not overwrite:

            print()
            print(
                "SKIP: output already exists:"
            )

            print(
                output_path
            )

            print()
            print(
                "Use --overwrite only if you "
                "really intend to replace it."
            )

            return None

        output_path.unlink()

    return temp_path


def finalize_output(
    temp_path,
    output_path,
):

    temp_path.replace(
        output_path
    )


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


def utc_now():

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def dataset_fingerprint(
    dataset,
):

    fingerprint = getattr(
        dataset,
        "_fingerprint",
        None,
    )

    if fingerprint is None:

        return None

    return str(
        fingerprint
    )


def build_dataset_kwargs(
    source_config,
):

    kwargs = {
        "path":
            source_config[
                "dataset"
            ],

        "split":
            source_config[
                "split"
            ],

        "streaming":
            True,
    }

    dataset_config = (
        source_config.get(
            "dataset_config"
        )
    )

    if dataset_config is not None:

        kwargs[
            "name"
        ] = dataset_config

    revision = (
        source_config.get(
            "revision"
        )
    )

    if revision is not None:

        kwargs[
            "revision"
        ] = revision

    return kwargs


def source_provenance(
    *,
    source_config,
    fingerprint,
):

    return {
        "dataset":
            source_config[
                "dataset"
            ],

        "dataset_config":
            source_config.get(
                "dataset_config"
            ),

        "split":
            source_config[
                "split"
            ],

        "requested_revision":
            source_config.get(
                "revision"
            ),

        "dataset_fingerprint":
            fingerprint,

        "datasets_library_version":
            hf_datasets.__version__,

        "shuffle_seed":
            int(
                source_config[
                    "shuffle_seed"
                ]
            ),

        "shuffle_buffer":
            int(
                source_config[
                    "shuffle_buffer"
                ]
            ),
    }


# ============================================================
# FineWeb-Edu
# ============================================================

def download_fineweb(
    config,
    overwrite,
):

    source_config = (
        config[
            "sources"
        ][
            "fineweb_edu"
        ]
    )

    output_path = (
        raw_directory(
            config
        )
        /
        source_config[
            "output_file"
        ]
    )

    temp_path = prepare_output(
        output_path,
        overwrite,
    )

    if temp_path is None:

        return None

    target_chars = int(
        source_config[
            "target_raw_characters"
        ]
    )

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 — FINEWEB-EDU DOWNLOAD"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Target raw characters: "
        f"{target_chars:,}"
    )

    print(
        "Output:"
    )

    print(
        output_path
    )

    print()
    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        **build_dataset_kwargs(
            source_config
        )
    )

    fingerprint = dataset_fingerprint(
        dataset
    )

    dataset = dataset.shuffle(
        seed=(
            source_config[
                "shuffle_seed"
            ]
        ),
        buffer_size=(
            source_config[
                "shuffle_buffer"
            ]
        ),
    )

    documents = 0
    total_chars = 0
    empty_skipped = 0

    started_at = utc_now()

    try:

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as output_file:

            for document in dataset:

                text = document.get(
                    "text",
                    "",
                )

                if (
                    not isinstance(
                        text,
                        str,
                    )
                    or
                    not text.strip()
                ):

                    empty_skipped += 1

                    continue

                remaining = (
                    target_chars
                    -
                    total_chars
                )

                contribution = text

                if (
                    len(
                        contribution
                    )
                    >
                    remaining
                ):

                    contribution = (
                        contribution[
                            :remaining
                        ]
                    )

                record = {
                    "text":
                        contribution,

                    "id":
                        document.get(
                            "id"
                        ),

                    "url":
                        document.get(
                            "url"
                        ),

                    "score":
                        document.get(
                            "score"
                        ),

                    "dump":
                        document.get(
                            "dump"
                        ),
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )

                output_file.write(
                    "\n"
                )

                documents += 1

                total_chars += len(
                    contribution
                )

                if (
                    documents
                    %
                    5_000
                    ==
                    0
                ):

                    percentage = (
                        total_chars
                        /
                        target_chars
                        *
                        100
                    )

                    print(
                        f"Documents: "
                        f"{documents:,} | "
                        f"Characters: "
                        f"{total_chars:,} | "
                        f"{percentage:.2f}%"
                    )

                if (
                    total_chars
                    >=
                    target_chars
                ):

                    break

        if (
            total_chars
            <
            target_chars
        ):

            raise RuntimeError(
                "FineWeb-Edu ended before "
                "target raw characters "
                "were reached."
            )

        finalize_output(
            temp_path,
            output_path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink()

    sha256 = file_sha256(
        output_path
    )

    print()
    print(
        "FineWeb-Edu complete"
    )

    print(
        "--------------------"
    )

    print(
        f"Documents: "
        f"{documents:,}"
    )

    print(
        f"Characters: "
        f"{total_chars:,}"
    )

    print(
        f"Empty skipped: "
        f"{empty_skipped:,}"
    )

    print(
        f"SHA-256: "
        f"{sha256}"
    )

    return {
        "documents":
            documents,

        "characters":
            total_chars,

        "empty_skipped":
            empty_skipped,

        "target_raw_characters":
            target_chars,

        "output":
            str(
                output_path
            ),

        "sha256":
            sha256,

        "download_started_at_utc":
            started_at,

        "download_completed_at_utc":
            utc_now(),

        "provenance":
            source_provenance(
                source_config=source_config,
                fingerprint=fingerprint,
            ),
    }


# ============================================================
# Wikipedia
# ============================================================

def download_wikipedia(
    config,
    overwrite,
):

    source_config = (
        config[
            "sources"
        ][
            "wikipedia"
        ]
    )

    output_path = (
        raw_directory(
            config
        )
        /
        source_config[
            "output_file"
        ]
    )

    temp_path = prepare_output(
        output_path,
        overwrite,
    )

    if temp_path is None:

        return None

    target_chars = int(
        source_config[
            "target_raw_characters"
        ]
    )

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 — WIKIPEDIA DOWNLOAD"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Target raw characters: "
        f"{target_chars:,}"
    )

    print(
        "Output:"
    )

    print(
        output_path
    )

    print()
    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        **build_dataset_kwargs(
            source_config
        )
    )

    fingerprint = dataset_fingerprint(
        dataset
    )

    dataset = dataset.shuffle(
        seed=(
            source_config[
                "shuffle_seed"
            ]
        ),
        buffer_size=(
            source_config[
                "shuffle_buffer"
            ]
        ),
    )

    documents = 0
    total_chars = 0
    empty_skipped = 0

    started_at = utc_now()

    try:

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as output_file:

            for document in dataset:

                text = document.get(
                    "text",
                    "",
                )

                if (
                    not isinstance(
                        text,
                        str,
                    )
                    or
                    not text.strip()
                ):

                    empty_skipped += 1

                    continue

                remaining = (
                    target_chars
                    -
                    total_chars
                )

                contribution = text

                if (
                    len(
                        contribution
                    )
                    >
                    remaining
                ):

                    contribution = (
                        contribution[
                            :remaining
                        ]
                    )

                record = {
                    "text":
                        contribution,

                    "id":
                        document.get(
                            "id"
                        ),

                    "url":
                        document.get(
                            "url"
                        ),

                    "title":
                        document.get(
                            "title"
                        ),
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )

                output_file.write(
                    "\n"
                )

                documents += 1

                total_chars += len(
                    contribution
                )

                if (
                    documents
                    %
                    500
                    ==
                    0
                ):

                    percentage = (
                        total_chars
                        /
                        target_chars
                        *
                        100
                    )

                    print(
                        f"Documents: "
                        f"{documents:,} | "
                        f"Characters: "
                        f"{total_chars:,} | "
                        f"{percentage:.2f}%"
                    )

                if (
                    total_chars
                    >=
                    target_chars
                ):

                    break

        if (
            total_chars
            <
            target_chars
        ):

            raise RuntimeError(
                "Wikipedia ended before "
                "target raw characters "
                "were reached."
            )

        finalize_output(
            temp_path,
            output_path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink()

    sha256 = file_sha256(
        output_path
    )

    print()
    print(
        "Wikipedia complete"
    )

    print(
        "------------------"
    )

    print(
        f"Documents: "
        f"{documents:,}"
    )

    print(
        f"Characters: "
        f"{total_chars:,}"
    )

    print(
        f"Empty skipped: "
        f"{empty_skipped:,}"
    )

    print(
        f"SHA-256: "
        f"{sha256}"
    )

    return {
        "documents":
            documents,

        "characters":
            total_chars,

        "empty_skipped":
            empty_skipped,

        "target_raw_characters":
            target_chars,

        "output":
            str(
                output_path
            ),

        "sha256":
            sha256,

        "download_started_at_utc":
            started_at,

        "download_completed_at_utc":
            utc_now(),

        "provenance":
            source_provenance(
                source_config=source_config,
                fingerprint=fingerprint,
            ),
    }


# ============================================================
# Project Gutenberg
# ============================================================

def download_gutenberg(
    config,
    overwrite,
):

    source_config = (
        config[
            "sources"
        ][
            "gutenberg"
        ]
    )

    output_path = (
        raw_directory(
            config
        )
        /
        source_config[
            "output_file"
        ]
    )

    temp_path = prepare_output(
        output_path,
        overwrite,
    )

    if temp_path is None:

        return None

    target_chars = int(
        source_config[
            "target_raw_characters"
        ]
    )

    max_chars_per_book = int(
        source_config[
            "max_characters_per_book"
        ]
    )

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 — PROJECT GUTENBERG DOWNLOAD"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Target raw characters: "
        f"{target_chars:,}"
    )

    print(
        f"Maximum chars/book: "
        f"{max_chars_per_book:,}"
    )

    print(
        "Output:"
    )

    print(
        output_path
    )

    print()
    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        **build_dataset_kwargs(
            source_config
        )
    )

    fingerprint = dataset_fingerprint(
        dataset
    )

    dataset = dataset.shuffle(
        seed=(
            source_config[
                "shuffle_seed"
            ]
        ),
        buffer_size=(
            source_config[
                "shuffle_buffer"
            ]
        ),
    )

    scanned = 0
    accepted = 0

    empty_skipped = 0
    non_english_skipped = 0

    total_chars = 0

    started_at = utc_now()

    try:

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as output_file:

            for document in dataset:

                scanned += 1

                text = document.get(
                    "text",
                    "",
                )

                if (
                    not isinstance(
                        text,
                        str,
                    )
                    or
                    not text.strip()
                ):

                    empty_skipped += 1

                    continue

                metadata = document.get(
                    "metadata",
                    {},
                )

                if not isinstance(
                    metadata,
                    dict,
                ):

                    metadata = {}

                language = metadata.get(
                    "language"
                )

                if (
                    language
                    is not None
                    and
                    language
                    !=
                    "en"
                ):

                    non_english_skipped += 1

                    continue

                contribution = text[
                    :max_chars_per_book
                ]

                remaining = (
                    target_chars
                    -
                    total_chars
                )

                if (
                    len(
                        contribution
                    )
                    >
                    remaining
                ):

                    contribution = (
                        contribution[
                            :remaining
                        ]
                    )

                record = {
                    "id":
                        document.get(
                            "id"
                        ),

                    "source":
                        document.get(
                            "source"
                        ),

                    "text":
                        contribution,

                    "metadata":
                        metadata,
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )

                output_file.write(
                    "\n"
                )

                accepted += 1

                total_chars += len(
                    contribution
                )

                if (
                    accepted
                    %
                    10
                    ==
                    0
                ):

                    percentage = (
                        total_chars
                        /
                        target_chars
                        *
                        100
                    )

                    print(
                        f"Books: "
                        f"{accepted:,} | "
                        f"Characters: "
                        f"{total_chars:,} | "
                        f"{percentage:.2f}%"
                    )

                if (
                    total_chars
                    >=
                    target_chars
                ):

                    break

        if (
            total_chars
            <
            target_chars
        ):

            raise RuntimeError(
                "Project Gutenberg ended "
                "before target raw "
                "characters were reached."
            )

        finalize_output(
            temp_path,
            output_path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink()

    sha256 = file_sha256(
        output_path
    )

    print()
    print(
        "Project Gutenberg complete"
    )

    print(
        "--------------------------"
    )

    print(
        f"Documents scanned: "
        f"{scanned:,}"
    )

    print(
        f"Books accepted: "
        f"{accepted:,}"
    )

    print(
        f"Characters: "
        f"{total_chars:,}"
    )

    print(
        f"Empty skipped: "
        f"{empty_skipped:,}"
    )

    print(
        f"Non-English skipped: "
        f"{non_english_skipped:,}"
    )

    print(
        f"SHA-256: "
        f"{sha256}"
    )

    return {
        "documents_scanned":
            scanned,

        "documents":
            accepted,

        "characters":
            total_chars,

        "empty_skipped":
            empty_skipped,

        "non_english_skipped":
            non_english_skipped,

        "target_raw_characters":
            target_chars,

        "max_characters_per_book":
            max_chars_per_book,

        "output":
            str(
                output_path
            ),

        "sha256":
            sha256,

        "download_started_at_utc":
            started_at,

        "download_completed_at_utc":
            utc_now(),

        "provenance":
            source_provenance(
                source_config=source_config,
                fingerprint=fingerprint,
            ),
    }


# ============================================================
# Manifest
# ============================================================

def save_manifest(
    config,
    source_results,
):

    output_dir = raw_directory(
        config
    )

    path = (
        output_dir
        /
        "download_manifest.json"
    )

    existing = {}

    if path.exists():

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            existing = json.load(
                file
            )

    manifest = {
        "version":
            config[
                "version"
            ],

        "dataset_version":
            config[
                "dataset_version"
            ],

        "experiment":
            config[
                "experiment"
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

        "tokenizer":
            config[
                "objective"
            ][
                "tokenizer"
            ],

        "seed":
            config[
                "reproducibility"
            ][
                "seed"
            ],

        "manifest_updated_at_utc":
            utc_now(),

        "sources":
            existing.get(
                "sources",
                {},
            ),
    }

    for (
        source_name,
        result,
    ) in source_results.items():

        if result is not None:

            manifest[
                "sources"
            ][
                source_name
            ] = result

    temporary = path.with_name(
        path.name
        +
        ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=4,
        )

    temporary.replace(
        path
    )

    print()
    print(
        "Download manifest:"
    )

    print(
        path
    )


# ============================================================
# Preflight
# ============================================================

def preflight(
    config,
):

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.4 SOURCE ACQUISITION PREFLIGHT"
    )

    print(
        "=" * 76
    )

    checks = {
        "Version = 0.4":
            (
                config.get(
                    "version"
                )
                ==
                "0.4"
            ),

        "Dataset identity":
            (
                config.get(
                    "dataset_version"
                )
                ==
                "general_pretraining_v0.4"
            ),

        "Training token target":
            (
                int(
                    config[
                        "objective"
                    ][
                        "target_training_tokens"
                    ]
                )
                ==
                250_000_000
            ),

        "Validation ratio":
            (
                float(
                    config[
                        "objective"
                    ][
                        "validation_ratio"
                    ]
                )
                ==
                0.10
            ),

        "Tokenizer frozen":
            (
                config[
                    "objective"
                ][
                    "tokenizer"
                ]
                ==
                "tokenizer_50m_v0.2"
            ),

        "Vocab frozen":
            (
                int(
                    config[
                        "objective"
                    ][
                        "vocab_size"
                    ]
                )
                ==
                16_000
            ),

        "Seed = 42":
            (
                int(
                    config[
                        "reproducibility"
                    ][
                        "seed"
                    ]
                )
                ==
                42
            ),

        "FineWeb target = 1.10B chars":
            (
                int(
                    config[
                        "sources"
                    ][
                        "fineweb_edu"
                    ][
                        "target_raw_characters"
                    ]
                )
                ==
                1_100_000_000
            ),

        "Wikipedia target = 220M chars":
            (
                int(
                    config[
                        "sources"
                    ][
                        "wikipedia"
                    ][
                        "target_raw_characters"
                    ]
                )
                ==
                220_000_000
            ),

        "Gutenberg target = 40M chars":
            (
                int(
                    config[
                        "sources"
                    ][
                        "gutenberg"
                    ][
                        "target_raw_characters"
                    ]
                )
                ==
                40_000_000
            ),
    }

    print()

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name:<38} "
            f"{passed}"
        )

    all_passed = all(
        checks.values()
    )

    print()

    if not all_passed:

        raise RuntimeError(
            "GENERAL-v0.4 acquisition "
            "preflight failed."
        )

    print(
        "SUCCESS: GENERAL-v0.4 "
        "source acquisition config passed."
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Download GENERAL-v0.4 "
            "data-scale source corpus."
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "all",
            "fineweb_edu",
            "wikipedia",
            "gutenberg",
        ],
        default="all",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing completed "
            "source file."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate the v0.4 acquisition "
            "configuration without "
            "downloading data."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    config = load_config()

    preflight(
        config
    )

    if args.preflight_only:

        return

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE 15B — GENERAL-v0.4 SOURCE ACQUISITION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Experiment: "
        f"{config['experiment']}"
    )

    print(
        f"Target training tokens: "
        f"{config['objective']['target_training_tokens']:,}"
    )

    print(
        f"Target total corpus tokens: "
        f"{config['objective']['target_total_tokens']:,}"
    )

    print(
        f"Tokenizer remains: "
        f"{config['objective']['tokenizer']}"
    )

    print()

    results = {}

    selected = (
        [
            "fineweb_edu",
            "wikipedia",
            "gutenberg",
        ]

        if args.source
        ==
        "all"

        else [
            args.source
        ]
    )

    for source_name in selected:

        if source_name == "fineweb_edu":

            results[
                source_name
            ] = download_fineweb(
                config,
                args.overwrite,
            )

        elif source_name == "wikipedia":

            results[
                source_name
            ] = download_wikipedia(
                config,
                args.overwrite,
            )

        elif source_name == "gutenberg":

            results[
                source_name
            ] = download_gutenberg(
                config,
                args.overwrite,
            )

    save_manifest(
        config,
        results,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: GENERAL-v0.4 "
        "SOURCE ACQUISITION COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()