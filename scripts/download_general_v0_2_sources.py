import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


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
    / "general_data_v0.2.json"
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
                f"SKIP: output already exists:"
            )

            print(
                output_path
            )

            print()

            print(
                "Use --overwrite if you "
                "really want to replace it."
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
        "=" * 72
    )

    print(
        "GENERAL-v0.2 — FINEWEB-EDU DOWNLOAD"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Target raw characters: "
        f"{target_chars:,}"
    )

    print(
        f"Output:"
    )

    print(
        output_path
    )

    print()

    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        source_config[
            "dataset"
        ],
        name=(
            source_config[
                "dataset_config"
            ]
        ),
        split=(
            source_config[
                "split"
            ]
        ),
        streaming=True,
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

                record = {
                    "text":
                        text,

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
                    text
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
                "the target was reached."
            )

        finalize_output(
            temp_path,
            output_path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink()

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

    return {
        "documents":
            documents,

        "characters":
            total_chars,

        "empty_skipped":
            empty_skipped,

        "output":
            str(
                output_path
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
        "=" * 72
    )

    print(
        "GENERAL-v0.2 — WIKIPEDIA DOWNLOAD"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Target raw characters: "
        f"{target_chars:,}"
    )

    print(
        f"Output:"
    )

    print(
        output_path
    )

    print()

    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        source_config[
            "dataset"
        ],
        source_config[
            "dataset_config"
        ],
        split=(
            source_config[
                "split"
            ]
        ),
        streaming=True,
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

                record = {
                    "text":
                        text,

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
                    text
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
                "the target was reached."
            )

        finalize_output(
            temp_path,
            output_path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink()

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

    return {
        "documents":
            documents,

        "characters":
            total_chars,

        "empty_skipped":
            empty_skipped,

        "output":
            str(
                output_path
            ),
    }


# ============================================================
# Gutenberg
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
        "=" * 72
    )

    print(
        "GENERAL-v0.2 — PROJECT GUTENBERG DOWNLOAD"
    )

    print(
        "=" * 72
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
        f"Output:"
    )

    print(
        output_path
    )

    print()

    print(
        "Loading streaming dataset..."
    )

    dataset_kwargs = {
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

    if (
        dataset_config
        is not None
    ):

        dataset_kwargs[
            "name"
        ] = dataset_config

    dataset = load_dataset(
        **dataset_kwargs
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
                "Gutenberg ended before "
                "the target was reached."
            )

        finalize_output(
            temp_path,
            output_path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink()

    print()
    print(
        "Gutenberg complete"
    )

    print(
        "------------------"
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

        "output":
            str(
                output_path
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

        "seed":
            config[
                "reproducibility"
            ][
                "seed"
            ],

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
        f"Download manifest:"
    )

    print(
        path
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Download General-v0.2 "
            "50M-token source corpus."
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

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    config = load_config()

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 12A.2 — GENERAL-v0.2 SOURCE EXPANSION"
    )

    print(
        "=" * 72
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
        f"Tokenizer remains: "
        f"{config['objective']['tokenizer']}"
    )

    results = {}

    selected = (
        [
            "fineweb_edu",
            "wikipedia",
            "gutenberg",
        ]
        if args.source == "all"
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
        "=" * 72
    )

    print(
        "SOURCE DOWNLOAD STAGE COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()