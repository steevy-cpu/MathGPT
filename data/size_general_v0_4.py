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
# Paths
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "general_data_v0.4.json"
)


V0_2_TRAIN_METADATA = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_v0_2_train_metadata.json"
)


INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general_v0.4"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general_v0.4_sized"
)


REPORT_PATH = (
    OUTPUT_DIR
    / "general_pretraining_sizing_report.json"
)


SOURCE_NAMES = [
    "fineweb_edu",
    "wikipedia",
    "gutenberg",
]


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


# ============================================================
# Inspect one cleaned split
# ============================================================

def inspect_split(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    documents = []

    total_characters = 0

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
                    f"{path} line "
                    f"{line_number:,}."
                )

            characters = len(
                text
            )

            documents.append(
                {
                    "line_number":
                        line_number,

                    "characters":
                        characters,
                }
            )

            total_characters += (
                characters
            )

    if not documents:

        raise RuntimeError(
            f"No documents found in "
            f"{path}."
        )

    return (
        documents,
        total_characters,
    )


# ============================================================
# Deterministic whole-document selection
# ============================================================

def select_documents(
    *,
    documents,
    target_characters,
    seed,
):

    rng = random.Random(
        seed
    )

    candidates = list(
        documents
    )

    rng.shuffle(
        candidates
    )

    selected = []

    selected_characters = 0

    for document in candidates:

        if (
            selected_characters
            >=
            target_characters
        ):

            break

        selected.append(
            document
        )

        selected_characters += (
            document[
                "characters"
            ]
        )

    # ========================================================
    # If dropping the last document gets us closer to the
    # target, drop it.
    # ========================================================

    if len(
        selected
    ) > 1:

        last_document = (
            selected[
                -1
            ]
        )

        with_last_distance = abs(
            selected_characters
            -
            target_characters
        )

        without_last_characters = (
            selected_characters
            -
            last_document[
                "characters"
            ]
        )

        without_last_distance = abs(
            without_last_characters
            -
            target_characters
        )

        if (
            without_last_distance
            <
            with_last_distance
        ):

            selected.pop()

            selected_characters = (
                without_last_characters
            )

    if not selected:

        raise RuntimeError(
            "Sizing selected zero documents."
        )

    selected_lines = {
        document[
            "line_number"
        ]

        for document
        in selected
    }

    return (
        selected_lines,
        selected_characters,
        len(
            selected
        ),
    )


# ============================================================
# Write selected documents
# ============================================================

def write_selected(
    *,
    input_path,
    output_path,
    selected_lines,
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

    documents_written = 0

    characters_written = 0

    try:

        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as input_file:

            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as output_file:

                for line_number, line in enumerate(
                    input_file,
                    start=1,
                ):

                    if (
                        line_number
                        not in
                        selected_lines
                    ):

                        continue

                    record = json.loads(
                        line
                    )

                    text = record[
                        "text"
                    ]

                    output_file.write(
                        line
                    )

                    documents_written += 1

                    characters_written += len(
                        text
                    )

        temporary_path.replace(
            output_path
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink()

    return {
        "documents":
            documents_written,

        "characters":
            characters_written,
    }


# ============================================================
# Main
# ============================================================

def main():

    config = load_json(
        CONFIG_PATH
    )

    old_metadata = load_json(
        V0_2_TRAIN_METADATA
    )

    seed = int(
        config[
            "reproducibility"
        ][
            "seed"
        ]
    )

    target_train_tokens = int(
        config[
            "objective"
        ][
            "target_training_tokens"
        ]
    )

    target_validation_tokens = int(
        config[
            "objective"
        ][
            "target_validation_tokens"
        ]
    )

    train_min = int(
        config[
            "objective"
        ][
            "training_token_tolerance_min"
        ]
    )

    train_max = int(
        config[
            "objective"
        ][
            "training_token_tolerance_max"
        ]
    )

    validation_min = int(
        config[
            "objective"
        ][
            "validation_token_tolerance_min"
        ]
    )

    validation_max = int(
        config[
            "objective"
        ][
            "validation_token_tolerance_max"
        ]
    )

    source_mix = (
        config[
            "source_mix"
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Protect completed output
    # ========================================================

    expected_outputs = [
        REPORT_PATH,
    ]

    for source_name in SOURCE_NAMES:

        expected_outputs.extend(
            [
                (
                    OUTPUT_DIR
                    /
                    f"{source_name}_train.jsonl"
                ),

                (
                    OUTPUT_DIR
                    /
                    f"{source_name}_validation.jsonl"
                ),
            ]
        )

    existing = [
        path

        for path
        in expected_outputs

        if path.exists()
    ]

    if existing:

        print()
        print(
            "GENERAL-v0.4 sized outputs "
            "already exist:"
        )

        for path in existing:

            print(
                f"  {path}"
            )

        print()
        print(
            "Refusing to overwrite."
        )

        raise SystemExit(
            0
        )

    print()
    print(
        "=" * 78
    )

    print(
        "STAGE 15C.1 — GENERAL-v0.4 "
        "DETERMINISTIC CORPUS SIZING"
    )

    print(
        "=" * 78
    )

    print()
    print(
        f"Target train tokens: "
        f"{target_train_tokens:,}"
    )

    print(
        f"Target validation tokens: "
        f"{target_validation_tokens:,}"
    )

    print()

    report = {
        "stage":
            "15C.1",

        "dataset_version":
            "general_pretraining_v0.4",

        "method":
            (
                "deterministic whole-document "
                "source-preserving sizing"
            ),

        "seed":
            seed,

        "target_train_tokens":
            target_train_tokens,

        "target_validation_tokens":
            target_validation_tokens,

        "historical_token_yield_source":
            str(
                V0_2_TRAIN_METADATA
            ),

        "sources":
            {},
    }

    projected_train_total = 0.0

    projected_validation_total = 0.0

    total_train_documents = 0

    total_validation_documents = 0

    total_train_characters = 0

    total_validation_characters = 0

    # ========================================================
    # Size each source independently
    # ========================================================

    for source_offset, source_name in enumerate(
        SOURCE_NAMES
    ):

        old_stats = (
            old_metadata[
                "sources"
            ][
                source_name
            ]
        )

        historical_characters = int(
            old_stats[
                "characters"
            ]
        )

        historical_tokens = int(
            old_stats[
                "tokens"
            ]
        )

        tokens_per_character = (
            historical_tokens
            /
            historical_characters
        )

        mix_fraction = float(
            source_mix[
                source_name
            ]
        )

        source_train_token_target = (
            target_train_tokens
            *
            mix_fraction
        )

        source_validation_token_target = (
            target_validation_tokens
            *
            mix_fraction
        )

        source_train_character_target = round(
            source_train_token_target
            /
            tokens_per_character
        )

        source_validation_character_target = round(
            source_validation_token_target
            /
            tokens_per_character
        )

        train_input = (
            INPUT_DIR
            /
            f"{source_name}_train.jsonl"
        )

        validation_input = (
            INPUT_DIR
            /
            f"{source_name}_validation.jsonl"
        )

        (
            train_documents,
            available_train_characters,
        ) = inspect_split(
            train_input
        )

        (
            validation_documents,
            available_validation_characters,
        ) = inspect_split(
            validation_input
        )

        if (
            available_train_characters
            <
            source_train_character_target
        ):

            raise RuntimeError(
                f"{source_name} does not have "
                "enough training characters."
            )

        if (
            available_validation_characters
            <
            source_validation_character_target
        ):

            raise RuntimeError(
                f"{source_name} does not have "
                "enough validation characters."
            )

        (
            selected_train_lines,
            selected_train_characters,
            selected_train_documents,
        ) = select_documents(
            documents=train_documents,

            target_characters=(
                source_train_character_target
            ),

            seed=(
                seed
                +
                source_offset
            ),
        )

        (
            selected_validation_lines,
            selected_validation_characters,
            selected_validation_documents,
        ) = select_documents(
            documents=validation_documents,

            target_characters=(
                source_validation_character_target
            ),

            seed=(
                seed
                +
                100
                +
                source_offset
            ),
        )

        train_output = (
            OUTPUT_DIR
            /
            f"{source_name}_train.jsonl"
        )

        validation_output = (
            OUTPUT_DIR
            /
            f"{source_name}_validation.jsonl"
        )

        train_write = write_selected(
            input_path=train_input,
            output_path=train_output,
            selected_lines=selected_train_lines,
        )

        validation_write = write_selected(
            input_path=validation_input,
            output_path=validation_output,
            selected_lines=(
                selected_validation_lines
            ),
        )

        # ====================================================
        # Writer cross-check
        # ====================================================

        writer_passed = (
            train_write[
                "documents"
            ]
            ==
            selected_train_documents

            and

            train_write[
                "characters"
            ]
            ==
            selected_train_characters

            and

            validation_write[
                "documents"
            ]
            ==
            selected_validation_documents

            and

            validation_write[
                "characters"
            ]
            ==
            selected_validation_characters
        )

        if not writer_passed:

            raise RuntimeError(
                f"{source_name} sizing writer "
                "accounting failed."
            )

        projected_train_tokens = (
            selected_train_characters
            *
            tokens_per_character
        )

        projected_validation_tokens = (
            selected_validation_characters
            *
            tokens_per_character
        )

        projected_train_total += (
            projected_train_tokens
        )

        projected_validation_total += (
            projected_validation_tokens
        )

        total_train_documents += (
            selected_train_documents
        )

        total_validation_documents += (
            selected_validation_documents
        )

        total_train_characters += (
            selected_train_characters
        )

        total_validation_characters += (
            selected_validation_characters
        )

        print()
        print(
            source_name
        )

        print(
            "-" * 52
        )

        print(
            f"Historical tokens/char: "
            f"{tokens_per_character:.6f}"
        )

        print()

        print(
            f"Train token target: "
            f"{source_train_token_target:,.0f}"
        )

        print(
            f"Train char target: "
            f"{source_train_character_target:,}"
        )

        print(
            f"Selected train chars: "
            f"{selected_train_characters:,}"
        )

        print(
            f"Projected train tokens: "
            f"{projected_train_tokens:,.0f}"
        )

        print()

        print(
            f"Validation token target: "
            f"{source_validation_token_target:,.0f}"
        )

        print(
            f"Validation char target: "
            f"{source_validation_character_target:,}"
        )

        print(
            f"Selected validation chars: "
            f"{selected_validation_characters:,}"
        )

        print(
            f"Projected validation tokens: "
            f"{projected_validation_tokens:,.0f}"
        )

        report[
            "sources"
        ][
            source_name
        ] = {
            "mix_fraction":
                mix_fraction,

            "historical_characters":
                historical_characters,

            "historical_tokens":
                historical_tokens,

            "historical_tokens_per_character":
                tokens_per_character,

            "train": {
                "available_characters":
                    available_train_characters,

                "target_tokens":
                    source_train_token_target,

                "target_characters":
                    source_train_character_target,

                "selected_documents":
                    selected_train_documents,

                "selected_characters":
                    selected_train_characters,

                "projected_tokens":
                    projected_train_tokens,

                "output":
                    str(
                        train_output
                    ),
            },

            "validation": {
                "available_characters":
                    available_validation_characters,

                "target_tokens":
                    source_validation_token_target,

                "target_characters":
                    source_validation_character_target,

                "selected_documents":
                    selected_validation_documents,

                "selected_characters":
                    selected_validation_characters,

                "projected_tokens":
                    projected_validation_tokens,

                "output":
                    str(
                        validation_output
                    ),
            },

            "writer_accounting_passed":
                writer_passed,
        }

    # ========================================================
    # Overall checks
    # ========================================================

    projected_train_rounded = round(
        projected_train_total
    )

    projected_validation_rounded = round(
        projected_validation_total
    )

    train_projection_passed = (
        train_min
        <=
        projected_train_rounded
        <=
        train_max
    )

    validation_projection_passed = (
        validation_min
        <=
        projected_validation_rounded
        <=
        validation_max
    )

    report[
        "overall"
    ] = {
        "train_documents":
            total_train_documents,

        "validation_documents":
            total_validation_documents,

        "train_characters":
            total_train_characters,

        "validation_characters":
            total_validation_characters,

        "projected_train_tokens":
            projected_train_rounded,

        "projected_validation_tokens":
            projected_validation_rounded,

        "training_projection_within_tolerance":
            train_projection_passed,

        "validation_projection_within_tolerance":
            validation_projection_passed,
    }

    temporary_report = REPORT_PATH.with_name(
        REPORT_PATH.name
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
            indent=4,
            ensure_ascii=False,
        )

    temporary_report.replace(
        REPORT_PATH
    )

    print()
    print(
        "=" * 78
    )

    print(
        "GENERAL-v0.4 SIZED CORPUS SUMMARY"
    )

    print(
        "=" * 78
    )

    print()

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
        f"Train characters: "
        f"{total_train_characters:,}"
    )

    print(
        f"Validation characters: "
        f"{total_validation_characters:,}"
    )

    print()

    print(
        f"Projected train tokens: "
        f"{projected_train_rounded:,}"
    )

    print(
        f"Projected validation tokens: "
        f"{projected_validation_rounded:,}"
    )

    print()

    print(
        f"Training projection within "
        f"245M–255M: "
        f"{train_projection_passed}"
    )

    print(
        f"Validation projection within "
        f"27M–29M: "
        f"{validation_projection_passed}"
    )

    print()

    print(
        "Sizing report:"
    )

    print(
        REPORT_PATH
    )

    all_passed = (
        train_projection_passed
        and
        validation_projection_passed
    )

    print()
    print(
        "=" * 78
    )

    if all_passed:

        print(
            "SUCCESS: GENERAL-v0.4 "
            "DETERMINISTIC SIZING PASSED."
        )

    else:

        raise RuntimeError(
            "GENERAL-v0.4 sizing "
            "failed projection checks."
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()