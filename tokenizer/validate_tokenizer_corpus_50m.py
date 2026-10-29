import json
import random
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "tokenizer_training_corpus_50m_v0.2.txt"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_50m_corpus_report.json"
)


EXPECTED_TOTAL_CHARS = 50_000_000

EXPECTED_SOURCES = {
    "fineweb_edu": 29_000_000,
    "wikipedia": 5_400_000,
    "gutenberg": 600_000,
    "openwebmath": 15_000_000,
}


RANDOM_SEED = 42
NUMBER_OF_SAMPLES = 5
SAMPLE_LENGTH = 600


def main():

    print()
    print("=" * 70)
    print("TOKENIZER 50M v0.2 CORPUS VALIDATION")
    print("=" * 70)

    # ========================================================
    # Load build report
    # ========================================================

    with open(
        REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    # ========================================================
    # Load corpus
    #
    # 50M characters is small enough for our training machine
    # and makes validation/simple sampling straightforward.
    # ========================================================

    print()
    print("Loading 50M corpus...")

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read()

    char_count = len(text)

    byte_count = (
        CORPUS_PATH
        .stat()
        .st_size
    )

    print()
    print("Basic integrity")
    print("---------------")

    print(
        f"Characters: "
        f"{char_count:,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{byte_count:,}"
    )

    exact_size_passed = (
        char_count
        == EXPECTED_TOTAL_CHARS
    )

    print(
        f"Exactly 50M characters: "
        f"{exact_size_passed}"
    )

    # ========================================================
    # Source report validation
    # ========================================================

    print()
    print("Source allocation")
    print("-----------------")

    source_targets_passed = True

    for (
        source_name,
        expected_chars,
    ) in EXPECTED_SOURCES.items():

        actual_chars = (
            report[
                "sources"
            ][
                source_name
            ][
                "characters_written"
            ]
        )

        passed = (
            actual_chars
            == expected_chars
        )

        source_targets_passed = (
            source_targets_passed
            and passed
        )

        print(
            f"{source_name:15} "
            f"{actual_chars:>12,} "
            f"expected "
            f"{expected_chars:>12,} "
            f"passed={passed}"
        )

    general_math_passed = (
        report[
            "general_characters"
        ]
        == 35_000_000
        and
        report[
            "math_characters"
        ]
        == 15_000_000
    )

    print()

    print(
        f"70/30 general-math split passed: "
        f"{general_math_passed}"
    )

    # ========================================================
    # Encoding / corruption checks
    # ========================================================

    print()
    print("Encoding / corruption checks")
    print("----------------------------")

    replacement_count = (
        text.count(
            "\ufffd"
        )
    )

    null_count = (
        text.count(
            "\x00"
        )
    )

    unexpected_control_count = 0

    for character in text:

        if character in (
            "\n",
            "\t",
        ):
            continue

        category = (
            unicodedata.category(
                character
            )
        )

        if category.startswith(
            "C"
        ):
            unexpected_control_count += 1

    replacement_passed = (
        replacement_count
        == 0
    )

    null_passed = (
        null_count
        == 0
    )

    control_passed = (
        unexpected_control_count
        == 0
    )

    print(
        f"Replacement characters: "
        f"{replacement_count:,}"
    )

    print(
        f"Null bytes/chars: "
        f"{null_count:,}"
    )

    print(
        f"Unexpected controls: "
        f"{unexpected_control_count:,}"
    )

    print(
        f"Unicode corruption passed: "
        f"{replacement_passed}"
    )

    print(
        f"Null check passed: "
        f"{null_passed}"
    )

    print(
        f"Control-character check passed: "
        f"{control_passed}"
    )

    # ========================================================
    # Remaining obvious noise
    # ========================================================

    print()
    print("Noise indicators")
    print("----------------")

    lower_text = text.lower()

    noise_patterns = {
        "HTML comments": "<!--",
        "script tags": "<script",
        "style tags": "<style",
        "permalink": "permalink",
        "show tags": "show tags",
        "back to top": "back to top",
    }

    noise_counts = {}

    for (
        name,
        pattern,
    ) in noise_patterns.items():

        count = lower_text.count(
            pattern
        )

        noise_counts[
            name
        ] = count

        print(
            f"{name:20} "
            f"{count:,}"
        )

    # These should definitely have been removed.
    structural_html_passed = (
        noise_counts[
            "HTML comments"
        ]
        == 0
        and
        noise_counts[
            "script tags"
        ]
        == 0
        and
        noise_counts[
            "style tags"
        ]
        == 0
    )

    print()

    print(
        f"Structural HTML cleanup passed: "
        f"{structural_html_passed}"
    )

    # ========================================================
    # Mathematical notation preservation
    # ========================================================

    print()
    print("Math preservation")
    print("-----------------")

    math_patterns = {
        "dollar signs": "$",
        "LaTeX frac": r"\frac",
        "LaTeX partial": r"\partial",
        "LaTeX begin": r"\begin",
        "LaTeX end": r"\end",
        "pi": "π",
        "summation": "∑",
        "less/equal": "≤",
        "greater/equal": "≥",
        "square root": "√",
    }

    math_counts = {}

    for (
        name,
        pattern,
    ) in math_patterns.items():

        count = text.count(
            pattern
        )

        math_counts[
            name
        ] = count

        print(
            f"{name:20} "
            f"{count:,}"
        )

    important_math_present = (
        math_counts[
            "LaTeX frac"
        ] > 0
        and
        math_counts[
            "LaTeX partial"
        ] > 0
        and
        math_counts[
            "dollar signs"
        ] > 0
    )

    print()

    print(
        f"Important math notation preserved: "
        f"{important_math_present}"
    )

    # ========================================================
    # Character diversity
    # ========================================================

    non_ascii_count = sum(
        1
        for character in text
        if ord(character) > 127
    )

    print()
    print("Character diversity")
    print("-------------------")

    print(
        f"Non-ASCII characters: "
        f"{non_ascii_count:,}"
    )

    print(
        f"Non-ASCII percentage: "
        f"{(
            non_ascii_count
            / char_count
            * 100
        ):.4f}%"
    )

    # ========================================================
    # Deterministic random samples
    # ========================================================

    print()
    print("=" * 70)
    print("RANDOM CORPUS SAMPLES")
    print("=" * 70)

    rng = random.Random(
        RANDOM_SEED
    )

    for sample_number in range(
        1,
        NUMBER_OF_SAMPLES + 1,
    ):

        start = rng.randint(
            0,
            char_count
            - SAMPLE_LENGTH
        )

        sample = text[
            start:
            start
            + SAMPLE_LENGTH
        ]

        print()
        print(
            f"Sample {sample_number}"
        )

        print(
            f"Character range: "
            f"{start:,} → "
            f"{start + SAMPLE_LENGTH:,}"
        )

        print("-" * 70)

        print(sample)

        print("-" * 70)

    # ========================================================
    # Final result
    # ========================================================

    all_passed = (
        exact_size_passed
        and source_targets_passed
        and general_math_passed
        and replacement_passed
        and null_passed
        and control_passed
        and structural_html_passed
        and important_math_present
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if all_passed:

        print(
            "SUCCESS: tokenizer_50m_v0.2 corpus "
            "passed validation."
        )

    else:

        print(
            "FAILURE: tokenizer_50m_v0.2 corpus "
            "failed one or more validation checks."
        )


if __name__ == "__main__":
    main()