from pathlib import Path
from collections import Counter
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent


CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "tokenizer_training_corpus.txt"
)


REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_training_corpus_report.json"
)


def main():

    print()
    print("Tokenizer Training Corpus Validation")
    print("====================================")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read()

    print()
    print("Basic statistics")
    print("----------------")

    print(
        f"Characters on disk: "
        f"{len(text):,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{len(text.encode('utf-8')):,}"
    )

    print(
        f"Lines: "
        f"{text.count(chr(10)) + 1:,}"
    )

    print()

    # --------------------------------------------------
    # Unicode health
    # --------------------------------------------------

    replacement_characters = (
        text.count("\uFFFD")
    )

    null_characters = (
        text.count("\x00")
    )

    print("Unicode health")
    print("--------------")

    print(
        f"Replacement characters (�): "
        f"{replacement_characters:,}"
    )

    print(
        f"Null characters: "
        f"{null_characters:,}"
    )

    print()

    # --------------------------------------------------
    # Math indicators
    # --------------------------------------------------

    math_indicators = {
        "$": text.count("$"),
        "\\frac": text.count("\\frac"),
        "\\partial": text.count("\\partial"),
        "\\sum": text.count("\\sum"),
        "\\sqrt": text.count("\\sqrt"),
        "\\int": text.count("\\int"),
        "=": text.count("="),
        "^": text.count("^"),
        "_": text.count("_"),
        "π": text.count("π"),
        "∑": text.count("∑"),
        "∫": text.count("∫"),
        "√": text.count("√"),
        "≤": text.count("≤"),
        "≥": text.count("≥"),
        "≠": text.count("≠"),
    }

    print("Math indicators")
    print("---------------")

    for symbol, count in (
        math_indicators.items()
    ):

        print(
            f"{symbol!r}: {count:,}"
        )

    print()

    # --------------------------------------------------
    # Character categories
    # --------------------------------------------------

    category_counts = Counter()

    for character in text:

        if character.isalpha():
            category_counts[
                "alphabetic"
            ] += 1

        elif character.isdigit():
            category_counts[
                "digits"
            ] += 1

        elif character.isspace():
            category_counts[
                "whitespace"
            ] += 1

        else:
            category_counts[
                "symbols_punctuation"
            ] += 1

    total = len(text)

    print("Character composition")
    print("---------------------")

    for category, count in (
        category_counts.items()
    ):

        ratio = (
            count / total
            if total
            else 0
        )

        print(
            f"{category}: "
            f"{count:,} "
            f"({ratio:.2%})"
        )

    print()

    # --------------------------------------------------
    # Check saved build report
    # --------------------------------------------------

    if REPORT_PATH.exists():

        with open(
            REPORT_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            report = json.load(file)

        print("Expected mixture")
        print("----------------")

        print(
            f"General characters: "
            f"{report['general_characters']:,}"
        )

        print(
            f"Math characters: "
            f"{report['math_characters']:,}"
        )

        print(
            f"General ratio: "
            f"{report['general_ratio']:.2%}"
        )

        print(
            f"Math ratio: "
            f"{report['math_ratio']:.2%}"
        )

    print()

    # --------------------------------------------------
    # Final checks
    # --------------------------------------------------

    passed = True

    if replacement_characters > 0:

        print(
            "WARNING: Unicode replacement "
            "characters were found."
        )

        passed = False

    if null_characters > 0:

        print(
            "WARNING: Null characters "
            "were found."
        )

        passed = False

    if len(text) < 10_000_000:

        print(
            "WARNING: Corpus is smaller "
            "than expected."
        )

        passed = False

    print()

    if passed:

        print(
            "SUCCESS: Tokenizer training "
            "corpus passed validation."
        )

    else:

        print(
            "WARNING: Corpus validation "
            "found issues."
        )


if __name__ == "__main__":
    main()