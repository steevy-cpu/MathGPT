from pathlib import Path

from text_cleaner import TextCleaner


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():

    config_path = (
        PROJECT_ROOT
        / "config"
        / "cleaning_rules_v0.1.json"
    )

    cleaner = TextCleaner(config_path)

    messy_text = (
        "Hello world!   \r\n"
        "\r\n"
        "\r\n"
        "Math symbols: ∫ x² dx   \n"
        "\x00"
        "This line contains a null character.\n"
        "\n"
        "\n"
        "\n"
        "Final line.   "
    )

    cleaned_text = cleaner.clean(
        messy_text
    )

    print()
    print("ORIGINAL")
    print("--------")
    print(repr(messy_text))

    print()
    print("CLEANED")
    print("-------")
    print(repr(cleaned_text))

    print()
    print("Readable cleaned text")
    print("---------------------")
    print(cleaned_text)


if __name__ == "__main__":
    main()