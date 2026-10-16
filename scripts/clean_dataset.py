# It connect generic + source_specific cleaning 
# But it is still conservative to gutenberg data
import json
import sys
from pathlib import Path

from data_writer import DataWriter
from text_cleaner import TextCleaner
from source_cleaners import (
    FineWebCleaner,
    WikipediaCleaner,
    GutenbergCleaner,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DatasetCleaner:

    def __init__(
        self,
        input_path,
        output_path,
        cleaning_config,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        self.text_cleaner = TextCleaner(
            cleaning_config
        )

        self.source_cleaners = {
            "fineweb_edu": FineWebCleaner(),
            "wikipedia": WikipediaCleaner(),
            "project_gutenberg": GutenbergCleaner(),
        }

        self.writer = DataWriter(
            self.output_path
        )

    def clean_document(self, document):
        """
        Apply generic cleaning followed by
        source-specific cleaning.
        """

        original_text = document.get(
            "text",
            "",
        )

        original_character_count = len(
            original_text
        )

        # First: generic cleaning.
        cleaned_text = self.text_cleaner.clean(
            original_text
        )

        # Second: source-specific cleaning.
        source = document.get(
            "source",
            "",
        )

        source_cleaner = self.source_cleaners.get(
            source
        )

        if source_cleaner is not None:
            cleaned_text = source_cleaner.clean(
                cleaned_text
            )

        cleaned_character_count = len(
            cleaned_text
        )

        document["text"] = cleaned_text

        document["cleaning_version"] = "0.1"

        document["original_character_count"] = (
            original_character_count
        )

        document["cleaned_character_count"] = (
            cleaned_character_count
        )

        return document

    def clean_dataset(self):
        """
        Clean every document in the input JSONL file.
        """

        self.writer.clear()

        total_documents = 0
        total_characters_before = 0
        total_characters_after = 0

        with open(
            self.input_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                document = json.loads(line)

                cleaned_document = (
                    self.clean_document(
                        document
                    )
                )

                total_documents += 1

                total_characters_before += (
                    cleaned_document[
                        "original_character_count"
                    ]
                )

                total_characters_after += (
                    cleaned_document[
                        "cleaned_character_count"
                    ]
                )

                self.writer.write_document(
                    cleaned_document
                )

        characters_removed = (
            total_characters_before
            - total_characters_after
        )

        print()
        print("Cleaning complete.")
        print("------------------")

        print(
            f"Documents cleaned: "
            f"{total_documents:,}"
        )

        print(
            f"Characters before: "
            f"{total_characters_before:,}"
        )

        print(
            f"Characters after: "
            f"{total_characters_after:,}"
        )

        print(
            f"Characters removed: "
            f"{characters_removed:,}"
        )

        print(
            f"Output: {self.output_path}"
        )


def main():

    if len(sys.argv) != 3:
        print(
            "Usage:"
        )

        print(
            "python scripts/clean_dataset.py "
            "<input.jsonl> <output.jsonl>"
        )

        return

    input_path = Path(
        sys.argv[1]
    )

    output_path = Path(
        sys.argv[2]
    )

    if not input_path.is_absolute():
        input_path = (
            PROJECT_ROOT / input_path
        )

    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT / output_path
        )

    config_path = (
        PROJECT_ROOT
        / "config"
        / "cleaning_rules_v0.1.json"
    )

    cleaner = DatasetCleaner(
        input_path=input_path,
        output_path=output_path,
        cleaning_config=config_path,
    )

    cleaner.clean_dataset()


if __name__ == "__main__":
    main()