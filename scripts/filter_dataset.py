import json
import sys
from pathlib import Path

from data_writer import DataWriter

# After cleaning and deduplication, is this document actually good enough to keep for training?

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DocumentFilter:

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.rules = self.load_rules()

    def load_rules(self):
        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        return config["rules"]

    def alphabetic_ratio(self, text):
        """
        Return the fraction of characters that are letters.
        """

        if not text:
            return 0.0

        alphabetic_count = sum(
            character.isalpha()
            for character in text
        )

        return alphabetic_count / len(text)

    def replacement_character_ratio(self, text):
        """
        Measure how much Unicode replacement-character noise exists.

        The replacement character is: �
        """

        if not text:
            return 0.0

        replacement_count = text.count("\uFFFD")

        return replacement_count / len(text)

    def check_document(self, document):
        """
        Return:
            keep_document, rejection_reason
        """

        text = document.get(
            "text",
            "",
        )

        if not text.strip():
            return False, "empty_text"

        min_characters = self.rules.get(
            "min_characters",
            200,
        )

        if len(text) < min_characters:
            return False, "too_short"

        minimum_alpha_ratio = self.rules.get(
            "min_alphabetic_ratio",
            0.30,
        )

        alpha_ratio = self.alphabetic_ratio(
            text
        )

        if alpha_ratio < minimum_alpha_ratio:
            return False, "low_alphabetic_ratio"

        maximum_replacement_ratio = self.rules.get(
            "max_replacement_character_ratio",
            0.01,
        )

        replacement_ratio = (
            self.replacement_character_ratio(
                text
            )
        )

        if replacement_ratio > maximum_replacement_ratio:
            return False, "too_many_replacement_characters"

        return True, None


class DatasetFilter:

    def __init__(
        self,
        input_path,
        output_path,
        filter_config,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        self.document_filter = DocumentFilter(
            filter_config
        )

        self.writer = DataWriter(
            self.output_path
        )

    def filter_dataset(self):
        self.writer.clear()

        documents_read = 0
        documents_kept = 0
        documents_rejected = 0

        rejection_reasons = {}

        with open(
            self.input_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                document = json.loads(line)

                documents_read += 1

                keep, reason = (
                    self.document_filter.check_document(
                        document
                    )
                )

                if not keep:
                    documents_rejected += 1

                    rejection_reasons[reason] = (
                        rejection_reasons.get(
                            reason,
                            0,
                        )
                        + 1
                    )

                    continue

                self.writer.write_document(
                    document
                )

                documents_kept += 1

        print()
        print("Document filtering complete.")
        print("----------------------------")

        print(
            f"Documents read: "
            f"{documents_read:,}"
        )

        print(
            f"Documents kept: "
            f"{documents_kept:,}"
        )

        print(
            f"Documents rejected: "
            f"{documents_rejected:,}"
        )

        if rejection_reasons:
            print()
            print("Rejection reasons")
            print("-----------------")

            for reason, count in rejection_reasons.items():
                print(
                    f"{reason}: {count:,}"
                )

        print()
        print(
            f"Output: "
            f"{self.output_path}"
        )


def main():

    if len(sys.argv) != 3:
        print(
            "Usage:"
        )

        print(
            "python scripts/filter_dataset.py "
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

    if not input_path.exists():
        print(
            f"Input dataset not found: "
            f"{input_path}"
        )
        return

    config_path = (
        PROJECT_ROOT
        / "config"
        / "filter_rules_v0.1.json"
    )

    dataset_filter = DatasetFilter(
        input_path=input_path,
        output_path=output_path,
        filter_config=config_path,
    )

    dataset_filter.filter_dataset()


if __name__ == "__main__":
    main()