import hashlib
import json
import sys
from pathlib import Path

from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DatasetDeduplicator:
    """
    Remove exact duplicate documents based on cleaned text.
    """

    def __init__(
        self,
        input_path,
        output_path,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        self.writer = DataWriter(
            self.output_path
        )

    def hash_text(self, text):
        """
        Create a stable SHA-256 hash from document text.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    def deduplicate(self):
        """
        Read the dataset and remove exact duplicate text.
        """

        self.writer.clear()

        seen_hashes = set()

        documents_read = 0
        documents_kept = 0
        duplicates_removed = 0

        with open(
            self.input_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                document = json.loads(line)

                documents_read += 1

                text = document.get(
                    "text",
                    "",
                )

                text_hash = self.hash_text(
                    text
                )

                if text_hash in seen_hashes:
                    duplicates_removed += 1
                    continue

                seen_hashes.add(
                    text_hash
                )

                self.writer.write_document(
                    document
                )

                documents_kept += 1

        print()
        print("Deduplication complete.")
        print("-----------------------")

        print(
            f"Documents read: "
            f"{documents_read:,}"
        )

        print(
            f"Documents kept: "
            f"{documents_kept:,}"
        )

        print(
            f"Exact duplicates removed: "
            f"{duplicates_removed:,}"
        )

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
            "python scripts/deduplicate_dataset.py "
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

    deduplicator = DatasetDeduplicator(
        input_path=input_path,
        output_path=output_path,
    )

    deduplicator.deduplicate()


if __name__ == "__main__":
    main()