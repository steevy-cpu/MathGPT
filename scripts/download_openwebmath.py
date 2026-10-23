import json
from pathlib import Path

from datasets import load_dataset

from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OpenWebMathDownloader:
    """
    Stream OpenWebMath documents and save them
    in the standard MathGPT JSONL format.
    """

    def __init__(
        self,
        output_path,
        target_characters=10_000_000,
        min_math_score=0.5,
        overwrite=True,
    ):
        self.output_path = Path(output_path)
        self.target_characters = target_characters
        self.min_math_score = min_math_score
        self.overwrite = overwrite
        
        self.writer = DataWriter(
            self.output_path
        )

    def load_stream(self):
        """
        Open OpenWebMath in streaming mode.
        """

        return load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True,
        )

    def parse_metadata(self, metadata):
        """
        OpenWebMath stores metadata as a JSON string.
        Convert it into a Python dictionary.
        """

        if not metadata:
            return {}

        try:
            return json.loads(metadata)

        except json.JSONDecodeError:
            return {}

    def convert_document(
        self,
        example,
        index,
    ):
        """
        Convert an OpenWebMath record into
        the standard MathGPT raw document format.
        """

        text = example.get(
            "text",
            "",
        )

        metadata = self.parse_metadata(
            example.get(
                "metadata",
                "",
            )
        )

        extraction_info = metadata.get(
            "extraction_info",
            {},
        )

        return {
            "id": f"openwebmath_{index:08d}",

            "source": "openwebmath",

            "source_version": "open-web-math",

            "url": example.get(
                "url",
                "",
            ),

            "license": "source_dependent",

            "category": "math",

            "date": example.get(
                "date",
                "",
            ),

            "found_math": extraction_info.get(
                "found_math",
                False,
            ),

            "math_score": extraction_info.get(
                "math_score",
                None,
            ),

            "perplexity_source": extraction_info.get(
                "perplexity",
                None,
            ),

            "character_count_source": len(
                text
            ),

            "text": text,
        }

    def download(self):
        """
        Stream documents until the character target
        has been reached.
        """

        if self.overwrite:
            self.writer.clear()

        dataset = self.load_stream()

        total_documents = 0
        total_characters = 0

        skipped_empty = 0
        skipped_low_score = 0

        print()
        print("Starting OpenWebMath download...")
        print(
            f"Target characters: "
            f"{self.target_characters:,}"
        )
        print(
            f"Minimum math score: "
            f"{self.min_math_score}"
        )
        print()

        for example in dataset:

            document = self.convert_document(
                example,
                total_documents,
            )

            text = document["text"]

            if not text.strip():
                skipped_empty += 1
                continue

            math_score = document.get(
                "math_score"
            )

            if math_score is None:
                skipped_low_score += 1
                continue

            if math_score < self.min_math_score:
                skipped_low_score += 1
                continue

            self.writer.write_document(
                document
            )

            total_documents += 1

            total_characters += (
                document[
                    "character_count_source"
                ]
            )

            print(
                f"\rDocuments: "
                f"{total_documents:,} | "
                f"Characters: "
                f"{total_characters:,} / "
                f"{self.target_characters:,}",
                end="",
            )

            if (
                total_characters
                >= self.target_characters
            ):
                break

        print()
        print()
        print("Download complete.")
        print("------------------")

        print(
            f"Documents saved: "
            f"{total_documents:,}"
        )

        print(
            f"Characters saved: "
            f"{total_characters:,}"
        )

        print(
            f"Low-score documents skipped: "
            f"{skipped_low_score:,}"
        )

        print(
            f"Empty documents skipped: "
            f"{skipped_empty:,}"
        )

        print(
            f"Output: "
            f"{self.output_path}"
        )


def main():

    output_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "math"
        / "openwebmath_sample.jsonl"
    )

    downloader = OpenWebMathDownloader(
        output_path=output_path,

        # Keep the first test small.
        target_characters=10_000_000,

        min_math_score=0.5,

        overwrite=True,
    )

    downloader.download()


if __name__ == "__main__":
    main()