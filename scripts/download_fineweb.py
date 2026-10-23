from pathlib import Path

from datasets import load_dataset

from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FineWebDownloader:

    def __init__(
        self,
        output_path,
        target_tokens=7_000_000,
        overwrite=True,
    ):

        self.output_path = Path(output_path)
        self.target_tokens = target_tokens
        self.overwrite = overwrite

        self.writer = DataWriter(self.output_path)

    def load_stream(self):
        """
        Stream FineWeb-Edu instead of downloading
        the full dataset.
        """

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        return dataset

    def convert_document(self, example, index):
        """
        Convert FineWeb format into our
        standard MathGPT raw format.
        """

        return {
            "id": f"fineweb_{index:08d}",

            "source": "fineweb_edu",

            "source_version": "sample-10BT",

            "url": example.get(
                "url",
                "",
            ),

            "license": "ODC-By-1.0",

            "category": "general",

            "source_id": example.get(
                "id",
                "",
            ),

            "language": example.get(
                "language",
                "unknown",
            ),

            "token_count_source": example.get(
                "token_count",
                0,
            ),

            "quality_score": example.get(
                "score",
                None,
            ),

            "text": example.get(
                "text",
                "",
            ),
        }

    def download(self):

        if self.overwrite:
            self.writer.clear()

        dataset = self.load_stream()

        total_tokens = 0
        total_documents = 0

        print()
        print("Starting FineWeb-Edu download...")
        print(
            f"Target tokens: {self.target_tokens:,}"
        )
        print()

        for example in dataset:

            document = self.convert_document(
                example,
                total_documents,
            )

            document_tokens = document[
                "token_count_source"
            ]

            # Skip records with missing or invalid
            # source token counts.
            if not isinstance(document_tokens, int):
                continue

            if document_tokens <= 0:
                continue

            # Skip empty documents.
            if not document["text"].strip():
                continue

            self.writer.write_document(
                document
            )

            total_documents += 1
            total_tokens += document_tokens

            print(
                f"\rDocuments: {total_documents:,} | "
                f"Tokens: {total_tokens:,} / "
                f"{self.target_tokens:,}",
                end="",
            )

            if total_tokens >= self.target_tokens:
                break

        print()
        print()
        print("Download complete.")
        print(
            f"Documents saved: "
            f"{total_documents:,}"
        )
        print(
            f"Estimated source tokens: "
            f"{total_tokens:,}"
        )
        print(
            f"Output: {self.output_path}"
        )


def main():

    output_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "general"
        / "fineweb_edu_sample.jsonl"
    )

    downloader = FineWebDownloader(
        output_path=output_path,

        # Tiny test first.
        target_tokens=7_000_000,

        overwrite=True,
    )

    downloader.download()


if __name__ == "__main__":
    main()