from pathlib import Path

from datasets import load_dataset

from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WikipediaDownloader:

    def __init__(
        self,
        output_path,
        target_tokens=2_000_000,
        overwrite=True,
    ):
        self.output_path = Path(output_path)
        self.target_tokens = target_tokens
        self.overwrite = overwrite

        self.writer = DataWriter(self.output_path)

    def load_stream(self):
        """
        Stream the English Wikipedia snapshot.
        """

        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )

        return dataset

    def estimate_tokens(self, text):
        """
        Temporary token estimate.

        We do not have our custom BPE tokenizer yet,
        so we use a rough character-based estimate.

        Rough rule:
        about 4 characters per token.
        """

        if not text:
            return 0

        return max(1, len(text) // 4)

    def convert_document(self, example, index):
        """
        Convert Wikipedia format into
        our standard MathGPT raw format.
        """

        text = example.get("text", "")

        return {
            "id": f"wikipedia_{index:08d}",
            "source": "wikipedia",
            "source_version": "20231101.en",
            "url": example.get("url", ""),
            "license": "CC-BY-SA",
            "category": "general",

            "source_id": example.get(
                "id",
                "",
            ),

            "title": example.get(
                "title",
                "",
            ),

            "language": "en",

            "token_count_source": self.estimate_tokens(
                text
            ),

            "text": text,
        }

    def download(self):

        if self.overwrite:
            self.writer.clear()

        dataset = self.load_stream()

        total_tokens = 0
        total_documents = 0

        print()
        print("Starting Wikipedia download...")
        print(
            f"Target tokens: {self.target_tokens:,}"
        )
        print()

        for example in dataset:

            document = self.convert_document(
                example,
                total_documents,
            )

            text = document["text"]

            if not text.strip():
                continue

            document_tokens = document[
                "token_count_source"
            ]

            if document_tokens <= 0:
                continue

            self.writer.write_document(
                document
            )

            total_documents += 1
            total_tokens += document_tokens

            print(
                f"\rDocuments: {total_documents:,} | "
                f"Estimated tokens: {total_tokens:,} / "
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
            f"Estimated tokens: "
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
        / "wikipedia_sample.jsonl"
    )

    downloader = WikipediaDownloader(
        output_path=output_path,
        target_tokens=2_000_000,
        overwrite=True,
    )

    downloader.download()


if __name__ == "__main__":
    main()