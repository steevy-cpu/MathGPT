from pathlib import Path

from datasets import load_dataset

from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FineWebDownloader:
    """
    Downloads a small streamed sample of FineWeb-Edu
    and saves it using the MathGPT raw JSONL format.
    """

    def __init__(self, output_path, max_documents=10):
        self.output_path = Path(output_path)
        self.max_documents = max_documents

        self.writer = DataWriter(self.output_path)

    def load_stream(self):
        """
        Open FineWeb-Edu in streaming mode.
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
        Convert a FineWeb-Edu record into
        the MathGPT raw document format.
        """

        document = {
            # source_id belongs to MathGPT
            "id": f"fineweb_{index:08d}",
            "source": "fineweb_edu",
            "source_version": "sample-10BT",
            "url": example.get("url", ""),
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
                None,
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

        return document

    def download(self):
        """
        Stream documents and save them locally.
        """

        dataset = self.load_stream()

        print(
            f"Downloading {self.max_documents} "
            f"FineWeb-Edu documents..."
        )

        count = 0

        for example in dataset:
            document = self.convert_document(
                example,
                count,
            )

            self.writer.write_document(document)

            count += 1

            print(
                f"Saved document {count}"
            )

            if count >= self.max_documents:
                break

        print()
        print(
            f"Finished. Saved {count} documents."
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
        max_documents=10,
    )

    downloader.download()


if __name__ == "__main__":
    main()