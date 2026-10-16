from pathlib import Path

from datasets import load_dataset

from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GutenbergDownloader:

    def __init__(
        self,
        output_path,
        target_tokens=100000,
        max_document_tokens=100000,
        overshoot_tokens=100000,
        overwrite=True,
    ):
        self.output_path = Path(output_path)
        self.target_tokens = target_tokens
        self.max_document_tokens = max_document_tokens
        self.overshoot_tokens = overshoot_tokens
        self.overwrite = overwrite

        self.writer = DataWriter(self.output_path)

    def load_stream(self):
        """
        Stream the Common Pile Project Gutenberg dataset.
        """

        dataset = load_dataset(
            "common-pile/project_gutenberg",
            split="train",
            streaming=True,
        )

        return dataset

    def estimate_tokens(self, text):
        """
        Temporary token estimate.

        Until our custom BPE tokenizer exists,
        estimate roughly 1 token per 4 characters.
        """

        if not text:
            return 0

        return max(1, len(text) // 4)

    def convert_document(self, example, index):
        """
        Convert Gutenberg format into
        the standard MathGPT raw document format.
        """

        text = example.get("text", "")
        metadata = example.get("metadata", {})

        return {
            "id": f"gutenberg_{index:08d}",

            "source": "project_gutenberg",

            "source_version": "common-pile",

            "url": metadata.get(
                "url",
                "",
            ),

            "license": metadata.get(
                "license",
                "unknown",
            ),

            "category": "general",

            "source_id": metadata.get(
                "url",
                "",
            ),

            "title": metadata.get(
                "title",
                "",
            ),

            "language": metadata.get(
                "language",
                "unknown",
            ),

            "added": example.get(
                "added",
                "",
            ),

            "token_count_source": self.estimate_tokens(
                text
            ),

            "text": text,
        }

    def download(self):
        
        total_tokens = 0
        total_documents = 0

        oversized_documents = 0
        documents_that_did_not_fit = 0
        
    
        print(
            f"Allowed overshoot: {self.overshoot_tokens:,}"
        )
                

        if self.overwrite:
            self.writer.clear()

        dataset = self.load_stream()

        total_tokens = 0
        total_documents = 0

        print()
        print("Starting Project Gutenberg download...")
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

            # Keep only English documents.
            if document["language"] != "en":
                continue

            # Keep only documents marked Public Domain.
            if document["license"] != "Public Domain":
                continue

            document_tokens = document[
                "token_count_source"
            ]

            if document_tokens <= 0:
                continue
            
            # Skip extremely large books for this acquisition build.
            if document_tokens > self.max_document_tokens:
                oversized_documents += 1

                continue
            
            
            maximum_allowed_tokens = (
                self.target_tokens
                + self.overshoot_tokens
            )

            # Do not allow the final corpus to exceed
            # the target plus the allowed overshoot.
            if (
                total_tokens + document_tokens
                > maximum_allowed_tokens
            ):
                continue
            

            self.writer.write_document(
                document
            )

            total_documents += 1
            total_tokens += document_tokens
            
            # Once we have reached or passed the target,
            # the acquisition test is complete.
            if total_tokens >= self.target_tokens:
                break

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

        print(
            f"Oversized documents skipped: "
            f"{oversized_documents:,}"
        )

        print(
            f"Documents skipped because they did not fit: "
            f"{documents_that_did_not_fit:,}"
        )
        
        overshoot = max(
            0,
            total_tokens - self.target_tokens,
        )

        print(
            f"Overshoot: {overshoot:,} tokens"
        )

def main():

    output_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "general"
        / "gutenberg_sample.jsonl"
    )

    downloader = GutenbergDownloader(
        output_path=output_path,
        target_tokens=100000,
        max_document_tokens=100000,
        overshoot_tokens=100000,
        overwrite=True,
    )

    downloader.download()


if __name__ == "__main__":
    main()