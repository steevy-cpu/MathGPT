import json
from pathlib import Path


class DataWriter:
    """
    Saves documents to a JSONL file.

    Each line represents one document.
    """

    REQUIRED_FIELDS = [
        "id",
        "source",
        "source_version",
        "url",
        "license",
        "category",
        "text",
    ]

    def __init__(self, output_path):
        self.output_path = Path(output_path)

        # Create the parent folders if they do not exist.
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def validate_document(self, document):
        """
        Check that a document contains all required fields.
        """

        for field in self.REQUIRED_FIELDS:
            if field not in document:
                raise ValueError(
                    f"Document is missing required field: {field}"
                )

    def write_document(self, document):
        """
        Validate and append one document to the JSONL file.
        """

        self.validate_document(document)

        with open(
            self.output_path,
            "a",
            encoding="utf-8",
        ) as file:
            json.dump(
                document,
                file,
                ensure_ascii=False,
            )

            file.write("\n")

    def write_documents(self, documents):
        """
        Save multiple documents.
        """

        for document in documents:
            self.write_document(document)