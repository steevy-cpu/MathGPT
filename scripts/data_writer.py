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

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def validate_document(self, document):
        """
        Make sure required fields exist.
        """

        for field in self.REQUIRED_FIELDS:
            if field not in document:
                raise ValueError(
                    f"Document is missing required field: {field}"
                )

    def clear(self):
        """
        Delete the existing output file if it exists.
        """

        if self.output_path.exists():
            self.output_path.unlink()

    def write_document(self, document):
        """
        Validate and append one document.
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