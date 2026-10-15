import json
from pathlib import Path


class DataWriter:
    """
    Saves documents to a JSONL file.

    Each line in the file represents one document.
    """

    def __init__(self, output_path):
        self.output_path = Path(output_path)

        # Make sure the parent folder exists.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_document(self, document):
        """
        Append one document to the JSONL file.
        """

        with open(self.output_path, "a", encoding="utf-8") as file:
            json.dump(document, file, ensure_ascii=False)
            file.write("\n")

    def write_documents(self, documents):
        """
        Save multiple documents.
        """

        for document in documents:
            self.write_document(document)