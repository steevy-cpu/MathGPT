import json
import random
import sys
from pathlib import Path

from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DatasetSplitter:
    """
    Split a JSONL dataset into training and validation sets.

    Splitting happens at the document level so that one document
    cannot appear in both training and validation.
    """

    def __init__(
        self,
        input_path,
        train_output_path,
        validation_output_path,
        config_path,
    ):
        self.input_path = Path(input_path)

        self.train_output_path = Path(
            train_output_path
        )

        self.validation_output_path = Path(
            validation_output_path
        )

        self.config_path = Path(config_path)

        self.rules = self.load_rules()

        self.train_writer = DataWriter(
            self.train_output_path
        )

        self.validation_writer = DataWriter(
            self.validation_output_path
        )

    def load_rules(self):
        """
        Load split settings from JSON.
        """

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        return config

    def load_documents(self):
        """
        Load all documents from the JSONL dataset.
        """

        documents = []

        with open(
            self.input_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                document = json.loads(line)
                documents.append(document)

        return documents

    def split(self):
        """
        Shuffle deterministically and create
        training and validation datasets.
        """

        documents = self.load_documents()

        total_documents = len(documents)

        if total_documents < 2:
            raise ValueError(
                "Dataset must contain at least "
                "2 documents to create a split."
            )

        validation_ratio = self.rules.get(
            "validation_ratio",
            0.05,
        )

        random_seed = self.rules.get(
            "random_seed",
            42,
        )

        shuffle_before_split = self.rules.get(
            "shuffle_before_split",
            True,
        )

        if shuffle_before_split:
            random_generator = random.Random(
                random_seed
            )

            random_generator.shuffle(
                documents
            )

        validation_count = round(
            total_documents * validation_ratio
        )

        # Always keep at least one validation document.
        validation_count = max(
            1,
            validation_count,
        )

        # Never allow validation to consume the whole dataset.
        validation_count = min(
            validation_count,
            total_documents - 1,
        )

        validation_documents = documents[
            :validation_count
        ]

        train_documents = documents[
            validation_count:
        ]

        self.train_writer.clear()
        self.validation_writer.clear()

        self.train_writer.write_documents(
            train_documents
        )

        self.validation_writer.write_documents(
            validation_documents
        )
        
        validation_documents = documents[:validation_count]
        train_documents = documents[validation_count:]
        
        validation_documents = documents[
            :validation_count]

        train_documents = documents[
            validation_count:
        ]

        # Calculate estimated token counts for each split.
        train_tokens = sum(
            document.get(
                "token_count_source",
                0,
            )
            for document in train_documents
        )

        validation_tokens = sum(
            document.get(
                "token_count_source",
                0,
            )
            for document in validation_documents
        )

        total_tokens = (
            train_tokens
            + validation_tokens
        )

        self.train_writer.clear()
        self.validation_writer.clear()

        self.train_writer.write_documents(
            train_documents
        )

        self.validation_writer.write_documents(
            validation_documents
        )

        print()
        print("Dataset split complete.")
        print("-----------------------")

        print(
            f"Total documents: "
            f"{total_documents:,}"
        )

        print(
            f"Training documents: "
            f"{len(train_documents):,}"
        )

        print(
            f"Validation documents: "
            f"{len(validation_documents):,}"
        )

        print(
            f"Configured validation ratio: "
            f"{validation_ratio:.2%}"
        )

        actual_validation_ratio = (
            len(validation_documents)
            / total_documents
        )

        print(
            f"Actual validation ratio: "
            f"{actual_validation_ratio:.2%}"
        )
        
        print()
        
        print("Token Estimates")
        print("---------------")

        print(
            f"Training tokens: "
            f"{train_tokens:,}"
        )

        print(
            f"Validation tokens: "
            f"{validation_tokens:,}"
        )

        if total_tokens > 0:
            validation_token_ratio = (
                validation_tokens
                / total_tokens
            )

            print(
                f"Validation token ratio: "
                f"{validation_token_ratio:.2%}"
            )
            

        print()

        print(
            f"Training output: "
            f"{self.train_output_path}"
        )

        print(
            f"Validation output: "
            f"{self.validation_output_path}"
        )
        
        
        


def main():

    if len(sys.argv) != 4:

        print("Usage:")

        print(
            "python scripts/split_dataset.py "
            "<input.jsonl> "
            "<train_output.jsonl> "
            "<validation_output.jsonl>"
        )

        return

    input_path = Path(
        sys.argv[1]
    )

    train_output_path = Path(
        sys.argv[2]
    )

    validation_output_path = Path(
        sys.argv[3]
    )

    if not input_path.is_absolute():
        input_path = (
            PROJECT_ROOT / input_path
        )

    if not train_output_path.is_absolute():
        train_output_path = (
            PROJECT_ROOT / train_output_path
        )

    if not validation_output_path.is_absolute():
        validation_output_path = (
            PROJECT_ROOT
            / validation_output_path
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
        / "split_rules_v0.1.json"
    )

    splitter = DatasetSplitter(
        input_path=input_path,
        train_output_path=train_output_path,
        validation_output_path=validation_output_path,
        config_path=config_path,
    )

    splitter.split()


if __name__ == "__main__":
    main()