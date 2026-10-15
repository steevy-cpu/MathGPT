from pathlib import Path
from data_writer import DataWriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

test_output = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "general"
    / "test_documents.jsonl"
)


documents = [
    {
        "id": "test_001",
        "source": "manual_test",
        "source_version": "v0.1",
        "url": "",
        "license": "test_only",
        "category": "general",
        "text": (
            "Linear algebra studies vectors, "
            "matrices, and transformations."
        ),
    },

    {
        "id": "test_002",
        "source": "manual_test",
        "source_version": "v0.1",
        "url": "",
        "license": "test_only",
        "category": "general",
        "text": (
            "Calculus studies change using "
            "derivatives and integrals."
        ),
    },
]


writer = DataWriter(test_output)

writer.write_documents(documents)

print("Saved test documents to:")
print(test_output)