import json
import sys
from pathlib import Path

from bpe_tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def analyze_file(tokenizer, file_path):
    total_documents = 0
    total_characters = 0
    total_byte_tokens = 0
    total_bpe_tokens = 0

    min_bpe_tokens = None
    max_bpe_tokens = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            document = json.loads(line)

            text = document.get(
                "text",
                "",
            )

            if not text:
                continue

            byte_ids = tokenizer.byte_tokenizer.encode(
                text
            )

            bpe_ids = tokenizer.encode(
                text
            )

            byte_count = len(byte_ids)
            bpe_count = len(bpe_ids)

            total_documents += 1
            total_characters += len(text)
            total_byte_tokens += byte_count
            total_bpe_tokens += bpe_count

            if min_bpe_tokens is None:
                min_bpe_tokens = bpe_count
            else:
                min_bpe_tokens = min(
                    min_bpe_tokens,
                    bpe_count,
                )

            max_bpe_tokens = max(
                max_bpe_tokens,
                bpe_count,
            )

    if total_documents == 0:
        return None

    compression_ratio = (
        1
        - (
            total_bpe_tokens
            / total_byte_tokens
        )
    )

    characters_per_bpe_token = (
        total_characters
        / total_bpe_tokens
    )

    average_bpe_tokens_per_document = (
        total_bpe_tokens
        / total_documents
    )

    return {
        "documents": total_documents,

        "characters": total_characters,

        "byte_tokens": total_byte_tokens,

        "bpe_tokens": total_bpe_tokens,

        "compression_ratio": compression_ratio,

        "characters_per_bpe_token": (
            characters_per_bpe_token
        ),

        "average_bpe_tokens_per_document": (
            average_bpe_tokens_per_document
        ),

        "min_bpe_tokens_per_document": (
            min_bpe_tokens
        ),

        "max_bpe_tokens_per_document": (
            max_bpe_tokens
        ),
    }


def print_stats(source_name, stats):
    print()
    print(source_name)
    print("-" * len(source_name))

    print(
        f"Documents: "
        f"{stats['documents']:,}"
    )

    print(
        f"Characters: "
        f"{stats['characters']:,}"
    )

    print(
        f"Byte tokens: "
        f"{stats['byte_tokens']:,}"
    )

    print(
        f"BPE tokens: "
        f"{stats['bpe_tokens']:,}"
    )

    print(
        f"Compression ratio: "
        f"{stats['compression_ratio']:.2%}"
    )

    print(
        f"Characters per BPE token: "
        f"{stats['characters_per_bpe_token']:.3f}"
    )

    print(
        f"Average BPE tokens/document: "
        f"{stats['average_bpe_tokens_per_document']:.2f}"
    )

    print(
        f"Smallest document: "
        f"{stats['min_bpe_tokens_per_document']:,} "
        f"BPE tokens"
    )

    print(
        f"Largest document: "
        f"{stats['max_bpe_tokens_per_document']:,} "
        f"BPE tokens"
    )


def main():

    artifact_directory = (
        PROJECT_ROOT
        / "tokenizer"
        / "artifacts"
    )

    tokenizer = BPETokenizer.from_files(
        config_path=(
            artifact_directory
            / "tokenizer_config.json"
        ),
        merges_path=(
            artifact_directory
            / "merges.json"
        ),
        vocab_path=(
            artifact_directory
            / "vocab.json"
        ),
    )

    datasets = {
        "FineWeb-Edu": (
            PROJECT_ROOT
            / "data"
            / "cleaned"
            / "general"
            / "fineweb_edu_train.jsonl"
        ),

        "Wikipedia": (
            PROJECT_ROOT
            / "data"
            / "cleaned"
            / "general"
            / "wikipedia_train.jsonl"
        ),

        "Project Gutenberg": (
            PROJECT_ROOT
            / "data"
            / "cleaned"
            / "general"
            / "gutenberg_train.jsonl"
        ),
    }

    print()
    print("MathGPT Tokenizer Corpus Statistics")
    print("===================================")

    combined = {
        "documents": 0,
        "characters": 0,
        "byte_tokens": 0,
        "bpe_tokens": 0,
    }

    for source_name, file_path in datasets.items():

        if not file_path.exists():
            print(
                f"\nWarning: missing dataset: "
                f"{file_path}"
            )
            continue

        stats = analyze_file(
            tokenizer,
            file_path,
        )

        if stats is None:
            continue

        print_stats(
            source_name,
            stats,
        )

        combined["documents"] += (
            stats["documents"]
        )

        combined["characters"] += (
            stats["characters"]
        )

        combined["byte_tokens"] += (
            stats["byte_tokens"]
        )

        combined["bpe_tokens"] += (
            stats["bpe_tokens"]
        )

    if combined["documents"] > 0:

        overall_compression = (
            1
            - (
                combined["bpe_tokens"]
                / combined["byte_tokens"]
            )
        )

        overall_characters_per_token = (
            combined["characters"]
            / combined["bpe_tokens"]
        )

        print()
        print("Combined")
        print("--------")

        print(
            f"Documents: "
            f"{combined['documents']:,}"
        )

        print(
            f"Characters: "
            f"{combined['characters']:,}"
        )

        print(
            f"Byte tokens: "
            f"{combined['byte_tokens']:,}"
        )

        print(
            f"BPE tokens: "
            f"{combined['bpe_tokens']:,}"
        )

        print(
            f"Compression ratio: "
            f"{overall_compression:.2%}"
        )

        print(
            f"Characters per BPE token: "
            f"{overall_characters_per_token:.3f}"
        )


if __name__ == "__main__":
    main()