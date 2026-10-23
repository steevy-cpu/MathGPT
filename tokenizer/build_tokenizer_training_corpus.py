import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42


# ============================================================
# Corpus targets
# ============================================================

TOTAL_TARGET = 10_000_000

GENERAL_TARGETS = {
    "fineweb_edu": 5_250_000,
    "wikipedia": 1_050_000,
    "gutenberg": 700_000,
}

MATH_TARGET = 3_000_000


# ============================================================
# Quality / size rules
# ============================================================

MATH_SCORE_THRESHOLD = 0.70

# General documents can be broken into chunks instead
# of throwing away the rest of large documents.
GENERAL_CHUNK_SIZE = 30_000

# OpenWebMath is handled differently:
# one webpage cannot contribute more than this amount.
MAX_MATH_CHARS_PER_DOCUMENT = 20_000


# ============================================================
# Input paths
# ============================================================

GENERAL_FILES = {
    "fineweb_edu": (
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "fineweb_edu_train.jsonl"
    ),

    "wikipedia": (
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "wikipedia_train.jsonl"
    ),

    "gutenberg": (
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "general"
        / "gutenberg_train.jsonl"
    ),
}


MATH_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "math"
    / "openwebmath_sample.jsonl"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "tokenizer_training_corpus.txt"
)


REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_training_corpus_report.json"
)


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):
    """
    Load JSON objects from a JSONL file.
    """

    documents = []

    if not path.exists():
        print(
            f"WARNING: file not found: {path}"
        )
        return documents

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            documents.append(
                json.loads(line)
            )

    return documents


def chunk_text(text, chunk_size):
    """
    Break a very large document into smaller pieces.

    This lets us use large Wikipedia pages/books without
    allowing one enormous sequence to act as one sample.
    """

    chunks = []

    for start in range(
        0,
        len(text),
        chunk_size,
    ):

        chunk = text[
            start:start + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def collect_to_target(
    pieces,
    target_characters,
    random_generator,
):
    """
    Shuffle text pieces and collect exactly as many
    characters as needed for the requested target.
    """

    pieces = pieces.copy()

    random_generator.shuffle(
        pieces
    )

    selected = []

    characters_collected = 0

    for text in pieces:

        remaining = (
            target_characters
            - characters_collected
        )

        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining]

        if not text:
            continue

        selected.append(text)

        characters_collected += len(text)

    return (
        selected,
        characters_collected,
    )


# ============================================================
# General corpus
# ============================================================

def prepare_general_source(
    source_name,
    path,
):
    """
    Load one general source and convert its documents
    into manageable chunks.
    """

    documents = load_jsonl(path)

    pieces = []

    raw_characters = 0

    for document in documents:

        text = document.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        raw_characters += len(text)

        chunks = chunk_text(
            text,
            GENERAL_CHUNK_SIZE,
        )

        pieces.extend(chunks)

    print(
        f"{source_name}: "
        f"{len(documents):,} documents | "
        f"{raw_characters:,} characters | "
        f"{len(pieces):,} chunks"
    )

    return pieces


# ============================================================
# Math corpus
# ============================================================

def prepare_math_documents():
    """
    Load OpenWebMath and enforce our math-quality policy.
    """

    documents = load_jsonl(
        MATH_FILE
    )

    accepted = []

    rejected_low_score = 0
    rejected_empty = 0

    original_accepted_characters = 0
    usable_characters = 0

    for document in documents:

        text = document.get(
            "text",
            ""
        ).strip()

        math_score = document.get(
            "math_score"
        )

        if not text:
            rejected_empty += 1
            continue

        if (
            math_score is None
            or math_score < MATH_SCORE_THRESHOLD
        ):
            rejected_low_score += 1
            continue

        original_accepted_characters += len(
            text
        )

        # Unlike books, we intentionally limit how much
        # one web page can influence BPE merge statistics.
        text = text[
            :MAX_MATH_CHARS_PER_DOCUMENT
        ]

        accepted.append(text)

        usable_characters += len(text)

    print()
    print("OpenWebMath filtering")
    print("---------------------")

    print(
        f"Documents read: "
        f"{len(documents):,}"
    )

    print(
        f"Accepted documents: "
        f"{len(accepted):,}"
    )

    print(
        f"Rejected low-score: "
        f"{rejected_low_score:,}"
    )

    print(
        f"Rejected empty: "
        f"{rejected_empty:,}"
    )

    print(
        f"Accepted characters before cap: "
        f"{original_accepted_characters:,}"
    )

    print(
        f"Usable characters after cap: "
        f"{usable_characters:,}"
    )

    return accepted


# ============================================================
# Build corpus
# ============================================================

def build_corpus():

    rng = random.Random(
        RANDOM_SEED
    )

    print()
    print("Preparing general corpus")
    print("========================")

    selected_items = []

    report = {
        "seed": RANDOM_SEED,
        "target_total_characters": TOTAL_TARGET,
        "sources": {},
    }

    total_general_characters = 0

    for source_name, target in (
        GENERAL_TARGETS.items()
    ):

        path = GENERAL_FILES[
            source_name
        ]

        pieces = prepare_general_source(
            source_name,
            path,
        )

        selected, collected = (
            collect_to_target(
                pieces,
                target,
                rng,
            )
        )

        for text in selected:

            selected_items.append(
                {
                    "source": source_name,
                    "text": text,
                }
            )

        total_general_characters += (
            collected
        )

        report["sources"][
            source_name
        ] = {
            "target_characters": target,
            "collected_characters": collected,
            "pieces_selected": len(
                selected
            ),
        }

    print()
    print("Preparing math corpus")
    print("=====================")

    math_documents = (
        prepare_math_documents()
    )

    math_selected, math_characters = (
        collect_to_target(
            math_documents,
            MATH_TARGET,
            rng,
        )
    )

    for text in math_selected:

        selected_items.append(
            {
                "source": "openwebmath",
                "text": text,
            }
        )

    report["sources"][
        "openwebmath"
    ] = {
        "target_characters": MATH_TARGET,
        "collected_characters": math_characters,
        "documents_selected": len(
            math_selected
        ),
        "minimum_math_score": (
            MATH_SCORE_THRESHOLD
        ),
        "max_characters_per_document": (
            MAX_MATH_CHARS_PER_DOCUMENT
        ),
    }

    # Shuffle source pieces together so the BPE trainer
    # does not see one giant block of each source.
    rng.shuffle(
        selected_items
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        for item in selected_items:

            file.write(
                item["text"]
            )

            file.write(
                "\n\n"
            )

    total_characters = (
        total_general_characters
        + math_characters
    )

    report[
        "general_characters"
    ] = total_general_characters

    report[
        "math_characters"
    ] = math_characters

    report[
        "total_characters"
    ] = total_characters

    if total_characters > 0:

        report[
            "general_ratio"
        ] = (
            total_general_characters
            / total_characters
        )

        report[
            "math_ratio"
        ] = (
            math_characters
            / total_characters
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()
    print(
        "Tokenizer Training Corpus Report"
    )
    print(
        "================================"
    )

    print()

    for source_name in [
        "fineweb_edu",
        "wikipedia",
        "gutenberg",
        "openwebmath",
    ]:

        stats = report[
            "sources"
        ][source_name]

        target = stats[
            "target_characters"
        ]

        collected = stats[
            "collected_characters"
        ]

        print(
            f"{source_name:15} "
            f"{collected:>10,} / "
            f"{target:>10,}"
        )

    print()

    print(
        f"General characters: "
        f"{total_general_characters:,}"
    )

    print(
        f"Math characters: "
        f"{math_characters:,}"
    )

    print(
        f"Total characters: "
        f"{total_characters:,}"
    )

    if total_characters > 0:

        print(
            f"General ratio: "
            f"{total_general_characters / total_characters:.2%}"
        )

        print(
            f"Math ratio: "
            f"{math_characters / total_characters:.2%}"
        )

    print()

    print(
        f"Corpus output: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Report output: "
        f"{REPORT_PATH}"
    )

    print()

    all_targets_met = True

    for stats in (
        report["sources"].values()
    ):

        if (
            stats["collected_characters"]
            < stats["target_characters"]
        ):
            all_targets_met = False

    if all_targets_met:

        print(
            "SUCCESS: All tokenizer corpus "
            "targets were reached."
        )

    else:

        print(
            "WARNING: One or more sources "
            "did not reach their target."
        )


def main():
    build_corpus()


if __name__ == "__main__":
    main()