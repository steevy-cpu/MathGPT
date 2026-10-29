import json
import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "tokenized"
    / "tokenizer_training_corpus_50m_v0.2.txt"
)
TEMP_OUTPUT_PATH = Path(str(OUTPUT_PATH) + ".tmp")

REPORT_PATH = (
    PROJECT_ROOT / "logs"
    / "tokenizer_50m_corpus_report.json"
)
TEMP_REPORT_PATH = Path(str(REPORT_PATH) + ".tmp")

TARGET_TOTAL_CHARS = 50_000_000
GENERAL_TARGET_CHARS = 35_000_000
MATH_TARGET_CHARS = 15_000_000

SOURCE_CONFIG = {
    "fineweb_edu": {
        "raw": (
            PROJECT_ROOT / "data" / "raw" / "general"
            / "fineweb_edu_50m_source.jsonl"
        ),
        "cleaned": (
            PROJECT_ROOT / "data" / "cleaned" / "general"
            / "fineweb_edu_train.jsonl"
        ),
        "target_chars": 29_000_000,
        "max_doc_chars": 30_000,
        "allow_chunks": False,
    },
    "wikipedia": {
        "raw": (
            PROJECT_ROOT / "data" / "raw" / "general"
            / "wikipedia_50m_source.jsonl"
        ),
        "cleaned": (
            PROJECT_ROOT / "data" / "cleaned" / "general"
            / "wikipedia_train.jsonl"
        ),
        "target_chars": 5_400_000,
        "max_doc_chars": 30_000,
        "allow_chunks": False,
    },
    "gutenberg": {
        "raw": (
            PROJECT_ROOT / "data" / "raw" / "general"
            / "gutenberg_50m_source.jsonl"
        ),
        "cleaned": (
            PROJECT_ROOT / "data" / "cleaned" / "general"
            / "gutenberg_train.jsonl"
        ),
        "target_chars": 600_000,
        "max_doc_chars": 30_000,
        "allow_chunks": True,
    },
    "openwebmath": {
        "raw": (
            PROJECT_ROOT / "data" / "raw" / "math"
            / "openwebmath_50m_source.jsonl"
        ),
        "target_chars": 15_000_000,
        "max_doc_chars": 20_000,
        "allow_chunks": False,
    },
}

GENERAL_SOURCES = ("fineweb_edu", "wikipedia", "gutenberg")
MATH_SOURCES = ("openwebmath",)

HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", flags=re.DOTALL)
SCRIPT_PATTERN = re.compile(
    r"<script\b[^>]*>.*?</script>",
    flags=(re.IGNORECASE | re.DOTALL),
)
STYLE_PATTERN = re.compile(
    r"<style\b[^>]*>.*?</style>",
    flags=(re.IGNORECASE | re.DOTALL),
)
SEPARATOR_LINE_PATTERN = re.compile(r"^[\s_\-=*#~]{6,}$")
LIKE_DISLIKE_PATTERN = re.compile(
    r"^\s*\d+\s+like\s+\d+\s+dislike\s*$",
    flags=re.IGNORECASE,
)
VIEWS_PATTERN = re.compile(
    r"^\s*\|?\s*\d+\s+views?\s*$",
    flags=re.IGNORECASE,
)
REPLACEMENT_CHARACTER_PATTERN = re.compile(r"\ufffd+")


def remove_control_characters(text):
    """Remove control/format chars while preserving newlines/tabs."""
    cleaned = []

    for character in text:
        if character in ("\n", "\t"):
            cleaned.append(character)
            continue

        if unicodedata.category(character).startswith("C"):
            continue

        cleaned.append(character)

    return "".join(cleaned)


def is_noise_line(line):
    """Conservatively detect obvious navigation / UI noise."""
    stripped = line.strip()

    if not stripped:
        return False

    lower = stripped.lower()

    exact_noise = {
        "back to top",
        "back to issues page",
        "show tags",
        "### show tags",
        "# show tags",
        "permalink",
        "#permalink",
    }

    if lower in exact_noise:
        return True

    if "permalink" in lower and len(stripped) <= 200:
        return True

    if LIKE_DISLIKE_PATTERN.fullmatch(stripped):
        return True

    if VIEWS_PATTERN.fullmatch(stripped):
        return True

    if SEPARATOR_LINE_PATTERN.fullmatch(stripped):
        return True

    return False


def clean_text(text):
    """
    Conservative tokenizer-corpus cleaner.

    Preserve LaTeX, Unicode math, equations, punctuation,
    prose, headings, citations, and code-like math notation.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # NFC keeps canonical Unicode normalization without the
    # compatibility folding that NFKC can apply to math text.
    text = unicodedata.normalize("NFC", text)

    # U+FFFD usually means text was already decoded incorrectly
    # upstream. Replace runs with a space to avoid joining words.
    text = REPLACEMENT_CHARACTER_PATTERN.sub(" ", text)

    text = HTML_COMMENT_PATTERN.sub("", text)
    text = SCRIPT_PATTERN.sub("", text)
    text = STYLE_PATTERN.sub("", text)

    text = remove_control_characters(text)

    cleaned_lines = []

    for line in text.split("\n"):
        # Keep leading indentation; remove trailing whitespace.
        line = line.rstrip()

        if is_noise_line(line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse very large blank-line runs.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def choose_source_path(config):
    """
    Prefer the dedicated v0.2 raw source file.

    Fall back to the older Stage-4 cleaned file only when the
    dedicated raw source is unavailable.
    """
    raw_path = config.get("raw")

    if raw_path is not None and raw_path.exists():
        return raw_path, "raw"

    cleaned_path = config.get("cleaned")

    if cleaned_path is not None and cleaned_path.exists():
        return cleaned_path, "cleaned"

    raise FileNotFoundError(
        "No usable source file found. "
        f"Checked raw={raw_path} and cleaned={cleaned_path}"
    )


def validate_source_configuration():
    """Validate targets and resolve every source before writing."""
    expected_total = sum(
        config["target_chars"]
        for config in SOURCE_CONFIG.values()
    )

    general_total = sum(
        SOURCE_CONFIG[name]["target_chars"]
        for name in GENERAL_SOURCES
    )

    math_total = sum(
        SOURCE_CONFIG[name]["target_chars"]
        for name in MATH_SOURCES
    )

    if expected_total != TARGET_TOTAL_CHARS:
        raise ValueError(
            "Source targets do not add to exactly "
            f"{TARGET_TOTAL_CHARS:,}. Got {expected_total:,}."
        )

    if general_total != GENERAL_TARGET_CHARS:
        raise ValueError(
            "General targets do not add to exactly "
            f"{GENERAL_TARGET_CHARS:,}. Got {general_total:,}."
        )

    if math_total != MATH_TARGET_CHARS:
        raise ValueError(
            "Math targets do not add to exactly "
            f"{MATH_TARGET_CHARS:,}. Got {math_total:,}."
        )

    resolved_sources = {}

    for source_name, config in SOURCE_CONFIG.items():
        source_path, source_type = choose_source_path(config)
        resolved_sources[source_name] = {
            "path": source_path,
            "type": source_type,
        }

    return expected_total, general_total, math_total, resolved_sources


def generate_chunks(text, max_chars, allow_chunks=False):
    """
    Yield usable pieces from a document.

    Normal sources contribute at most max_chars once.
    Gutenberg can yield multiple max_chars chunks.
    """
    if not text:
        return

    if not allow_chunks:
        yield text[:max_chars]
        return

    start = 0

    while start < len(text):
        chunk = text[start:start + max_chars]

        if chunk.strip():
            yield chunk

        start += max_chars


def add_source(
    source_name,
    config,
    output_file,
    corpus_chars_written,
):
    """Add exactly target_chars characters from one source."""
    source_path, source_type = choose_source_path(config)

    target_chars = config["target_chars"]
    max_doc_chars = config["max_doc_chars"]
    allow_chunks = config.get("allow_chunks", False)

    print()
    print("=" * 65)
    print(f"Adding source: {source_name}")
    print("=" * 65)
    print(f"Input type: {source_type}")
    print(f"Input path: {source_path}")
    print(f"Target chars: {target_chars:,}")

    documents_read = 0
    documents_used = 0
    chunks_used = 0

    raw_chars_seen = 0
    cleaned_chars_seen = 0
    replacement_chars_seen = 0

    source_chars_written = 0

    with open(source_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if source_chars_written >= target_chars:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            documents_read += 1

            text = record.get("text", "")

            if not isinstance(text, str):
                continue

            raw_chars_seen += len(text)
            replacement_chars_seen += text.count("\ufffd")

            cleaned = clean_text(text)

            if not cleaned:
                continue

            cleaned_chars_seen += len(cleaned)
            document_contributed = False

            for chunk in generate_chunks(
                cleaned,
                max_doc_chars,
                allow_chunks=allow_chunks,
            ):
                if source_chars_written >= target_chars:
                    break

                separator = ""

                if corpus_chars_written > 0:
                    separator = "\n\n"

                piece = separator + chunk

                remaining = (
                    target_chars
                    - source_chars_written
                )

                if len(piece) > remaining:
                    piece = piece[:remaining]

                output_file.write(piece)

                written = len(piece)

                source_chars_written += written
                corpus_chars_written += written

                chunks_used += 1
                document_contributed = True

            if document_contributed:
                documents_used += 1

            if documents_read % 1000 == 0:
                percentage = (
                    source_chars_written
                    / target_chars
                    * 100
                )

                print(
                    f"Documents read: {documents_read:,} | "
                    f"Written: {source_chars_written:,} "
                    f"({percentage:.2f}%)"
                )

    if source_chars_written != target_chars:
        raise RuntimeError(
            f"{source_name} did not contain enough usable text.\n"
            f"Needed: {target_chars:,}\n"
            f"Got:    {source_chars_written:,}"
        )

    removed_chars = (
        raw_chars_seen
        - cleaned_chars_seen
    )

    print()
    print(f"{source_name} complete")
    print(f"Documents read: {documents_read:,}")
    print(f"Documents used: {documents_used:,}")
    print(f"Chunks used: {chunks_used:,}")
    print(f"Raw chars examined: {raw_chars_seen:,}")
    print(f"Cleaned chars examined: {cleaned_chars_seen:,}")
    print(f"Cleaning removed: {removed_chars:,}")
    print(
        "Replacement chars found upstream: "
        f"{replacement_chars_seen:,}"
    )
    print(f"Characters written: {source_chars_written:,}")

    report = {
        "source": source_name,
        "input_path": str(source_path),
        "input_type": source_type,
        "target_chars": target_chars,
        "characters_written": source_chars_written,
        "documents_read": documents_read,
        "documents_used": documents_used,
        "chunks_used": chunks_used,
        "raw_chars_examined": raw_chars_seen,
        "cleaned_chars_examined": cleaned_chars_seen,
        "cleaning_removed_chars": removed_chars,
        "replacement_characters_found_upstream": (
            replacement_chars_seen
        ),
    }

    return corpus_chars_written, report


def inspect_output_file(path):
    """Verify the completed corpus without loading all of it at once."""
    total_chars = 0
    replacement_chars = 0
    null_chars = 0

    with open(path, "r", encoding="utf-8") as file:
        while True:
            block = file.read(1_000_000)

            if not block:
                break

            total_chars += len(block)
            replacement_chars += block.count("\ufffd")
            null_chars += block.count("\x00")

    return {
        "characters": total_chars,
        "replacement_characters": replacement_chars,
        "null_characters": null_chars,
        "utf8_bytes": path.stat().st_size,
    }


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale temporary files from an interrupted build.
    if TEMP_OUTPUT_PATH.exists():
        TEMP_OUTPUT_PATH.unlink()

    if TEMP_REPORT_PATH.exists():
        TEMP_REPORT_PATH.unlink()

    (
        expected_total,
        general_total,
        math_total,
        resolved_sources,
    ) = validate_source_configuration()

    print()
    print("=" * 65)
    print("BUILDING TOKENIZER 50M v0.2 CORPUS")
    print("=" * 65)
    print()
    print(f"Target total: {expected_total:,} characters")
    print(f"General target: {general_total:,} characters")
    print(f"Math target: {math_total:,} characters")

    print()
    print("Resolved source files")
    print("---------------------")

    for source_name, info in resolved_sources.items():
        print(
            f"{source_name:15} "
            f"[{info['type']}] "
            f"{info['path']}"
        )

    reports = {}
    corpus_chars_written = 0

    try:
        # Build to a temporary file so a failed rebuild does not
        # overwrite the last known-good final corpus.
        with open(
            TEMP_OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as output_file:
            for source_name, config in SOURCE_CONFIG.items():
                (
                    corpus_chars_written,
                    source_report,
                ) = add_source(
                    source_name=source_name,
                    config=config,
                    output_file=output_file,
                    corpus_chars_written=(
                        corpus_chars_written
                    ),
                )

                reports[source_name] = source_report

        inspection = inspect_output_file(
            TEMP_OUTPUT_PATH
        )

        actual_file_chars = inspection["characters"]
        replacement_chars = inspection[
            "replacement_characters"
        ]
        null_chars = inspection["null_characters"]
        file_bytes = inspection["utf8_bytes"]

        build_passed = (
            corpus_chars_written
            == TARGET_TOTAL_CHARS
        )

        file_size_passed = (
            actual_file_chars
            == TARGET_TOTAL_CHARS
        )

        replacement_passed = (
            replacement_chars == 0
        )

        null_passed = (
            null_chars == 0
        )

        report = {
            "version": "tokenizer_50m_v0.2",
            "target_characters": TARGET_TOTAL_CHARS,
            "actual_file_characters": actual_file_chars,
            "utf8_file_bytes": file_bytes,
            "general_characters": general_total,
            "math_characters": math_total,
            "general_percentage": (
                general_total
                / TARGET_TOTAL_CHARS
                * 100
            ),
            "math_percentage": (
                math_total
                / TARGET_TOTAL_CHARS
                * 100
            ),
            "replacement_characters_in_final": (
                replacement_chars
            ),
            "null_characters_in_final": null_chars,
            "sources": reports,
        }

        print()
        print("=" * 65)
        print("FINAL CORPUS REPORT")
        print("=" * 65)
        print()

        for source_name, source_report in reports.items():
            print(
                f"{source_name:15} "
                f"{source_report['characters_written']:>12,}"
            )

        print()
        print(f"General chars: {general_total:,}")
        print(f"Math chars: {math_total:,}")
        print(
            f"Total chars written: "
            f"{corpus_chars_written:,}"
        )
        print(
            f"Verified file chars: "
            f"{actual_file_chars:,}"
        )
        print(f"UTF-8 file bytes: {file_bytes:,}")
        print(
            f"Replacement chars in final: "
            f"{replacement_chars:,}"
        )
        print(
            f"Null chars in final: "
            f"{null_chars:,}"
        )

        print()
        print(
            f"Internal count passed: "
            f"{build_passed}"
        )
        print(
            f"Independent 50M check passed: "
            f"{file_size_passed}"
        )
        print(
            f"Replacement-char check passed: "
            f"{replacement_passed}"
        )
        print(
            f"Null-char check passed: "
            f"{null_passed}"
        )

        all_passed = (
            build_passed
            and file_size_passed
            and replacement_passed
            and null_passed
        )

        if not all_passed:
            raise AssertionError(
                "The temporary 50M corpus failed one or "
                "more final integrity checks."
            )

        with open(
            TEMP_REPORT_PATH,
            "w",
            encoding="utf-8",
        ) as report_file:
            json.dump(
                report,
                report_file,
                indent=2,
                ensure_ascii=False,
            )

        # Final files are replaced only after successful checks.
        TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)
        TEMP_REPORT_PATH.replace(REPORT_PATH)

        print()
        print("Corpus:")
        print(OUTPUT_PATH)

        print()
        print("Report:")
        print(REPORT_PATH)

        print()
        print(
            "SUCCESS: tokenizer_50m_v0.2 training corpus "
            "contains exactly 50,000,000 clean characters."
        )

    except Exception:
        print()
        print(
            "BUILD FAILED: final corpus was NOT replaced."
        )
        print(
            "Temporary output, if present: "
            f"{TEMP_OUTPUT_PATH}"
        )
        raise


if __name__ == "__main__":
    main()
