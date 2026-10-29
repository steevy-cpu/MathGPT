import json
from pathlib import Path

from bpe_tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_50m_v0.2"
)

CONFIG_PATH = ARTIFACT_DIRECTORY / "tokenizer_config.json"
MERGES_PATH = ARTIFACT_DIRECTORY / "merges.json"
VOCAB_PATH = ARTIFACT_DIRECTORY / "vocab.json"

REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_50m_v0.2_evaluation.json"
)

EXPECTED_MERGES = 15_741
EXPECTED_SERIALIZED_VOCAB = 15_997
EXPECTED_LOGICAL_VOCAB = 16_000
EXPECTED_ARTIFACT_NAME = "tokenizer_50m_v0.2"


# IMPORTANT:
# These examples are intentionally identical to the original
# tokenizer_10m_v0.1 evaluation so the comparison is fair.

GENERAL_EXAMPLES = [
    (
        "Artificial intelligence systems learn patterns "
        "from data and use those patterns to make predictions."
    ),
    (
        "The quick brown fox jumps over the lazy dog while "
        "the sun sets behind the mountains."
    ),
    (
        "Computer security requires authentication, "
        "authorization, encryption, and careful monitoring."
    ),
    (
        "A scientist should compare experimental results "
        "with the original hypothesis before drawing conclusions."
    ),
    (
        "Neural networks contain layers of interconnected "
        "parameters that transform input information."
    ),
]

MATH_EXAMPLES = [
    "The derivative of x squared is 2x.",
    "For a matrix A, the product A^T A is symmetric.",
    "The probability of an event satisfies 0 <= P(A) <= 1.",
    (
        "The integral of x squared from 0 to 1 "
        "is equal to one third."
    ),
    (
        "If f(x) = x^2 + 3x + 2, then "
        "f'(x) = 2x + 3."
    ),
]

LATEX_EXAMPLES = [
    r"\frac{x^2}{2}",
    r"\int_0^1 x^2 \, dx = \frac{1}{3}",
    r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
    r"\sqrt{x^2 + y^2}",
    r"\frac{\partial^2 z}{\partial x \partial y}",
    r"A^T A",
    r"\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}",
]

UNICODE_EXAMPLES = [
    "π ≤ 3.14159",
    "α + β = γ",
    "∫ x² dx",
    "∑ₙ n²",
    "x ≠ y",
    "√2 ≥ 1",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_tokenizer():
    return BPETokenizer.from_files(
        config_path=CONFIG_PATH,
        merges_path=MERGES_PATH,
        vocab_path=VOCAB_PATH,
    )


def evaluate_artifact_integrity():
    config = load_json(CONFIG_PATH)
    merges = load_json(MERGES_PATH)
    vocab = load_json(VOCAB_PATH)

    checks = {
        "merge_count": len(merges) == EXPECTED_MERGES,
        "serialized_vocab": (
            len(vocab) == EXPECTED_SERIALIZED_VOCAB
        ),
        "logical_vocab": (
            config.get("vocab_size") == EXPECTED_LOGICAL_VOCAB
        ),
        "artifact_name": (
            config.get("artifact_name") == EXPECTED_ARTIFACT_NAME
        ),
    }

    print()
    print("5M.7.1 Artifact Integrity")
    print("=========================")
    print(f"Merge rules: {len(merges):,}")
    print(f"Serialized byte/BPE tokens: {len(vocab):,}")
    print(
        "Logical vocabulary size: "
        f"{config.get('vocab_size'):,}"
    )
    print(
        f"Special tokens: "
        f"{len(config.get('special_tokens', {})):,}"
    )
    print(
        f"Artifact name: "
        f"{config.get('artifact_name')}"
    )
    print()

    for name, passed in checks.items():
        print(f"{name:22} passed={passed}")

    return {
        "passed": all(checks.values()),
        "merge_count": len(merges),
        "serialized_vocab_count": len(vocab),
        "logical_vocab_size": config.get("vocab_size"),
        "artifact_name": config.get("artifact_name"),
        "checks": checks,
    }


def evaluate_round_trip(tokenizer, examples, category):
    print()
    print(f"5M.7.2 Round-Trip: {category}")
    print("=" * (len(category) + 21))

    results = []

    for text in examples:
        token_ids = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        decoded = tokenizer.decode(token_ids)
        passed = decoded == text

        results.append(
            {
                "text": text,
                "token_count": len(token_ids),
                "passed": passed,
            }
        )

        print()
        print(f"Original: {text}")
        print(f"Decoded:  {decoded}")
        print(f"Match:    {passed}")

    return results


def evaluate_compression(tokenizer, examples, category):
    print()
    print(f"5M.7.3 Compression: {category}")
    print("=" * (len(category) + 20))

    results = []
    total_byte_tokens = 0
    total_bpe_tokens = 0

    for text in examples:
        byte_count = len(text.encode("utf-8"))

        bpe_ids = tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )

        bpe_count = len(bpe_ids)

        total_byte_tokens += byte_count
        total_bpe_tokens += bpe_count

        reduction = (
            1 - (bpe_count / byte_count)
            if byte_count > 0
            else 0.0
        )

        characters_per_token = (
            len(text) / bpe_count
            if bpe_count > 0
            else 0.0
        )

        results.append(
            {
                "text": text,
                "byte_tokens": byte_count,
                "bpe_tokens": bpe_count,
                "reduction": reduction,
                "characters_per_token": characters_per_token,
            }
        )

        print()
        print(f"Text: {text}")
        print(f"Byte tokens: {byte_count:,}")
        print(f"BPE tokens:  {bpe_count:,}")
        print(f"Reduction:   {reduction:.2%}")
        print(f"Chars/token: {characters_per_token:.2f}")

    overall_reduction = (
        1 - (total_bpe_tokens / total_byte_tokens)
        if total_byte_tokens > 0
        else 0.0
    )

    print()
    print(f"{category} TOTAL")
    print("-" * (len(category) + 6))
    print(f"Byte tokens: {total_byte_tokens:,}")
    print(f"BPE tokens: {total_bpe_tokens:,}")
    print(f"Overall reduction: {overall_reduction:.2%}")

    return {
        "examples": results,
        "total_byte_tokens": total_byte_tokens,
        "total_bpe_tokens": total_bpe_tokens,
        "overall_reduction": overall_reduction,
    }


def decode_vocab_token(byte_values):
    try:
        return bytes(byte_values).decode("utf-8")
    except UnicodeDecodeError:
        return None


def inspect_learned_tokens():
    vocab = load_json(VOCAB_PATH)

    decoded_tokens = []

    for token_id_string, byte_values in vocab.items():
        token_id = int(token_id_string)

        if token_id < 259:
            continue

        decoded = decode_vocab_token(byte_values)

        if decoded is not None:
            decoded_tokens.append((token_id, decoded))

    longest = sorted(
        decoded_tokens,
        key=lambda item: len(item[1]),
        reverse=True,
    )[:30]

    math_markers = [
        "\\",
        "frac",
        "sqrt",
        "partial",
        "sum",
        "int",
        "^",
        "_",
        "{",
        "}",
        "matrix",
        "deriv",
        "equation",
        "function",
        "math",
    ]

    math_tokens = [
        (token_id, token_text)
        for token_id, token_text in decoded_tokens
        if any(
            marker in token_text.lower()
            for marker in math_markers
        )
    ]

    math_tokens = sorted(
        math_tokens,
        key=lambda item: len(item[1]),
        reverse=True,
    )[:50]

    print()
    print("5M.7.4 Learned Token Inspection")
    print("================================")

    print()
    print("Longest learned tokens")
    print("----------------------")

    for token_id, token_text in longest:
        print(f"{token_id:>5}: {token_text!r}")

    print()
    print("Math / LaTeX related tokens")
    print("---------------------------")

    for token_id, token_text in math_tokens:
        print(f"{token_id:>5}: {token_text!r}")

    return {
        "longest_tokens": [
            {
                "token_id": token_id,
                "text": token_text,
            }
            for token_id, token_text in longest
        ],
        "math_tokens": [
            {
                "token_id": token_id,
                "text": token_text,
            }
            for token_id, token_text in math_tokens
        ],
    }


def save_report(report):
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
            ensure_ascii=False,
        )

    print()
    print("Evaluation report saved to:")
    print(REPORT_PATH)


def main():
    print()
    print("MathGPT 50M v0.2 Tokenizer Evaluation")
    print("======================================")

    integrity = evaluate_artifact_integrity()
    tokenizer = load_tokenizer()

    categories = {
        "general": (
            "General English",
            GENERAL_EXAMPLES,
        ),
        "math": (
            "Math Prose",
            MATH_EXAMPLES,
        ),
        "latex": (
            "LaTeX",
            LATEX_EXAMPLES,
        ),
        "unicode": (
            "Unicode Math",
            UNICODE_EXAMPLES,
        ),
    }

    round_trip = {}
    compression = {}

    for key, (label, examples) in categories.items():
        round_trip[key] = evaluate_round_trip(
            tokenizer,
            examples,
            label,
        )

    for key, (label, examples) in categories.items():
        compression[key] = evaluate_compression(
            tokenizer,
            examples,
            label,
        )

    learned_tokens = inspect_learned_tokens()

    all_round_trip_passed = all(
        result["passed"]
        for category_results in round_trip.values()
        for result in category_results
    )

    final_pass = (
        integrity["passed"]
        and all_round_trip_passed
    )

    report = {
        "tokenizer": "tokenizer_50m_v0.2",
        "artifact_integrity": integrity,
        "round_trip": round_trip,
        "compression": compression,
        "learned_tokens": learned_tokens,
        "all_round_trip_passed": all_round_trip_passed,
        "evaluation_passed": final_pass,
    }

    save_report(report)

    print()
    print("=" * 60)
    print("FINAL TOKENIZER EVALUATION")
    print("=" * 60)

    print(
        f"Artifact integrity: "
        f"{integrity['passed']}"
    )

    print(
        f"All round-trip tests: "
        f"{all_round_trip_passed}"
    )

    print()
    print("Compression summary")
    print("-------------------")

    print(
        "General English: "
        f"{compression['general']['overall_reduction']:.2%}"
    )

    print(
        "Math prose: "
        f"{compression['math']['overall_reduction']:.2%}"
    )

    print(
        "LaTeX: "
        f"{compression['latex']['overall_reduction']:.2%}"
    )

    print(
        "Unicode math: "
        f"{compression['unicode']['overall_reduction']:.2%}"
    )

    print()

    if final_pass:
        print(
            "SUCCESS: tokenizer_50m_v0.2 "
            "passed Stage 5M.7 evaluation."
        )
    else:
        print(
            "WARNING: tokenizer_50m_v0.2 "
            "failed one or more Stage 5M.7 checks."
        )


if __name__ == "__main__":
    main()
