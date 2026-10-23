import json
from pathlib import Path


from bpe_tokenizer import BPETokenizer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_10m_v0.1"
)


CONFIG_PATH = (
    ARTIFACT_DIRECTORY
    / "tokenizer_config.json"
)


MERGES_PATH = (
    ARTIFACT_DIRECTORY
    / "merges.json"
)


VOCAB_PATH = (
    ARTIFACT_DIRECTORY
    / "vocab.json"
)


REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_16k_evaluation.json"
)


# ============================================================
# Expected tokenizer structure
# ============================================================

EXPECTED_MERGES = 15_741

EXPECTED_SERIALIZED_VOCAB = 15_997

EXPECTED_LOGICAL_VOCAB = 16_000


# ============================================================
# Evaluation examples
# ============================================================

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

    (
        "For a matrix A, the product "
        "A^T A is symmetric."
    ),

    (
        "The probability of an event satisfies "
        "0 <= P(A) <= 1."
    ),

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


# ============================================================
# Load artifacts
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_tokenizer():

    return BPETokenizer.from_files(
        config_path=CONFIG_PATH,
        merges_path=MERGES_PATH,
        vocab_path=VOCAB_PATH,
    )


# ============================================================
# 5K.1 Artifact integrity
# ============================================================

def evaluate_artifact_integrity():

    config = load_json(
        CONFIG_PATH
    )

    merges = load_json(
        MERGES_PATH
    )

    vocab = load_json(
        VOCAB_PATH
    )

    merge_count = len(
        merges
    )

    serialized_vocab_count = len(
        vocab
    )

    logical_vocab_size = config.get(
        "vocab_size"
    )

    print()
    print("5K.1 Artifact Integrity")
    print("=======================")

    print(
        f"Merge rules: "
        f"{merge_count:,}"
    )

    print(
        f"Serialized byte/BPE tokens: "
        f"{serialized_vocab_count:,}"
    )

    print(
        f"Logical vocabulary size: "
        f"{logical_vocab_size:,}"
    )

    print(
        f"Special tokens: "
        f"{len(config.get('special_tokens', {})):,}"
    )

    merge_pass = (
        merge_count
        == EXPECTED_MERGES
    )

    serialized_vocab_pass = (
        serialized_vocab_count
        == EXPECTED_SERIALIZED_VOCAB
    )

    logical_vocab_pass = (
        logical_vocab_size
        == EXPECTED_LOGICAL_VOCAB
    )

    passed = (
        merge_pass
        and serialized_vocab_pass
        and logical_vocab_pass
    )

    print()

    print(
        f"Merge count correct: "
        f"{merge_pass}"
    )

    print(
        f"Serialized vocab correct: "
        f"{serialized_vocab_pass}"
    )

    print(
        f"Logical vocab correct: "
        f"{logical_vocab_pass}"
    )

    return {
        "passed": passed,
        "merge_count": merge_count,
        "serialized_vocab_count": (
            serialized_vocab_count
        ),
        "logical_vocab_size": (
            logical_vocab_size
        ),
    }


# ============================================================
# 5K.2 Round-trip tests
# ============================================================

def evaluate_round_trip(
    tokenizer,
    examples,
    category,
):

    print()
    print(
        f"5K.2 Round-Trip: {category}"
    )

    print(
        "=" * (
            len(category) + 19
        )
    )

    results = []

    for text in examples:

        token_ids = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        decoded = tokenizer.decode(
            token_ids
        )

        passed = (
            decoded == text
        )

        results.append(
            {
                "text": text,
                "token_count": len(
                    token_ids
                ),
                "passed": passed,
            }
        )

        print()
        print(
            f"Original: {text}"
        )

        print(
            f"Decoded:  {decoded}"
        )

        print(
            f"Match:    {passed}"
        )

    return results


# ============================================================
# 5K.3 Compression analysis
# ============================================================

def evaluate_compression(
    tokenizer,
    examples,
    category,
):

    print()
    print(
        f"5K.3 Compression: {category}"
    )

    print(
        "=" * (
            len(category) + 18
        )
    )

    results = []

    total_byte_tokens = 0
    total_bpe_tokens = 0

    for text in examples:

        byte_count = len(
            text.encode(
                "utf-8"
            )
        )

        bpe_ids = tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )

        bpe_count = len(
            bpe_ids
        )

        total_byte_tokens += (
            byte_count
        )

        total_bpe_tokens += (
            bpe_count
        )

        if byte_count > 0:

            reduction = (
                1
                - (
                    bpe_count
                    / byte_count
                )
            )

        else:

            reduction = 0.0

        characters_per_token = (
            len(text)
            / bpe_count
            if bpe_count > 0
            else 0
        )

        results.append(
            {
                "text": text,
                "byte_tokens": byte_count,
                "bpe_tokens": bpe_count,
                "reduction": reduction,
                "characters_per_token": (
                    characters_per_token
                ),
            }
        )

        print()
        print(
            f"Text: {text}"
        )

        print(
            f"Byte tokens: "
            f"{byte_count:,}"
        )

        print(
            f"BPE tokens:  "
            f"{bpe_count:,}"
        )

        print(
            f"Reduction:   "
            f"{reduction:.2%}"
        )

        print(
            f"Chars/token: "
            f"{characters_per_token:.2f}"
        )

    overall_reduction = (
        1
        - (
            total_bpe_tokens
            / total_byte_tokens
        )
        if total_byte_tokens > 0
        else 0
    )

    print()
    print(
        f"{category} TOTAL"
    )

    print(
        "-" * (
            len(category) + 6
        )
    )

    print(
        f"Byte tokens: "
        f"{total_byte_tokens:,}"
    )

    print(
        f"BPE tokens: "
        f"{total_bpe_tokens:,}"
    )

    print(
        f"Overall reduction: "
        f"{overall_reduction:.2%}"
    )

    return {
        "examples": results,
        "total_byte_tokens": (
            total_byte_tokens
        ),
        "total_bpe_tokens": (
            total_bpe_tokens
        ),
        "overall_reduction": (
            overall_reduction
        ),
    }


# ============================================================
# 5K.4 Inspect learned vocabulary
# ============================================================

def decode_vocab_token(
    token_id,
    byte_values,
):

    try:

        raw_bytes = bytes(
            byte_values
        )

        return raw_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        return None


def inspect_learned_tokens():

    vocab = load_json(
        VOCAB_PATH
    )

    decoded_tokens = []

    for token_id_string, byte_values in (
        vocab.items()
    ):

        token_id = int(
            token_id_string
        )

        # IDs below 259 are raw byte tokens.
        # We care primarily about learned BPE tokens.
        if token_id < 259:
            continue

        decoded = decode_vocab_token(
            token_id,
            byte_values,
        )

        if decoded is None:
            continue

        decoded_tokens.append(
            (
                token_id,
                decoded,
            )
        )

    # --------------------------------------------------------
    # Longest learned tokens
    # --------------------------------------------------------

    longest = sorted(
        decoded_tokens,
        key=lambda item: len(
            item[1]
        ),
        reverse=True,
    )[:30]

    # --------------------------------------------------------
    # Math-related learned tokens
    # --------------------------------------------------------

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

    math_tokens = []

    for token_id, token_text in (
        decoded_tokens
    ):

        lowercase = (
            token_text.lower()
        )

        if any(
            marker in lowercase
            for marker in math_markers
        ):

            math_tokens.append(
                (
                    token_id,
                    token_text,
                )
            )

    # Prefer longer useful math pieces.
    math_tokens = sorted(
        math_tokens,
        key=lambda item: len(
            item[1]
        ),
        reverse=True,
    )[:50]

    print()
    print(
        "5K.4 Learned Token Inspection"
    )

    print(
        "============================="
    )

    print()
    print(
        "Longest learned tokens"
    )

    print(
        "----------------------"
    )

    for token_id, text in longest:

        print(
            f"{token_id:>5}: "
            f"{text!r}"
        )

    print()
    print(
        "Math / LaTeX related tokens"
    )

    print(
        "---------------------------"
    )

    for token_id, text in math_tokens:

        print(
            f"{token_id:>5}: "
            f"{text!r}"
        )

    return {
        "longest_tokens": [
            {
                "token_id": token_id,
                "text": text,
            }
            for token_id, text in longest
        ],

        "math_tokens": [
            {
                "token_id": token_id,
                "text": text,
            }
            for token_id, text
            in math_tokens
        ],
    }


# ============================================================
# Final report
# ============================================================

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
    print(
        f"Evaluation report saved to:"
    )

    print(
        REPORT_PATH
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "MathGPT 16K Tokenizer Evaluation"
    )

    print(
        "================================"
    )

    integrity = (
        evaluate_artifact_integrity()
    )

    tokenizer = (
        load_tokenizer()
    )

    round_trip = {}

    round_trip["general"] = (
        evaluate_round_trip(
            tokenizer,
            GENERAL_EXAMPLES,
            "General English",
        )
    )

    round_trip["math"] = (
        evaluate_round_trip(
            tokenizer,
            MATH_EXAMPLES,
            "Math Prose",
        )
    )

    round_trip["latex"] = (
        evaluate_round_trip(
            tokenizer,
            LATEX_EXAMPLES,
            "LaTeX",
        )
    )

    round_trip["unicode"] = (
        evaluate_round_trip(
            tokenizer,
            UNICODE_EXAMPLES,
            "Unicode Math",
        )
    )

    compression = {}

    compression["general"] = (
        evaluate_compression(
            tokenizer,
            GENERAL_EXAMPLES,
            "General English",
        )
    )

    compression["math"] = (
        evaluate_compression(
            tokenizer,
            MATH_EXAMPLES,
            "Math Prose",
        )
    )

    compression["latex"] = (
        evaluate_compression(
            tokenizer,
            LATEX_EXAMPLES,
            "LaTeX",
        )
    )

    compression["unicode"] = (
        evaluate_compression(
            tokenizer,
            UNICODE_EXAMPLES,
            "Unicode Math",
        )
    )

    learned_tokens = (
        inspect_learned_tokens()
    )

    all_round_trip_passed = all(
        result["passed"]
        for category_results
        in round_trip.values()
        for result
        in category_results
    )

    final_pass = (
        integrity["passed"]
        and all_round_trip_passed
    )

    report = {
        "tokenizer": (
            "tokenizer_10m_v0.1"
        ),

        "artifact_integrity": (
            integrity
        ),

        "round_trip": (
            round_trip
        ),

        "compression": (
            compression
        ),

        "learned_tokens": (
            learned_tokens
        ),

        "all_round_trip_passed": (
            all_round_trip_passed
        ),

        "evaluation_passed": (
            final_pass
        ),
    }

    save_report(
        report
    )

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL TOKENIZER EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Artifact integrity: "
        f"{integrity['passed']}"
    )

    print(
        f"All round-trip tests: "
        f"{all_round_trip_passed}"
    )

    print()

    print(
        "Compression summary"
    )

    print(
        "-------------------"
    )

    print(
        f"General English: "
        f"{compression['general']['overall_reduction']:.2%}"
    )

    print(
        f"Math prose: "
        f"{compression['math']['overall_reduction']:.2%}"
    )

    print(
        f"LaTeX: "
        f"{compression['latex']['overall_reduction']:.2%}"
    )

    print(
        f"Unicode math: "
        f"{compression['unicode']['overall_reduction']:.2%}"
    )

    print()

    if final_pass:

        print(
            "SUCCESS: The 16K tokenizer "
            "passed Stage 5K evaluation."
        )

    else:

        print(
            "WARNING: The tokenizer failed "
            "one or more Stage 5K checks."
        )


if __name__ == "__main__":
    main()