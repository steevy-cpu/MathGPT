import json
import re
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V1_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_10m_v0.1"
)

V2_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_50m_v0.2"
)

V1_REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_16k_evaluation.json"
)

V2_REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_50m_v0.2_evaluation.json"
)

OUTPUT_REPORT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "tokenizer_comparison_v0.1_vs_v0.2.json"
)


# ============================================================
# Constants
# ============================================================

LEARNED_TOKEN_START_ID = 259

CATEGORIES = [
    "general",
    "math",
    "latex",
    "unicode",
]

CATEGORY_LABELS = {
    "general": "General English",
    "math": "Math prose",
    "latex": "LaTeX",
    "unicode": "Unicode math",
}


# ============================================================
# Heuristic audit patterns
#
# IMPORTANT:
# "Suspicious" does NOT automatically mean "bad".
# These are review flags for likely source/UI artifacts.
# ============================================================

SUSPICIOUS_PATTERNS = {
    "nbsp": re.compile(
        "\u00a0"
    ),

    "replacement_char": re.compile(
        "\ufffd"
    ),

    "html_comment": re.compile(
        r"<!--",
        re.IGNORECASE,
    ),

    "script_tag": re.compile(
        r"<script",
        re.IGNORECASE,
    ),

    "style_tag": re.compile(
        r"<style",
        re.IGNORECASE,
    ),

    "permalink": re.compile(
        r"permalink",
        re.IGNORECASE,
    ),

    "show_tags": re.compile(
        r"show\s+tags",
        re.IGNORECASE,
    ),

    "back_to_top": re.compile(
        r"back\s+to\s+top",
        re.IGNORECASE,
    ),

    "originally_posted": re.compile(
        r"originally\s+posted\s+by",
        re.IGNORECASE,
    ),

    "public_html": re.compile(
        r"public_html",
        re.IGNORECASE,
    ),

    "like_dislike_ui": re.compile(
        r"\d+\s+like\s+\d+\s+dislike",
        re.IGNORECASE,
    ),

    "views_ui": re.compile(
        r"\b\d+\s+views?\b",
        re.IGNORECASE,
    ),

    "long_space_run": re.compile(
        r"[ \t]{8,}"
    ),

    "separator_run": re.compile(
        r"[_=\-*#~]{8,}"
    ),
}


MATH_PATTERNS = {
    "latex_backslash": re.compile(
        r"\\"
    ),

    "frac": re.compile(
        r"frac",
        re.IGNORECASE,
    ),

    "partial": re.compile(
        r"partial",
        re.IGNORECASE,
    ),

    "sqrt": re.compile(
        r"sqrt",
        re.IGNORECASE,
    ),

    "begin": re.compile(
        r"\\?begin",
        re.IGNORECASE,
    ),

    "end": re.compile(
        r"\\?end",
        re.IGNORECASE,
    ),

    "matrix": re.compile(
        r"matrix",
        re.IGNORECASE,
    ),

    "derivative": re.compile(
        r"deriv",
        re.IGNORECASE,
    ),

    "equation": re.compile(
        r"equation",
        re.IGNORECASE,
    ),

    "function": re.compile(
        r"function",
        re.IGNORECASE,
    ),

    "integral": re.compile(
        r"integr",
        re.IGNORECASE,
    ),

    "summation": re.compile(
        r"sum",
        re.IGNORECASE,
    ),

    "operatorname": re.compile(
        r"operatorname",
        re.IGNORECASE,
    ),

    "unicode_math": re.compile(
        r"[π∑∫√≤≥≠αβγ²ₙ]"
    ),
}


# ============================================================
# Helpers
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_decoded_learned_tokens(
    tokenizer_directory,
):
    """
    Load learned BPE tokens as decoded UTF-8 strings.

    Returns:
        list of dicts:
        {
            "token_id": int,
            "text": str,
            "bytes": tuple[int, ...]
        }

    Tokens that are not individually valid UTF-8 are skipped
    from the text-based audit but still remain valid byte-level
    tokenizer tokens.
    """

    vocab_path = (
        tokenizer_directory
        / "vocab.json"
    )

    vocab = load_json(
        vocab_path
    )

    decoded_tokens = []

    invalid_utf8_count = 0

    for (
        token_id_string,
        byte_values,
    ) in vocab.items():

        token_id = int(
            token_id_string
        )

        if (
            token_id
            < LEARNED_TOKEN_START_ID
        ):
            continue

        byte_tuple = tuple(
            byte_values
        )

        try:

            text = bytes(
                byte_values
            ).decode(
                "utf-8"
            )

        except UnicodeDecodeError:

            invalid_utf8_count += 1
            continue

        decoded_tokens.append(
            {
                "token_id": token_id,
                "text": text,
                "bytes": byte_tuple,
            }
        )

    return (
        decoded_tokens,
        invalid_utf8_count,
    )


def audit_suspicious_tokens(
    decoded_tokens,
):
    """
    Flag learned tokens that match likely web/UI noise.

    One token may match more than one pattern.
    """

    matches = {
        name: []
        for name
        in SUSPICIOUS_PATTERNS
    }

    suspicious_token_ids = set()

    for token in decoded_tokens:

        token_id = token[
            "token_id"
        ]

        token_text = token[
            "text"
        ]

        for (
            pattern_name,
            pattern,
        ) in SUSPICIOUS_PATTERNS.items():

            if pattern.search(
                token_text
            ):

                matches[
                    pattern_name
                ].append(
                    {
                        "token_id": (
                            token_id
                        ),
                        "text": (
                            token_text
                        ),
                    }
                )

                suspicious_token_ids.add(
                    token_id
                )

    return {
        "unique_suspicious_tokens": (
            len(
                suspicious_token_ids
            )
        ),

        "matches_by_pattern": (
            matches
        ),

        "match_counts": {
            name: len(
                items
            )
            for name, items
            in matches.items()
        },
    }


def audit_math_tokens(
    decoded_tokens,
):
    """
    Count learned tokens containing useful math-related patterns.

    This is a heuristic coverage measure, not a semantic score.
    """

    matches = {
        name: []
        for name
        in MATH_PATTERNS
    }

    math_token_ids = set()

    for token in decoded_tokens:

        token_id = token[
            "token_id"
        ]

        token_text = token[
            "text"
        ]

        for (
            pattern_name,
            pattern,
        ) in MATH_PATTERNS.items():

            if pattern.search(
                token_text
            ):

                matches[
                    pattern_name
                ].append(
                    {
                        "token_id": (
                            token_id
                        ),
                        "text": (
                            token_text
                        ),
                    }
                )

                math_token_ids.add(
                    token_id
                )

    return {
        "unique_math_related_tokens": (
            len(
                math_token_ids
            )
        ),

        "matches_by_pattern": (
            matches
        ),

        "match_counts": {
            name: len(
                items
            )
            for name, items
            in matches.items()
        },
    }


def compression_summary(
    report,
):
    """
    Extract the comparable benchmark results.
    """

    result = {}

    for category in CATEGORIES:

        category_data = (
            report[
                "compression"
            ][
                category
            ]
        )

        result[
            category
        ] = {
            "byte_tokens": (
                category_data[
                    "total_byte_tokens"
                ]
            ),

            "bpe_tokens": (
                category_data[
                    "total_bpe_tokens"
                ]
            ),

            "reduction": (
                category_data[
                    "overall_reduction"
                ]
            ),
        }

    return result


def all_benchmark_totals(
    compression,
):
    """
    Sum all four benchmark groups.
    """

    byte_tokens = sum(
        compression[
            category
        ][
            "byte_tokens"
        ]
        for category
        in CATEGORIES
    )

    bpe_tokens = sum(
        compression[
            category
        ][
            "bpe_tokens"
        ]
        for category
        in CATEGORIES
    )

    reduction = (
        1
        - (
            bpe_tokens
            / byte_tokens
        )
    )

    return {
        "byte_tokens": (
            byte_tokens
        ),

        "bpe_tokens": (
            bpe_tokens
        ),

        "reduction": (
            reduction
        ),
    }


def exact_vocab_overlap(
    v1_tokens,
    v2_tokens,
):
    """
    Compare learned token byte sequences.

    Token IDs are not compared because the same byte sequence
    can receive different IDs in independently trained BPEs.
    """

    v1_bytes = {
        token[
            "bytes"
        ]
        for token
        in v1_tokens
    }

    v2_bytes = {
        token[
            "bytes"
        ]
        for token
        in v2_tokens
    }

    overlap = (
        v1_bytes
        & v2_bytes
    )

    union = (
        v1_bytes
        | v2_bytes
    )

    return {
        "v1_unique_decoded_tokens": (
            len(
                v1_bytes
            )
        ),

        "v2_unique_decoded_tokens": (
            len(
                v2_bytes
            )
        ),

        "exact_overlap": (
            len(
                overlap
            )
        ),

        "jaccard_similarity": (
            len(
                overlap
            )
            / len(
                union
            )
            if union
            else 0.0
        ),

        "v1_overlap_percentage": (
            len(
                overlap
            )
            / len(
                v1_bytes
            )
            if v1_bytes
            else 0.0
        ),

        "v2_overlap_percentage": (
            len(
                overlap
            )
            / len(
                v2_bytes
            )
            if v2_bytes
            else 0.0
        ),
    }


# ============================================================
# Display helpers
# ============================================================

def print_compression_comparison(
    v1_compression,
    v2_compression,
):

    print()
    print("=" * 76)
    print(
        "5M.8.1 COMPRESSION COMPARISON"
    )
    print("=" * 76)

    print()

    print(
        f"{'Category':18}"
        f"{'v0.1':>12}"
        f"{'v0.2':>12}"
        f"{'Δ pp':>12}"
        f"{'Winner':>14}"
    )

    print(
        "-" * 68
    )

    for category in CATEGORIES:

        v1_value = (
            v1_compression[
                category
            ][
                "reduction"
            ]
        )

        v2_value = (
            v2_compression[
                category
            ][
                "reduction"
            ]
        )

        delta_pp = (
            v2_value
            - v1_value
        ) * 100

        if (
            abs(delta_pp)
            < 0.005
        ):
            winner = "Tie"

        elif (
            delta_pp
            > 0
        ):
            winner = "v0.2"

        else:
            winner = "v0.1"

        print(
            f"{CATEGORY_LABELS[category]:18}"
            f"{v1_value:>11.2%}"
            f"{v2_value:>12.2%}"
            f"{delta_pp:>+11.2f}"
            f"{winner:>14}"
        )


def print_noise_audit(
    name,
    audit,
    total_learned_tokens,
):

    count = audit[
        "unique_suspicious_tokens"
    ]

    percentage = (
        count
        / total_learned_tokens
        * 100
    )

    print()
    print(
        f"{name} suspicious learned tokens: "
        f"{count:,} / "
        f"{total_learned_tokens:,} "
        f"({percentage:.3f}%)"
    )

    print(
        "Pattern counts:"
    )

    for (
        pattern_name,
        count,
    ) in audit[
        "match_counts"
    ].items():

        if count > 0:

            print(
                f"  {pattern_name:22} "
                f"{count:,}"
            )


def print_math_audit(
    name,
    audit,
    total_learned_tokens,
):

    count = audit[
        "unique_math_related_tokens"
    ]

    percentage = (
        count
        / total_learned_tokens
        * 100
    )

    print()
    print(
        f"{name} math-related learned tokens: "
        f"{count:,} / "
        f"{total_learned_tokens:,} "
        f"({percentage:.3f}%)"
    )

    print(
        "Pattern counts:"
    )

    for (
        pattern_name,
        count,
    ) in audit[
        "match_counts"
    ].items():

        if count > 0:

            print(
                f"  {pattern_name:22} "
                f"{count:,}"
            )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "MathGPT Tokenizer Comparison"
    )

    print(
        "v0.1 (10M) vs v0.2 (50M)"
    )

    print(
        "=" * 76
    )

    # --------------------------------------------------------
    # Load evaluation reports
    # --------------------------------------------------------

    v1_report = load_json(
        V1_REPORT_PATH
    )

    v2_report = load_json(
        V2_REPORT_PATH
    )

    # --------------------------------------------------------
    # Basic evaluation status
    # --------------------------------------------------------

    print()
    print(
        "Evaluation status"
    )

    print(
        "-----------------"
    )

    print(
        f"v0.1 evaluation passed: "
        f"{v1_report.get('evaluation_passed')}"
    )

    print(
        f"v0.2 evaluation passed: "
        f"{v2_report.get('evaluation_passed')}"
    )

    print(
        f"v0.1 round-trip passed: "
        f"{v1_report.get('all_round_trip_passed')}"
    )

    print(
        f"v0.2 round-trip passed: "
        f"{v2_report.get('all_round_trip_passed')}"
    )

    # --------------------------------------------------------
    # Compression
    # --------------------------------------------------------

    v1_compression = (
        compression_summary(
            v1_report
        )
    )

    v2_compression = (
        compression_summary(
            v2_report
        )
    )

    print_compression_comparison(
        v1_compression,
        v2_compression,
    )

    v1_total = (
        all_benchmark_totals(
            v1_compression
        )
    )

    v2_total = (
        all_benchmark_totals(
            v2_compression
        )
    )

    overall_token_savings = (
        v1_total[
            "bpe_tokens"
        ]
        - v2_total[
            "bpe_tokens"
        ]
    )

    overall_token_savings_percent = (
        overall_token_savings
        / v1_total[
            "bpe_tokens"
        ]
        * 100
    )

    print()
    print(
        "All benchmark groups combined"
    )

    print(
        "-----------------------------"
    )

    print(
        f"Common byte baseline: "
        f"{v1_total['byte_tokens']:,}"
    )

    print(
        f"v0.1 BPE tokens: "
        f"{v1_total['bpe_tokens']:,}"
    )

    print(
        f"v0.2 BPE tokens: "
        f"{v2_total['bpe_tokens']:,}"
    )

    print(
        f"v0.1 overall reduction: "
        f"{v1_total['reduction']:.2%}"
    )

    print(
        f"v0.2 overall reduction: "
        f"{v2_total['reduction']:.2%}"
    )

    print(
        f"v0.2 token savings vs v0.1: "
        f"{overall_token_savings:,} "
        f"({overall_token_savings_percent:.2f}%)"
    )

    # --------------------------------------------------------
    # Load learned vocabularies
    # --------------------------------------------------------

    (
        v1_tokens,
        v1_invalid_utf8,
    ) = load_decoded_learned_tokens(
        V1_DIR
    )

    (
        v2_tokens,
        v2_invalid_utf8,
    ) = load_decoded_learned_tokens(
        V2_DIR
    )

    total_learned_tokens = (
        15_741
    )

    # --------------------------------------------------------
    # Vocabulary overlap
    # --------------------------------------------------------

    overlap = exact_vocab_overlap(
        v1_tokens,
        v2_tokens,
    )

    print()
    print("=" * 76)
    print(
        "5M.8.2 LEARNED VOCABULARY OVERLAP"
    )
    print("=" * 76)

    print()

    print(
        f"v0.1 individually valid UTF-8 learned tokens: "
        f"{len(v1_tokens):,}"
    )

    print(
        f"v0.2 individually valid UTF-8 learned tokens: "
        f"{len(v2_tokens):,}"
    )

    print(
        f"v0.1 partial/non-standalone UTF-8 tokens: "
        f"{v1_invalid_utf8:,}"
    )

    print(
        f"v0.2 partial/non-standalone UTF-8 tokens: "
        f"{v2_invalid_utf8:,}"
    )

    print()

    print(
        f"Exact learned-token overlap: "
        f"{overlap['exact_overlap']:,}"
    )

    print(
        f"v0.1 vocab retained in v0.2: "
        f"{overlap['v1_overlap_percentage']:.2%}"
    )

    print(
        f"v0.2 vocab shared with v0.1: "
        f"{overlap['v2_overlap_percentage']:.2%}"
    )

    print(
        f"Jaccard similarity: "
        f"{overlap['jaccard_similarity']:.2%}"
    )

    # --------------------------------------------------------
    # Noise audit
    # --------------------------------------------------------

    v1_noise = audit_suspicious_tokens(
        v1_tokens
    )

    v2_noise = audit_suspicious_tokens(
        v2_tokens
    )

    print()
    print("=" * 76)
    print(
        "5M.8.3 HEURISTIC NOISE AUDIT"
    )
    print("=" * 76)

    print_noise_audit(
        "v0.1",
        v1_noise,
        total_learned_tokens,
    )

    print_noise_audit(
        "v0.2",
        v2_noise,
        total_learned_tokens,
    )

    # --------------------------------------------------------
    # Math-token audit
    # --------------------------------------------------------

    v1_math = audit_math_tokens(
        v1_tokens
    )

    v2_math = audit_math_tokens(
        v2_tokens
    )

    print()
    print("=" * 76)
    print(
        "5M.8.4 MATH-TOKEN COVERAGE AUDIT"
    )
    print("=" * 76)

    print_math_audit(
        "v0.1",
        v1_math,
        total_learned_tokens,
    )

    print_math_audit(
        "v0.2",
        v2_math,
        total_learned_tokens,
    )

    # --------------------------------------------------------
    # Decision signals
    # --------------------------------------------------------

    v2_compression_better = (
        v2_total[
            "bpe_tokens"
        ]
        < v1_total[
            "bpe_tokens"
        ]
    )

    v2_noise_not_worse = (
        v2_noise[
            "unique_suspicious_tokens"
        ]
        <= v1_noise[
            "unique_suspicious_tokens"
        ]
    )

    v2_math_coverage_not_worse = (
        v2_math[
            "unique_math_related_tokens"
        ]
        >= v1_math[
            "unique_math_related_tokens"
        ]
    )

    both_correct = (
        bool(
            v1_report.get(
                "evaluation_passed"
            )
        )
        and
        bool(
            v2_report.get(
                "evaluation_passed"
            )
        )
    )

    print()
    print("=" * 76)
    print(
        "5M.8.5 DECISION SIGNALS"
    )
    print("=" * 76)

    print()

    print(
        f"Both tokenizers functionally correct: "
        f"{both_correct}"
    )

    print(
        f"v0.2 uses fewer benchmark tokens: "
        f"{v2_compression_better}"
    )

    print(
        f"v0.2 suspicious-token count <= v0.1: "
        f"{v2_noise_not_worse}"
    )

    print(
        f"v0.2 math-token coverage >= v0.1: "
        f"{v2_math_coverage_not_worse}"
    )

    # --------------------------------------------------------
    # Save comparison report
    # --------------------------------------------------------

    comparison_report = {
        "v0.1": {
            "name": "tokenizer_10m_v0.1",
            "evaluation_passed": (
                v1_report.get(
                    "evaluation_passed"
                )
            ),
            "compression": (
                v1_compression
            ),
            "combined_benchmark": (
                v1_total
            ),
            "noise_audit": (
                v1_noise
            ),
            "math_audit": (
                v1_math
            ),
            "invalid_utf8_learned_tokens": (
                v1_invalid_utf8
            ),
        },

        "v0.2": {
            "name": "tokenizer_50m_v0.2",
            "evaluation_passed": (
                v2_report.get(
                    "evaluation_passed"
                )
            ),
            "compression": (
                v2_compression
            ),
            "combined_benchmark": (
                v2_total
            ),
            "noise_audit": (
                v2_noise
            ),
            "math_audit": (
                v2_math
            ),
            "invalid_utf8_learned_tokens": (
                v2_invalid_utf8
            ),
        },

        "vocabulary_overlap": (
            overlap
        ),

        "overall_token_savings": {
            "tokens": (
                overall_token_savings
            ),
            "percentage_vs_v0.1": (
                overall_token_savings_percent
                / 100
            ),
        },

        "decision_signals": {
            "both_functionally_correct": (
                both_correct
            ),
            "v0.2_better_compression": (
                v2_compression_better
            ),
            "v0.2_noise_not_worse": (
                v2_noise_not_worse
            ),
            "v0.2_math_coverage_not_worse": (
                v2_math_coverage_not_worse
            ),
        },
    }

    OUTPUT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison_report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(
        "Comparison report saved to:"
    )

    print(
        OUTPUT_REPORT_PATH
    )


if __name__ == "__main__":
    main()
