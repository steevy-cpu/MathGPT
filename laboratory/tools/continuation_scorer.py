import argparse
import json
import math
import sys

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

import numpy as np


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Laboratory imports
# ============================================================

from laboratory.core.runtime import (
    LAB_TOOL_VERSION,
    MARS_PROTOCOL_VERSION,
    LaboratoryRuntime,
    token_statistics,
    token_text,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.2 continuation_scorer — "
            "deterministically score candidate "
            "continuations and identify their "
            "first discriminative token."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--continuation",
        required=True,
        action="append",
    )

    parser.add_argument(
        "--preferred-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Common-prefix helper
# ============================================================

def common_prefix(
    sequences,
):

    if not sequences:

        return []

    prefix = list(
        sequences[
            0
        ]
    )

    for sequence in sequences[
        1:
    ]:

        maximum = min(
            len(
                prefix
            ),
            len(
                sequence
            ),
        )

        index = 0

        while (
            index
            <
            maximum
            and
            prefix[
                index
            ]
            ==
            sequence[
                index
            ]
        ):

            index += 1

        prefix = (
            prefix[
                :index
            ]
        )

        if not prefix:

            break

    return prefix


# ============================================================
# Score continuation
# ============================================================

def score_continuation(
    *,
    runtime,
    prompt,
    continuation,
):

    tokenizer = (
        runtime.tokenizer
    )

    model = (
        runtime.model
    )

    prompt_ids = tokenizer.encode(
        prompt,
        add_bos=False,
        add_eos=False,
    )

    continuation_ids = tokenizer.encode(
        continuation,
        add_bos=False,
        add_eos=False,
    )

    if not prompt_ids:

        prompt_ids = [
            tokenizer.bos_token_id
        ]

    if not continuation_ids:

        raise ValueError(
            "Continuation tokenized "
            "to an empty sequence."
        )

    context_ids = list(
        prompt_ids
    )

    token_results = []

    total_log_probability = (
        0.0
    )

    # ========================================================
    # Autoregressive teacher-forced scoring
    # ========================================================

    for (
        position,
        target_token_id,
    ) in enumerate(
        continuation_ids
    ):

        model_context = (
            context_ids[
                -model.context_length:
            ]
        )

        input_ids = np.asarray(
            [
                model_context
            ],
            dtype=np.int64,
        )

        logits = model.forward(
            input_ids
        )

        next_token_logits = (
            logits[
                0,
                -1,
                :
            ]
        )

        statistics = token_statistics(
            next_token_logits,
            target_token_id,
        )

        total_log_probability += (
            statistics[
                "log_probability"
            ]
        )

        token_results.append(
            {
                "position":
                    int(
                        position
                    ),

                "token_id":
                    int(
                        target_token_id
                    ),

                "token_text":
                    token_text(
                        tokenizer,
                        target_token_id,
                    ),

                "probability":
                    float(
                        statistics[
                            "probability"
                        ]
                    ),

                "log_probability":
                    float(
                        statistics[
                            "log_probability"
                        ]
                    ),

                "logit":
                    float(
                        statistics[
                            "logit"
                        ]
                    ),

                "rank":
                    int(
                        statistics[
                            "rank"
                        ]
                    ),
            }
        )

        context_ids.append(
            int(
                target_token_id
            )
        )

    token_count = len(
        continuation_ids
    )

    mean_log_probability = (
        total_log_probability
        /
        token_count
    )

    mean_nll = (
        -mean_log_probability
    )

    try:

        perplexity = math.exp(
            mean_nll
        )

    except OverflowError:

        perplexity = float(
            "inf"
        )

    ranks = [
        result[
            "rank"
        ]

        for result
        in token_results
    ]

    return {
        "prompt":
            prompt,

        "continuation":
            continuation,

        "prompt_token_ids":
            [
                int(
                    token_id
                )

                for token_id
                in prompt_ids
            ],

        "continuation_token_ids":
            [
                int(
                    token_id
                )

                for token_id
                in continuation_ids
            ],

        "continuation_tokens":
            [
                token_text(
                    tokenizer,
                    token_id,
                )

                for token_id
                in continuation_ids
            ],

        "continuation_token_count":
            int(
                token_count
            ),

        "sum_log_probability":
            float(
                total_log_probability
            ),

        "mean_log_probability":
            float(
                mean_log_probability
            ),

        "mean_nll":
            float(
                mean_nll
            ),

        "perplexity":
            float(
                perplexity
            ),

        # Literal first token.
        #
        # This is preserved for compatibility,
        # but may be shared punctuation/space.
        "first_token_id":
            int(
                continuation_ids[
                    0
                ]
            ),

        "first_token_text":
            token_text(
                tokenizer,
                continuation_ids[
                    0
                ],
            ),

        "first_token_probability":
            float(
                token_results[
                    0
                ][
                    "probability"
                ]
            ),

        "first_token_rank":
            int(
                token_results[
                    0
                ][
                    "rank"
                ]
            ),

        "mean_token_rank":
            float(
                np.mean(
                    ranks
                )
            ),

        "median_token_rank":
            float(
                np.median(
                    ranks
                )
            ),

        "top1_fraction":
            float(
                np.mean(
                    [
                        rank <= 1
                        for rank
                        in ranks
                    ]
                )
            ),

        "top10_fraction":
            float(
                np.mean(
                    [
                        rank <= 10
                        for rank
                        in ranks
                    ]
                )
            ),

        "top100_fraction":
            float(
                np.mean(
                    [
                        rank <= 100
                        for rank
                        in ranks
                    ]
                )
            ),

        "token_results":
            token_results,
    }


# ============================================================
# First discriminative-token analysis
# ============================================================

def build_discriminative_analysis(
    *,
    runtime,
    prompt,
    candidate_results,
):

    tokenizer = (
        runtime.tokenizer
    )

    model = (
        runtime.model
    )

    continuation_sequences = [
        candidate[
            "continuation_token_ids"
        ]

        for candidate
        in candidate_results
    ]

    shared_prefix_ids = common_prefix(
        continuation_sequences
    )

    shared_prefix_count = len(
        shared_prefix_ids
    )

    prompt_ids = tokenizer.encode(
        prompt,
        add_bos=False,
        add_eos=False,
    )

    if not prompt_ids:

        prompt_ids = [
            tokenizer.bos_token_id
        ]

    # This exactly reproduces the autoregressive
    # context used by continuation scoring:
    #
    # prompt tokens
    # +
    # shared continuation prefix tokens
    context_ids = (
        list(
            prompt_ids
        )
        +
        list(
            shared_prefix_ids
        )
    )

    model_context = (
        context_ids[
            -model.context_length:
        ]
    )

    input_ids = np.asarray(
        [
            model_context
        ],
        dtype=np.int64,
    )

    logits = model.forward(
        input_ids
    )

    next_token_logits = (
        logits[
            0,
            -1,
            :
        ]
    )

    discriminative_candidates = []

    for candidate in candidate_results:

        continuation_ids = (
            candidate[
                "continuation_token_ids"
            ]
        )

        candidate_index = (
            candidate[
                "candidate_index"
            ]
        )

        if (
            shared_prefix_count
            >=
            len(
                continuation_ids
            )
        ):

            discriminative_candidates.append(
                {
                    "candidate_index":
                        int(
                            candidate_index
                        ),

                    "has_discriminative_token":
                        False,

                    "reason":
                        (
                            "Candidate ends inside "
                            "the shared prefix."
                        ),
                }
            )

            continue

        target_token_id = int(
            continuation_ids[
                shared_prefix_count
            ]
        )

        statistics = token_statistics(
            next_token_logits,
            target_token_id,
        )

        discriminative_candidates.append(
            {
                "candidate_index":
                    int(
                        candidate_index
                    ),

                "has_discriminative_token":
                    True,

                "token_position":
                    int(
                        shared_prefix_count
                    ),

                "token_id":
                    target_token_id,

                "token_text":
                    token_text(
                        tokenizer,
                        target_token_id,
                    ),

                "rank":
                    int(
                        statistics[
                            "rank"
                        ]
                    ),

                "probability":
                    float(
                        statistics[
                            "probability"
                        ]
                    ),

                "log_probability":
                    float(
                        statistics[
                            "log_probability"
                        ]
                    ),

                "logit":
                    float(
                        statistics[
                            "logit"
                        ]
                    ),
            }
        )

    return {
        "shared_prefix_token_count":
            int(
                shared_prefix_count
            ),

        "shared_prefix_token_ids":
            [
                int(
                    token_id
                )

                for token_id
                in shared_prefix_ids
            ],

        "shared_prefix_tokens":
            [
                token_text(
                    tokenizer,
                    token_id,
                )

                for token_id
                in shared_prefix_ids
            ],

        "scoring_context_token_ids":
            [
                int(
                    token_id
                )

                for token_id
                in model_context
            ],

        "candidates":
            discriminative_candidates,
    }


# ============================================================
# Preference comparison
# ============================================================

def build_preference_comparisons(
    *,
    candidate_results,
    preferred_index,
):

    if preferred_index is None:

        return []

    if not (
        0
        <=
        preferred_index
        <
        len(
            candidate_results
        )
    ):

        raise ValueError(
            "--preferred-index is outside "
            "the candidate range."
        )

    preferred = (
        candidate_results[
            preferred_index
        ]
    )

    comparisons = []

    for (
        candidate_index,
        candidate,
    ) in enumerate(
        candidate_results
    ):

        if (
            candidate_index
            ==
            preferred_index
        ):

            continue

        # Positive means preferred has LOWER NLL.
        margin = (
            candidate[
                "mean_nll"
            ]
            -
            preferred[
                "mean_nll"
            ]
        )

        comparisons.append(
            {
                "preferred_index":
                    int(
                        preferred_index
                    ),

                "comparison_index":
                    int(
                        candidate_index
                    ),

                "preferred_continuation":
                    preferred[
                        "continuation"
                    ],

                "comparison_continuation":
                    candidate[
                        "continuation"
                    ],

                "preferred_mean_nll":
                    float(
                        preferred[
                            "mean_nll"
                        ]
                    ),

                "comparison_mean_nll":
                    float(
                        candidate[
                            "mean_nll"
                        ]
                    ),

                "margin":
                    float(
                        margin
                    ),

                "preferred_wins":
                    bool(
                        margin > 0.0
                    ),
            }
        )

    return comparisons


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print()
    print(
        "=" * 76
    )

    print(
        "L6.2 — CONTINUATION SCORER"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Permission level: P1"
    )

    print(
        "Evidence level: E1 "
        "(behavioral measurement)"
    )

    runtime = LaboratoryRuntime(
        checkpoint_path=(
            args.checkpoint
        ),

        expected_checkpoint_sha256=(
            args.expected_sha256
        ),
    )

    candidate_results = []

    for (
        candidate_index,
        continuation,
    ) in enumerate(
        args.continuation
    ):

        result = score_continuation(
            runtime=runtime,
            prompt=args.prompt,
            continuation=continuation,
        )

        result[
            "candidate_index"
        ] = int(
            candidate_index
        )

        candidate_results.append(
            result
        )

    discriminative_analysis = (
        build_discriminative_analysis(
            runtime=runtime,
            prompt=args.prompt,
            candidate_results=(
                candidate_results
            ),
        )
    )

    comparisons = (
        build_preference_comparisons(
            candidate_results=(
                candidate_results
            ),

            preferred_index=(
                args.preferred_index
            ),
        )
    )

    checks = {
        "candidate_count_positive":
            (
                len(
                    candidate_results
                )
                >
                0
            ),

        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "discriminative_analysis_created":
            (
                "candidates"
                in
                discriminative_analysis
            ),
    }

    all_checks_passed = all(
        checks.values()
    )

    report = {
        "mars_protocol":
            MARS_PROTOCOL_VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "tool": {
            "name":
                "continuation_scorer",

            "version":
                LAB_TOOL_VERSION,

            "permission_level":
                "P1",

            "evidence_level":
                "E1",
        },

        "status":
            (
                "success"
                if all_checks_passed
                else "failed"
            ),

        "model":
            runtime.model_record(),

        "prompt":
            args.prompt,

        "candidate_count":
            len(
                candidate_results
            ),

        "preferred_index":
            args.preferred_index,

        "candidates":
            candidate_results,

        "discriminative_analysis":
            discriminative_analysis,

        "comparisons":
            comparisons,

        "checks":
            checks,
    }

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Prompt"
    )

    print(
        "------"
    )

    print(
        args.prompt
    )

    for candidate in candidate_results:

        print()
        print(
            "-" * 76
        )

        print(
            f"Candidate "
            f"{candidate['candidate_index']}"
        )

        print(
            "-" * 76
        )

        print()

        print(
            repr(
                candidate[
                    "continuation"
                ]
            )
        )

        print()

        print(
            "Literal first token:"
        )

        print(
            (
                f"  {repr(candidate['first_token_text'])} "
                f"id={candidate['first_token_id']} "
                f"rank={candidate['first_token_rank']}"
            )
        )

        print()

        print(
            f"Mean NLL: "
            f"{candidate['mean_nll']:.8f}"
        )

        print(
            f"Perplexity: "
            f"{candidate['perplexity']:,.2f}"
        )

    # ========================================================
    # Discriminative token
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "FIRST DISCRIMINATIVE TOKEN ANALYSIS"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Shared prefix tokens: "
        f"{discriminative_analysis['shared_prefix_tokens']}"
    )

    print(
        f"Shared prefix token IDs: "
        f"{discriminative_analysis['shared_prefix_token_ids']}"
    )

    print()

    for candidate in (
        discriminative_analysis[
            "candidates"
        ]
    ):

        print(
            f"Candidate "
            f"{candidate['candidate_index']}:"
        )

        if not candidate[
            "has_discriminative_token"
        ]:

            print(
                "  No discriminative token."
            )

            continue

        print(
            (
                f"  token="
                f"{repr(candidate['token_text'])} "
                f"id={candidate['token_id']}"
            )
        )

        print(
            f"  rank={candidate['rank']:,}"
        )

        print(
            f"  probability="
            f"{candidate['probability']:.10f}"
        )

        print()

    # ========================================================
    # Preference
    # ========================================================

    if comparisons:

        print()
        print(
            "=" * 76
        )

        print(
            "FULL-CONTINUATION PREFERENCE"
        )

        print(
            "=" * 76
        )

        for comparison in comparisons:

            print()

            print(
                f"Preferred: "
                f"{repr(comparison['preferred_continuation'])}"
            )

            print(
                f"Comparison: "
                f"{repr(comparison['comparison_continuation'])}"
            )

            print(
                f"Margin: "
                f"{comparison['margin']:+.8f}"
            )

            print(
                f"Preferred wins: "
                f"{comparison['preferred_wins']}"
            )

    print()
    print(
        "Checks"
    )

    print(
        "------"
    )

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    if args.output is not None:

        output_path = Path(
            args.output
        )

        if not output_path.is_absolute():

            output_path = (
                PROJECT_ROOT
                /
                output_path
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
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
            "JSON artifact:"
        )

        print(
            output_path
        )

    if not all_checks_passed:

        raise RuntimeError(
            "L6.2 continuation scorer failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.2 CONTINUATION "
        "SCORER PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()