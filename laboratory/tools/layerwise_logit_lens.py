import argparse
import json
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
    readonly_layer_norm,
    token_statistics,
    token_text,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.4 layerwise_logit_lens — "
            "measure target-token evidence "
            "across MathGPT depth."
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
        "--append-token-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Append exact token IDs after "
            "the encoded prompt."
        ),
    )

    parser.add_argument(
        "--target-token-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Target vocabulary token to track. "
            "Repeat for multiple targets."
        ),
    )

    parser.add_argument(
        "--preferred-target-index",
        type=int,
        default=0,
        help=(
            "Index of the target considered "
            "preferred/correct."
        ),
    )

    parser.add_argument(
        "--scorer-artifact",
        type=str,
        default=None,
        help=(
            "Optional L6.2 JSON artifact. "
            "If supplied, discriminative target "
            "tokens are validated against it."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=(
            "Number of highest-ranked vocabulary "
            "tokens to show at every stage."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# L6.2 artifact helper
# ============================================================

def load_scorer_artifact(
    path,
):

    path = Path(
        path
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        artifact = json.load(
            file
        )

    return (
        path,
        artifact,
    )


def scorer_target_records(
    scorer_artifact,
):

    analysis = (
        scorer_artifact[
            "discriminative_analysis"
        ]
    )

    records = []

    for candidate in (
        analysis[
            "candidates"
        ]
    ):

        if not candidate.get(
            "has_discriminative_token",
            False,
        ):

            continue

        records.append(
            {
                "candidate_index":
                    int(
                        candidate[
                            "candidate_index"
                        ]
                    ),

                "token_id":
                    int(
                        candidate[
                            "token_id"
                        ]
                    ),

                "token_text":
                    candidate[
                        "token_text"
                    ],

                "rank":
                    int(
                        candidate[
                            "rank"
                        ]
                    ),

                "probability":
                    float(
                        candidate[
                            "probability"
                        ]
                    ),
            }
        )

    return records


# ============================================================
# Top-k vocabulary helper
# ============================================================

def top_k_tokens(
    *,
    tokenizer,
    logits,
    k,
):

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    k = min(
        int(
            k
        ),
        logits.shape[
            0
        ],
    )

    if k <= 0:

        return []

    # Stable descending ordering.
    order = np.argsort(
        -logits,
        kind="stable",
    )[
        :k
    ]

    maximum = np.max(
        logits
    )

    shifted = (
        logits
        -
        maximum
    )

    exp_logits = np.exp(
        shifted
    )

    denominator = np.sum(
        exp_logits
    )

    records = []

    for (
        rank_index,
        token_id,
    ) in enumerate(
        order,
        start=1,
    ):

        probability = (
            exp_logits[
                token_id
            ]
            /
            denominator
        )

        records.append(
            {
                "rank":
                    int(
                        rank_index
                    ),

                "token_id":
                    int(
                        token_id
                    ),

                "token_text":
                    token_text(
                        tokenizer,
                        token_id,
                    ),

                "logit":
                    float(
                        logits[
                            token_id
                        ]
                    ),

                "probability":
                    float(
                        probability
                    ),
            }
        )

    return records


# ============================================================
# Project one representation to vocabulary
# ============================================================

def project_representation(
    *,
    runtime,
    representation,
    already_normalized,
):

    representation = np.asarray(
        representation,
        dtype=np.float32,
    )

    if representation.ndim != 3:

        raise ValueError(
            "Representation must have shape "
            "(batch, sequence, d_model)."
        )

    if already_normalized:

        normalized = (
            representation
        )

    else:

        # Standard logit-lens readout:
        #
        # intermediate residual
        #      ↓
        # trained final LayerNorm
        #      ↓
        # tied output embedding
        normalized = readonly_layer_norm(
            runtime.model.final_norm,
            representation,
        )

    embedding_weight = (
        runtime
        .model
        .input_embedding
        .token_embedding
        .weight
    )

    logits = (
        normalized
        @
        embedding_weight.T
    )

    return logits.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Target measurements for one stage
# ============================================================

def measure_stage(
    *,
    runtime,
    stage_name,
    logits,
    target_token_ids,
    preferred_target_index,
    top_k,
):

    next_logits = (
        logits[
            0,
            -1,
            :
        ]
    )

    target_records = []

    for (
        target_index,
        token_id,
    ) in enumerate(
        target_token_ids
    ):

        statistics = token_statistics(
            next_logits,
            token_id,
        )

        target_records.append(
            {
                "target_index":
                    int(
                        target_index
                    ),

                "token_id":
                    int(
                        token_id
                    ),

                "token_text":
                    token_text(
                        runtime.tokenizer,
                        token_id,
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

    target_winner_index = int(
        max(
            range(
                len(
                    target_records
                )
            ),
            key=lambda index: (
                target_records[
                    index
                ][
                    "logit"
                ]
            ),
        )
    )

    preferred = (
        target_records[
            preferred_target_index
        ]
    )

    pairwise = []

    for (
        target_index,
        target_record,
    ) in enumerate(
        target_records
    ):

        if (
            target_index
            ==
            preferred_target_index
        ):

            continue

        logit_margin = (
            preferred[
                "logit"
            ]
            -
            target_record[
                "logit"
            ]
        )

        log_probability_margin = (
            preferred[
                "log_probability"
            ]
            -
            target_record[
                "log_probability"
            ]
        )

        if (
            target_record[
                "probability"
            ]
            >
            0.0
        ):

            probability_ratio = (
                preferred[
                    "probability"
                ]
                /
                target_record[
                    "probability"
                ]
            )

        else:

            probability_ratio = (
                float(
                    "inf"
                )
            )

        pairwise.append(
            {
                "preferred_target_index":
                    int(
                        preferred_target_index
                    ),

                "comparison_target_index":
                    int(
                        target_index
                    ),

                "logit_margin":
                    float(
                        logit_margin
                    ),

                "log_probability_margin":
                    float(
                        log_probability_margin
                    ),

                "probability_ratio":
                    float(
                        probability_ratio
                    ),

                "preferred_wins":
                    bool(
                        logit_margin
                        >
                        0.0
                    ),
            }
        )

    return {
        "stage":
            stage_name,

        "targets":
            target_records,

        "target_winner_index":
            target_winner_index,

        "target_winner_token_id":
            int(
                target_records[
                    target_winner_index
                ][
                    "token_id"
                ]
            ),

        "target_winner_text":
            target_records[
                target_winner_index
            ][
                "token_text"
            ],

        "pairwise_against_preferred":
            pairwise,

        "top_tokens":
            top_k_tokens(
                tokenizer=(
                    runtime.tokenizer
                ),
                logits=next_logits,
                k=top_k,
            ),
    }


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
        "L6.4 — LAYER-WISE LOGIT LENS"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Permission level: P2"
    )

    print(
        "Evidence level: E2 "
        "(internal correlation)"
    )

    # ========================================================
    # Runtime
    # ========================================================

    runtime = LaboratoryRuntime(
        checkpoint_path=(
            args.checkpoint
        ),

        expected_checkpoint_sha256=(
            args.expected_sha256
        ),
    )

    # ========================================================
    # Optional L6.2 artifact
    # ========================================================

    scorer_path = None
    scorer_artifact = None
    scorer_records = []

    if (
        args.scorer_artifact
        is not None
    ):

        (
            scorer_path,
            scorer_artifact,
        ) = load_scorer_artifact(
            args.scorer_artifact
        )

        scorer_records = (
            scorer_target_records(
                scorer_artifact
            )
        )

    # ========================================================
    # Target token IDs
    # ========================================================

    explicit_target_ids = [
        int(
            value
        )

        for value
        in args.target_token_id
    ]

    scorer_target_ids = [
        int(
            record[
                "token_id"
            ]
        )

        for record
        in scorer_records
    ]

    if explicit_target_ids:

        target_token_ids = (
            explicit_target_ids
        )

        if (
            scorer_target_ids
            and
            explicit_target_ids
            !=
            scorer_target_ids
        ):

            raise ValueError(
                "Explicit target token IDs "
                "do not match the L6.2 "
                "discriminative target IDs."
            )

    elif scorer_target_ids:

        target_token_ids = (
            scorer_target_ids
        )

    else:

        raise ValueError(
            "Provide --target-token-id or "
            "--scorer-artifact."
        )

    if len(
        target_token_ids
    ) < 2:

        raise ValueError(
            "Logit-lens comparison requires "
            "at least two target tokens."
        )

    for token_id in target_token_ids:

        if not (
            0
            <=
            token_id
            <
            runtime.model.vocab_size
        ):

            raise ValueError(
                f"Target token ID {token_id} "
                "is outside vocabulary."
            )

    if not (
        0
        <=
        args.preferred_target_index
        <
        len(
            target_token_ids
        )
    ):

        raise ValueError(
            "preferred-target-index is "
            "outside target range."
        )

    # ========================================================
    # Exact input context
    # ========================================================

    prompt_token_ids = (
        runtime.encode_prompt(
            args.prompt
        )
    )

    appended_token_ids = [
        int(
            value
        )

        for value
        in args.append_token_id
    ]

    context_token_ids = (
        list(
            prompt_token_ids
        )
        +
        appended_token_ids
    )

    if (
        len(
            context_token_ids
        )
        >
        runtime.model.context_length
    ):

        raise ValueError(
            "Input exceeds context length."
        )

    input_ids = np.asarray(
        [
            context_token_ids
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Trace model
    # ========================================================

    traced_logits, trace = (
        runtime.forward_with_trace(
            input_ids
        )
    )

    # ========================================================
    # Define stages
    # ========================================================

    stage_specs = [
        {
            "name":
                "embedding",

            "trace_key":
                "embedding_output",

            "already_normalized":
                False,
        }
    ]

    for block_index in range(
        runtime.model.num_layers
    ):

        stage_specs.append(
            {
                "name":
                    f"block_{block_index}",

                "trace_key":
                    f"block_{block_index}.output",

                "already_normalized":
                    False,
            }
        )

    stage_specs.append(
        {
            "name":
                "final_layernorm",

            "trace_key":
                "final_layernorm",

            "already_normalized":
                True,
        }
    )

    # ========================================================
    # Measure each stage
    # ========================================================

    stage_results = []

    stage_logits = {}

    for stage_spec in stage_specs:

        representation = (
            trace[
                stage_spec[
                    "trace_key"
                ]
            ]
        )

        projected_logits = (
            project_representation(
                runtime=runtime,

                representation=(
                    representation
                ),

                already_normalized=(
                    stage_spec[
                        "already_normalized"
                    ]
                ),
            )
        )

        stage_logits[
            stage_spec[
                "name"
            ]
        ] = (
            projected_logits
        )

        result = measure_stage(
            runtime=runtime,

            stage_name=(
                stage_spec[
                    "name"
                ]
            ),

            logits=(
                projected_logits
            ),

            target_token_ids=(
                target_token_ids
            ),

            preferred_target_index=(
                args.preferred_target_index
            ),

            top_k=(
                args.top_k
            ),
        )

        stage_results.append(
            result
        )

    # ========================================================
    # Internal consistency checks
    # ========================================================

    final_lens_logits = (
        stage_logits[
            "final_layernorm"
        ]
    )

    final_logits_exact = bool(
        np.array_equal(
            final_lens_logits,
            traced_logits,
        )
    )

    final_logits_max_difference = float(
        np.max(
            np.abs(
                final_lens_logits.astype(
                    np.float64
                )
                -
                traced_logits.astype(
                    np.float64
                )
            )
        )
    )

    # Block 4 readout applies the final LN manually.
    # It should therefore equal the actual final LN.
    last_block_name = (
        f"block_"
        f"{runtime.model.num_layers - 1}"
    )

    last_block_normalized = (
        readonly_layer_norm(
            runtime.model.final_norm,
            trace[
                (
                    f"block_"
                    f"{runtime.model.num_layers - 1}"
                    f".output"
                )
            ],
        )
    )

    final_norm_trace = (
        trace[
            "final_layernorm"
        ]
    )

    last_block_norm_exact = bool(
        np.array_equal(
            last_block_normalized,
            final_norm_trace,
        )
    )

    last_block_norm_max_difference = float(
        np.max(
            np.abs(
                last_block_normalized.astype(
                    np.float64
                )
                -
                final_norm_trace.astype(
                    np.float64
                )
            )
        )
    )

    # ========================================================
    # Cross-check against L6.2
    # ========================================================

    scorer_crosscheck = {
        "available":
            bool(
                scorer_artifact
                is not None
            ),

        "passed":
            None,

        "details":
            [],
    }

    if scorer_artifact is not None:

        scorer_checkpoint_hash = (
            scorer_artifact[
                "model"
            ][
                "checkpoint_sha256"
            ]
        )

        checkpoint_matches = (
            scorer_checkpoint_hash
            ==
            runtime.checkpoint_sha256
        )

        scorer_prompt = (
            scorer_artifact[
                "prompt"
            ]
        )

        prompt_matches = (
            scorer_prompt
            ==
            args.prompt
        )

        scorer_shared_prefix = (
            scorer_artifact[
                "discriminative_analysis"
            ][
                "shared_prefix_token_ids"
            ]
        )

        context_prefix_matches = (
            [
                int(
                    value
                )

                for value
                in scorer_shared_prefix
            ]
            ==
            appended_token_ids
        )

        final_result = (
            stage_results[
                -1
            ]
        )

        final_by_token = {
            int(
                target[
                    "token_id"
                ]
            ):
                target

            for target
            in final_result[
                "targets"
            ]
        }

        detail_checks = []

        for scorer_record in scorer_records:

            token_id = int(
                scorer_record[
                    "token_id"
                ]
            )

            lens_record = (
                final_by_token[
                    token_id
                ]
            )

            rank_matches = (
                int(
                    scorer_record[
                        "rank"
                    ]
                )
                ==
                int(
                    lens_record[
                        "rank"
                    ]
                )
            )

            probability_difference = abs(
                float(
                    scorer_record[
                        "probability"
                    ]
                )
                -
                float(
                    lens_record[
                        "probability"
                    ]
                )
            )

            probability_matches = (
                probability_difference
                <=
                1e-12
            )

            detail_checks.append(
                {
                    "token_id":
                        token_id,

                    "token_text":
                        scorer_record[
                            "token_text"
                        ],

                    "rank_matches":
                        bool(
                            rank_matches
                        ),

                    "scorer_rank":
                        int(
                            scorer_record[
                                "rank"
                            ]
                        ),

                    "lens_rank":
                        int(
                            lens_record[
                                "rank"
                            ]
                        ),

                    "probability_matches":
                        bool(
                            probability_matches
                        ),

                    "probability_difference":
                        float(
                            probability_difference
                        ),
                }
            )

        scorer_crosscheck[
            "details"
        ] = (
            detail_checks
        )

        scorer_crosscheck[
            "checkpoint_matches"
        ] = (
            checkpoint_matches
        )

        scorer_crosscheck[
            "prompt_matches"
        ] = (
            prompt_matches
        )

        scorer_crosscheck[
            "shared_prefix_matches_appended_tokens"
        ] = (
            context_prefix_matches
        )

        scorer_crosscheck[
            "passed"
        ] = bool(
            checkpoint_matches
            and
            prompt_matches
            and
            context_prefix_matches
            and
            all(
                item[
                    "rank_matches"
                ]
                and
                item[
                    "probability_matches"
                ]

                for item
                in detail_checks
            )
        )

    # ========================================================
    # Checks
    # ========================================================

    targets_finite = all(
        np.isfinite(
            target[
                "probability"
            ]
        )
        and
        np.isfinite(
            target[
                "logit"
            ]
        )

        for stage
        in stage_results

        for target
        in stage[
            "targets"
        ]
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "target_count_at_least_two":
            (
                len(
                    target_token_ids
                )
                >=
                2
            ),

        "target_statistics_finite":
            bool(
                targets_finite
            ),

        "last_block_manual_norm_matches_final_norm":
            last_block_norm_exact,

        "final_logit_lens_matches_model_logits":
            final_logits_exact,
    }

    if scorer_artifact is not None:

        checks[
            "l6_2_crosscheck"
        ] = bool(
            scorer_crosscheck[
                "passed"
            ]
        )

    all_checks_passed = all(
        checks.values()
    )

    # ========================================================
    # Report
    # ========================================================

    report = {
        "mars_protocol":
            MARS_PROTOCOL_VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "tool": {
            "name":
                "layerwise_logit_lens",

            "version":
                LAB_TOOL_VERSION,

            "permission_level":
                "P2",

            "evidence_level":
                "E2",
        },

        "status":
            (
                "success"
                if all_checks_passed
                else "failed"
            ),

        "model":
            runtime.model_record(),

        "input": {
            "prompt":
                args.prompt,

            "prompt_token_ids":
                [
                    int(
                        value
                    )

                    for value
                    in prompt_token_ids
                ],

            "appended_token_ids":
                appended_token_ids,

            "context_token_ids":
                [
                    int(
                        value
                    )

                    for value
                    in context_token_ids
                ],
        },

        "targets": [
            {
                "target_index":
                    int(
                        index
                    ),

                "token_id":
                    int(
                        token_id
                    ),

                "token_text":
                    token_text(
                        runtime.tokenizer,
                        token_id,
                    ),

                "preferred":
                    bool(
                        index
                        ==
                        args.preferred_target_index
                    ),
            }

            for (
                index,
                token_id,
            ) in enumerate(
                target_token_ids
            )
        ],

        "logit_lens": {
            "method":
                (
                    "Intermediate residual streams "
                    "are passed through the trained "
                    "final LayerNorm and the tied "
                    "token-embedding output projection."
                ),

            "stages":
                stage_results,
        },

        "consistency": {
            "last_block_name":
                last_block_name,

            "last_block_norm_exact":
                last_block_norm_exact,

            "last_block_norm_max_difference":
                last_block_norm_max_difference,

            "final_logits_exact":
                final_logits_exact,

            "final_logits_max_difference":
                final_logits_max_difference,

            "l6_2_crosscheck":
                scorer_crosscheck,
        },

        "checks":
            checks,
    }

    # ========================================================
    # Output
    # ========================================================

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

    # ========================================================
    # Display
    # ========================================================

    print(
        "Checkpoint"
    )

    print(
        "----------"
    )

    print(
        runtime.checkpoint_path
    )

    print()

    print(
        "Input context"
    )

    print(
        "-------------"
    )

    print(
        repr(
            args.prompt
        )
    )

    print()

    print(
        f"Appended token IDs: "
        f"{appended_token_ids}"
    )

    if appended_token_ids:

        print(
            "Appended token text:"
        )

        for token_id in appended_token_ids:

            print(
                (
                    f"  id={token_id:<5} "
                    f"text="
                    f"{repr(token_text(runtime.tokenizer, token_id))}"
                )
            )

    print()

    print(
        "Tracked targets"
    )

    print(
        "---------------"
    )

    for (
        index,
        token_id,
    ) in enumerate(
        target_token_ids
    ):

        marker = (
            " [PREFERRED]"
            if
            index
            ==
            args.preferred_target_index
            else
            ""
        )

        print(
            (
                f"[{index}] "
                f"id={token_id:<5} "
                f"text="
                f"{repr(token_text(runtime.tokenizer, token_id))}"
                f"{marker}"
            )
        )

    # ========================================================
    # Layer table
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "LAYER-WISE TARGET EVIDENCE"
    )

    print(
        "=" * 76
    )

    print()

    header = (
        f"{'Stage':<18}"
    )

    for (
        index,
        token_id,
    ) in enumerate(
        target_token_ids
    ):

        label = token_text(
            runtime.tokenizer,
            token_id,
        )

        header += (
            f"{label + ' rank':>14}"
        )

    header += (
        f"{'Winner':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for stage in stage_results:

        row = (
            f"{stage['stage']:<18}"
        )

        for target in stage[
            "targets"
        ]:

            row += (
                f"{target['rank']:>14,}"
            )

        row += (
            f"{stage['target_winner_text']:>14}"
        )

        print(
            row
        )

    # ========================================================
    # Pairwise margins
    # ========================================================

    print()
    print(
        "Preferred-target margins"
    )

    print(
        "------------------------"
    )

    for stage in stage_results:

        for comparison in (
            stage[
                "pairwise_against_preferred"
            ]
        ):

            print(
                (
                    f"{stage['stage']:<18} "
                    f"logit_margin="
                    f"{comparison['logit_margin']:+.6f} "
                    f"ratio="
                    f"{comparison['probability_ratio']:.4f} "
                    f"preferred_wins="
                    f"{comparison['preferred_wins']}"
                )
            )

    # ========================================================
    # Top predictions
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        f"TOP-{args.top_k} VOCABULARY TOKENS BY STAGE"
    )

    print(
        "=" * 76
    )

    for stage in stage_results:

        print()
        print(
            stage[
                "stage"
            ]
        )

        for token in stage[
            "top_tokens"
        ]:

            print(
                (
                    f"  #{token['rank']:<3} "
                    f"id={token['token_id']:<5} "
                    f"text={repr(token['token_text']):<18} "
                    f"p={token['probability']:.8f}"
                )
            )

    # ========================================================
    # Consistency
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "INSTRUMENT CONSISTENCY"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Manual final norm exact: "
        f"{last_block_norm_exact}"
    )

    print(
        f"Manual final norm max diff: "
        f"{last_block_norm_max_difference}"
    )

    print()

    print(
        f"Final lens logits exact: "
        f"{final_logits_exact}"
    )

    print(
        f"Final lens max diff: "
        f"{final_logits_max_difference}"
    )

    if scorer_artifact is not None:

        print()

        print(
            f"L6.2 artifact:"
        )

        print(
            scorer_path
        )

        print(
            f"L6.2 cross-check passed: "
            f"{scorer_crosscheck['passed']}"
        )

        for detail in (
            scorer_crosscheck[
                "details"
            ]
        ):

            print(
                (
                    f"  {repr(detail['token_text'])}: "
                    f"rank_match="
                    f"{detail['rank_matches']} "
                    f"probability_match="
                    f"{detail['probability_matches']}"
                )
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

    print()

    print(
        "JSON artifact:"
    )

    print(
        output_path
    )

    # ========================================================
    # Acceptance
    # ========================================================

    if not all_checks_passed:

        raise RuntimeError(
            "L6.4 layerwise logit lens failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.4 LAYER-WISE "
        "LOGIT LENS PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()