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
# Imports
# ============================================================

from laboratory.core.runtime import (
    LAB_TOOL_VERSION,
    MARS_PROTOCOL_VERSION,
    LaboratoryRuntime,
    token_statistics,
    token_text,
)

from model.causal_attention import (
    causal_attention,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.8 attention_head_ablation — "
            "temporarily zero individual "
            "attention heads and measure "
            "causal effects."
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
    )

    parser.add_argument(
        "--scorer-artifact",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--attention-artifact",
        type=str,
        default=None,
        help=(
            "Optional L6.5 artifact. "
            "If supplied, focus_mass is "
            "shown beside causal effects."
        ),
    )

    parser.add_argument(
        "--preferred-target-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--comparison-target-index",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--layer",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--head",
        action="append",
        type=int,
        default=[],
        help=(
            "Specific head to test. "
            "Repeat for multiple heads. "
            "If omitted, all heads in the "
            "selected layer are tested."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# JSON
# ============================================================

def load_json(
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

        content = json.load(
            file
        )

    return (
        path,
        content,
    )


# ============================================================
# Targets
# ============================================================

def extract_targets(
    scorer_artifact,
):

    candidates = (
        scorer_artifact[
            "discriminative_analysis"
        ][
            "candidates"
        ]
    )

    records = []

    for candidate in candidates:

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
            }
        )

    return records


# ============================================================
# Softmax
# ============================================================

def softmax(
    logits,
):

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    maximum = np.max(
        logits
    )

    values = np.exp(
        logits
        -
        maximum
    )

    return (
        values
        /
        np.sum(
            values
        )
    )


# ============================================================
# Distribution comparison
# ============================================================

def distribution_metrics(
    baseline_logits,
    intervention_logits,
):

    baseline_logits = np.asarray(
        baseline_logits,
        dtype=np.float64,
    )

    intervention_logits = np.asarray(
        intervention_logits,
        dtype=np.float64,
    )

    baseline_p = softmax(
        baseline_logits
    )

    intervention_p = softmax(
        intervention_logits
    )

    epsilon = 1e-30

    kl = float(
        np.sum(
            baseline_p
            *
            (
                np.log(
                    np.maximum(
                        baseline_p,
                        epsilon,
                    )
                )
                -
                np.log(
                    np.maximum(
                        intervention_p,
                        epsilon,
                    )
                )
            )
        )
    )

    probability_l1 = float(
        np.sum(
            np.abs(
                baseline_p
                -
                intervention_p
            )
        )
    )

    logit_l2 = float(
        np.linalg.norm(
            baseline_logits
            -
            intervention_logits
        )
    )

    baseline_top1 = int(
        np.argmax(
            baseline_logits
        )
    )

    intervention_top1 = int(
        np.argmax(
            intervention_logits
        )
    )

    return {
        "kl_baseline_to_intervention":
            kl,

        "probability_l1_distance":
            probability_l1,

        "logit_l2_difference":
            logit_l2,

        "baseline_top1_token_id":
            baseline_top1,

        "intervention_top1_token_id":
            intervention_top1,

        "top1_changed":
            bool(
                baseline_top1
                !=
                intervention_top1
            ),
    }


# ============================================================
# Target measurements
# ============================================================

def measure_targets(
    *,
    runtime,
    logits,
    preferred_token_id,
    comparison_token_id,
):

    next_logits = (
        logits[
            0,
            -1,
            :
        ]
    )

    preferred = token_statistics(
        next_logits,
        preferred_token_id,
    )

    comparison = token_statistics(
        next_logits,
        comparison_token_id,
    )

    margin = float(
        preferred[
            "logit"
        ]
        -
        comparison[
            "logit"
        ]
    )

    ratio = float(
        preferred[
            "probability"
        ]
        /
        comparison[
            "probability"
        ]
    )

    return {
        "preferred": {
            "token_id":
                int(
                    preferred_token_id
                ),

            "token_text":
                token_text(
                    runtime.tokenizer,
                    preferred_token_id,
                ),

            **preferred,
        },

        "comparison": {
            "token_id":
                int(
                    comparison_token_id
                ),

            "token_text":
                token_text(
                    runtime.tokenizer,
                    comparison_token_id,
                ),

            **comparison,
        },

        "logit_margin":
            margin,

        "probability_ratio":
            ratio,

        "preferred_wins":
            bool(
                margin
                >
                0.0
            ),
    }


# ============================================================
# Attention artifact
# ============================================================

def attention_focus_lookup(
    attention_artifact,
):

    if attention_artifact is None:

        return {}

    heads = (
        attention_artifact[
            "attention"
        ][
            "heads"
        ]
    )

    return {
        head[
            "head_id"
        ]:
            float(
                head[
                    "focus_mass"
                ]
            )

        for head
        in heads
    }


# ============================================================
# Manual attention with head scaling
# ============================================================

def attention_forward_with_head_scale(
    *,
    attention,
    x,
    target_head,
    head_scale,
):

    batch_size = int(
        x.shape[
            0
        ]
    )

    sequence_length = int(
        x.shape[
            1
        ]
    )

    # ========================================================
    # Q / K / V
    # ========================================================

    query = (
        attention
        .query_projection
        .forward(
            x
        )
    )

    key = (
        attention
        .key_projection
        .forward(
            x
        )
    )

    value = (
        attention
        .value_projection
        .forward(
            x
        )
    )

    # ========================================================
    # Split heads
    # ========================================================

    query_heads = (
        attention.split_heads(
            query
        )
    )

    key_heads = (
        attention.split_heads(
            key
        )
    )

    value_heads = (
        attention.split_heads(
            value
        )
    )

    # ========================================================
    # Flatten B × H
    # ========================================================

    query_flat = (
        query_heads.reshape(
            batch_size
            *
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    key_flat = (
        key_heads.reshape(
            batch_size
            *
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    value_flat = (
        value_heads.reshape(
            batch_size
            *
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    # ========================================================
    # Scaled dot-product attention
    # ========================================================

    (
        attention_output_flat,
        attention_weights_flat,
    ) = causal_attention(
        query_flat,
        key_flat,
        value_flat,
    )

    attention_output_heads = (
        attention_output_flat.reshape(
            batch_size,
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    attention_weights = (
        attention_weights_flat.reshape(
            batch_size,
            attention.num_heads,
            sequence_length,
            sequence_length,
        )
    )

    original_head_output = np.array(
        attention_output_heads[
            :,
            target_head,
            :,
            :
        ],
        copy=True,
    )

    # ========================================================
    # Intervention
    #
    # This happens BEFORE combine_heads()
    # and BEFORE Wo.
    # ========================================================

    attention_output_heads[
        :,
        target_head,
        :,
        :
    ] *= np.float32(
        head_scale
    )

    intervened_head_output = (
        attention_output_heads[
            :,
            target_head,
            :,
            :
        ]
    )

    # ========================================================
    # Recombine + Wo
    # ========================================================

    combined = (
        attention.combine_heads(
            attention_output_heads
        )
    )

    output = (
        attention
        .output_projection
        .forward(
            combined
        )
    )

    record = {
        "head":
            int(
                target_head
            ),

        "scale":
            float(
                head_scale
            ),

        "original_last_token_l2":
            float(
                np.linalg.norm(
                    original_head_output[
                        0,
                        -1,
                        :
                    ].astype(
                        np.float64
                    )
                )
            ),

        "intervened_last_token_l2":
            float(
                np.linalg.norm(
                    intervened_head_output[
                        0,
                        -1,
                        :
                    ].astype(
                        np.float64
                    )
                )
            ),
    }

    return (
        output.astype(
            np.float32,
            copy=False,
        ),
        attention_weights,
        record,
    )


# ============================================================
# Full forward with one head intervention
# ============================================================

def forward_with_head_scale(
    *,
    runtime,
    token_ids,
    target_layer,
    target_head,
    head_scale,
):

    model = (
        runtime.model
    )

    token_ids = np.asarray(
        token_ids,
        dtype=np.int64,
    )

    if token_ids.ndim == 1:

        token_ids = (
            token_ids[
                None,
                :
            ]
        )

    x = (
        model
        .input_embedding
        .forward(
            token_ids
        )
    )

    intervention_record = None

    # ========================================================
    # Transformer stack
    # ========================================================

    for (
        block_index,
        block,
    ) in enumerate(
        model.blocks
    ):

        normalized_x = (
            block.norm1.forward(
                x
            )
        )

        if (
            block_index
            ==
            target_layer
        ):

            (
                attention_output,
                _,
                intervention_record,
            ) = (
                attention_forward_with_head_scale(
                    attention=(
                        block.attention
                    ),

                    x=(
                        normalized_x
                    ),

                    target_head=(
                        target_head
                    ),

                    head_scale=(
                        head_scale
                    ),
                )
            )

        else:

            (
                attention_output,
                _,
            ) = (
                block
                .attention
                .forward(
                    normalized_x
                )
            )

        attention_residual = (
            x
            +
            attention_output
        )

        normalized_attention = (
            block.norm2.forward(
                attention_residual
            )
        )

        mlp_output = (
            block.mlp.forward(
                normalized_attention
            )
        )

        x = (
            attention_residual
            +
            mlp_output
        )

    # ========================================================
    # Final LayerNorm + tied projection
    # ========================================================

    hidden = (
        model.final_norm.forward(
            x
        )
    )

    embedding_weight = (
        model
        .input_embedding
        .token_embedding
        .weight
    )

    logits = (
        hidden
        @
        embedding_weight.T
    )

    return (
        logits.astype(
            np.float32,
            copy=False,
        ),
        intervention_record,
    )


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
        "L6.8 — ATTENTION HEAD ABLATION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Permission level: P3"
    )

    print(
        "Evidence level: E3 "
        "(head-level causal intervention)"
    )

    print()

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
    # Layer / heads
    # ========================================================

    if not (
        0
        <=
        args.layer
        <
        runtime.model.num_layers
    ):

        raise ValueError(
            f"Layer must be between 0 and "
            f"{runtime.model.num_layers - 1}."
        )

    if args.head:

        heads_to_test = [
            int(
                head
            )

            for head
            in args.head
        ]

    else:

        heads_to_test = list(
            range(
                runtime.model.num_heads
            )
        )

    if len(
        set(
            heads_to_test
        )
    ) != len(
        heads_to_test
    ):

        raise ValueError(
            "Duplicate head IDs supplied."
        )

    for head in heads_to_test:

        if not (
            0
            <=
            head
            <
            runtime.model.num_heads
        ):

            raise ValueError(
                f"Head {head} is invalid."
            )

    # ========================================================
    # L6.2 scorer artifact
    # ========================================================

    (
        scorer_path,
        scorer_artifact,
    ) = load_json(
        args.scorer_artifact
    )

    if (
        scorer_artifact[
            "model"
        ][
            "checkpoint_sha256"
        ]
        !=
        runtime.checkpoint_sha256
    ):

        raise RuntimeError(
            "L6.2 checkpoint mismatch."
        )

    if (
        scorer_artifact[
            "prompt"
        ]
        !=
        args.prompt
    ):

        raise RuntimeError(
            "L6.2 prompt mismatch."
        )

    expected_prefix = [
        int(
            token_id
        )

        for token_id
        in scorer_artifact[
            "discriminative_analysis"
        ][
            "shared_prefix_token_ids"
        ]
    ]

    appended_token_ids = [
        int(
            token_id
        )

        for token_id
        in args.append_token_id
    ]

    if (
        expected_prefix
        !=
        appended_token_ids
    ):

        raise RuntimeError(
            "Appended token IDs do not match "
            "the L6.2 discriminative context."
        )

    target_records = (
        extract_targets(
            scorer_artifact
        )
    )

    if len(
        target_records
    ) < 2:

        raise RuntimeError(
            "At least two targets required."
        )

    preferred_token_id = int(
        target_records[
            args.preferred_target_index
        ][
            "token_id"
        ]
    )

    comparison_token_id = int(
        target_records[
            args.comparison_target_index
        ][
            "token_id"
        ]
    )

    # ========================================================
    # Optional L6.5 artifact
    # ========================================================

    attention_path = None
    attention_artifact = None
    focus_lookup = {}

    if (
        args.attention_artifact
        is not None
    ):

        (
            attention_path,
            attention_artifact,
        ) = load_json(
            args.attention_artifact
        )

        if (
            attention_artifact[
                "model"
            ][
                "checkpoint_sha256"
            ]
            !=
            runtime.checkpoint_sha256
        ):

            raise RuntimeError(
                "L6.5 checkpoint mismatch."
            )

        focus_lookup = (
            attention_focus_lookup(
                attention_artifact
            )
        )

    # ========================================================
    # Context
    # ========================================================

    prompt_token_ids = (
        runtime.encode_prompt(
            args.prompt
        )
    )

    context_token_ids = (
        list(
            prompt_token_ids
        )
        +
        appended_token_ids
    )

    input_ids = np.asarray(
        [
            context_token_ids
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Baseline
    # ========================================================

    baseline_logits = (
        runtime.model.forward(
            input_ids
        )
    )

    baseline_targets = (
        measure_targets(
            runtime=runtime,

            logits=baseline_logits,

            preferred_token_id=(
                preferred_token_id
            ),

            comparison_token_id=(
                comparison_token_id
            ),
        )
    )

    # ========================================================
    # Scale-1 control
    #
    # Test first requested head.
    # ========================================================

    control_head = (
        heads_to_test[
            0
        ]
    )

    (
        control_logits,
        _,
    ) = forward_with_head_scale(
        runtime=runtime,

        token_ids=input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            control_head
        ),

        head_scale=1.0,
    )

    control_exact = bool(
        np.array_equal(
            baseline_logits,
            control_logits,
        )
    )

    control_max_difference = float(
        np.max(
            np.abs(
                baseline_logits.astype(
                    np.float64
                )
                -
                control_logits.astype(
                    np.float64
                )
            )
        )
    )

    # ========================================================
    # Test each head
    # ========================================================

    results = []

    for head in heads_to_test:

        (
            ablated_logits,
            intervention_record,
        ) = forward_with_head_scale(
            runtime=runtime,

            token_ids=input_ids,

            target_layer=(
                args.layer
            ),

            target_head=(
                head
            ),

            head_scale=0.0,
        )

        ablated_targets = (
            measure_targets(
                runtime=runtime,

                logits=(
                    ablated_logits
                ),

                preferred_token_id=(
                    preferred_token_id
                ),

                comparison_token_id=(
                    comparison_token_id
                ),
            )
        )

        margin_delta = float(
            ablated_targets[
                "logit_margin"
            ]
            -
            baseline_targets[
                "logit_margin"
            ]
        )

        preferred_rank_delta = int(
            ablated_targets[
                "preferred"
            ][
                "rank"
            ]
            -
            baseline_targets[
                "preferred"
            ][
                "rank"
            ]
        )

        comparison_rank_delta = int(
            ablated_targets[
                "comparison"
            ][
                "rank"
            ]
            -
            baseline_targets[
                "comparison"
            ][
                "rank"
            ]
        )

        distribution = (
            distribution_metrics(
                baseline_logits[
                    0,
                    -1,
                    :
                ],

                ablated_logits[
                    0,
                    -1,
                    :
                ],
            )
        )

        if margin_delta < 0.0:

            causal_direction = (
                "supports_preferred_margin"
            )

        elif margin_delta > 0.0:

            causal_direction = (
                "suppresses_preferred_margin"
            )

        else:

            causal_direction = (
                "no_detected_margin_effect"
            )

        head_id = (
            f"L{args.layer}H{head}"
        )

        results.append(
            {
                "layer":
                    int(
                        args.layer
                    ),

                "head":
                    int(
                        head
                    ),

                "head_id":
                    head_id,

                "attention_focus_mass":
                    focus_lookup.get(
                        head_id
                    ),

                "intervention":
                    intervention_record,

                "ablated":
                    ablated_targets,

                "effect": {
                    "logit_margin_delta":
                        margin_delta,

                    "absolute_logit_margin_delta":
                        float(
                            abs(
                                margin_delta
                            )
                        ),

                    "preferred_rank_delta":
                        preferred_rank_delta,

                    "comparison_rank_delta":
                        comparison_rank_delta,

                    "causal_direction":
                        causal_direction,

                    "preferred_still_wins":
                        bool(
                            ablated_targets[
                                "preferred_wins"
                            ]
                        ),

                    "distribution_change":
                        distribution,
                },
            }
        )

    # ========================================================
    # Rankings
    # ========================================================

    causal_ranking = sorted(
        results,
        key=lambda result: (
            result[
                "effect"
            ][
                "absolute_logit_margin_delta"
            ]
        ),
        reverse=True,
    )

    focus_ranking = sorted(
        [
            result

            for result
            in results

            if (
                result[
                    "attention_focus_mass"
                ]
                is not None
            )
        ],
        key=lambda result: (
            result[
                "attention_focus_mass"
            ]
        ),
        reverse=True,
    )

    # ========================================================
    # Integrity
    # ========================================================

    measurements_finite = all(
        np.isfinite(
            result[
                "effect"
            ][
                "logit_margin_delta"
            ]
        )

        for result
        in results
    )

    all_heads_zeroed = all(
        (
            result[
                "intervention"
            ][
                "intervened_last_token_l2"
            ]
            ==
            0.0
        )

        for result
        in results
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "control_forward_matches_gpt":
            control_exact,

        "head_count_positive":
            (
                len(
                    results
                )
                >
                0
            ),

        "all_head_ablations_zeroed":
            all_heads_zeroed,

        "measurements_finite":
            bool(
                measurements_finite
            ),
    }

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
                "attention_head_ablation",

            "version":
                LAB_TOOL_VERSION,

            "permission_level":
                "P3",

            "evidence_level":
                "E3",
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
                prompt_token_ids,

            "appended_token_ids":
                appended_token_ids,

            "context_token_ids":
                context_token_ids,
        },

        "targets": {
            "preferred_token_id":
                preferred_token_id,

            "preferred_token_text":
                token_text(
                    runtime.tokenizer,
                    preferred_token_id,
                ),

            "comparison_token_id":
                comparison_token_id,

            "comparison_token_text":
                token_text(
                    runtime.tokenizer,
                    comparison_token_id,
                ),
        },

        "layer":
            int(
                args.layer
            ),

        "heads_tested":
            heads_to_test,

        "baseline":
            baseline_targets,

        "control_verification": {
            "head":
                int(
                    control_head
                ),

            "scale":
                1.0,

            "exact":
                control_exact,

            "max_absolute_difference":
                control_max_difference,
        },

        "results":
            results,

        "rankings": {
            "by_absolute_causal_margin_effect":
                [
                    {
                        "rank":
                            int(
                                rank
                            ),

                        "head_id":
                            item[
                                "head_id"
                            ],

                        "focus_mass":
                            item[
                                "attention_focus_mass"
                            ],

                        "margin_delta":
                            item[
                                "effect"
                            ][
                                "logit_margin_delta"
                            ],

                        "causal_direction":
                            item[
                                "effect"
                            ][
                                "causal_direction"
                            ],
                    }

                    for (
                        rank,
                        item,
                    ) in enumerate(
                        causal_ranking,
                        start=1,
                    )
                ],

            "by_attention_focus_mass":
                [
                    {
                        "rank":
                            int(
                                rank
                            ),

                        "head_id":
                            item[
                                "head_id"
                            ],

                        "focus_mass":
                            item[
                                "attention_focus_mass"
                            ],

                        "margin_delta":
                            item[
                                "effect"
                            ][
                                "logit_margin_delta"
                            ],
                    }

                    for (
                        rank,
                        item,
                    ) in enumerate(
                        focus_ranking,
                        start=1,
                    )
                ],
        },

        "artifacts": {
            "scorer_artifact":
                str(
                    scorer_path
                ),

            "attention_artifact":
                (
                    None
                    if attention_path is None
                    else str(
                        attention_path
                    )
                ),
        },

        "checks":
            checks,
    }

    # ========================================================
    # Save
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

    report[
        "artifacts"
    ][
        "json"
    ] = str(
        output_path
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
        "Experiment"
    )

    print(
        "----------"
    )

    print(
        f"Layer: {args.layer}"
    )

    print(
        f"Heads tested: "
        f"{heads_to_test}"
    )

    print()

    print(
        f"Baseline margin: "
        f"{baseline_targets['logit_margin']:+.8f}"
    )

    print(
        (
            f"Baseline preferred rank: "
            f"{baseline_targets['preferred']['rank']:,}"
        )
    )

    print(
        (
            f"Baseline comparison rank: "
            f"{baseline_targets['comparison']['rank']:,}"
        )
    )

    print()
    print(
        "Control verification"
    )

    print(
        "--------------------"
    )

    print(
        f"Scale-1 exact: "
        f"{control_exact}"
    )

    print(
        f"Max absolute difference: "
        f"{control_max_difference}"
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "HEAD-LEVEL CAUSAL RESULTS"
    )

    print(
        "=" * 90
    )

    print()

    print(
        (
            f"{'Head':<8}"
            f"{'Focus':>12}"
            f"{'Margin':>14}"
            f"{'Δ Margin':>14}"
            f"{'Pref Rank':>14}"
            f"{'KL':>12}"
            f"{'Direction':>24}"
        )
    )

    print(
        "-" * 98
    )

    for result in results:

        focus = (
            result[
                "attention_focus_mass"
            ]
        )

        focus_text = (
            "n/a"
            if focus is None
            else
            f"{focus:.6f}"
        )

        effect = (
            result[
                "effect"
            ]
        )

        ablated = (
            result[
                "ablated"
            ]
        )

        print(
            (
                f"{result['head_id']:<8}"
                f"{focus_text:>12}"
                f"{ablated['logit_margin']:>+14.6f}"
                f"{effect['logit_margin_delta']:>+14.6f}"
                f"{ablated['preferred']['rank']:>14,}"
                f"{effect['distribution_change']['kl_baseline_to_intervention']:>12.6f}"
                f"{effect['causal_direction']:>24}"
            )
        )

    # ========================================================
    # Causal ranking
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "HEADS RANKED BY |CAUSAL MARGIN EFFECT|"
    )

    print(
        "=" * 76
    )

    print()

    for (
        rank,
        result,
    ) in enumerate(
        causal_ranking,
        start=1,
    ):

        print(
            (
                f"#{rank:<2} "
                f"{result['head_id']:<6} "
                f"focus="
                f"{result['attention_focus_mass']} "
                f"Δmargin="
                f"{result['effect']['logit_margin_delta']:+.8f}"
            )
        )

    # ========================================================
    # Attention ranking comparison
    # ========================================================

    if focus_ranking:

        print()
        print(
            "=" * 76
        )

        print(
            "HEADS RANKED BY ATTENTION FOCUS MASS"
        )

        print(
            "=" * 76
        )

        print()

        for (
            rank,
            result,
        ) in enumerate(
            focus_ranking,
            start=1,
        ):

            print(
                (
                    f"#{rank:<2} "
                    f"{result['head_id']:<6} "
                    f"focus="
                    f"{result['attention_focus_mass']:.6f} "
                    f"Δmargin="
                    f"{result['effect']['logit_margin_delta']:+.8f}"
                )
            )

    # ========================================================
    # Checks
    # ========================================================

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

    if not all_checks_passed:

        raise RuntimeError(
            "L6.8 attention head "
            "ablation failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.8 ATTENTION "
        "HEAD ABLATION PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()