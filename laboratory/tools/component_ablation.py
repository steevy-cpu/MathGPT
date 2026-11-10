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
    token_statistics,
    token_text,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.7 component_ablation — "
            "temporarily remove one internal "
            "MathGPT component during inference."
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
        "--component",
        required=True,
        type=str,
        help=(
            "Component to ablate, for example "
            "block_0.attention_output or "
            "block_3.mlp_output."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# JSON helper
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

        return (
            path,
            json.load(
                file
            ),
        )


# ============================================================
# Extract target tokens from L6.2
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

    results = []

    for candidate in candidates:

        if not candidate.get(
            "has_discriminative_token",
            False,
        ):

            continue

        results.append(
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

    return results


# ============================================================
# Component parser
# ============================================================

def parse_component(
    component,
    num_layers,
):

    parts = component.split(
        "."
    )

    if len(
        parts
    ) != 2:

        raise ValueError(
            "Component must have format "
            "'block_N.attention_output' or "
            "'block_N.mlp_output'."
        )

    block_name = (
        parts[
            0
        ]
    )

    component_name = (
        parts[
            1
        ]
    )

    if not block_name.startswith(
        "block_"
    ):

        raise ValueError(
            "Component must begin with block_N."
        )

    try:

        block_index = int(
            block_name.split(
                "_"
            )[
                1
            ]
        )

    except ValueError as error:

        raise ValueError(
            "Invalid block index."
        ) from error

    if not (
        0
        <=
        block_index
        <
        num_layers
    ):

        raise ValueError(
            f"Block index must be between "
            f"0 and {num_layers - 1}."
        )

    if component_name not in {
        "attention_output",
        "mlp_output",
    }:

        raise ValueError(
            "Supported components are "
            "attention_output and mlp_output."
        )

    return (
        block_index,
        component_name,
    )


# ============================================================
# Manual forward with optional intervention
# ============================================================

def forward_with_component_scale(
    *,
    runtime,
    token_ids,
    target_block_index,
    target_component,
    scale,
):

    model = (
        runtime.model
    )

    token_ids = np.asarray(
        token_ids,
        dtype=np.int64,
    )

    if token_ids.ndim == 1:

        token_ids = token_ids[
            None,
            :
        ]

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

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        normalized_x = (
            block.norm1.forward(
                x
            )
        )

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

        if (
            block_index
            ==
            target_block_index
            and
            target_component
            ==
            "attention_output"
        ):

            original_output = np.array(
                attention_output,
                copy=True,
            )

            attention_output = (
                attention_output
                *
                np.float32(
                    scale
                )
            )

            intervention_record = {
                "block_index":
                    int(
                        block_index
                    ),

                "component":
                    target_component,

                "scale":
                    float(
                        scale
                    ),

                "original_last_token_l2":
                    float(
                        np.linalg.norm(
                            original_output[
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
                            attention_output[
                                0,
                                -1,
                                :
                            ].astype(
                                np.float64
                            )
                        )
                    ),
            }

        attention_residual = (
            x
            +
            attention_output
        )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

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

        if (
            block_index
            ==
            target_block_index
            and
            target_component
            ==
            "mlp_output"
        ):

            original_output = np.array(
                mlp_output,
                copy=True,
            )

            mlp_output = (
                mlp_output
                *
                np.float32(
                    scale
                )
            )

            intervention_record = {
                "block_index":
                    int(
                        block_index
                    ),

                "component":
                    target_component,

                "scale":
                    float(
                        scale
                    ),

                "original_last_token_l2":
                    float(
                        np.linalg.norm(
                            original_output[
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
                            mlp_output[
                                0,
                                -1,
                                :
                            ].astype(
                                np.float64
                            )
                        )
                    ),
            }

        x = (
            attention_residual
            +
            mlp_output
        )

    # ========================================================
    # Final output
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
# Distribution comparison
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

    shifted = (
        logits
        -
        maximum
    )

    exp_values = np.exp(
        shifted
    )

    return (
        exp_values
        /
        np.sum(
            exp_values
        )
    )


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

    baseline_probability = (
        softmax(
            baseline_logits
        )
    )

    intervention_probability = (
        softmax(
            intervention_logits
        )
    )

    epsilon = 1e-30

    kl_baseline_to_intervention = float(
        np.sum(
            baseline_probability
            *
            (
                np.log(
                    np.maximum(
                        baseline_probability,
                        epsilon,
                    )
                )
                -
                np.log(
                    np.maximum(
                        intervention_probability,
                        epsilon,
                    )
                )
            )
        )
    )

    probability_l1 = float(
        np.sum(
            np.abs(
                baseline_probability
                -
                intervention_probability
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
            kl_baseline_to_intervention,

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
# Target measurement
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

    margin = (
        preferred[
            "logit"
        ]
        -
        comparison[
            "logit"
        ]
    )

    probability_ratio = (
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
            float(
                margin
            ),

        "probability_ratio":
            float(
                probability_ratio
            ),

        "preferred_wins":
            bool(
                margin
                >
                0.0
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
        "L6.7 — COMPONENT ABLATION"
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
        "(causal intervention)"
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
    # Validate component
    # ========================================================

    (
        target_block_index,
        target_component,
    ) = parse_component(
        args.component,
        runtime.model.num_layers,
    )

    # ========================================================
    # Load scorer artifact
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
            "Appended tokens do not match "
            "L6.2 discriminative context."
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
    # Exact context
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
    # Normal GPT baseline
    # ========================================================

    baseline_logits = (
        runtime.model.forward(
            input_ids
        )
    )

    # ========================================================
    # Control manual forward
    #
    # scale = 1.0 must reproduce GPT.forward()
    # exactly before we trust intervention mode.
    # ========================================================

    (
        control_logits,
        control_record,
    ) = forward_with_component_scale(
        runtime=runtime,
        token_ids=input_ids,
        target_block_index=(
            target_block_index
        ),
        target_component=(
            target_component
        ),
        scale=1.0,
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
    # Ablation
    #
    # scale = 0
    # ========================================================

    (
        ablated_logits,
        intervention_record,
    ) = forward_with_component_scale(
        runtime=runtime,
        token_ids=input_ids,
        target_block_index=(
            target_block_index
        ),
        target_component=(
            target_component
        ),
        scale=0.0,
    )

    # ========================================================
    # Measurements
    # ========================================================

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

    ablated_targets = (
        measure_targets(
            runtime=runtime,
            logits=ablated_logits,
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

    distribution_change = (
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

    # ========================================================
    # Interpret intervention direction
    # ========================================================

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

    # ========================================================
    # Integrity
    # ========================================================

    all_measurements_finite = all(
        np.isfinite(
            value
        )

        for value in [
            baseline_targets[
                "logit_margin"
            ],
            ablated_targets[
                "logit_margin"
            ],
            margin_delta,
            distribution_change[
                "kl_baseline_to_intervention"
            ],
            distribution_change[
                "probability_l1_distance"
            ],
            distribution_change[
                "logit_l2_difference"
            ],
        ]
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

        "intervention_record_created":
            (
                intervention_record
                is not None
            ),

        "ablation_zeroed_component":
            (
                intervention_record
                is not None
                and
                intervention_record[
                    "intervened_last_token_l2"
                ]
                ==
                0.0
            ),

        "measurements_finite":
            bool(
                all_measurements_finite
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
                "component_ablation",

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

        "intervention": {
            "type":
                "component_ablation",

            "component":
                args.component,

            "block_index":
                int(
                    target_block_index
                ),

            "component_type":
                target_component,

            "scale":
                0.0,

            "record":
                intervention_record,
        },

        "control_verification": {
            "scale":
                1.0,

            "exact":
                control_exact,

            "max_absolute_difference":
                control_max_difference,

            "record":
                control_record,
        },

        "baseline":
            baseline_targets,

        "ablated":
            ablated_targets,

        "effect": {
            "logit_margin_delta":
                margin_delta,

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
                distribution_change,
        },

        "interpretation_rule": {
            "negative_margin_delta":
                (
                    "Ablation reduced the preferred "
                    "minus comparison margin. "
                    "The component causally supports "
                    "that margin for this prompt."
                ),

            "positive_margin_delta":
                (
                    "Ablation increased the preferred "
                    "minus comparison margin. "
                    "The component causally suppresses "
                    "that margin for this prompt."
                ),

            "scope_warning":
                (
                    "This result applies to this "
                    "specific intervention and prompt. "
                    "It does not establish the "
                    "component's general function."
                ),
        },

        "artifacts": {
            "scorer_artifact":
                str(
                    scorer_path
                ),

            "json":
                str(
                    Path(
                        args.output
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
        "Intervention"
    )

    print(
        "------------"
    )

    print(
        f"Component: "
        f"{args.component}"
    )

    print(
        "Scale: 0.0 "
        "(complete temporary ablation)"
    )

    print()

    print(
        f"Original component last-token L2: "
        f"{intervention_record['original_last_token_l2']:.8f}"
    )

    print(
        f"Ablated component last-token L2: "
        f"{intervention_record['intervened_last_token_l2']:.8f}"
    )

    # ========================================================
    # Control
    # ========================================================

    print()
    print(
        "Control verification"
    )

    print(
        "--------------------"
    )

    print(
        f"Scale-1 control exact: "
        f"{control_exact}"
    )

    print(
        f"Max absolute difference: "
        f"{control_max_difference}"
    )

    # ========================================================
    # Baseline
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "BASELINE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        (
            f"{repr(baseline_targets['preferred']['token_text'])} "
            f"rank="
            f"{baseline_targets['preferred']['rank']:,} "
            f"p="
            f"{baseline_targets['preferred']['probability']:.10f}"
        )
    )

    print(
        (
            f"{repr(baseline_targets['comparison']['token_text'])} "
            f"rank="
            f"{baseline_targets['comparison']['rank']:,} "
            f"p="
            f"{baseline_targets['comparison']['probability']:.10f}"
        )
    )

    print()

    print(
        f"Logit margin: "
        f"{baseline_targets['logit_margin']:+.8f}"
    )

    print(
        f"Probability ratio: "
        f"{baseline_targets['probability_ratio']:.6f}"
    )

    # ========================================================
    # Ablated
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "ABLATION RESULT"
    )

    print(
        "=" * 76
    )

    print()

    print(
        (
            f"{repr(ablated_targets['preferred']['token_text'])} "
            f"rank="
            f"{ablated_targets['preferred']['rank']:,} "
            f"p="
            f"{ablated_targets['preferred']['probability']:.10f}"
        )
    )

    print(
        (
            f"{repr(ablated_targets['comparison']['token_text'])} "
            f"rank="
            f"{ablated_targets['comparison']['rank']:,} "
            f"p="
            f"{ablated_targets['comparison']['probability']:.10f}"
        )
    )

    print()

    print(
        f"Logit margin: "
        f"{ablated_targets['logit_margin']:+.8f}"
    )

    print(
        f"Probability ratio: "
        f"{ablated_targets['probability_ratio']:.6f}"
    )

    # ========================================================
    # Effect
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "CAUSAL EFFECT"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Margin delta: "
        f"{margin_delta:+.8f}"
    )

    print(
        f"Preferred rank delta: "
        f"{preferred_rank_delta:+d}"
    )

    print(
        f"Comparison rank delta: "
        f"{comparison_rank_delta:+d}"
    )

    print()

    print(
        f"Direction: "
        f"{causal_direction}"
    )

    print(
        f"Preferred still wins: "
        f"{ablated_targets['preferred_wins']}"
    )

    print()

    print(
        "Distribution change:"
    )

    print(
        (
            f"  KL(base || ablated): "
            f"{distribution_change['kl_baseline_to_intervention']:.8f}"
        )
    )

    print(
        (
            f"  probability L1: "
            f"{distribution_change['probability_l1_distance']:.8f}"
        )
    )

    print(
        (
            f"  logit L2: "
            f"{distribution_change['logit_l2_difference']:.8f}"
        )
    )

    print(
        (
            f"  top-1 changed: "
            f"{distribution_change['top1_changed']}"
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
            "L6.7 component ablation failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.7 COMPONENT "
        "ABLATION PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()