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
    token_text,
)

from laboratory.tools.attention_head_ablation import (
    load_json,
    extract_targets,
    measure_targets,
    distribution_metrics,
    forward_with_head_scale,
    attention_focus_lookup,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.9 activation_scaler — "
            "perform controlled dose-response "
            "scaling of one attention head."
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
    )

    parser.add_argument(
        "--head-ablation-artifact",
        type=str,
        default=None,
        help=(
            "Optional L6.8 artifact used to "
            "cross-check the scale=0 result."
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
        required=True,
        type=int,
    )

    parser.add_argument(
        "--scale",
        action="append",
        type=float,
        default=None,
        help=(
            "Head activation scale. "
            "Repeat for multiple values. "
            "L6.9 v0.1 restricts scaling "
            "to the interval [0, 1]."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# Head-ablation cross-check
# ============================================================

def find_head_ablation_result(
    artifact,
    layer,
    head,
):

    head_id = (
        f"L{layer}H{head}"
    )

    for result in artifact[
        "results"
    ]:

        if (
            result[
                "head_id"
            ]
            ==
            head_id
        ):

            return result

    raise RuntimeError(
        f"{head_id} was not found in "
        "the supplied L6.8 artifact."
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
        "L6.9 — ACTIVATION SCALER"
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
        "(controlled causal dose-response)"
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
    # Validate layer/head
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

    if not (
        0
        <=
        args.head
        <
        runtime.model.num_heads
    ):

        raise ValueError(
            f"Head must be between 0 and "
            f"{runtime.model.num_heads - 1}."
        )

    head_id = (
        f"L{args.layer}H{args.head}"
    )

    # ========================================================
    # Scales
    # ========================================================

    if args.scale is None:

        scales = [
            1.00,
            0.75,
            0.50,
            0.25,
            0.00,
        ]

    else:

        scales = [
            float(
                value
            )

            for value
            in args.scale
        ]

    if len(
        scales
    ) < 3:

        raise ValueError(
            "L6.9 requires at least "
            "three scale values."
        )

    if len(
        scales
    ) != len(
        set(
            scales
        )
    ):

        raise ValueError(
            "Duplicate scale values "
            "are not allowed."
        )

    for scale in scales:

        if not np.isfinite(
            scale
        ):

            raise ValueError(
                "Scale values must be finite."
            )

        if not (
            0.0
            <=
            scale
            <=
            1.0
        ):

            raise ValueError(
                "L6.9 v0.1 permits scales "
                "only between 0.0 and 1.0."
            )

    if 1.0 not in scales:

        raise ValueError(
            "Scale 1.0 is required as "
            "the no-intervention control."
        )

    if 0.0 not in scales:

        raise ValueError(
            "Scale 0.0 is required to "
            "connect L6.9 to L6.8 ablation."
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
            "the validated L6.2 context."
        )

    # ========================================================
    # Targets
    # ========================================================

    target_records = (
        extract_targets(
            scorer_artifact
        )
    )

    if len(
        target_records
    ) < 2:

        raise RuntimeError(
            "At least two discriminative "
            "targets are required."
        )

    if not (
        0
        <=
        args.preferred_target_index
        <
        len(
            target_records
        )
    ):

        raise ValueError(
            "Preferred target index invalid."
        )

    if not (
        0
        <=
        args.comparison_target_index
        <
        len(
            target_records
        )
    ):

        raise ValueError(
            "Comparison target index invalid."
        )

    if (
        args.preferred_target_index
        ==
        args.comparison_target_index
    ):

        raise ValueError(
            "Preferred and comparison "
            "targets must differ."
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
    # Optional L6.5 attention artifact
    # ========================================================

    attention_path = None
    attention_artifact = None
    focus_mass = None

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

        focus_mass = (
            focus_lookup.get(
                head_id
            )
        )

    # ========================================================
    # Optional L6.8 head-ablation artifact
    # ========================================================

    ablation_path = None
    ablation_artifact = None
    previous_ablation_result = None

    if (
        args.head_ablation_artifact
        is not None
    ):

        (
            ablation_path,
            ablation_artifact,
        ) = load_json(
            args.head_ablation_artifact
        )

        if (
            ablation_artifact[
                "model"
            ][
                "checkpoint_sha256"
            ]
            !=
            runtime.checkpoint_sha256
        ):

            raise RuntimeError(
                "L6.8 checkpoint mismatch."
            )

        if (
            int(
                ablation_artifact[
                    "layer"
                ]
            )
            !=
            args.layer
        ):

            raise RuntimeError(
                "L6.8 layer mismatch."
            )

        previous_ablation_result = (
            find_head_ablation_result(
                ablation_artifact,
                args.layer,
                args.head,
            )
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
    # True baseline
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

    baseline_margin = float(
        baseline_targets[
            "logit_margin"
        ]
    )

    # ========================================================
    # Scale sweep
    # ========================================================

    scale_results = []

    for scale in scales:

        (
            scaled_logits,
            intervention_record,
        ) = forward_with_head_scale(
            runtime=runtime,

            token_ids=input_ids,

            target_layer=(
                args.layer
            ),

            target_head=(
                args.head
            ),

            head_scale=(
                scale
            ),
        )

        targets = (
            measure_targets(
                runtime=runtime,

                logits=scaled_logits,

                preferred_token_id=(
                    preferred_token_id
                ),

                comparison_token_id=(
                    comparison_token_id
                ),
            )
        )

        margin_delta = float(
            targets[
                "logit_margin"
            ]
            -
            baseline_margin
        )

        distribution = (
            distribution_metrics(
                baseline_logits[
                    0,
                    -1,
                    :
                ],

                scaled_logits[
                    0,
                    -1,
                    :
                ],
            )
        )

        exact_baseline_match = (
            bool(
                np.array_equal(
                    baseline_logits,
                    scaled_logits,
                )
            )
            if scale == 1.0
            else None
        )

        max_baseline_difference = (
            float(
                np.max(
                    np.abs(
                        baseline_logits.astype(
                            np.float64
                        )
                        -
                        scaled_logits.astype(
                            np.float64
                        )
                    )
                )
            )
            if scale == 1.0
            else None
        )

        scale_results.append(
            {
                "scale":
                    float(
                        scale
                    ),

                "intervention":
                    intervention_record,

                "targets":
                    targets,

                "effect": {
                    "logit_margin_delta":
                        margin_delta,

                    "preferred_rank_delta":
                        int(
                            targets[
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
                        ),

                    "comparison_rank_delta":
                        int(
                            targets[
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
                        ),

                    "distribution_change":
                        distribution,
                },

                "baseline_control": {
                    "applicable":
                        bool(
                            scale
                            ==
                            1.0
                        ),

                    "exact":
                        exact_baseline_match,

                    "max_absolute_difference":
                        max_baseline_difference,
                },
            }
        )

    # ========================================================
    # Find endpoint results
    # ========================================================

    result_by_scale = {
        float(
            result[
                "scale"
            ]
        ):
            result

        for result
        in scale_results
    }

    scale_one_result = (
        result_by_scale[
            1.0
        ]
    )

    scale_zero_result = (
        result_by_scale[
            0.0
        ]
    )

    # ========================================================
    # Dose-response analysis
    # ========================================================

    sorted_results = sorted(
        scale_results,
        key=lambda result: (
            result[
                "scale"
            ]
        ),
    )

    sorted_scales = np.asarray(
        [
            result[
                "scale"
            ]

            for result
            in sorted_results
        ],
        dtype=np.float64,
    )

    sorted_margins = np.asarray(
        [
            result[
                "targets"
            ][
                "logit_margin"
            ]

            for result
            in sorted_results
        ],
        dtype=np.float64,
    )

    tolerance = 1e-10

    margin_nondecreasing_with_scale = all(
        (
            sorted_margins[
                index + 1
            ]
            >=
            sorted_margins[
                index
            ]
            -
            tolerance
        )

        for index
        in range(
            len(
                sorted_margins
            )
            -
            1
        )
    )

    margin_nonincreasing_with_scale = all(
        (
            sorted_margins[
                index + 1
            ]
            <=
            sorted_margins[
                index
            ]
            +
            tolerance
        )

        for index
        in range(
            len(
                sorted_margins
            )
            -
            1
        )
    )

    slope, intercept = np.polyfit(
        sorted_scales,
        sorted_margins,
        deg=1,
    )

    predicted_margins = (
        slope
        *
        sorted_scales
        +
        intercept
    )

    residual_sum_squares = float(
        np.sum(
            (
                sorted_margins
                -
                predicted_margins
            )
            **
            2
        )
    )

    total_sum_squares = float(
        np.sum(
            (
                sorted_margins
                -
                np.mean(
                    sorted_margins
                )
            )
            **
            2
        )
    )

    if total_sum_squares > 0.0:

        r_squared = float(
            1.0
            -
            (
                residual_sum_squares
                /
                total_sum_squares
            )
        )

    else:

        r_squared = 1.0

    if (
        np.std(
            sorted_scales
        )
        >
        0.0
        and
        np.std(
            sorted_margins
        )
        >
        0.0
    ):

        correlation = float(
            np.corrcoef(
                sorted_scales,
                sorted_margins,
            )[
                0,
                1
            ]
        )

    else:

        correlation = 0.0

    # ========================================================
    # Preference-flip intervals
    # ========================================================

    flip_intervals = []

    for index in range(
        len(
            sorted_results
        )
        -
        1
    ):

        lower = (
            sorted_results[
                index
            ]
        )

        upper = (
            sorted_results[
                index + 1
            ]
        )

        lower_margin = float(
            lower[
                "targets"
            ][
                "logit_margin"
            ]
        )

        upper_margin = float(
            upper[
                "targets"
            ][
                "logit_margin"
            ]
        )

        if (
            lower_margin
            ==
            0.0
            or
            upper_margin
            ==
            0.0
            or
            (
                lower_margin
                <
                0.0
                <
                upper_margin
            )
            or
            (
                upper_margin
                <
                0.0
                <
                lower_margin
            )
        ):

            flip_intervals.append(
                {
                    "lower_scale":
                        float(
                            lower[
                                "scale"
                            ]
                        ),

                    "lower_margin":
                        lower_margin,

                    "upper_scale":
                        float(
                            upper[
                                "scale"
                            ]
                        ),

                    "upper_margin":
                        upper_margin,
                }
            )

    # ========================================================
    # L6.8 cross-check
    # ========================================================

    ablation_crosscheck = {
        "available":
            bool(
                previous_ablation_result
                is not None
            ),

        "passed":
            None,
    }

    if (
        previous_ablation_result
        is not None
    ):

        previous_margin = float(
            previous_ablation_result[
                "ablated"
            ][
                "logit_margin"
            ]
        )

        previous_delta = float(
            previous_ablation_result[
                "effect"
            ][
                "logit_margin_delta"
            ]
        )

        current_margin = float(
            scale_zero_result[
                "targets"
            ][
                "logit_margin"
            ]
        )

        current_delta = float(
            scale_zero_result[
                "effect"
            ][
                "logit_margin_delta"
            ]
        )

        margin_difference = abs(
            current_margin
            -
            previous_margin
        )

        delta_difference = abs(
            current_delta
            -
            previous_delta
        )

        ablation_crosscheck = {
            "available":
                True,

            "passed":
                bool(
                    margin_difference
                    <=
                    1e-12
                    and
                    delta_difference
                    <=
                    1e-12
                ),

            "previous_margin":
                previous_margin,

            "current_margin":
                current_margin,

            "margin_difference":
                float(
                    margin_difference
                ),

            "previous_margin_delta":
                previous_delta,

            "current_margin_delta":
                current_delta,

            "margin_delta_difference":
                float(
                    delta_difference
                ),
        }

    # ========================================================
    # Checks
    # ========================================================

    all_measurements_finite = all(
        np.isfinite(
            result[
                "targets"
            ][
                "logit_margin"
            ]
        )

        and
        np.isfinite(
            result[
                "effect"
            ][
                "logit_margin_delta"
            ]
        )

        for result
        in scale_results
    )

    scale_one_exact = bool(
        scale_one_result[
            "baseline_control"
        ][
            "exact"
        ]
    )

    scale_zero_l2 = float(
        scale_zero_result[
            "intervention"
        ][
            "intervened_last_token_l2"
        ]
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "scale_count_at_least_three":
            (
                len(
                    scales
                )
                >=
                3
            ),

        "scale_one_control_exact":
            scale_one_exact,

        "scale_zero_fully_ablates_head":
            (
                scale_zero_l2
                ==
                0.0
            ),

        "all_measurements_finite":
            bool(
                all_measurements_finite
            ),
    }

    if (
        previous_ablation_result
        is not None
    ):

        checks[
            "l6_8_zero_scale_crosscheck"
        ] = bool(
            ablation_crosscheck[
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
                "activation_scaler",

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

        "target_head": {
            "layer":
                int(
                    args.layer
                ),

            "head":
                int(
                    args.head
                ),

            "head_id":
                head_id,

            "attention_focus_mass":
                focus_mass,
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

        "baseline":
            baseline_targets,

        "scale_results":
            scale_results,

        "dose_response": {
            "scales_sorted":
                [
                    float(
                        value
                    )

                    for value
                    in sorted_scales
                ],

            "margins_sorted":
                [
                    float(
                        value
                    )

                    for value
                    in sorted_margins
                ],

            "margin_nondecreasing_with_scale":
                bool(
                    margin_nondecreasing_with_scale
                ),

            "margin_nonincreasing_with_scale":
                bool(
                    margin_nonincreasing_with_scale
                ),

            "linear_slope":
                float(
                    slope
                ),

            "linear_intercept":
                float(
                    intercept
                ),

            "linear_r_squared":
                float(
                    r_squared
                ),

            "scale_margin_correlation":
                float(
                    correlation
                ),

            "zero_to_one_margin_change":
                float(
                    scale_one_result[
                        "targets"
                    ][
                        "logit_margin"
                    ]
                    -
                    scale_zero_result[
                        "targets"
                    ][
                        "logit_margin"
                    ]
                ),

            "preference_flip_intervals":
                flip_intervals,
        },

        "crosschecks": {
            "l6_8_ablation":
                ablation_crosscheck,
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

            "head_ablation_artifact":
                (
                    None
                    if ablation_path is None
                    else str(
                        ablation_path
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
        "Target head"
    )

    print(
        "-----------"
    )

    print(
        f"{head_id}"
    )

    if focus_mass is not None:

        print(
            f"Attention focus mass: "
            f"{focus_mass:.6f}"
        )

    print()

    print(
        f"Preferred: "
        f"{repr(token_text(runtime.tokenizer, preferred_token_id))}"
    )

    print(
        f"Comparison: "
        f"{repr(token_text(runtime.tokenizer, comparison_token_id))}"
    )

    print()

    print(
        f"Baseline margin: "
        f"{baseline_margin:+.8f}"
    )

    # ========================================================
    # Dose-response table
    # ========================================================

    print()
    print(
        "=" * 92
    )

    print(
        "HEAD ACTIVATION DOSE-RESPONSE"
    )

    print(
        "=" * 92
    )

    print()

    print(
        (
            f"{'Scale':>8}"
            f"{'Margin':>14}"
            f"{'Δ Margin':>14}"
            f"{'Four Rank':>14}"
            f"{'Fiv Rank':>14}"
            f"{'Ratio':>12}"
            f"{'KL':>12}"
        )
    )

    print(
        "-" * 88
    )

    for result in scale_results:

        targets = (
            result[
                "targets"
            ]
        )

        effect = (
            result[
                "effect"
            ]
        )

        print(
            (
                f"{result['scale']:>8.2f}"
                f"{targets['logit_margin']:>+14.6f}"
                f"{effect['logit_margin_delta']:>+14.6f}"
                f"{targets['preferred']['rank']:>14,}"
                f"{targets['comparison']['rank']:>14,}"
                f"{targets['probability_ratio']:>12.4f}"
                f"{effect['distribution_change']['kl_baseline_to_intervention']:>12.6f}"
            )
        )

    # ========================================================
    # Dose-response analysis
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "DOSE-RESPONSE ANALYSIS"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Margin nondecreasing with scale: "
        f"{margin_nondecreasing_with_scale}"
    )

    print(
        f"Margin nonincreasing with scale: "
        f"{margin_nonincreasing_with_scale}"
    )

    print(
        f"Linear slope: "
        f"{slope:+.8f}"
    )

    print(
        f"Linear R²: "
        f"{r_squared:.8f}"
    )

    print(
        f"Scale/margin correlation: "
        f"{correlation:+.8f}"
    )

    print(
        (
            f"Scale 0 → 1 margin change: "
            f"{report['dose_response']['zero_to_one_margin_change']:+.8f}"
        )
    )

    if flip_intervals:

        print()
        print(
            "Preference-flip interval(s):"
        )

        for interval in flip_intervals:

            print(
                (
                    f"  scale "
                    f"{interval['lower_scale']:.2f} "
                    f"(margin "
                    f"{interval['lower_margin']:+.6f}) "
                    f"→ "
                    f"{interval['upper_scale']:.2f} "
                    f"(margin "
                    f"{interval['upper_margin']:+.6f})"
                )
            )

    else:

        print()
        print(
            "No preference flip observed "
            "within tested scales."
        )

    # ========================================================
    # Cross-check
    # ========================================================

    if (
        previous_ablation_result
        is not None
    ):

        print()
        print(
            "L6.8 zero-scale cross-check"
        )

        print(
            "--------------------------"
        )

        print(
            f"Passed: "
            f"{ablation_crosscheck['passed']}"
        )

        print(
            (
                f"Margin difference: "
                f"{ablation_crosscheck['margin_difference']}"
            )
        )

        print(
            (
                f"Δ-margin difference: "
                f"{ablation_crosscheck['margin_delta_difference']}"
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
            "L6.9 activation scaler failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.9 ACTIVATION "
        "SCALER PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()