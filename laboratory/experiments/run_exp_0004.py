import argparse
import hashlib
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
)

from laboratory.tools.continuation_scorer import (
    score_continuation,
    build_discriminative_analysis,
    build_preference_comparisons,
)

from laboratory.tools.attention_head_ablation import (
    forward_with_head_scale,
    measure_targets,
    distribution_metrics,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "EXP-0004 — test whether mature "
            "v0.3 competing attention-head "
            "effects generalize across contexts."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--expected-sha256",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--config",
        default=(
            "laboratory/experiments/"
            "exp_0004_competing_heads.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "laboratory/artifacts/"
            "exp_0004_competing_heads"
        ),
    )

    parser.add_argument(
        "--summary",
        default=(
            "laboratory/artifacts/"
            "exp_0004_competing_heads_summary.json"
        ),
    )

    parser.add_argument(
        "--probe-id",
        action="append",
        default=[],
        help=(
            "Optional probe ID to run. "
            "Repeat for multiple probes. "
            "If omitted, all probes run."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Paths / JSON
# ============================================================

def project_path(
    value,
):

    path = Path(
        value
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    return path


def load_json(
    path,
):

    path = project_path(
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

        return json.load(
            file
        )


def save_json(
    path,
    value,
):

    path = project_path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        +
        ".tmp"
    )

    if temporary.exists():

        temporary.unlink()

    try:

        with open(
            temporary,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                value,
                file,
                ensure_ascii=False,
                indent=4,
            )

        temporary.replace(
            path
        )

    finally:

        if temporary.exists():

            temporary.unlink()


# ============================================================
# SHA-256
# ============================================================

def sha256_file(
    path,
):

    path = Path(
        path
    )

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024
                *
                1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# Display helpers
# ============================================================

def format_signed(
    value,
):

    if value is None:

        return "N/A"

    return (
        f"{float(value):+.6f}"
    )


def format_percentage(
    value,
):

    if value is None:

        return "N/A"

    return (
        f"{float(value):.2%}"
    )


# ============================================================
# Summary statistics
# ============================================================

def summarize_effects(
    records,
):

    if not records:

        return {
            "count":
                0,

            "mean_delta":
                None,

            "median_delta":
                None,

            "mean_absolute_delta":
                None,

            "supports_preferred_count":
                0,

            "suppresses_preferred_count":
                0,

            "no_effect_count":
                0,

            "supports_preferred_rate":
                None,

            "suppresses_preferred_rate":
                None,
        }

    values = np.asarray(
        [
            record[
                "logit_margin_delta"
            ]

            for record
            in records
        ],
        dtype=np.float64,
    )

    supports = int(
        np.sum(
            values
            <
            0.0
        )
    )

    suppresses = int(
        np.sum(
            values
            >
            0.0
        )
    )

    no_effect = int(
        len(values)
        -
        supports
        -
        suppresses
    )

    return {
        "count":
            int(
                len(
                    values
                )
            ),

        "mean_delta":
            float(
                np.mean(
                    values
                )
            ),

        "median_delta":
            float(
                np.median(
                    values
                )
            ),

        "mean_absolute_delta":
            float(
                np.mean(
                    np.abs(
                        values
                    )
                )
            ),

        "supports_preferred_count":
            supports,

        "suppresses_preferred_count":
            suppresses,

        "no_effect_count":
            no_effect,

        "supports_preferred_rate":
            float(
                supports
                /
                len(
                    values
                )
            ),

        "suppresses_preferred_rate":
            float(
                suppresses
                /
                len(
                    values
                )
            ),
    }


# ============================================================
# Experiment definition checks
# ============================================================

def validate_experiment_definition(
    *,
    config,
    probe_config,
):

    checks = {
        "experiment_id":
            (
                config.get(
                    "experiment_id"
                )
                ==
                "EXP-0004"
            ),

        "probe_config_id":
            (
                probe_config.get(
                    "experiment_id"
                )
                ==
                "EXP-0002"
            ),

        "anchor_probe_defined":
            (
                config.get(
                    "anchor_probe_id"
                )
                is not None
            ),

        "heads_defined":
            (
                isinstance(
                    config.get(
                        "heads"
                    ),
                    list,
                )
                and
                len(
                    config.get(
                        "heads"
                    )
                )
                >
                0
            ),

        "probes_defined":
            (
                isinstance(
                    probe_config.get(
                        "probes"
                    ),
                    list,
                )
                and
                len(
                    probe_config.get(
                        "probes"
                    )
                )
                >
                0
            ),
    }

    if not all(
        checks.values()
    ):

        raise RuntimeError(
            "EXP-0004 definition failed: "
            f"{checks}"
        )

    return checks


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    checkpoint_path = project_path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    config = load_json(
        args.config
    )

    probe_config = load_json(
        config[
            "probe_config"
        ]
    )

    definition_checks = (
        validate_experiment_definition(
            config=config,
            probe_config=probe_config,
        )
    )

    output_dir = project_path(
        args.output_dir
    )

    summary_path = project_path(
        args.summary
    )

    # ========================================================
    # Checkpoint fingerprint before loading model
    # ========================================================

    actual_sha256 = sha256_file(
        checkpoint_path
    )

    if (
        actual_sha256
        !=
        args.expected_sha256
    ):

        raise RuntimeError(
            "Checkpoint SHA-256 mismatch.\n"
            f"Expected: {args.expected_sha256}\n"
            f"Actual:   {actual_sha256}"
        )

    # ========================================================
    # Target heads
    # ========================================================

    heads = list(
        config[
            "heads"
        ]
    )

    head_ids = [
        item[
            "head_id"
        ]

        for item
        in heads
    ]

    if len(
        set(
            head_ids
        )
    ) != len(
        head_ids
    ):

        raise RuntimeError(
            "Duplicate target heads."
        )

    for head_spec in heads:

        expected_head_id = (
            f"L{int(head_spec['layer'])}"
            f"H{int(head_spec['head'])}"
        )

        if (
            head_spec[
                "head_id"
            ]
            !=
            expected_head_id
        ):

            raise RuntimeError(
                "Head identity mismatch for "
                f"{head_spec}."
            )

    # ========================================================
    # Select probes
    # ========================================================

    probes = list(
        probe_config[
            "probes"
        ]
    )

    if args.probe_id:

        requested = set(
            args.probe_id
        )

        probes = [
            probe

            for probe
            in probes

            if probe[
                "id"
            ]
            in requested
        ]

        found = {
            probe[
                "id"
            ]

            for probe
            in probes
        }

        missing = (
            requested
            -
            found
        )

        if missing:

            raise ValueError(
                "Unknown probe IDs: "
                f"{sorted(missing)}"
            )

    if not probes:

        raise RuntimeError(
            "No probes selected."
        )

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "=" * 84
    )

    print(
        "EXP-0004 — MATURE v0.3 "
        "COMPETING HEAD GENERALIZATION"
    )

    print(
        "=" * 84
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        checkpoint_path
    )

    print()
    print(
        "SHA-256:"
    )

    print(
        actual_sha256
    )

    print()
    print(
        "Target heads:"
    )

    for head in heads:

        print(
            f"  {head['head_id']:<6} "
            f"{head['anchor_role']}"
        )

    print()
    print(
        f"Selected probes: "
        f"{len(probes)}"
    )

    # ========================================================
    # Load checkpoint ONCE
    # ========================================================

    print()
    print(
        "Loading mature v0.3 checkpoint once..."
    )

    runtime = LaboratoryRuntime(
        checkpoint_path=(
            checkpoint_path
        ),

        expected_checkpoint_sha256=(
            args.expected_sha256
        ),
    )

    # ========================================================
    # Validate target head bounds against model
    # ========================================================

    for head_spec in heads:

        layer = int(
            head_spec[
                "layer"
            ]
        )

        head = int(
            head_spec[
                "head"
            ]
        )

        if not (
            0
            <=
            layer
            <
            runtime.model.num_layers
        ):

            raise ValueError(
                f"Invalid layer {layer}."
            )

        if not (
            0
            <=
            head
            <
            runtime.model.num_heads
        ):

            raise ValueError(
                f"Invalid head {head}."
            )

    if args.preflight_only:

        print()
        print(
            "Definition checks:"
        )

        for name, passed in (
            definition_checks.items()
        ):

            print(
                f"  {name}: {passed}"
            )

        print()
        print(
            "Model identity checks:"
        )

        for name, passed in (
            runtime
            .identity_checks
            .items()
        ):

            print(
                f"  {name}: {passed}"
            )

        print()
        print(
            "=" * 84
        )

        print(
            "SUCCESS: EXP-0004 PREFLIGHT PASSED."
        )

        print(
            "No causal interventions were run."
        )

        print(
            "=" * 84
        )

        return

    # ========================================================
    # Output protection
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        summary_path.exists()
        and
        not args.overwrite
    ):

        raise FileExistsError(
            f"Summary already exists: "
            f"{summary_path}\n"
            "Use --overwrite to rebuild."
        )

    # ========================================================
    # Run probes
    # ========================================================

    experiment_records = []

    all_effect_records = []

    for index, probe in enumerate(
        probes,
        start=1,
    ):

        probe_id = (
            probe[
                "id"
            ]
        )

        prompt = (
            probe[
                "prompt"
            ]
        )

        preferred_text = (
            probe[
                "preferred"
            ]
        )

        comparison_text = (
            probe[
                "comparison"
            ]
        )

        output_path = (
            output_dir
            /
            (
                f"{probe_id.lower()}"
                "_competing_heads.json"
            )
        )

        if (
            output_path.exists()
            and
            not args.overwrite
        ):

            raise FileExistsError(
                f"Artifact already exists: "
                f"{output_path}"
            )

        print()
        print(
            "=" * 84
        )

        print(
            f"[{index}/{len(probes)}] "
            f"{probe_id}"
        )

        print(
            "=" * 84
        )

        print(
            f"Category: "
            f"{probe['category']}"
        )

        print(
            f"Prompt: "
            f"{prompt}"
        )

        print(
            f"Preferred: "
            f"{repr(preferred_text)}"
        )

        print(
            f"Comparison: "
            f"{repr(comparison_text)}"
        )

        # ====================================================
        # Behavioral continuation scoring
        # ====================================================

        candidate_results = []

        for candidate_index, continuation in enumerate(
            [
                preferred_text,
                comparison_text,
            ]
        ):

            candidate = score_continuation(
                runtime=runtime,
                prompt=prompt,
                continuation=continuation,
            )

            candidate[
                "candidate_index"
            ] = int(
                candidate_index
            )

            candidate_results.append(
                candidate
            )

        discriminative = (
            build_discriminative_analysis(
                runtime=runtime,
                prompt=prompt,
                candidate_results=(
                    candidate_results
                ),
            )
        )

        preference_comparisons = (
            build_preference_comparisons(
                candidate_results=(
                    candidate_results
                ),
                preferred_index=0,
            )
        )

        candidates_by_index = {
            int(
                item[
                    "candidate_index"
                ]
            ):
                item

            for item
            in discriminative[
                "candidates"
            ]

            if item.get(
                "has_discriminative_token",
                False,
            )
        }

        if (
            0
            not in candidates_by_index
            or
            1
            not in candidates_by_index
        ):

            raise RuntimeError(
                f"{probe_id}: both candidates "
                "must have discriminative tokens."
            )

        preferred_token_id = int(
            candidates_by_index[
                0
            ][
                "token_id"
            ]
        )

        comparison_token_id = int(
            candidates_by_index[
                1
            ][
                "token_id"
            ]
        )

        context_ids = [
            int(
                token_id
            )

            for token_id
            in discriminative[
                "scoring_context_token_ids"
            ]
        ]

        input_ids = np.asarray(
            [
                context_ids
            ],
            dtype=np.int64,
        )

        # ====================================================
        # Baseline discriminative margin
        # ====================================================

        baseline_logits = (
            runtime
            .model
            .forward(
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

        print()
        print(
            f"Baseline discriminative margin: "
            f"{baseline_targets['logit_margin']:+.6f}"
        )

        print(
            f"Preferred wins: "
            f"{baseline_targets['preferred_wins']}"
        )

        # ====================================================
        # Head interventions
        # ====================================================

        head_results = []

        for head_spec in heads:

            layer = int(
                head_spec[
                    "layer"
                ]
            )

            head = int(
                head_spec[
                    "head"
                ]
            )

            head_id = (
                head_spec[
                    "head_id"
                ]
            )

            # --------------------------------------------
            # Scale-1 exactness control
            # --------------------------------------------

            (
                control_logits,
                _,
            ) = forward_with_head_scale(
                runtime=runtime,
                token_ids=input_ids,
                target_layer=layer,
                target_head=head,
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

            # --------------------------------------------
            # Zero-ablation
            # --------------------------------------------

            (
                ablated_logits,
                intervention_record,
            ) = forward_with_head_scale(
                runtime=runtime,
                token_ids=input_ids,
                target_layer=layer,
                target_head=head,
                head_scale=0.0,
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

            result = {
                "layer":
                    layer,

                "head":
                    head,

                "head_id":
                    head_id,

                "anchor_role":
                    head_spec[
                        "anchor_role"
                    ],

                "control_scale_1_exact":
                    control_exact,

                "control_scale_1_max_difference":
                    control_max_difference,

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

            head_results.append(
                result
            )

            all_effect_records.append(
                {
                    "probe_id":
                        probe_id,

                    "category":
                        probe[
                            "category"
                        ],

                    "head_id":
                        head_id,

                    "logit_margin_delta":
                        margin_delta,

                    "absolute_logit_margin_delta":
                        float(
                            abs(
                                margin_delta
                            )
                        ),

                    "causal_direction":
                        causal_direction,
                }
            )

            print(
                f"{head_id:<6} "
                f"delta={margin_delta:+.6f} "
                f"{causal_direction}"
            )

        # ====================================================
        # Per-probe integrity
        # ====================================================

        checks = {
            "runtime_identity":
                all(
                    runtime
                    .identity_checks
                    .values()
                ),

            "two_discriminative_candidates":
                (
                    0
                    in candidates_by_index
                    and
                    1
                    in candidates_by_index
                ),

            "all_scale_1_controls_exact":
                all(
                    result[
                        "control_scale_1_exact"
                    ]

                    for result
                    in head_results
                ),

            "all_effects_finite":
                all(
                    np.isfinite(
                        result[
                            "effect"
                        ][
                            "logit_margin_delta"
                        ]
                    )

                    for result
                    in head_results
                ),

            "all_heads_zeroed":
                all(
                    np.isclose(
                        result[
                            "intervention"
                        ][
                            "intervened_last_token_l2"
                        ],
                        0.0,
                    )

                    for result
                    in head_results
                ),
        }

        if not all(
            checks.values()
        ):

            raise RuntimeError(
                f"{probe_id} failed "
                f"integrity checks: "
                f"{checks}"
            )

        # ====================================================
        # Per-probe artifact
        # ====================================================

        probe_report = {
            "mars_protocol":
                MARS_PROTOCOL_VERSION,

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "experiment_id":
                config[
                    "experiment_id"
                ],

            "status":
                "success",

            "model":
                runtime.model_record(),

            "probe":
                probe,

            "continuation_scoring": {
                "candidates":
                    candidate_results,

                "discriminative_analysis":
                    discriminative,

                "preference_comparisons":
                    preference_comparisons,
            },

            "baseline":
                baseline_targets,

            "head_results":
                head_results,

            "checks":
                checks,
        }

        save_json(
            output_path,
            probe_report,
        )

        experiment_records.append(
            {
                "probe_id":
                    probe_id,

                "category":
                    probe[
                        "category"
                    ],

                "prompt":
                    prompt,

                "preferred":
                    preferred_text,

                "comparison":
                    comparison_text,

                "baseline_margin":
                    float(
                        baseline_targets[
                            "logit_margin"
                        ]
                    ),

                "baseline_preferred_wins":
                    bool(
                        baseline_targets[
                            "preferred_wins"
                        ]
                    ),

                "artifact":
                    str(
                        output_path
                    ),

                "head_effects": {
                    item[
                        "head_id"
                    ]:
                        float(
                            item[
                                "effect"
                            ][
                                "logit_margin_delta"
                            ]
                        )

                    for item
                    in head_results
                },
            }
        )

    # ========================================================
    # Aggregate by head
    # ========================================================

    head_summaries = {}

    for head_spec in heads:

        head_id = (
            head_spec[
                "head_id"
            ]
        )

        all_records = [
            record

            for record
            in all_effect_records

            if record[
                "head_id"
            ]
            ==
            head_id
        ]

        addition_records = [
            record

            for record
            in all_records

            if record[
                "category"
            ]
            ==
            "addition"
        ]

        control_records = [
            record

            for record
            in all_records

            if record[
                "category"
            ]
            !=
            "addition"
        ]

        anchor_records = [
            record

            for record
            in all_records

            if record[
                "probe_id"
            ]
            ==
            config[
                "anchor_probe_id"
            ]
        ]

        head_summaries[
            head_id
        ] = {
            "anchor_role":
                head_spec[
                    "anchor_role"
                ],

            "all":
                summarize_effects(
                    all_records
                ),

            "addition":
                summarize_effects(
                    addition_records
                ),

            "controls":
                summarize_effects(
                    control_records
                ),

            "anchor_probe_delta":
                (
                    None
                    if not anchor_records
                    else float(
                        anchor_records[
                            0
                        ][
                            "logit_margin_delta"
                        ]
                    )
                ),
        }

    # ========================================================
    # Hypothesis diagnostics
    #
    # These are observations, not pass/fail integrity checks.
    # ========================================================

    def group_mean(
        head_id,
        group,
    ):

        head = head_summaries.get(
            head_id
        )

        if head is None:

            return None

        return head[
            group
        ][
            "mean_delta"
        ]

    def group_mean_abs(
        head_id,
        group,
    ):

        head = head_summaries.get(
            head_id
        )

        if head is None:

            return None

        return head[
            group
        ][
            "mean_absolute_delta"
        ]

    l1h0_add = group_mean(
        "L1H0",
        "addition",
    )

    l3h3_add = group_mean(
        "L3H3",
        "addition",
    )

    l4h0_add = group_mean(
        "L4H0",
        "addition",
    )

    l4h1_add_abs = group_mean_abs(
        "L4H1",
        "addition",
    )

    l4h0_add_abs = group_mean_abs(
        "L4H0",
        "addition",
    )

    hypothesis_diagnostics = {
        "L1H0_mean_addition_supports_preferred":
            (
                l1h0_add
                is not None
                and
                l1h0_add
                <
                0.0
            ),

        "L3H3_mean_addition_suppresses_preferred":
            (
                l3h3_add
                is not None
                and
                l3h3_add
                >
                0.0
            ),

        "L4H0_mean_addition_suppresses_preferred":
            (
                l4h0_add
                is not None
                and
                l4h0_add
                >
                0.0
            ),

        "L4H1_weaker_than_L4H0_on_addition":
            (
                l4h1_add_abs
                is not None
                and
                l4h0_add_abs
                is not None
                and
                l4h1_add_abs
                <
                l4h0_add_abs
            ),
    }

    # ========================================================
    # Final integrity
    # ========================================================

    final_checks = {
        "checkpoint_sha256_verified":
            (
                actual_sha256
                ==
                args.expected_sha256
            ),

        "runtime_identity":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "definition_checks":
            all(
                definition_checks.values()
            ),

        "all_selected_probes_completed":
            (
                len(
                    experiment_records
                )
                ==
                len(
                    probes
                )
            ),

        "expected_effect_record_count":
            (
                len(
                    all_effect_records
                )
                ==
                len(
                    probes
                )
                *
                len(
                    heads
                )
            ),

        "all_effects_finite":
            all(
                np.isfinite(
                    record[
                        "logit_margin_delta"
                    ]
                )

                for record
                in all_effect_records
            ),
    }

    if not all(
        final_checks.values()
    ):

        raise RuntimeError(
            "EXP-0004 final integrity "
            f"checks failed: {final_checks}"
        )

    # ========================================================
    # Final summary artifact
    # ========================================================

    summary = {
        "mars_protocol":
            MARS_PROTOCOL_VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment_id":
            config[
                "experiment_id"
            ],

        "title":
            config[
                "title"
            ],

        "question":
            config[
                "question"
            ],

        "status":
            "success",

        "tool_version":
            LAB_TOOL_VERSION,

        "evidence_level":
            "E3",

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_sha256":
            actual_sha256,

        "model":
            runtime.model_record(),

        "anchor_probe_id":
            config[
                "anchor_probe_id"
            ],

        "selected_probe_ids": [
            record[
                "probe_id"
            ]

            for record
            in experiment_records
        ],

        "probe_count":
            len(
                experiment_records
            ),

        "head_count":
            len(
                heads
            ),

        "heads":
            heads,

        "probes":
            experiment_records,

        "head_summaries":
            head_summaries,

        "hypothesis_diagnostics":
            hypothesis_diagnostics,

        "interpretation_rule": {
            "negative_logit_margin_delta":
                (
                    "The intact head causally "
                    "supports the preferred "
                    "discriminative-token margin."
                ),

            "positive_logit_margin_delta":
                (
                    "The intact head causally "
                    "suppresses the preferred "
                    "discriminative-token margin."
                ),
        },

        "checks":
            final_checks,
    }

    save_json(
        summary_path,
        summary,
    )

    # ========================================================
    # Display summary
    # ========================================================

    print()
    print(
        "=" * 84
    )

    print(
        "EXP-0004 SUMMARY"
    )

    print(
        "=" * 84
    )

    print()

    for head_id, result in (
        head_summaries.items()
    ):

        overall = (
            result[
                "all"
            ]
        )

        addition = (
            result[
                "addition"
            ]
        )

        controls = (
            result[
                "controls"
            ]
        )

        print(
            f"{head_id}"
        )

        print(
            f"  all mean delta: "
            f"{format_signed(overall['mean_delta'])}"
        )

        print(
            f"  addition mean delta: "
            f"{format_signed(addition['mean_delta'])}"
        )

        print(
            f"  addition preferred-support rate: "
            f"{format_percentage(addition['supports_preferred_rate'])}"
        )

        print(
            f"  control mean delta: "
            f"{format_signed(controls['mean_delta'])}"
        )

        print(
            f"  control preferred-support rate: "
            f"{format_percentage(controls['supports_preferred_rate'])}"
        )

        print(
            f"  anchor probe delta: "
            f"{format_signed(result['anchor_probe_delta'])}"
        )

        print()

    print(
        "Hypothesis diagnostics"
    )

    print(
        "----------------------"
    )

    for name, value in (
        hypothesis_diagnostics.items()
    ):

        print(
            f"{name}: {value}"
        )

    print()
    print(
        "Integrity checks"
    )

    print(
        "----------------"
    )

    for name, value in (
        final_checks.items()
    ):

        print(
            f"{name}: {value}"
        )

    print()
    print(
        "Summary artifact:"
    )

    print(
        summary_path
    )

    print()
    print(
        "=" * 84
    )

    print(
        "SUCCESS: EXP-0004 COMPLETE."
    )

    print(
        "=" * 84
    )


if __name__ == "__main__":

    main()
