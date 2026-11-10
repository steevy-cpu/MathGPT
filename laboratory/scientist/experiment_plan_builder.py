import argparse
import hashlib
import json
import sys

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path


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


if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


# ============================================================
# Identity
# ============================================================

ENGINE_VERSION = (
    "L9A.1"
)


EXPECTED_HANDOFF_VERSION = (
    "L8D.1"
)


EXPECTED_EXPERIMENT_ID = (
    "EXP-0005"
)


DEFAULT_HANDOFF_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "research_action_handoff.json"
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan.json"
)


# ============================================================
# EXP-0005 preregistered design
# ============================================================

FIXED_CANDIDATES = {
    "four":
        " four.",

    "five":
        " five.",
}


TARGET_HEADS = [
    {
        "layer": 1,
        "head": 0,
        "head_id": "L1H0",
        "role": "primary_four_direction_candidate",
    },
    {
        "layer": 4,
        "head": 0,
        "head_id": "L4H0",
        "role": "primary_five_direction_candidate",
    },
    {
        "layer": 3,
        "head": 3,
        "head_id": "L3H3",
        "role": "secondary_context_sensitive_candidate",
    },
    {
        "layer": 4,
        "head": 1,
        "head_id": "L4H1",
        "role": "weak_control",
    },
]


PROBES = [
    {
        "probe_id": "F01",
        "category": "arithmetic_words",
        "prompt": "Two plus two equals",
        "semantically_correct": "four",
    },
    {
        "probe_id": "F02",
        "category": "arithmetic_words",
        "prompt": "Two plus three equals",
        "semantically_correct": "five",
    },
    {
        "probe_id": "F03",
        "category": "arithmetic_words",
        "prompt": "Three plus two equals",
        "semantically_correct": "five",
    },
    {
        "probe_id": "F04",
        "category": "arithmetic_words",
        "prompt": "One plus three equals",
        "semantically_correct": "four",
    },
    {
        "probe_id": "F05",
        "category": "numeric_successor",
        "prompt": "Three is followed by",
        "semantically_correct": "four",
    },
    {
        "probe_id": "F06",
        "category": "numeric_successor",
        "prompt": "Four is followed by",
        "semantically_correct": "five",
    },
    {
        "probe_id": "F07",
        "category": "number_sequence",
        "prompt": "One, two, three,",
        "semantically_correct": "four",
    },
    {
        "probe_id": "F08",
        "category": "number_sequence",
        "prompt": "Two, three, four,",
        "semantically_correct": "five",
    },
    {
        "probe_id": "F09",
        "category": "quantity_language",
        "prompt": "Two groups of two make",
        "semantically_correct": "four",
    },
    {
        "probe_id": "F10",
        "category": "quantity_language",
        "prompt": "Two and three make",
        "semantically_correct": "five",
    },
    {
        "probe_id": "F11",
        "category": "arithmetic_symbolic",
        "prompt": "2 + 2 =",
        "semantically_correct": "four",
    },
    {
        "probe_id": "F12",
        "category": "arithmetic_symbolic",
        "prompt": "2 + 3 =",
        "semantically_correct": "five",
    },
]


# ============================================================
# Helpers
# ============================================================

def utc_now():

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


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


def atomic_write_json(
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

    temporary = (
        path.with_name(
            path.name
            +
            ".tmp"
        )
    )

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

        file.write(
            "\n"
        )

    temporary.replace(
        path
    )


def file_sha256(
    path,
):

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


def canonical_json_sha256(
    value,
):

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L9A — compile the approved "
            "research direction into a concrete "
            "preregistered EXP-0005 experiment "
            "plan without executing it."
        )
    )

    parser.add_argument(
        "--handoff",
        default=str(
            DEFAULT_HANDOFF_PATH
        ),
        help=(
            "Path to the L8D research-action "
            "handoff."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Path for the L9A experiment plan."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing plan after the "
            "research handoff changes."
        ),
    )

    return parser.parse_args()


# ============================================================
# Handoff validation
# ============================================================

def validate_handoff(
    handoff,
):

    h = handoff.get(
        "handoff",
        {},
    )

    experiment = h.get(
        "experiment",
        {},
    )

    permission = h.get(
        "permission_gate",
        {},
    )

    execution = h.get(
        "execution_gate",
        {},
    )

    checks = {
        "handoff_version":
            (
                handoff.get(
                    "engine_version"
                )
                ==
                EXPECTED_HANDOFF_VERSION
            ),

        "handoff_checks_present":
            bool(
                handoff.get(
                    "overall_checks"
                )
            ),

        "handoff_checks_passed":
            (
                bool(
                    handoff.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    handoff[
                        "overall_checks"
                    ].values()
                )
            ),

        "experiment_is_exp_0005":
            (
                experiment.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "experiment_registered":
            (
                experiment.get(
                    "registered"
                )
                is True
            ),

        "permission_is_p3":
            (
                permission.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "human_approval_required":
            (
                permission.get(
                    "human_approval_required"
                )
                is True
            ),

        "execution_remains_blocked":
            (
                execution.get(
                    "may_execute_now"
                )
                is False
            ),

        "execution_not_attempted":
            (
                execution.get(
                    "execution_attempted"
                )
                is False
            ),

        "knowledge_not_mutated":
            (
                execution.get(
                    "knowledge_store_mutated"
                )
                is False
            ),
    }

    failed = [
        name

        for name, passed
        in checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "L9A handoff validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Probe validation
# ============================================================

def validate_probe_design():

    probe_ids = [
        probe[
            "probe_id"
        ]

        for probe
        in PROBES
    ]

    four_correct = [
        probe

        for probe
        in PROBES

        if probe[
            "semantically_correct"
        ]
        ==
        "four"
    ]

    five_correct = [
        probe

        for probe
        in PROBES

        if probe[
            "semantically_correct"
        ]
        ==
        "five"
    ]

    categories = {
        probe[
            "category"
        ]

        for probe
        in PROBES
    }

    checks = {
        "probe_ids_unique":
            (
                len(
                    probe_ids
                )
                ==
                len(
                    set(
                        probe_ids
                    )
                )
            ),

        "probe_count_is_12":
            (
                len(
                    PROBES
                )
                ==
                12
            ),

        "four_correct_count_is_6":
            (
                len(
                    four_correct
                )
                ==
                6
            ),

        "five_correct_count_is_6":
            (
                len(
                    five_correct
                )
                ==
                6
            ),

        "multiple_semantic_categories":
            (
                len(
                    categories
                )
                >=
                5
            ),

        "all_prompts_nonempty":
            all(
                bool(
                    probe.get(
                        "prompt"
                    )
                )

                for probe
                in PROBES
            ),

        "all_correct_labels_valid":
            all(
                probe.get(
                    "semantically_correct"
                )
                in
                {
                    "four",
                    "five",
                }

                for probe
                in PROBES
            ),
    }

    failed = [
        name

        for name, passed
        in checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "EXP-0005 probe design failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Build preregistered plan
# ============================================================

def build_plan(
    *,
    handoff,
    handoff_sha256,
):

    h = (
        handoff[
            "handoff"
        ]
    )

    registered_experiment = (
        h[
            "experiment"
        ][
            "record"
        ]
    )

    if not isinstance(
        registered_experiment,
        dict,
    ):

        raise RuntimeError(
            "EXP-0005 registered experiment "
            "record is missing."
        )

    model_scope = (
        registered_experiment.get(
            "model_scope"
        )
    )

    if not isinstance(
        model_scope,
        dict,
    ):

        raise RuntimeError(
            "EXP-0005 model_scope missing."
        )

    checkpoint_sha256 = (
        model_scope.get(
            "checkpoint_sha256"
        )
    )

    if (
        not isinstance(
            checkpoint_sha256,
            str,
        )
        or
        len(
            checkpoint_sha256
        )
        !=
        64
    ):

        raise RuntimeError(
            "EXP-0005 checkpoint SHA-256 "
            "is invalid."
        )

    # --------------------------------------------------------
    # Fixed orientation
    # --------------------------------------------------------
    #
    # EXP-0004 used preferred-vs-comparison orientation,
    # which changed when the semantically correct answer
    # changed. EXP-0005 removes that confound.
    #
    # Every probe measures:
    #
    #   M = logit(four) - logit(five)
    #
    # Head ablation effect:
    #
    #   delta = M_ablated - M_baseline
    #
    # Therefore:
    #   delta < 0 => intact head supports FOUR over FIVE
    #   delta > 0 => intact head supports FIVE over FOUR
    #
    # This interpretation is invariant to which candidate is
    # semantically correct for a given prompt.
    # --------------------------------------------------------

    measurement_definition = {
        "candidate_four":
            FIXED_CANDIDATES[
                "four"
            ],

        "candidate_five":
            FIXED_CANDIDATES[
                "five"
            ],

        "canonical_margin":
            (
                "logit(four_discriminative_token) "
                "- "
                "logit(five_discriminative_token)"
            ),

        "ablation_delta":
            (
                "ablated_canonical_margin "
                "- baseline_canonical_margin"
            ),

        "negative_delta_interpretation":
            (
                "The intact head causally "
                "supports four relative to five."
            ),

        "positive_delta_interpretation":
            (
                "The intact head causally "
                "supports five relative to four."
            ),

        "zero_delta_interpretation":
            (
                "No measured directional effect "
                "on the fixed four-versus-five "
                "margin."
            ),
    }

    # --------------------------------------------------------
    # Preregistered hypothesis tests
    # --------------------------------------------------------

    decision_rules = {
        "directional_consistency_threshold":
            0.75,

        "minimum_consistent_probes":
            9,

        "total_probe_count":
            12,

        "primary_tests": {
            "L1H0": {
                "predicted_direction":
                    "four",

                "passing_sign":
                    "negative",

                "criterion":
                    (
                        "At least 9 of 12 probes "
                        "must have negative "
                        "ablation delta."
                    ),
            },

            "L4H0": {
                "predicted_direction":
                    "five",

                "passing_sign":
                    "positive",

                "criterion":
                    (
                        "At least 9 of 12 probes "
                        "must have positive "
                        "ablation delta."
                    ),
            },
        },

        "balance_checks": {
            "four_correct_subset_size":
                6,

            "five_correct_subset_size":
                6,

            "purpose":
                (
                    "A fixed lexical/numerical "
                    "direction should remain "
                    "similar even when semantic "
                    "correctness flips between "
                    "four and five."
                ),
        },

        "weak_control": {
            "head_id":
                "L4H1",

            "criterion":
                (
                    "Its mean absolute ablation "
                    "delta should remain smaller "
                    "than both primary-head mean "
                    "absolute deltas."
                ),
        },

        "outcome_labels": {
            "both_primary_heads_pass":
                (
                    "supports_fixed_four_five_"
                    "direction_hypothesis"
                ),

            "exactly_one_primary_head_passes":
                (
                    "partial_support_fixed_"
                    "direction_hypothesis"
                ),

            "neither_primary_head_passes":
                (
                    "does_not_support_fixed_"
                    "direction_hypothesis"
                ),
        },
    }

    plan_core = {
        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "title":
            registered_experiment.get(
                "title"
            ),

        "question":
            registered_experiment.get(
                "question"
            ),

        "hypothesis_id":
            h[
                "hypothesis"
            ][
                "hypothesis_id"
            ],

        "hypothesis_statement":
            (
                "L1H0 preferentially supports "
                "four relative to five while "
                "L4H0 preferentially supports "
                "five relative to four across "
                "multiple numerical contexts, "
                "independent of which candidate "
                "is semantically correct."
            ),

        "model_scope":
            model_scope,

        "required_permission":
            "P3",

        "target_heads":
            TARGET_HEADS,

        "fixed_candidates":
            FIXED_CANDIDATES,

        "measurement_definition":
            measurement_definition,

        "probe_design": {
            "design_type":
                (
                    "balanced_fixed_candidate_"
                    "context_transfer"
                ),

            "probe_count":
                len(
                    PROBES
                ),

            "four_correct_count":
                sum(
                    1

                    for probe
                    in PROBES

                    if probe[
                        "semantically_correct"
                    ]
                    ==
                    "four"
                ),

            "five_correct_count":
                sum(
                    1

                    for probe
                    in PROBES

                    if probe[
                        "semantically_correct"
                    ]
                    ==
                    "five"
                ),

            "probes":
                PROBES,
        },

        "intervention": {
            "type":
                "attention_head_ablation",

            "baseline_scale":
                1.0,

            "ablation_scale":
                0.0,

            "scale_one_exactness_control":
                True,

            "model_weights_persistently_modified":
                False,
        },

        "decision_rules":
            decision_rules,

        "required_outputs": [
            (
                "Per-probe baseline canonical "
                "four-minus-five margin."
            ),
            (
                "Per-head, per-probe ablated "
                "canonical margin."
            ),
            (
                "Per-head, per-probe ablation "
                "delta."
            ),
            (
                "Directional sign consistency "
                "for each target head."
            ),
            (
                "Four-correct versus five-correct "
                "subset summaries."
            ),
            (
                "Weak-control comparison against "
                "the primary heads."
            ),
            (
                "Checkpoint/runtime identity "
                "checks."
            ),
            (
                "Machine-readable summary artifact "
                "for Stage L7B ingestion."
            ),
        ],

        "execution_constraints": {
            "execution_authorized":
                False,

            "human_approval_required":
                True,

            "approval_status":
                h[
                    "permission_gate"
                ][
                    "approval_status"
                ],

            "must_not_execute_during_L9A":
                True,

            "checkpoint_sha256_must_match":
                checkpoint_sha256,

            "load_model_once":
                True,

            "recommended_thread_limits": {
                "OMP_NUM_THREADS":
                    1,

                "OPENBLAS_NUM_THREADS":
                    1,

                "MKL_NUM_THREADS":
                    1,
            },

            "recommended_niceness":
                15,
        },
    }

    plan_id_seed = (
        handoff[
            "handoff"
        ][
            "request_id"
        ]
        +
        "|"
        +
        canonical_json_sha256(
            plan_core
        )
    )

    plan_id = (
        "PLAN-"
        +
        hashlib.sha256(
            plan_id_seed.encode(
                "utf-8"
            )
        ).hexdigest()[
            :16
        ].upper()
    )

    return {
        "plan_id":
            plan_id,

        "plan":
            plan_core,

        "source_handoff": {
            "request_id":
                h[
                    "request_id"
                ],

            "handoff_sha256":
                handoff_sha256,

            "knowledge_state_sha256":
                h[
                    "source_decision"
                ][
                    "knowledge_state_sha256"
                ],
        },
    }


# ============================================================
# Plan validation
# ============================================================

def validate_plan(
    plan_package,
):

    plan = (
        plan_package[
            "plan"
        ]
    )

    heads = {
        head[
            "head_id"
        ]

        for head
        in plan[
            "target_heads"
        ]
    }

    probes = (
        plan[
            "probe_design"
        ][
            "probes"
        ]
    )

    checks = {
        "plan_id_present":
            (
                isinstance(
                    plan_package.get(
                        "plan_id"
                    ),
                    str,
                )
                and
                plan_package[
                    "plan_id"
                ].startswith(
                    "PLAN-"
                )
            ),

        "experiment_id":
            (
                plan.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "hypothesis_present":
            bool(
                plan.get(
                    "hypothesis_id"
                )
            ),

        "checkpoint_sha256_valid":
            (
                isinstance(
                    plan.get(
                        "model_scope",
                        {},
                    ).get(
                        "checkpoint_sha256"
                    ),
                    str,
                )
                and
                len(
                    plan[
                        "model_scope"
                    ][
                        "checkpoint_sha256"
                    ]
                )
                ==
                64
            ),

        "required_permission_p3":
            (
                plan.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "primary_heads_present":
            (
                {
                    "L1H0",
                    "L4H0",
                }
                <=
                heads
            ),

        "weak_control_present":
            (
                "L4H1"
                in
                heads
            ),

        "fixed_candidates":
            (
                plan.get(
                    "fixed_candidates"
                )
                ==
                FIXED_CANDIDATES
            ),

        "balanced_probe_count":
            (
                len(
                    probes
                )
                ==
                12
                and
                plan[
                    "probe_design"
                ][
                    "four_correct_count"
                ]
                ==
                6
                and
                plan[
                    "probe_design"
                ][
                    "five_correct_count"
                ]
                ==
                6
            ),

        "canonical_margin_fixed":
            (
                "four_discriminative_token"
                in
                plan[
                    "measurement_definition"
                ][
                    "canonical_margin"
                ]
                and
                "five_discriminative_token"
                in
                plan[
                    "measurement_definition"
                ][
                    "canonical_margin"
                ]
            ),

        "execution_not_authorized":
            (
                plan[
                    "execution_constraints"
                ][
                    "execution_authorized"
                ]
                is False
            ),

        "human_approval_required":
            (
                plan[
                    "execution_constraints"
                ][
                    "human_approval_required"
                ]
                is True
            ),

        "no_persistent_weight_modification":
            (
                plan[
                    "intervention"
                ][
                    "model_weights_persistently_modified"
                ]
                is False
            ),
    }

    failed = [
        name

        for name, passed
        in checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "L9A experiment-plan validation "
            f"failed: {failed}"
        )

    return checks


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    handoff_path = project_path(
        args.handoff
    )

    output_path = project_path(
        args.output
    )

    if (
        output_path.exists()
        and
        not args.overwrite
    ):

        raise FileExistsError(
            f"Experiment plan already exists: "
            f"{output_path}\n"
            "Use --overwrite after the research "
            "handoff or plan design changes."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L9A — EXPERIMENT "
        "PLAN COMPILER"
    )

    print(
        "=" * 76
    )

    handoff = load_json(
        handoff_path
    )

    handoff_sha256 = file_sha256(
        handoff_path
    )

    handoff_checks = validate_handoff(
        handoff
    )

    probe_checks = validate_probe_design()

    print()
    print(
        "Source handoff"
    )

    print(
        "--------------"
    )

    print(
        f"Request ID: "
        f"{handoff['handoff']['request_id']}"
    )

    print(
        f"Experiment: "
        f"{handoff['handoff']['experiment']['experiment_id']}"
    )

    print(
        f"Permission: "
        f"{handoff['handoff']['permission_gate']['required_permission']}"
    )

    print(
        f"Approval status: "
        f"{handoff['handoff']['permission_gate']['approval_status']}"
    )

    print()

    print(
        "Handoff checks"
    )

    print(
        "--------------"
    )

    for name, passed in (
        handoff_checks.items()
    ):

        print(
            f"{name:<38} "
            f"{passed}"
        )

    print()
    print(
        "Probe-design checks"
    )

    print(
        "-------------------"
    )

    for name, passed in (
        probe_checks.items()
    ):

        print(
            f"{name:<38} "
            f"{passed}"
        )

    package = build_plan(
        handoff=handoff,
        handoff_sha256=(
            handoff_sha256
        ),
    )

    plan_checks = validate_plan(
        package
    )

    output = {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "deterministic_preregistered_"
                "experiment_planning"
            ),

        "policy": {
            "executes_experiment":
                False,

            "grants_p3_approval":
                False,

            "mutates_knowledge_store":
                False,

            "mutates_experiment_registry":
                False,

            "changes_model_weights":
                False,
        },

        "source_handoff_path":
            str(
                handoff_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "handoff_checks":
            handoff_checks,

        "probe_design_checks":
            probe_checks,

        "experiment_plan":
            package,

        "overall_checks":
            plan_checks,
    }

    atomic_write_json(
        output_path,
        output,
    )

    saved = load_json(
        output_path
    )

    if (
        saved.get(
            "engine_version"
        )
        !=
        ENGINE_VERSION
    ):

        raise RuntimeError(
            "Saved L9A plan identity mismatch."
        )

    if not all(
        saved[
            "overall_checks"
        ].values()
    ):

        raise RuntimeError(
            "Saved L9A experiment plan "
            "contains failed checks."
        )

    plan = (
        package[
            "plan"
        ]
    )

    print()
    print(
        "=" * 76
    )

    print(
        "L9A EXPERIMENT PLAN"
    )

    print(
        "=" * 76
    )

    print()
    print(
        f"Plan ID:"
    )

    print(
        f"  {package['plan_id']}"
    )

    print()
    print(
        f"Experiment:"
    )

    print(
        f"  {plan['experiment_id']}: "
        f"{plan['title']}"
    )

    print()
    print(
        f"Hypothesis:"
    )

    print(
        f"  {plan['hypothesis_id']}"
    )

    print()
    print(
        "Fixed comparison:"
    )

    print(
        "  FOUR vs FIVE"
    )

    print()
    print(
        f"Probes:"
    )

    print(
        f"  {plan['probe_design']['probe_count']} "
        f"(6 four-correct / 6 five-correct)"
    )

    print()
    print(
        "Primary heads:"
    )

    print(
        "  L1H0 → predicted FOUR direction"
    )

    print(
        "  L4H0 → predicted FIVE direction"
    )

    print()
    print(
        "Preregistered threshold:"
    )

    print(
        "  >= 9/12 directional consistency"
    )

    print()
    print(
        "Required permission:"
    )

    print(
        f"  {plan['required_permission']}"
    )

    print()
    print(
        "Execution authorized:"
    )

    print(
        f"  {plan['execution_constraints']['execution_authorized']}"
    )

    print()
    print(
        "Overall checks"
    )

    print(
        "--------------"
    )

    for name, passed in (
        plan_checks.items()
    ):

        print(
            f"{name:<40} "
            f"{passed}"
        )

    print()
    print(
        "Experiment plan:"
    )

    print(
        output_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L9A "
        "EXPERIMENT PLAN COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()
