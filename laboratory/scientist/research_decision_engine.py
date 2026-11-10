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
    "L8C.1"
)


EXPECTED_EVALUATOR_VERSION = (
    "L8B.1"
)


EXPECTED_SNAPSHOT_VERSION = (
    "L7D.1"
)


DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)


DEFAULT_EVALUATION_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "hypothesis_evaluation.json"
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "research_decision.json"
)


# ============================================================
# Deterministic policy order
#
# Lower values rank ahead of higher values.
# These are ordering rules, not confidence scores.
# ============================================================

PRIORITY_ORDER = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


ACTION_ORDER = {
    "execute_registered_discriminating_test": 0,
    "design_discriminating_experiment": 1,
    "design_replication_experiment": 2,
    "design_replication_or_falsification_test": 3,
    "design_initial_test": 4,
    "monitor": 5,
}


HYPOTHESIS_STATUS_ORDER = {
    "active": 0,
    "unresolved": 1,
    "proposed": 2,
    "weakened": 3,
    "supported": 4,
    "falsified": 5,
    "retired": 6,
}


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


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L8C — deterministic research "
            "decision engine for the MathGPT "
            "AI Scientist."
        )
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
        ),
        help=(
            "Path to the L7D scientist context."
        ),
    )

    parser.add_argument(
        "--evaluation",
        default=str(
            DEFAULT_EVALUATION_PATH
        ),
        help=(
            "Path to the L8B hypothesis evaluation."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Path for the L8C research decision artifact."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing research decision artifact."
        ),
    )

    return parser.parse_args()


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    snapshot,
    snapshot_sha256,
    evaluation,
):

    inputs = evaluation.get(
        "inputs",
        {},
    )

    checks = {
        "snapshot_version":
            (
                snapshot.get(
                    "snapshot_version"
                )
                ==
                EXPECTED_SNAPSHOT_VERSION
            ),

        "snapshot_handoff_checks":
            (
                bool(
                    snapshot.get(
                        "handoff_checks"
                    )
                )
                and
                all(
                    snapshot[
                        "handoff_checks"
                    ].values()
                )
            ),

        "evaluation_version":
            (
                evaluation.get(
                    "engine_version"
                )
                ==
                EXPECTED_EVALUATOR_VERSION
            ),

        "evaluation_checks":
            (
                bool(
                    evaluation.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    evaluation[
                        "overall_checks"
                    ].values()
                )
            ),

        "exact_snapshot_sha_matches_evaluation":
            (
                inputs.get(
                    "scientist_context_sha256"
                )
                ==
                snapshot_sha256
            ),

        "knowledge_state_matches":
            (
                inputs.get(
                    "knowledge_state_sha256"
                )
                ==
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        "evaluations_present":
            isinstance(
                evaluation.get(
                    "hypothesis_evaluations"
                ),
                list,
            ),

        "evaluation_does_not_mutate_store":
            (
                evaluation.get(
                    "policy",
                    {},
                ).get(
                    "mutates_knowledge_store"
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
            "L8C input validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Snapshot indexes
# ============================================================

def build_planned_experiment_index(
    snapshot,
):

    index = {}

    for experiment in (
        snapshot.get(
            "experiments",
            {},
        ).get(
            "planned",
            [],
        )
    ):

        experiment_id = (
            experiment[
                "experiment_id"
            ]
        )

        if experiment_id in index:

            raise RuntimeError(
                "Duplicate planned experiment: "
                f"{experiment_id}"
            )

        index[
            experiment_id
        ] = experiment

    return index


# ============================================================
# Candidate construction
# ============================================================

def build_candidate(
    *,
    evaluation,
    snapshot,
    planned_experiment_index,
):

    hypothesis_id = (
        evaluation[
            "hypothesis_id"
        ]
    )

    current_status = (
        evaluation[
            "current_status"
        ]
    )

    next_action = (
        evaluation[
            "next_action"
        ]
    )

    action = next_action.get(
        "action"
    )

    priority = next_action.get(
        "priority"
    )

    experiment_id = next_action.get(
        "experiment_id"
    )

    if priority not in PRIORITY_ORDER:

        raise RuntimeError(
            f"{hypothesis_id}: unknown priority "
            f"{priority!r}."
        )

    if action not in ACTION_ORDER:

        raise RuntimeError(
            f"{hypothesis_id}: unknown action "
            f"{action!r}."
        )

    if current_status not in HYPOTHESIS_STATUS_ORDER:

        raise RuntimeError(
            f"{hypothesis_id}: unknown hypothesis "
            f"status {current_status!r}."
        )

    registered_experiment = None

    if experiment_id is not None:

        registered_experiment = (
            planned_experiment_index.get(
                experiment_id
            )
        )

        if registered_experiment is None:

            raise RuntimeError(
                f"{hypothesis_id}: next action "
                f"references unregistered planned "
                f"experiment {experiment_id}."
            )

    research_state_next = (
        snapshot.get(
            "research_focus",
            {},
        ).get(
            "next_recommended_experiment"
        )
    )

    matches_registered_recommendation = (
        experiment_id is not None
        and
        experiment_id
        ==
        research_state_next
    )

    # --------------------------------------------------------
    # Lexicographic decision key
    #
    # This is not a confidence/value score. It is a fixed,
    # transparent ordering policy:
    #
    # 1. higher declared priority
    # 2. executable registered test before design-only work
    # 3. already-recommended registered experiment
    # 4. active hypotheses before weaker lifecycle states
    # 5. stable ID tie-breaker
    # --------------------------------------------------------

    decision_key = (
        PRIORITY_ORDER[
            priority
        ],

        ACTION_ORDER[
            action
        ],

        (
            0
            if matches_registered_recommendation
            else 1
        ),

        HYPOTHESIS_STATUS_ORDER[
            current_status
        ],

        hypothesis_id,
    )

    return {
        "hypothesis_id":
            hypothesis_id,

        "hypothesis_title":
            evaluation.get(
                "title"
            ),

        "current_status":
            current_status,

        "proposed_status":
            evaluation.get(
                "evaluation",
                {},
            ).get(
                "proposed_status"
            ),

        "evidence_classification":
            evaluation.get(
                "evidence_summary",
                {},
            ).get(
                "classification"
            ),

        "action":
            action,

        "priority":
            priority,

        "experiment_id":
            experiment_id,

        "registered_experiment":
            registered_experiment,

        "matches_research_state_recommendation":
            matches_registered_recommendation,

        "action_reason":
            next_action.get(
                "reason"
            ),

        "decision_key": [
            decision_key[0],
            decision_key[1],
            decision_key[2],
            decision_key[3],
            decision_key[4],
        ],

        "_sort_key":
            decision_key,
    }


# ============================================================
# Selection
# ============================================================

def choose_research_action(
    candidates,
):

    if not candidates:

        raise RuntimeError(
            "No research-action candidates were produced."
        )

    ordered = sorted(
        candidates,
        key=lambda item:
            item[
                "_sort_key"
            ],
    )

    selected = ordered[
        0
    ]

    alternatives = ordered[
        1:
    ]

    return (
        selected,
        alternatives,
    )


# ============================================================
# Public record cleanup
# ============================================================

def public_candidate(
    candidate,
):

    return {
        key:
            value

        for key, value
        in candidate.items()

        if key
        !=
        "_sort_key"
    }


# ============================================================
# Selection rationale
# ============================================================

def build_selection_rationale(
    selected,
    alternatives,
):

    reasons = []

    reasons.append(
        (
            "The selected action has deterministic "
            f"priority '{selected['priority']}'."
        )
    )

    if (
        selected[
            "experiment_id"
        ]
        is not None
    ):

        reasons.append(
            (
                "It is executable through an already "
                "registered planned experiment rather "
                "than requiring a new experiment design."
            )
        )

    if selected[
        "matches_research_state_recommendation"
    ]:

        reasons.append(
            (
                "The registered experiment matches the "
                "current research-state recommendation."
            )
        )

    if selected[
        "current_status"
    ] == "active":

        reasons.append(
            (
                "It advances an active hypothesis rather "
                "than a lower-priority lifecycle state."
            )
        )

    if alternatives:

        reasons.append(
            (
                "Other candidate actions remain recorded "
                "as alternatives and are not discarded."
            )
        )

    return reasons


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    snapshot_path = project_path(
        args.snapshot
    )

    evaluation_path = project_path(
        args.evaluation
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
            f"Research decision already exists: "
            f"{output_path}\n"
            "Use --overwrite after the scientific "
            "state or hypothesis evaluation changes."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L8C — RESEARCH DECISION ENGINE"
    )

    print(
        "=" * 76
    )

    snapshot = load_json(
        snapshot_path
    )

    evaluation = load_json(
        evaluation_path
    )

    snapshot_sha256 = file_sha256(
        snapshot_path
    )

    evaluation_sha256 = file_sha256(
        evaluation_path
    )

    checks = validate_inputs(
        snapshot=snapshot,
        snapshot_sha256=(
            snapshot_sha256
        ),
        evaluation=evaluation,
    )

    print()
    print(
        "Input identity"
    )

    print(
        "--------------"
    )

    print(
        "Knowledge state:"
    )

    print(
        snapshot[
            "knowledge_state_sha256"
        ]
    )

    print()
    print(
        "Scientist context SHA:"
    )

    print(
        snapshot_sha256
    )

    print()
    print(
        "Hypothesis evaluation SHA:"
    )

    print(
        evaluation_sha256
    )

    print()
    print(
        "Input checks"
    )

    print(
        "------------"
    )

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    planned_experiment_index = (
        build_planned_experiment_index(
            snapshot
        )
    )

    # ========================================================
    # Build candidate actions
    # ========================================================

    candidates = []

    print()
    print(
        "Research-action candidates"
    )

    print(
        "--------------------------"
    )

    for hypothesis_evaluation in (
        evaluation[
            "hypothesis_evaluations"
        ]
    ):

        candidate = build_candidate(
            evaluation=(
                hypothesis_evaluation
            ),
            snapshot=snapshot,
            planned_experiment_index=(
                planned_experiment_index
            ),
        )

        candidates.append(
            candidate
        )

        print()
        print(
            candidate[
                "hypothesis_id"
            ]
        )

        print(
            f"  status: "
            f"{candidate['current_status']}"
        )

        print(
            f"  evidence: "
            f"{candidate['evidence_classification']}"
        )

        print(
            f"  priority: "
            f"{candidate['priority']}"
        )

        print(
            f"  action: "
            f"{candidate['action']}"
        )

        print(
            f"  experiment: "
            f"{candidate['experiment_id']}"
        )

    # ========================================================
    # Select one action
    # ========================================================

    selected, alternatives = (
        choose_research_action(
            candidates
        )
    )

    rationale = (
        build_selection_rationale(
            selected,
            alternatives,
        )
    )

    # ========================================================
    # Overall checks
    # ========================================================

    evaluation_ids = {
        item[
            "hypothesis_id"
        ]

        for item
        in evaluation[
            "hypothesis_evaluations"
        ]
    }

    candidate_ids = {
        item[
            "hypothesis_id"
        ]

        for item
        in candidates
    }

    selected_experiment_is_registered = (
        selected[
            "experiment_id"
        ]
        is None
        or
        selected[
            "experiment_id"
        ]
        in
        planned_experiment_index
    )

    selected_is_first_by_policy = (
        selected[
            "_sort_key"
        ]
        ==
        min(
            candidate[
                "_sort_key"
            ]

            for candidate
            in candidates
        )
    )

    overall_checks = {
        "inputs_valid":
            all(
                checks.values()
            ),

        "candidate_for_every_hypothesis":
            (
                candidate_ids
                ==
                evaluation_ids
            ),

        "exactly_one_action_selected":
            (
                selected
                is not None
            ),

        "selected_action_follows_policy":
            selected_is_first_by_policy,

        "selected_experiment_reference_valid":
            selected_experiment_is_registered,

        "knowledge_store_not_mutated":
            True,

        "experiment_not_executed":
            True,

        "no_confidence_score_generated":
            True,
    }

    failed = [
        name

        for name, passed
        in overall_checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "L8C research decision failed: "
            f"{failed}"
        )

    # ========================================================
    # Output
    # ========================================================

    output = {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "deterministic_single_action_"
                "research_selection"
            ),

        "inputs": {
            "scientist_context_path":
                str(
                    snapshot_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "scientist_context_sha256":
                snapshot_sha256,

            "knowledge_state_sha256":
                snapshot[
                    "knowledge_state_sha256"
                ],

            "hypothesis_evaluation_path":
                str(
                    evaluation_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "hypothesis_evaluation_sha256":
                evaluation_sha256,
        },

        "policy": {
            "mutates_knowledge_store":
                False,

            "executes_experiment":
                False,

            "generates_confidence_score":
                False,

            "selection_method":
                "deterministic_lexicographic_policy",

            "ordering": [
                "declared_priority",
                "action_executability",
                "research_state_recommendation_match",
                "hypothesis_lifecycle_status",
                "stable_hypothesis_id_tiebreaker",
            ],

            "priority_order":
                PRIORITY_ORDER,

            "action_order":
                ACTION_ORDER,

            "hypothesis_status_order":
                HYPOTHESIS_STATUS_ORDER,
        },

        "candidate_count":
            len(
                candidates
            ),

        "candidates": [
            public_candidate(
                candidate
            )

            for candidate
            in sorted(
                candidates,
                key=lambda item:
                    item[
                        "_sort_key"
                    ],
            )
        ],

        "selected_action":
            public_candidate(
                selected
            ),

        "selection_rationale":
            rationale,

        "alternative_actions": [
            public_candidate(
                candidate
            )

            for candidate
            in alternatives
        ],

        "overall_checks":
            overall_checks,
    }

    atomic_write_json(
        output_path,
        output,
    )

    # ========================================================
    # Verify saved output
    # ========================================================

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
            "Saved research decision identity mismatch."
        )

    if not all(
        saved[
            "overall_checks"
        ].values()
    ):

        raise RuntimeError(
            "Saved research decision contains failed checks."
        )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "L8C RESEARCH DECISION SUMMARY"
    )

    print(
        "=" * 76
    )

    print()
    print(
        "Selected hypothesis:"
    )

    print(
        f"  {selected['hypothesis_id']}: "
        f"{selected['hypothesis_title']}"
    )

    print()
    print(
        "Selected action:"
    )

    print(
        f"  {selected['action']}"
    )

    print()
    print(
        "Priority:"
    )

    print(
        f"  {selected['priority']}"
    )

    print()
    print(
        "Selected experiment:"
    )

    print(
        f"  {selected['experiment_id']}"
    )

    print()
    print(
        "Rationale:"
    )

    for reason in rationale:

        print(
            f"  - {reason}"
        )

    print()
    print(
        "Alternative actions retained:"
    )

    print(
        f"  {len(alternatives)}"
    )

    print()
    print(
        "Overall checks"
    )

    print(
        "--------------"
    )

    for name, passed in (
        overall_checks.items()
    ):

        print(
            f"{name:<42} "
            f"{passed}"
        )

    print()
    print(
        "Research decision:"
    )

    print(
        output_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L8C RESEARCH DECISION COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()
