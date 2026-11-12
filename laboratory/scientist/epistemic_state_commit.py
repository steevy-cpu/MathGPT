import argparse
import copy
import json
import sys

from datetime import datetime, timezone
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# L7 imports
# ============================================================

from laboratory.knowledge.store import (
    KnowledgeStore,
)

from laboratory.knowledge.query import (
    KnowledgeQuery,
)

from laboratory.knowledge.snapshot import (
    SNAPSHOT_VERSION,
    build_snapshot,
    validate_snapshot,
    sha256_json,
)


# ============================================================
# Identity
# ============================================================

COMMIT_VERSION = "L9F.1"

EXPECTED_EVALUATOR_VERSION = "L8B.1"
EXPECTED_SNAPSHOT_VERSION = "L7D.1"

DEFAULT_EVALUATION_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "hypothesis_evaluation.json"
)

DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)

DEFAULT_SNAPSHOT_REPORT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "knowledge_snapshot_report.json"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "epistemic_state_commit.json"
)


# ============================================================
# Allowed deterministic transitions
# ============================================================

ALLOWED_TRANSITIONS = {
    ("proposed", "active"),
    ("unresolved", "active"),
    ("active", "supported"),
    ("active", "weakened"),
    ("proposed", "weakened"),
    ("unresolved", "weakened"),
    ("weakened", "falsified"),
    ("active", "falsified"),
}


MUTATING_DECISIONS = {
    "promote",
    "weaken",
    "demote",
}


# ============================================================
# Helpers
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def project_path(value):
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def relative_project_path(path):
    path = Path(path).resolve()

    try:
        return str(
            path.relative_to(
                PROJECT_ROOT.resolve()
            )
        )

    except ValueError:
        return str(path)


def load_json(path):
    path = project_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def atomic_write_json(
    path,
    value,
):
    path = project_path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_name(
            path.name + ".tmp"
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

        file.write("\n")

    temporary.replace(path)


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


def index_by(
    records,
    field,
):
    return {
        record[field]:
            record

        for record
        in records
    }


def capture_backups(
    paths,
):
    backups = {}

    for path in paths:
        path = Path(path)

        backups[path] = (
            path.read_bytes()
            if path.exists()
            else None
        )

    return backups


def restore_backups(
    backups,
):
    for (
        path,
        content,
    ) in backups.items():

        path = Path(path)

        if content is None:

            if path.exists():
                path.unlink()

        else:

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_bytes(
                content
            )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L9F — transactionally commit "
            "validated L8B hypothesis-status "
            "transitions into the L7 scientific "
            "knowledge layer."
        )
    )

    parser.add_argument(
        "--evaluation",
        default=str(
            DEFAULT_EVALUATION_PATH
        ),
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
        ),
    )

    parser.add_argument(
        "--snapshot-report",
        default=str(
            DEFAULT_SNAPSHOT_REPORT_PATH
        ),
    )

    parser.add_argument(
        "--report",
        default=str(
            DEFAULT_REPORT_PATH
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and display proposed "
            "epistemic transitions without "
            "writing to L7."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the L9F report after a "
            "new evaluator result. This flag "
            "does not bypass transition checks."
        ),
    )

    return parser.parse_args()


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    store,
    snapshot,
    evaluation,
):

    live_data = store.load_all()

    live_knowledge_sha = (
        sha256_json(
            live_data
        )
    )

    evaluation_inputs = (
        evaluation.get(
            "inputs",
            {},
        )
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

        "evaluator_version":
            (
                evaluation.get(
                    "engine_version"
                )
                ==
                EXPECTED_EVALUATOR_VERSION
            ),

        "evaluation_checks_present":
            bool(
                evaluation.get(
                    "overall_checks"
                )
            ),

        "evaluation_checks_passed":
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

        "evaluation_did_not_mutate_store":
            (
                evaluation.get(
                    "policy",
                    {},
                ).get(
                    "mutates_knowledge_store"
                )
                is False
            ),

        "evaluation_generated_no_confidence":
            (
                evaluation.get(
                    "policy",
                    {},
                ).get(
                    "generates_numerical_confidence"
                )
                is False
            ),

        "snapshot_matches_live_knowledge":
            (
                snapshot.get(
                    "knowledge_state_sha256"
                )
                ==
                live_knowledge_sha
            ),

        "evaluation_matches_live_knowledge":
            (
                evaluation_inputs.get(
                    "knowledge_state_sha256"
                )
                ==
                live_knowledge_sha
            ),

        "evaluations_present":
            (
                isinstance(
                    evaluation.get(
                        "hypothesis_evaluations"
                    ),
                    list,
                )
                and
                len(
                    evaluation[
                        "hypothesis_evaluations"
                    ]
                )
                >
                0
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
            "L9F input validation failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "live_knowledge_sha256":
            live_knowledge_sha,
    }


# ============================================================
# Transition extraction
# ============================================================

def extract_transitions(
    *,
    evaluation,
    hypotheses,
):

    hypothesis_index = index_by(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    transitions = []

    retained = []

    for result in (
        evaluation[
            "hypothesis_evaluations"
        ]
    ):

        hypothesis_id = (
            result[
                "hypothesis_id"
            ]
        )

        require(
            hypothesis_id
            in
            hypothesis_index,
            (
                "L9F evaluator references "
                f"unknown hypothesis "
                f"{hypothesis_id}."
            ),
        )

        live_record = (
            hypothesis_index[
                hypothesis_id
            ]
        )

        live_status = (
            live_record[
                "status"
            ]
        )

        evaluated_current_status = (
            result[
                "current_status"
            ]
        )

        proposed_status = (
            result[
                "evaluation"
            ][
                "proposed_status"
            ]
        )

        decision = (
            result[
                "evaluation"
            ][
                "decision"
            ]
        )

        require(
            live_status
            ==
            evaluated_current_status,
            (
                f"{hypothesis_id}: live status "
                f"{live_status!r} does not match "
                "the evaluator's input status "
                f"{evaluated_current_status!r}."
            ),
        )

        if proposed_status == live_status:

            retained.append(
                {
                    "hypothesis_id":
                        hypothesis_id,

                    "status":
                        live_status,

                    "decision":
                        decision,

                    "reason_code":
                        result[
                            "evaluation"
                        ][
                            "reason_code"
                        ],
                }
            )

            continue

        require(
            decision
            in
            MUTATING_DECISIONS,
            (
                f"{hypothesis_id}: evaluator "
                "proposes a status mutation "
                f"using non-mutating decision "
                f"{decision!r}."
            ),
        )

        require(
            (
                live_status,
                proposed_status,
            )
            in
            ALLOWED_TRANSITIONS,
            (
                f"{hypothesis_id}: transition "
                f"{live_status!r} -> "
                f"{proposed_status!r} is not "
                "allowed by L9F policy."
            ),
        )

        transitions.append(
            {
                "hypothesis_id":
                    hypothesis_id,

                "from_status":
                    live_status,

                "to_status":
                    proposed_status,

                "decision":
                    decision,

                "reason_code":
                    result[
                        "evaluation"
                    ][
                        "reason_code"
                    ],

                "reason":
                    result[
                        "evaluation"
                    ][
                        "reason"
                    ],

                "evidence_summary":
                    copy.deepcopy(
                        result[
                            "evidence_summary"
                        ]
                    ),
            }
        )

    return {
        "transitions":
            transitions,

        "retained":
            retained,
    }


# ============================================================
# Research-state membership helpers
# ============================================================

def remove_from_list(
    state,
    field,
    hypothesis_id,
):
    values = list(
        state.get(
            field,
            [],
        )
    )

    values = [
        value

        for value
        in values

        if value
        !=
        hypothesis_id
    ]

    state[
        field
    ] = values


def add_to_list(
    state,
    field,
    hypothesis_id,
):
    values = list(
        state.get(
            field,
            [],
        )
    )

    if hypothesis_id not in values:
        values.append(
            hypothesis_id
        )

    state[
        field
    ] = sorted(
        values
    )


def update_research_state_membership(
    *,
    state,
    hypothesis_id,
    new_status,
):

    for field in (
        "active_hypotheses",
        "weakened_hypotheses",
        "supported_hypotheses",
        "falsified_hypotheses",
    ):
        remove_from_list(
            state,
            field,
            hypothesis_id,
        )

    if new_status == "active":
        add_to_list(
            state,
            "active_hypotheses",
            hypothesis_id,
        )

    elif new_status == "weakened":
        add_to_list(
            state,
            "weakened_hypotheses",
            hypothesis_id,
        )

    elif new_status == "supported":
        add_to_list(
            state,
            "supported_hypotheses",
            hypothesis_id,
        )

    elif new_status == "falsified":
        add_to_list(
            state,
            "falsified_hypotheses",
            hypothesis_id,
        )


# ============================================================
# Apply transitions to in-memory collections
# ============================================================

def apply_transitions(
    *,
    hypotheses,
    research_state,
    transitions,
):

    hypotheses = copy.deepcopy(
        hypotheses
    )

    research_state = copy.deepcopy(
        research_state
    )

    hypothesis_index = index_by(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    state = (
        research_state[
            "state"
        ]
    )

    changed_at = utc_now()

    for transition in transitions:

        hypothesis_id = (
            transition[
                "hypothesis_id"
            ]
        )

        record = (
            hypothesis_index[
                hypothesis_id
            ]
        )

        record[
            "status"
        ] = transition[
            "to_status"
        ]

        record[
            "updated_at_utc"
        ] = changed_at

        # Keep numerical confidence absent.
        confidence = record.setdefault(
            "confidence",
            {},
        )

        confidence[
            "score"
        ] = None

        confidence[
            "label"
        ] = (
            "replicated_causal_support"
            if transition[
                "to_status"
            ]
            ==
            "supported"
            else
            transition[
                "to_status"
            ]
        )

        confidence[
            "method"
        ] = "L8B.1_deterministic_policy"

        confidence[
            "note"
        ] = transition[
            "reason"
        ]

        update_research_state_membership(
            state=state,
            hypothesis_id=hypothesis_id,
            new_status=(
                transition[
                    "to_status"
                ]
            ),
        )

    state[
        "last_epistemic_update"
    ] = {
        "updated_at_utc":
            changed_at,

        "commit_version":
            COMMIT_VERSION,

        "transition_count":
            len(
                transitions
            ),

        "hypothesis_ids":
            [
                transition[
                    "hypothesis_id"
                ]

                for transition
                in transitions
            ],
    }

    hypotheses[
        "updated_at_utc"
    ] = changed_at

    research_state[
        "updated_at_utc"
    ] = changed_at

    return {
        "hypotheses":
            hypotheses,

        "research_state":
            research_state,
    }


# ============================================================
# Integrated-state validation
# ============================================================

def validate_committed_state(
    *,
    store,
    transitions,
):

    validation = store.validate()

    require(
        validation[
            "passed"
        ],
        (
            "L9F committed knowledge store "
            "failed validation."
        ),
    )

    data = store.load_all()

    hypothesis_index = index_by(
        data[
            "hypotheses"
        ][
            "records"
        ],
        "hypothesis_id",
    )

    state = (
        data[
            "research_state"
        ][
            "state"
        ]
    )

    checks = {
        "knowledge_store_valid":
            bool(
                validation[
                    "passed"
                ]
            ),

        "no_numerical_confidence_scores":
            all(
                record.get(
                    "confidence",
                    {},
                ).get(
                    "score"
                )
                is None

                for record
                in data[
                    "hypotheses"
                ][
                    "records"
                ]
            ),
    }

    for transition in transitions:

        hypothesis_id = (
            transition[
                "hypothesis_id"
            ]
        )

        to_status = (
            transition[
                "to_status"
            ]
        )

        checks[
            f"{hypothesis_id}_status_committed"
        ] = (
            hypothesis_index[
                hypothesis_id
            ][
                "status"
            ]
            ==
            to_status
        )

        membership_field = {
            "active":
                "active_hypotheses",

            "weakened":
                "weakened_hypotheses",

            "supported":
                "supported_hypotheses",

            "falsified":
                "falsified_hypotheses",
        }.get(
            to_status
        )

        if membership_field is not None:

            checks[
                f"{hypothesis_id}_state_membership"
            ] = (
                hypothesis_id
                in
                state.get(
                    membership_field,
                    [],
                )
            )

        if to_status != "active":

            checks[
                f"{hypothesis_id}_not_active"
            ] = (
                hypothesis_id
                not in
                state.get(
                    "active_hypotheses",
                    [],
                )
            )

    failed = [
        name

        for name, passed
        in checks.items()

        if not passed
    ]

    if failed:
        raise RuntimeError(
            "L9F committed-state validation "
            f"failed: {failed}"
        )

    return {
        "store_validation":
            validation,

        "checks":
            checks,
    }


# ============================================================
# Snapshot regeneration
# ============================================================

def regenerate_snapshot(
    *,
    store,
    snapshot_path,
    snapshot_report_path,
):

    knowledge = KnowledgeQuery(
        store=store
    )

    snapshot = build_snapshot(
        store,
        knowledge,
    )

    validation = validate_snapshot(
        snapshot
    )

    require(
        validation[
            "passed"
        ],
        (
            "L9F regenerated scientist "
            "snapshot failed validation."
        ),
    )

    atomic_write_json(
        snapshot_path,
        snapshot,
    )

    saved = load_json(
        snapshot_path
    )

    saved_validation = (
        validate_snapshot(
            saved
        )
    )

    require(
        saved_validation[
            "passed"
        ],
        (
            "L9F saved scientist snapshot "
            "failed validation."
        ),
    )

    report = {
        "stage":
            "L7D",

        "snapshot_version":
            SNAPSHOT_VERSION,

        "created_at_utc":
            utc_now(),

        "snapshot_path":
            relative_project_path(
                snapshot_path
            ),

        "knowledge_state_sha256":
            snapshot[
                "knowledge_state_sha256"
            ],

        "knowledge_counts":
            snapshot[
                "knowledge_counts"
            ],

        "active_hypothesis_count":
            len(
                snapshot[
                    "hypotheses"
                ][
                    "active"
                ]
            ),

        "supported_hypothesis_count":
            len(
                snapshot[
                    "hypotheses"
                ][
                    "supported"
                ]
            ),

        "open_question_count":
            len(
                snapshot[
                    "open_questions"
                ]
            ),

        "next_recommended_experiment":
            snapshot[
                "research_focus"
            ][
                "next_recommended_experiment"
            ],

        "snapshot_validation":
            saved_validation,

        "passed":
            True,
    }

    atomic_write_json(
        snapshot_report_path,
        report,
    )

    return {
        "snapshot":
            snapshot,

        "validation":
            saved_validation,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    evaluation_path = project_path(
        args.evaluation
    )

    snapshot_path = project_path(
        args.snapshot
    )

    snapshot_report_path = project_path(
        args.snapshot_report
    )

    report_path = project_path(
        args.report
    )

    if (
        report_path.exists()
        and
        not args.overwrite
        and
        not args.dry_run
    ):
        raise FileExistsError(
            "L9F commit report already exists: "
            f"{report_path}\n"
            "Use --overwrite after a new "
            "evaluator result."
        )

    print()
    print(
        "=" * 80
    )

    print(
        "STAGE L9F — EPISTEMIC "
        "STATE COMMIT"
    )

    print(
        "=" * 80
    )

    store = KnowledgeStore()

    snapshot = load_json(
        snapshot_path
    )

    evaluation = load_json(
        evaluation_path
    )

    input_validation = (
        validate_inputs(
            store=store,
            snapshot=snapshot,
            evaluation=evaluation,
        )
    )

    print()
    print(
        "Input checks"
    )

    print(
        "------------"
    )

    for (
        name,
        passed,
    ) in input_validation[
        "checks"
    ].items():

        print(
            f"{name:<44} "
            f"{passed}"
        )

    current_hypotheses = (
        store.load(
            "hypotheses"
        )
    )

    current_research_state = (
        store.load(
            "research_state"
        )
    )

    extracted = extract_transitions(
        evaluation=evaluation,
        hypotheses=current_hypotheses,
    )

    transitions = (
        extracted[
            "transitions"
        ]
    )

    retained = (
        extracted[
            "retained"
        ]
    )

    print()
    print(
        "Proposed epistemic transitions"
    )

    print(
        "------------------------------"
    )

    if transitions:

        for transition in transitions:

            print()

            print(
                transition[
                    "hypothesis_id"
                ]
            )

            print(
                f"  "
                f"{transition['from_status']} "
                f"-> "
                f"{transition['to_status']}"
            )

            print(
                f"  decision: "
                f"{transition['decision']}"
            )

            print(
                f"  reason: "
                f"{transition['reason_code']}"
            )

    else:

        print(
            "  None"
        )

    print()
    print(
        "Retained states"
    )

    print(
        "---------------"
    )

    for item in retained:

        print(
            f"  {item['hypothesis_id']}: "
            f"{item['status']} "
            f"({item['decision']})"
        )

    prepared = apply_transitions(
        hypotheses=current_hypotheses,
        research_state=(
            current_research_state
        ),
        transitions=transitions,
    )

    if args.dry_run:

        print()
        print(
            "Dry-run result"
        )

        print(
            "--------------"
        )

        print(
            f"Transitions to commit: "
            f"{len(transitions)}"
        )

        print(
            "Knowledge files modified: False"
        )

        print()
        print(
            "=" * 80
        )

        print(
            "SUCCESS: L9F DRY RUN PASSED."
        )

        print(
            "No epistemic state was mutated."
        )

        print(
            "=" * 80
        )

        return

    # ========================================================
    # Transaction
    # ========================================================

    paths = [
        store.collection_path(
            "hypotheses"
        ),
        store.collection_path(
            "research_state"
        ),
        snapshot_path,
        snapshot_report_path,
        report_path,
    ]

    backups = capture_backups(
        paths
    )

    try:

        store.save(
            "hypotheses",
            prepared[
                "hypotheses"
            ],
        )

        store.save(
            "research_state",
            prepared[
                "research_state"
            ],
        )

        committed_validation = (
            validate_committed_state(
                store=store,
                transitions=transitions,
            )
        )

        snapshot_result = (
            regenerate_snapshot(
                store=store,
                snapshot_path=(
                    snapshot_path
                ),
                snapshot_report_path=(
                    snapshot_report_path
                ),
            )
        )

        new_knowledge_sha = (
            snapshot_result[
                "snapshot"
            ][
                "knowledge_state_sha256"
            ]
        )

        report = {
            "stage":
                "L9F",

            "commit_version":
                COMMIT_VERSION,

            "created_at_utc":
                utc_now(),

            "source_evaluation":
                relative_project_path(
                    evaluation_path
                ),

            "old_knowledge_state_sha256":
                input_validation[
                    "live_knowledge_sha256"
                ],

            "new_knowledge_state_sha256":
                new_knowledge_sha,

            "transition_count":
                len(
                    transitions
                ),

            "transitions":
                transitions,

            "retained":
                retained,

            "policy": {
                "numerical_confidence_generated":
                    False,

                "only_evaluator_proposed_"
                "transitions_committed":
                    True,

                "transactional_rollback":
                    True,
            },

            "committed_state_checks":
                committed_validation[
                    "checks"
                ],

            "snapshot_validation":
                snapshot_result[
                    "validation"
                ],

            "passed":
                True,
        }

        atomic_write_json(
            report_path,
            report,
        )

    except Exception:

        restore_backups(
            backups
        )

        raise

    # ========================================================
    # Final display
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "L9F EPISTEMIC COMMIT SUMMARY"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Committed transitions: "
        f"{len(transitions)}"
    )

    for transition in transitions:

        print(
            f"  "
            f"{transition['hypothesis_id']}: "
            f"{transition['from_status']} "
            f"-> "
            f"{transition['to_status']}"
        )

    print()

    print(
        "Old knowledge fingerprint:"
    )

    print(
        "  "
        +
        input_validation[
            "live_knowledge_sha256"
        ]
    )

    print()

    print(
        "New knowledge fingerprint:"
    )

    print(
        "  "
        +
        snapshot_result[
            "snapshot"
        ][
            "knowledge_state_sha256"
        ]
    )

    print()

    print(
        "Scientist state:"
    )

    print(
        f"  active hypotheses: "
        f"{len(snapshot_result['snapshot']['hypotheses']['active'])}"
    )

    print(
        f"  supported hypotheses: "
        f"{len(snapshot_result['snapshot']['hypotheses']['supported'])}"
    )

    print(
        f"  weakened hypotheses: "
        f"{len(snapshot_result['snapshot']['hypotheses']['weakened'])}"
    )

    print()

    print(
        "Commit report:"
    )

    print(
        f"  {report_path}"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "SUCCESS: STAGE L9F "
        "EPISTEMIC STATE COMMIT COMPLETE."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
