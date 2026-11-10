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
    "L8D.1"
)


EXPECTED_DECISION_VERSION = (
    "L8C.1"
)


DEFAULT_DECISION_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "research_decision.json"
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "research_action_handoff.json"
)


# ============================================================
# Permission model
#
# P0 = metadata / identity inspection
# P1 = behavioral measurement
# P2 = observational internal inspection
# P3 = temporary causal intervention
#
# Conservative rule:
# A registered discriminating test selected for execution
# defaults to P3 unless the registered experiment explicitly
# declares a required_permission.
#
# This can over-gate a future lower-risk experiment, but it
# must never under-gate a causal intervention.
# ============================================================

VALID_PERMISSIONS = {
    "P0",
    "P1",
    "P2",
    "P3",
}


ACTION_PERMISSION_DEFAULTS = {
    "execute_registered_discriminating_test":
        "P3",
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
            "Stage L8D — convert the selected "
            "research decision into a safe "
            "execution handoff without running "
            "the experiment."
        )
    )

    parser.add_argument(
        "--decision",
        default=str(
            DEFAULT_DECISION_PATH
        ),
        help=(
            "Path to the L8C research decision."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Path for the L8D research-action "
            "handoff artifact."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing L8D handoff "
            "after the research decision changes."
        ),
    )

    return parser.parse_args()


# ============================================================
# Input validation
# ============================================================

def validate_decision(
    decision,
):

    selected = (
        decision.get(
            "selected_action"
        )
    )

    checks = {
        "decision_version":
            (
                decision.get(
                    "engine_version"
                )
                ==
                EXPECTED_DECISION_VERSION
            ),

        "decision_checks_present":
            bool(
                decision.get(
                    "overall_checks"
                )
            ),

        "decision_checks_passed":
            (
                bool(
                    decision.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    decision[
                        "overall_checks"
                    ].values()
                )
            ),

        "decision_did_not_execute":
            (
                decision.get(
                    "policy",
                    {},
                ).get(
                    "executes_experiment"
                )
                is False
            ),

        "decision_did_not_mutate_store":
            (
                decision.get(
                    "policy",
                    {},
                ).get(
                    "mutates_knowledge_store"
                )
                is False
            ),

        "selected_action_present":
            isinstance(
                selected,
                dict,
            ),

        "knowledge_state_present":
            (
                isinstance(
                    decision.get(
                        "inputs",
                        {},
                    ).get(
                        "knowledge_state_sha256"
                    ),
                    str,
                )
                and
                len(
                    decision[
                        "inputs"
                    ][
                        "knowledge_state_sha256"
                    ]
                )
                ==
                64
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
            "L8D decision validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Permission resolution
# ============================================================

def resolve_permission(
    selected_action,
):

    action = selected_action.get(
        "action"
    )

    experiment = selected_action.get(
        "registered_experiment"
    )

    explicit_permission = None

    if isinstance(
        experiment,
        dict,
    ):

        explicit_permission = (
            experiment.get(
                "required_permission"
            )
        )

    if explicit_permission is not None:

        if (
            explicit_permission
            not in
            VALID_PERMISSIONS
        ):

            raise RuntimeError(
                "Registered experiment declares "
                "invalid required_permission: "
                f"{explicit_permission!r}"
            )

        return {
            "required_permission":
                explicit_permission,

            "resolution_source":
                "registered_experiment_metadata",

            "conservative_default_used":
                False,
        }

    default_permission = (
        ACTION_PERMISSION_DEFAULTS.get(
            action
        )
    )

    if default_permission is None:

        return {
            "required_permission":
                None,

            "resolution_source":
                "no_execution_permission_resolved",

            "conservative_default_used":
                False,
        }

    return {
        "required_permission":
            default_permission,

        "resolution_source":
            "conservative_action_default",

        "conservative_default_used":
            True,
    }


# ============================================================
# Handoff construction
# ============================================================

def build_handoff(
    *,
    decision,
    decision_sha256,
):

    selected = (
        decision[
            "selected_action"
        ]
    )

    experiment_id = (
        selected.get(
            "experiment_id"
        )
    )

    registered_experiment = (
        selected.get(
            "registered_experiment"
        )
    )

    action = selected.get(
        "action"
    )

    permission = (
        resolve_permission(
            selected
        )
    )

    required_permission = (
        permission[
            "required_permission"
        ]
    )

    executable_registered_action = (
        experiment_id is not None
        and
        isinstance(
            registered_experiment,
            dict,
        )
        and
        registered_experiment.get(
            "experiment_id"
        )
        ==
        experiment_id
        and
        registered_experiment.get(
            "status"
        )
        ==
        "planned"
    )

    if executable_registered_action:

        if required_permission is None:

            raise RuntimeError(
                "Executable registered experiment "
                "has no resolved permission level."
            )

        approval_required = (
            required_permission
            ==
            "P3"
        )

        approval_status = (
            "awaiting_human_approval"
            if approval_required
            else
            "permission_resolved"
        )

        execution_status = (
            "blocked_pending_approval"
            if approval_required
            else
            "ready_for_execution_planning"
        )

    else:

        approval_required = False
        approval_status = (
            "not_applicable"
        )

        execution_status = (
            "experiment_design_required"
        )

    knowledge_state_sha256 = (
        decision[
            "inputs"
        ][
            "knowledge_state_sha256"
        ]
    )

    request_seed = (
        f"{knowledge_state_sha256}|"
        f"{selected.get('hypothesis_id')}|"
        f"{action}|"
        f"{experiment_id}"
    )

    request_hash = hashlib.sha256(
        request_seed.encode(
            "utf-8"
        )
    ).hexdigest()

    request_id = (
        "REQ-"
        +
        request_hash[
            :16
        ].upper()
    )

    return {
        "request_id":
            request_id,

        "hypothesis": {
            "hypothesis_id":
                selected.get(
                    "hypothesis_id"
                ),

            "title":
                selected.get(
                    "hypothesis_title"
                ),

            "current_status":
                selected.get(
                    "current_status"
                ),

            "proposed_status":
                selected.get(
                    "proposed_status"
                ),

            "evidence_classification":
                selected.get(
                    "evidence_classification"
                ),
        },

        "research_action": {
            "action":
                action,

            "priority":
                selected.get(
                    "priority"
                ),

            "reason":
                selected.get(
                    "action_reason"
                ),
        },

        "experiment": {
            "experiment_id":
                experiment_id,

            "registered":
                executable_registered_action,

            "record":
                registered_experiment,
        },

        "permission_gate": {
            "required_permission":
                required_permission,

            "resolution_source":
                permission[
                    "resolution_source"
                ],

            "conservative_default_used":
                permission[
                    "conservative_default_used"
                ],

            "human_approval_required":
                approval_required,

            "approval_status":
                approval_status,
        },

        "execution_gate": {
            "status":
                execution_status,

            "may_execute_now":
                False,

            "execution_attempted":
                False,

            "knowledge_store_mutated":
                False,

            "experiment_status_mutated":
                False,
        },

        "source_decision": {
            "engine_version":
                decision.get(
                    "engine_version"
                ),

            "decision_sha256":
                decision_sha256,

            "knowledge_state_sha256":
                knowledge_state_sha256,
        },
    }


# ============================================================
# Overall validation
# ============================================================

def validate_handoff(
    handoff,
):

    experiment_id = (
        handoff.get(
            "experiment",
            {},
        ).get(
            "experiment_id"
        )
    )

    registered = (
        handoff.get(
            "experiment",
            {},
        ).get(
            "registered"
        )
    )

    required_permission = (
        handoff.get(
            "permission_gate",
            {},
        ).get(
            "required_permission"
        )
    )

    approval_required = (
        handoff.get(
            "permission_gate",
            {},
        ).get(
            "human_approval_required"
        )
    )

    may_execute = (
        handoff.get(
            "execution_gate",
            {},
        ).get(
            "may_execute_now"
        )
    )

    checks = {
        "request_id_present":
            (
                isinstance(
                    handoff.get(
                        "request_id"
                    ),
                    str,
                )
                and
                handoff[
                    "request_id"
                ].startswith(
                    "REQ-"
                )
            ),

        "selected_hypothesis_present":
            bool(
                handoff.get(
                    "hypothesis",
                    {},
                ).get(
                    "hypothesis_id"
                )
            ),

        "research_action_present":
            bool(
                handoff.get(
                    "research_action",
                    {},
                ).get(
                    "action"
                )
            ),

        "selected_experiment_registered":
            (
                experiment_id is None
                or
                registered is True
            ),

        "permission_valid":
            (
                required_permission is None
                or
                required_permission
                in
                VALID_PERMISSIONS
            ),

        "p3_requires_human_approval":
            (
                required_permission
                !=
                "P3"
                or
                approval_required
                is True
            ),

        "execution_blocked":
            (
                may_execute
                is False
            ),

        "experiment_not_executed":
            (
                handoff.get(
                    "execution_gate",
                    {},
                ).get(
                    "execution_attempted"
                )
                is False
            ),

        "knowledge_store_not_mutated":
            (
                handoff.get(
                    "execution_gate",
                    {},
                ).get(
                    "knowledge_store_mutated"
                )
                is False
            ),

        "experiment_registry_not_mutated":
            (
                handoff.get(
                    "execution_gate",
                    {},
                ).get(
                    "experiment_status_mutated"
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
            "L8D handoff validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    decision_path = project_path(
        args.decision
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
            f"Research-action handoff already "
            f"exists: {output_path}\n"
            "Use --overwrite after the research "
            "decision changes."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L8D — EXECUTION GATE / "
        "RESEARCH ACTION HANDOFF"
    )

    print(
        "=" * 76
    )

    decision = load_json(
        decision_path
    )

    decision_sha256 = file_sha256(
        decision_path
    )

    input_checks = (
        validate_decision(
            decision
        )
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
        decision[
            "inputs"
        ][
            "knowledge_state_sha256"
        ]
    )

    print()

    print(
        "Research decision SHA:"
    )

    print(
        decision_sha256
    )

    print()

    print(
        "Input checks"
    )

    print(
        "------------"
    )

    for name, passed in (
        input_checks.items()
    ):

        print(
            f"{name:<38} "
            f"{passed}"
        )

    handoff = build_handoff(
        decision=decision,
        decision_sha256=(
            decision_sha256
        ),
    )

    handoff_checks = (
        validate_handoff(
            handoff
        )
    )

    output = {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "deterministic_safe_"
                "research_action_handoff"
            ),

        "policy": {
            "executes_experiment":
                False,

            "mutates_knowledge_store":
                False,

            "mutates_experiment_registry":
                False,

            "p3_requires_human_approval":
                True,

            "conservative_permission_default":
                (
                    "Registered discriminating "
                    "tests default to P3 when no "
                    "experiment-specific permission "
                    "metadata is available."
                ),
        },

        "input_checks":
            input_checks,

        "handoff":
            handoff,

        "overall_checks":
            handoff_checks,
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
            "Saved L8D handoff identity mismatch."
        )

    if not all(
        saved[
            "overall_checks"
        ].values()
    ):

        raise RuntimeError(
            "Saved L8D handoff contains "
            "failed checks."
        )

    # ========================================================
    # Display
    # ========================================================

    h = handoff

    print()
    print(
        "=" * 76
    )

    print(
        "L8D RESEARCH ACTION HANDOFF"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Request ID:"
    )

    print(
        f"  {h['request_id']}"
    )

    print()

    print(
        "Selected hypothesis:"
    )

    print(
        f"  {h['hypothesis']['hypothesis_id']}: "
        f"{h['hypothesis']['title']}"
    )

    print()

    print(
        "Selected action:"
    )

    print(
        f"  {h['research_action']['action']}"
    )

    print()

    print(
        "Selected experiment:"
    )

    print(
        f"  {h['experiment']['experiment_id']}"
    )

    print()

    print(
        "Required permission:"
    )

    print(
        f"  {h['permission_gate']['required_permission']}"
    )

    print()

    print(
        "Human approval required:"
    )

    print(
        f"  {h['permission_gate']['human_approval_required']}"
    )

    print()

    print(
        "Approval status:"
    )

    print(
        f"  {h['permission_gate']['approval_status']}"
    )

    print()

    print(
        "Execution status:"
    )

    print(
        f"  {h['execution_gate']['status']}"
    )

    print()

    print(
        "Overall checks"
    )

    print(
        "--------------"
    )

    for name, passed in (
        handoff_checks.items()
    ):

        print(
            f"{name:<40} "
            f"{passed}"
        )

    print()

    print(
        "Research-action handoff:"
    )

    print(
        output_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L8D "
        "EXECUTION GATE COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()
