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
    "L9B.1"
)


EXPECTED_PLAN_VERSION = (
    "L9A.1"
)


EXPECTED_EXPERIMENT_ID = (
    "EXP-0005"
)


DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan.json"
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec.json"
)


RUNNER_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "experiments"
    / "run_exp_0005.py"
)


APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0005.json"
)


ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0005_fixed_four_five"
)


SUMMARY_ARTIFACT = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0005_fixed_four_five_summary.json"
)


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


def relative_project_path(
    path,
):

    path = Path(
        path
    ).resolve()

    return str(
        path.relative_to(
            PROJECT_ROOT.resolve()
        )
    )


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
            "Stage L9B — compile the "
            "preregistered EXP-0005 plan into "
            "an exact execution specification "
            "without executing or approving it."
        )
    )

    parser.add_argument(
        "--plan",
        default=str(
            DEFAULT_PLAN_PATH
        ),
        help=(
            "Path to the L9A experiment plan."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Path for the L9B execution "
            "specification."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing execution "
            "specification after the experiment "
            "plan changes."
        ),
    )

    return parser.parse_args()


# ============================================================
# L9A plan validation
# ============================================================

def validate_plan_artifact(
    plan_artifact,
):

    package = (
        plan_artifact.get(
            "experiment_plan",
            {},
        )
    )

    plan = (
        package.get(
            "plan",
            {},
        )
    )

    checks = {
        "plan_version":
            (
                plan_artifact.get(
                    "engine_version"
                )
                ==
                EXPECTED_PLAN_VERSION
            ),

        "plan_checks_present":
            bool(
                plan_artifact.get(
                    "overall_checks"
                )
            ),

        "plan_checks_passed":
            (
                bool(
                    plan_artifact.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    plan_artifact[
                        "overall_checks"
                    ].values()
                )
            ),

        "experiment_is_exp_0005":
            (
                plan.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "permission_is_p3":
            (
                plan.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "execution_not_authorized":
            (
                plan.get(
                    "execution_constraints",
                    {},
                ).get(
                    "execution_authorized"
                )
                is False
            ),

        "human_approval_required":
            (
                plan.get(
                    "execution_constraints",
                    {},
                ).get(
                    "human_approval_required"
                )
                is True
            ),

        "approval_still_pending":
            (
                plan.get(
                    "execution_constraints",
                    {},
                ).get(
                    "approval_status"
                )
                ==
                "awaiting_human_approval"
            ),

        "fixed_four_five_candidates":
            (
                plan.get(
                    "fixed_candidates",
                    {},
                ).get(
                    "four"
                )
                ==
                " four."
                and
                plan.get(
                    "fixed_candidates",
                    {},
                ).get(
                    "five"
                )
                ==
                " five."
            ),

        "probe_count_is_12":
            (
                plan.get(
                    "probe_design",
                    {},
                ).get(
                    "probe_count"
                )
                ==
                12
            ),

        "balanced_correctness":
            (
                plan.get(
                    "probe_design",
                    {},
                ).get(
                    "four_correct_count"
                )
                ==
                6
                and
                plan.get(
                    "probe_design",
                    {},
                ).get(
                    "five_correct_count"
                )
                ==
                6
            ),

        "checkpoint_sha256_present":
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

        "checkpoint_path_present":
            bool(
                plan.get(
                    "model_scope",
                    {},
                ).get(
                    "checkpoint"
                )
            ),

        "source_handoff_present":
            bool(
                package.get(
                    "source_handoff"
                )
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
            "L9B plan validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Build exact execution specification
# ============================================================

def build_execution_spec(
    *,
    plan_artifact,
    plan_sha256,
):

    package = (
        plan_artifact[
            "experiment_plan"
        ]
    )

    plan = (
        package[
            "plan"
        ]
    )

    probes = (
        plan[
            "probe_design"
        ][
            "probes"
        ]
    )

    target_heads = (
        plan[
            "target_heads"
        ]
    )

    model_scope = (
        plan[
            "model_scope"
        ]
    )

    checkpoint_path = (
        model_scope[
            "checkpoint"
        ]
    )

    checkpoint_sha256 = (
        model_scope[
            "checkpoint_sha256"
        ]
    )

    # ========================================================
    # Runner contract
    # ========================================================

    runner_contract = {
        "runner_path":
            relative_project_path(
                RUNNER_PATH
            ),

        "runner_must_exist_before_execution":
            True,

        "runner_required_arguments": [
            "--execution-spec",
            "--approval",
        ],

        "execution_spec_argument":
            relative_project_path(
                DEFAULT_OUTPUT_PATH
            ),

        "approval_argument":
            relative_project_path(
                APPROVAL_PATH
            ),

        "must_verify_before_model_load": [
            (
                "Execution specification "
                "identity and SHA-256."
            ),
            (
                "P3 approval artifact identity "
                "and approval status."
            ),
            (
                "Experiment ID EXP-0005."
            ),
            (
                "Checkpoint path and SHA-256."
            ),
            (
                "Plan ID and source plan SHA-256."
            ),
        ],

        "must_verify_after_model_load": [
            (
                "Runtime model identity matches "
                "the registered checkpoint."
            ),
            (
                "Scale=1 head intervention "
                "reproduces the baseline within "
                "the established exactness "
                "tolerance."
            ),
        ],

        "must_not": [
            (
                "Modify checkpoint weights "
                "persistently."
            ),
            (
                "Update the knowledge store "
                "during experiment execution."
            ),
            (
                "Update hypothesis status "
                "during experiment execution."
            ),
            (
                "Change preregistered probes, "
                "heads, margin orientation, or "
                "decision thresholds."
            ),
        ],
    }

    # ========================================================
    # Exact experimental payload
    # ========================================================

    experiment_payload = {
        "experiment_id":
            plan[
                "experiment_id"
            ],

        "plan_id":
            package[
                "plan_id"
            ],

        "hypothesis_id":
            plan[
                "hypothesis_id"
            ],

        "checkpoint": {
            "path":
                checkpoint_path,

            "sha256":
                checkpoint_sha256,

            "training_step":
                model_scope.get(
                    "training_step"
                ),

            "tokens_seen":
                model_scope.get(
                    "tokens_seen"
                ),
        },

        "fixed_candidates":
            plan[
                "fixed_candidates"
            ],

        "measurement_definition":
            plan[
                "measurement_definition"
            ],

        "target_heads":
            target_heads,

        "probes":
            probes,

        "intervention":
            plan[
                "intervention"
            ],

        "decision_rules":
            plan[
                "decision_rules"
            ],
    }

    # ========================================================
    # Runtime constraints
    # ========================================================

    plan_runtime = (
        plan[
            "execution_constraints"
        ]
    )

    runtime_constraints = {
        "load_model_once":
            bool(
                plan_runtime.get(
                    "load_model_once",
                    True,
                )
            ),

        "environment": {
            "OMP_NUM_THREADS":
                "1",

            "OPENBLAS_NUM_THREADS":
                "1",

            "MKL_NUM_THREADS":
                "1",
        },

        "nice_value":
            int(
                plan_runtime.get(
                    "recommended_niceness",
                    15,
                )
            ),

        "checkpoint_sha256_must_match":
            checkpoint_sha256,

        "deterministic_probe_order":
            [
                probe[
                    "probe_id"
                ]

                for probe
                in probes
            ],

        "deterministic_head_order":
            [
                head[
                    "head_id"
                ]

                for head
                in target_heads
            ],
    }

    # ========================================================
    # Artifact contract
    # ========================================================

    artifact_contract = {
        "artifact_directory":
            relative_project_path(
                ARTIFACT_DIRECTORY
            ),

        "summary_artifact":
            relative_project_path(
                SUMMARY_ARTIFACT
            ),

        "per_probe_filename_template":
            (
                "{probe_id_lower}_"
                "fixed_four_five_heads.json"
            ),

        "required_summary_fields": [
            "experiment_id",
            "title",
            "question",
            "status",
            "evidence_level",
            "checkpoint",
            "checkpoint_sha256",
            "model",
            "plan_id",
            "execution_spec_id",
            "selected_probe_ids",
            "probe_count",
            "head_count",
            "heads",
            "probes",
            "head_summaries",
            "subset_summaries",
            "preregistered_tests",
            "hypothesis_diagnostics",
            "interpretation_rule",
            "checks",
        ],

        "required_evidence_level":
            "E3",

        "ingestion_target":
            (
                "Stage L7B evidence ledger "
                "after successful completion."
            ),
    }

    # ========================================================
    # Approval contract
    # ========================================================

    approval_contract = {
        "required_permission":
            "P3",

        "human_approval_required":
            True,

        "approval_artifact_path":
            relative_project_path(
                APPROVAL_PATH
            ),

        "approval_artifact_exists_at_L9B":
            APPROVAL_PATH.exists(),

        "approval_status_required_for_execution":
            "approved",

        "approval_must_bind_to": [
            "EXP-0005",
            package[
                "plan_id"
            ],
            plan_sha256,
            checkpoint_sha256,
        ],

        "execution_authorized":
            False,

        "may_execute_now":
            False,
    }

    # ========================================================
    # Exact command contract
    #
    # This is a command TEMPLATE only.
    # L9B never executes it.
    # ========================================================

    execution_command = {
        "environment_prefix": (
            "OMP_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1"
        ),

        "nice_prefix":
            "nice -n 15",

        "python":
            "python",

        "runner":
            relative_project_path(
                RUNNER_PATH
            ),

        "arguments": [
            "--execution-spec",
            relative_project_path(
                DEFAULT_OUTPUT_PATH
            ),
            "--approval",
            relative_project_path(
                APPROVAL_PATH
            ),
        ],

        "shell_preview": (
            "OMP_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 "
            "nice -n 15 "
            "python "
            f"{relative_project_path(RUNNER_PATH)} "
            "--execution-spec "
            f"{relative_project_path(DEFAULT_OUTPUT_PATH)} "
            "--approval "
            f"{relative_project_path(APPROVAL_PATH)}"
        ),

        "executed_by_L9B":
            False,
    }

    spec_core = {
        "experiment_payload":
            experiment_payload,

        "runner_contract":
            runner_contract,

        "runtime_constraints":
            runtime_constraints,

        "artifact_contract":
            artifact_contract,

        "approval_contract":
            approval_contract,

        "execution_command":
            execution_command,

        "source_plan": {
            "path":
                relative_project_path(
                    DEFAULT_PLAN_PATH
                ),

            "sha256":
                plan_sha256,

            "plan_id":
                package[
                    "plan_id"
                ],

            "knowledge_state_sha256":
                package[
                    "source_handoff"
                ][
                    "knowledge_state_sha256"
                ],
        },
    }

    execution_spec_id = (
        "SPEC-"
        +
        canonical_json_sha256(
            spec_core
        )[
            :16
        ].upper()
    )

    return {
        "execution_spec_id":
            execution_spec_id,

        **spec_core,
    }


# ============================================================
# Specification validation
# ============================================================

def validate_execution_spec(
    spec,
):

    payload = (
        spec[
            "experiment_payload"
        ]
    )

    approval = (
        spec[
            "approval_contract"
        ]
    )

    command = (
        spec[
            "execution_command"
        ]
    )

    runner = (
        spec[
            "runner_contract"
        ]
    )

    artifacts = (
        spec[
            "artifact_contract"
        ]
    )

    runtime = (
        spec[
            "runtime_constraints"
        ]
    )

    head_ids = [
        head[
            "head_id"
        ]

        for head
        in payload[
            "target_heads"
        ]
    ]

    probe_ids = [
        probe[
            "probe_id"
        ]

        for probe
        in payload[
            "probes"
        ]
    ]

    checks = {
        "execution_spec_id_present":
            (
                isinstance(
                    spec.get(
                        "execution_spec_id"
                    ),
                    str,
                )
                and
                spec[
                    "execution_spec_id"
                ].startswith(
                    "SPEC-"
                )
            ),

        "experiment_id":
            (
                payload.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "plan_id_present":
            (
                isinstance(
                    payload.get(
                        "plan_id"
                    ),
                    str,
                )
                and
                payload[
                    "plan_id"
                ].startswith(
                    "PLAN-"
                )
            ),

        "checkpoint_sha_valid":
            (
                isinstance(
                    payload.get(
                        "checkpoint",
                        {},
                    ).get(
                        "sha256"
                    ),
                    str,
                )
                and
                len(
                    payload[
                        "checkpoint"
                    ][
                        "sha256"
                    ]
                )
                ==
                64
            ),

        "four_target_heads":
            (
                len(
                    head_ids
                )
                ==
                4
                and
                len(
                    set(
                        head_ids
                    )
                )
                ==
                4
            ),

        "primary_heads_present":
            (
                {
                    "L1H0",
                    "L4H0",
                }
                <=
                set(
                    head_ids
                )
            ),

        "weak_control_present":
            (
                "L4H1"
                in
                head_ids
            ),

        "twelve_unique_probes":
            (
                len(
                    probe_ids
                )
                ==
                12
                and
                len(
                    set(
                        probe_ids
                    )
                )
                ==
                12
            ),

        "probe_order_frozen":
            (
                runtime[
                    "deterministic_probe_order"
                ]
                ==
                probe_ids
            ),

        "head_order_frozen":
            (
                runtime[
                    "deterministic_head_order"
                ]
                ==
                head_ids
            ),

        "thread_limits_are_one":
            (
                all(
                    value
                    ==
                    "1"

                    for value
                    in runtime[
                        "environment"
                    ].values()
                )
            ),

        "nice_value_is_15":
            (
                runtime[
                    "nice_value"
                ]
                ==
                15
            ),

        "p3_required":
            (
                approval[
                    "required_permission"
                ]
                ==
                "P3"
            ),

        "human_approval_required":
            (
                approval[
                    "human_approval_required"
                ]
                is True
            ),

        "execution_not_authorized":
            (
                approval[
                    "execution_authorized"
                ]
                is False
                and
                approval[
                    "may_execute_now"
                ]
                is False
            ),

        "command_not_executed":
            (
                command[
                    "executed_by_L9B"
                ]
                is False
            ),

        "runner_contract_requires_approval":
            (
                "--approval"
                in
                runner[
                    "runner_required_arguments"
                ]
            ),

        "artifact_summary_registered":
            bool(
                artifacts.get(
                    "summary_artifact"
                )
            ),

        "evidence_level_e3":
            (
                artifacts[
                    "required_evidence_level"
                ]
                ==
                "E3"
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
            "L9B execution specification "
            f"failed validation: {failed}"
        )

    return checks


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    plan_path = project_path(
        args.plan
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
            f"Execution specification already "
            f"exists: {output_path}\n"
            "Use --overwrite after the L9A plan "
            "changes."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L9B — EXPERIMENT "
        "EXECUTION SPECIFICATION"
    )

    print(
        "=" * 76
    )

    plan_artifact = load_json(
        plan_path
    )

    plan_sha256 = file_sha256(
        plan_path
    )

    plan_checks = (
        validate_plan_artifact(
            plan_artifact
        )
    )

    print()
    print(
        "Source plan"
    )

    print(
        "-----------"
    )

    print(
        f"Plan SHA-256:"
    )

    print(
        plan_sha256
    )

    print()

    print(
        "Plan checks"
    )

    print(
        "-----------"
    )

    for name, passed in (
        plan_checks.items()
    ):

        print(
            f"{name:<40} "
            f"{passed}"
        )

    spec = build_execution_spec(
        plan_artifact=plan_artifact,
        plan_sha256=(
            plan_sha256
        ),
    )

    spec_checks = (
        validate_execution_spec(
            spec
        )
    )

    output = {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "deterministic_blocked_"
                "execution_specification"
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

        "source_plan_path":
            relative_project_path(
                plan_path
            ),

        "plan_checks":
            plan_checks,

        "execution_spec":
            spec,

        "overall_checks":
            spec_checks,
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
            "Saved L9B execution "
            "specification identity mismatch."
        )

    if not all(
        saved[
            "overall_checks"
        ].values()
    ):

        raise RuntimeError(
            "Saved L9B execution "
            "specification contains failed "
            "checks."
        )

    s = (
        spec
    )

    print()
    print(
        "=" * 76
    )

    print(
        "L9B EXECUTION SPECIFICATION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Execution Spec ID:"
    )

    print(
        f"  {s['execution_spec_id']}"
    )

    print()

    print(
        "Experiment:"
    )

    print(
        f"  {s['experiment_payload']['experiment_id']}"
    )

    print()

    print(
        "Plan:"
    )

    print(
        f"  {s['experiment_payload']['plan_id']}"
    )

    print()

    print(
        "Checkpoint:"
    )

    print(
        f"  {s['experiment_payload']['checkpoint']['path']}"
    )

    print()

    print(
        "Target heads:"
    )

    for head in (
        s[
            "experiment_payload"
        ][
            "target_heads"
        ]
    ):

        print(
            f"  {head['head_id']} — "
            f"{head['role']}"
        )

    print()

    print(
        "Probe count:"
    )

    print(
        f"  {len(s['experiment_payload']['probes'])}"
    )

    print()

    print(
        "Required permission:"
    )

    print(
        f"  {s['approval_contract']['required_permission']}"
    )

    print()

    print(
        "Approval status:"
    )

    print(
        "  awaiting_human_approval"
    )

    print()

    print(
        "Execution authorized:"
    )

    print(
        f"  {s['approval_contract']['execution_authorized']}"
    )

    print()

    print(
        "Future runner:"
    )

    print(
        f"  {s['runner_contract']['runner_path']}"
    )

    print()

    print(
        "Future command template:"
    )

    print(
        "  "
        +
        s[
            "execution_command"
        ][
            "shell_preview"
        ]
    )

    print()

    print(
        "Overall checks"
    )

    print(
        "--------------"
    )

    for name, passed in (
        spec_checks.items()
    ):

        print(
            f"{name:<42} "
            f"{passed}"
        )

    print()

    print(
        "Execution specification:"
    )

    print(
        output_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L9B "
        "EXECUTION SPECIFICATION COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()
