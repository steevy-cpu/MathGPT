import argparse
import hashlib
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
# Identity
# ============================================================

ENGINE_VERSION = "L10C.1"
EXPECTED_PLAN_VERSION = "L10B.1"
EXPECTED_EXPERIMENT_ID = "EXP-0006"
EXPECTED_HYPOTHESIS_ID = "HYP-V03-001"

DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0006.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec_exp_0006.json"
)

RUNNER_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "experiments"
    / "run_exp_0006.py"
)

APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0006.json"
)

ARTIFACT_DIRECTORY = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0006_cross_pair_correctness"
)

SUMMARY_ARTIFACT = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0006_cross_pair_correctness_summary.json"
)


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

    return str(
        path.relative_to(
            PROJECT_ROOT.resolve()
        )
    )


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

    temporary = path.with_name(
        path.name + ".tmp"
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


def file_sha256(path):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L10C — compile the frozen "
            "EXP-0006 preregistered plan into "
            "an exact execution specification "
            "without executing the experiment "
            "or granting P3 approval."
        )
    )

    parser.add_argument(
        "--plan",
        default=str(
            DEFAULT_PLAN_PATH
        ),
        help=(
            "Path to the L10B EXP-0006 "
            "preregistered plan."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Path for the L10C EXP-0006 "
            "execution specification."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing execution "
            "specification only before execution "
            "has begun."
        ),
    )

    return parser.parse_args()


# ============================================================
# L10B plan validation
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

    probe_design = (
        plan.get(
            "probe_design",
            {},
        )
    )

    pair_design = (
        plan.get(
            "candidate_pair_design",
            {},
        )
    )

    decision_rules = (
        plan.get(
            "decision_rules",
            {},
        )
    )

    execution_policy = (
        plan.get(
            "execution_policy",
            {},
        )
    )

    probes = (
        probe_design.get(
            "probes",
            [],
        )
    )

    pairs = (
        pair_design.get(
            "pairs",
            [],
        )
    )

    probe_ids = [
        probe.get(
            "probe_id"
        )

        for probe
        in probes
    ]

    pair_ids = [
        pair.get(
            "pair_id"
        )

        for pair
        in pairs
    ]

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

        "experiment_is_exp_0006":
            (
                plan.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "hypothesis_is_hyp_v03_001":
            (
                plan.get(
                    "hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
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
                execution_policy.get(
                    "execution_authorized"
                )
                is False
            ),

        "human_approval_required":
            (
                execution_policy.get(
                    "human_approval_required"
                )
                is True
            ),

        "approval_must_bind_exact_plan":
            (
                execution_policy.get(
                    "approval_must_bind_exact_plan"
                )
                is True
            ),

        "approval_must_bind_exact_spec":
            (
                execution_policy.get(
                    "approval_must_bind_exact_execution_spec"
                )
                is True
            ),

        "probe_count_is_16":
            (
                probe_design.get(
                    "probe_count"
                )
                ==
                16
            ),

        "sixteen_unique_probes":
            (
                len(
                    probe_ids
                )
                ==
                16
                and
                len(
                    set(
                        probe_ids
                    )
                )
                ==
                16
            ),

        "balanced_correctness":
            (
                probe_design.get(
                    "lower_correct_count"
                )
                ==
                8
                and
                probe_design.get(
                    "upper_correct_count"
                )
                ==
                8
            ),

        "balanced_format":
            (
                probe_design.get(
                    "word_probe_count"
                )
                ==
                8
                and
                probe_design.get(
                    "symbol_probe_count"
                )
                ==
                8
            ),

        "pair_count_is_4":
            (
                pair_design.get(
                    "pair_count"
                )
                ==
                4
            ),

        "four_unique_pairs":
            (
                len(
                    pair_ids
                )
                ==
                4
                and
                len(
                    set(
                        pair_ids
                    )
                )
                ==
                4
            ),

        "expected_pairs_present":
            (
                set(
                    pair_ids
                )
                ==
                {
                    "P23",
                    "P34",
                    "P45",
                    "P56",
                }
            ),

        "positive_control_is_p45":
            (
                pair_design.get(
                    "positive_control_pair"
                )
                ==
                "P45"
                and
                decision_rules.get(
                    "positive_control",
                    {},
                ).get(
                    "pair_id"
                )
                ==
                "P45"
            ),

        "global_threshold_75_percent":
            (
                decision_rules.get(
                    "global_directional_threshold"
                )
                ==
                0.75
                and
                decision_rules.get(
                    "minimum_global_consistent_probes"
                )
                ==
                12
            ),

        "per_pair_threshold_75_percent":
            (
                decision_rules.get(
                    "per_pair_directional_threshold"
                )
                ==
                0.75
                and
                decision_rules.get(
                    "minimum_per_pair_consistent_probes"
                )
                ==
                3
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

        "source_proposal_present":
            bool(
                plan_artifact.get(
                    "source_proposal"
                )
            ),

        "plan_does_not_execute":
            (
                execution_policy.get(
                    "plan_does_not_execute"
                )
                is True
            ),

        "plan_does_not_mutate_knowledge":
            (
                execution_policy.get(
                    "plan_does_not_mutate_knowledge_store"
                )
                is True
            ),

        "plan_does_not_create_evidence":
            (
                execution_policy.get(
                    "plan_does_not_create_evidence"
                )
                is True
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
            "L10C plan validation failed: "
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

    pairs = (
        plan[
            "candidate_pair_design"
        ][
            "pairs"
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

    probe_ids = [
        probe[
            "probe_id"
        ]

        for probe
        in probes
    ]

    head_ids = [
        head[
            "head_id"
        ]

        for head
        in target_heads
    ]

    pair_ids = [
        pair[
            "pair_id"
        ]

        for pair
        in pairs
    ]

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
                "identity and exact file SHA-256."
            ),
            (
                "P3 approval artifact identity, "
                "decision, and exact binding."
            ),
            (
                "Experiment ID EXP-0006."
            ),
            (
                "Hypothesis ID HYP-V03-001."
            ),
            (
                "Checkpoint path and SHA-256."
            ),
            (
                "Plan ID and source-plan "
                "SHA-256."
            ),
            (
                "Frozen probe, pair, and head "
                "ordering."
            ),
            (
                "Frozen lower-minus-upper margin "
                "orientation."
            ),
            (
                "P45 positive-control decision "
                "rule."
            ),
        ],

        "must_verify_after_model_load": [
            (
                "Runtime model identity matches "
                "the registered mature v0.3 "
                "checkpoint."
            ),
            (
                "Every candidate pair has a "
                "valid first discriminative "
                "token position."
            ),
            (
                "Scale=1 head intervention "
                "reproduces baseline within the "
                "established exactness "
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
                "Create or edit scientific "
                "findings during execution."
            ),
            (
                "Change preregistered candidate "
                "pairs."
            ),
            (
                "Change preregistered prompts or "
                "candidate strings."
            ),
            (
                "Change head order, probe order, "
                "margin orientation, thresholds, "
                "or outcome labels."
            ),
            (
                "Interpret primary outcomes when "
                "the P45 positive control fails."
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

        "hypothesis_status_at_preregistration":
            plan.get(
                "hypothesis_status_at_preregistration"
            ),

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

        "candidate_pairs":
            pairs,

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

        "scientific_interpretation_guardrails":
            plan[
                "scientific_interpretation_guardrails"
            ],
    }

    # ========================================================
    # Runtime constraints
    # ========================================================

    runtime_constraints = {
        "load_model_once":
            True,

        "environment": {
            "OMP_NUM_THREADS":
                "1",

            "OPENBLAS_NUM_THREADS":
                "1",

            "MKL_NUM_THREADS":
                "1",
        },

        "nice_value":
            15,

        "checkpoint_sha256_must_match":
            checkpoint_sha256,

        "deterministic_pair_order":
            pair_ids,

        "deterministic_probe_order":
            probe_ids,

        "deterministic_head_order":
            head_ids,

        "expected_pair_count":
            4,

        "expected_probe_count":
            16,

        "expected_head_count":
            4,

        "expected_effect_record_count":
            (
                len(
                    probes
                )
                *
                len(
                    target_heads
                )
            ),

        "scale_one_exactness_required":
            True,

        "persistent_weight_modification":
            False,
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
                "cross_pair_heads.json"
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
            "approval_id",
            "candidate_pairs",
            "selected_probe_ids",
            "probe_count",
            "head_count",
            "heads",
            "probes",
            "head_summaries",
            "pair_summaries",
            "correctness_summaries",
            "format_summaries",
            "preregistered_tests",
            "positive_control",
            "token_bias_diagnostics",
            "hypothesis_diagnostics",
            "interpretation_rule",
            "checks",
        ],

        "required_evidence_level":
            "E3",

        "ingestion_target":
            (
                "Stage L7 evidence ledger only "
                "after successful completion and "
                "post-run result validation."
            ),

        "runner_may_write_only_under": [
            relative_project_path(
                ARTIFACT_DIRECTORY
            ),
            relative_project_path(
                SUMMARY_ARTIFACT
            ),
        ],
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

        "approval_artifact_exists_at_L10C":
            APPROVAL_PATH.exists(),

        "approval_status_required_for_execution":
            "approved",

        "approval_scope_required":
            (
                "temporary_causal_intervention_only"
            ),

        "approval_must_bind_to": {
            "experiment_id":
                EXPECTED_EXPERIMENT_ID,

            "hypothesis_id":
                EXPECTED_HYPOTHESIS_ID,

            "plan_id":
                package[
                    "plan_id"
                ],

            "source_plan_sha256":
                plan_sha256,

            "checkpoint_sha256":
                checkpoint_sha256,
        },

        "execution_authorized":
            False,

        "may_execute_now":
            False,
    }

    # ========================================================
    # Exact command contract
    #
    # Template only. L10C never executes it.
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

        "executed_by_L10C":
            False,
    }

    # ========================================================
    # Frozen source-plan binding
    # ========================================================

    source_plan = {
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

        "plan_fingerprint_sha256":
            package[
                "plan_fingerprint_sha256"
            ],

        "source_proposal_sha256":
            plan_artifact[
                "source_proposal"
            ][
                "sha256"
            ],

        "source_proposal_id":
            plan_artifact[
                "source_proposal"
            ][
                "proposal_id"
            ],

        "knowledge_state_sha256":
            plan_artifact[
                "source_proposal"
            ][
                "source_knowledge_state_sha256"
            ],
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

        "source_plan":
            source_plan,
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

    decision_rules = (
        payload[
            "decision_rules"
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

    pair_ids = [
        pair[
            "pair_id"
        ]

        for pair
        in payload[
            "candidate_pairs"
        ]
    ]

    positive_control = (
        decision_rules[
            "positive_control"
        ]
    )

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

        "hypothesis_id":
            (
                payload.get(
                    "hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
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

        "expected_primary_heads_present":
            (
                {
                    "L1H0",
                    "L3H3",
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

        "four_candidate_pairs":
            (
                len(
                    pair_ids
                )
                ==
                4
                and
                set(
                    pair_ids
                )
                ==
                {
                    "P23",
                    "P34",
                    "P45",
                    "P56",
                }
            ),

        "sixteen_unique_probes":
            (
                len(
                    probe_ids
                )
                ==
                16
                and
                len(
                    set(
                        probe_ids
                    )
                )
                ==
                16
            ),

        "pair_order_frozen":
            (
                runtime[
                    "deterministic_pair_order"
                ]
                ==
                pair_ids
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

        "effect_record_count_64":
            (
                runtime[
                    "expected_effect_record_count"
                ]
                ==
                64
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

        "positive_control_is_p45":
            (
                positive_control.get(
                    "pair_id"
                )
                ==
                "P45"
            ),

        "global_threshold_is_12_of_16":
            (
                decision_rules.get(
                    "minimum_global_consistent_probes"
                )
                ==
                12
            ),

        "per_pair_threshold_is_3_of_4":
            (
                decision_rules.get(
                    "minimum_per_pair_consistent_probes"
                )
                ==
                3
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
                    "executed_by_L10C"
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

        "runner_contract_requires_spec":
            (
                "--execution-spec"
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

        "persistent_weights_unchanged":
            (
                runtime[
                    "persistent_weight_modification"
                ]
                is False
            ),

        "scale_one_control_required":
            (
                runtime[
                    "scale_one_exactness_required"
                ]
                is True
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
            "L10C execution specification "
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
            "EXP-0006 execution specification "
            f"already exists: {output_path}\n"
            "Use --overwrite only before "
            "execution begins."
        )

    print()
    print(
        "=" * 80
    )

    print(
        "STAGE L10C — EXP-0006 "
        "EXECUTION SPECIFICATION"
    )

    print(
        "=" * 80
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
        "Plan SHA-256:"
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
            f"{name:<46} "
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
                "exp_0006_execution_specification"
            ),

        "policy": {
            "executes_experiment":
                False,

            "grants_p3_approval":
                False,

            "loads_model":
                False,

            "performs_intervention":
                False,

            "creates_evidence":
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

    require(
        saved.get(
            "engine_version"
        )
        ==
        ENGINE_VERSION,
        (
            "Saved L10C execution "
            "specification identity mismatch."
        ),
    )

    require(
        all(
            saved[
                "overall_checks"
            ].values()
        ),
        (
            "Saved L10C execution "
            "specification contains failed "
            "checks."
        ),
    )

    s = saved[
        "execution_spec"
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "L10C EXP-0006 EXECUTION "
        "SPECIFICATION"
    )

    print(
        "=" * 80
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
        "Hypothesis:"
    )

    print(
        f"  {s['experiment_payload']['hypothesis_id']}"
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
        "Candidate pairs:"
    )

    for pair in (
        s[
            "experiment_payload"
        ][
            "candidate_pairs"
        ]
    ):

        print(
            f"  {pair['pair_id']} — "
            f"{pair['label']} "
            f"({pair['role']})"
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
            f"{head['hypothesis_role']}"
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
        "Expected causal effects:"
    )

    print(
        f"  {s['runtime_constraints']['expected_effect_record_count']}"
    )

    print()
    print(
        "Positive control:"
    )

    print(
        "  P45 four-vs-five"
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
        saved[
            "overall_checks"
        ].items()
    ):

        print(
            f"{name:<44} "
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
        "=" * 80
    )

    print(
        "SUCCESS: STAGE L10C EXP-0006 "
        "EXECUTION SPECIFICATION COMPLETE."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
