import argparse
import copy
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
)


# ============================================================
# Identity
# ============================================================

INTAKE_VERSION = "L10F.1"

EXPECTED_EXPERIMENT_ID = "EXP-0006"
EXPECTED_HYPOTHESIS_ID = "HYP-V03-001"
EXPECTED_EVIDENCE_LEVEL = "E3"

EXPECTED_PLAN_VERSION = "L10B.1"
EXPECTED_SPEC_VERSION = "L10C.1"
EXPECTED_APPROVAL_VERSION = "P3A.1"

FINDING_MAIN = "FND-0008"
FINDING_TOKEN_BIAS = "FND-0009"
FINDING_L3H3 = "FND-0010"

EVIDENCE_RECORD_ID = "EVR-EXP-0006"
EXTRACTION_ADAPTER = "exp_0006_v1"


# ============================================================
# Default paths
# ============================================================

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0006_cross_pair_correctness_summary.json"
)

DEFAULT_APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0006.json"
)

DEFAULT_SPEC_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec_exp_0006.json"
)

DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0006.json"
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

DEFAULT_INTAKE_REPORT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "exp_0006_result_intake_report.json"
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


def file_sha256(path):
    path = Path(path)

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


def capture_backups(paths):
    backups = {}

    for path in paths:
        path = Path(path)

        backups[path] = (
            path.read_bytes()
            if path.exists()
            else None
        )

    return backups


def restore_backups(backups):
    for path, content in backups.items():

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
            "Stage L10F — verify the completed "
            "EXP-0006 E3 result and reintegrate "
            "its preregistered and mechanistic "
            "findings into the Stage-L7 "
            "scientific knowledge layer."
        )
    )

    parser.add_argument(
        "--summary",
        default=str(
            DEFAULT_SUMMARY_PATH
        ),
    )

    parser.add_argument(
        "--approval",
        default=str(
            DEFAULT_APPROVAL_PATH
        ),
    )

    parser.add_argument(
        "--execution-spec",
        default=str(
            DEFAULT_SPEC_PATH
        ),
    )

    parser.add_argument(
        "--plan",
        default=str(
            DEFAULT_PLAN_PATH
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
            DEFAULT_INTAKE_REPORT_PATH
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate EXP-0006 and display "
            "the proposed L7 changes without "
            "writing them."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permit replacement of an existing "
            "EXP-0006 integration only when "
            "the exact frozen identities still "
            "verify."
        ),
    )

    return parser.parse_args()


# ============================================================
# Frozen result verification
# ============================================================

def validate_frozen_inputs(
    *,
    summary,
    summary_path,
    approval,
    approval_path,
    spec_artifact,
    spec_path,
    plan_artifact,
    plan_path,
):

    summary_sha256 = file_sha256(
        summary_path
    )

    approval_sha256 = file_sha256(
        approval_path
    )

    spec_sha256 = file_sha256(
        spec_path
    )

    plan_sha256 = file_sha256(
        plan_path
    )

    spec = spec_artifact.get(
        "execution_spec",
        {},
    )

    payload = spec.get(
        "experiment_payload",
        {},
    )

    source_plan = spec.get(
        "source_plan",
        {},
    )

    plan_package = plan_artifact.get(
        "experiment_plan",
        {},
    )

    plan = plan_package.get(
        "plan",
        {},
    )

    summary_checks = summary.get(
        "checks",
        {},
    )

    preregistered = summary.get(
        "preregistered_tests",
        {},
    )

    positive_control = summary.get(
        "positive_control",
        {},
    )

    token_bias = summary.get(
        "token_bias_diagnostics",
        {},
    )

    binding = approval.get(
        "binding",
        {},
    )

    expected_checkpoint_sha = (
        payload.get(
            "checkpoint",
            {},
        ).get(
            "sha256"
        )
    )

    checkpoint_path = project_path(
        payload[
            "checkpoint"
        ][
            "path"
        ]
    )

    require(
        checkpoint_path.exists(),
        (
            "EXP-0006 checkpoint does not "
            f"exist: {checkpoint_path}"
        ),
    )

    actual_checkpoint_sha = (
        file_sha256(
            checkpoint_path
        )
    )

    checks = {
        "summary_experiment_id":
            (
                summary.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "summary_hypothesis_id":
            (
                summary.get(
                    "hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "summary_status_success":
            (
                summary.get(
                    "status"
                )
                ==
                "success"
            ),

        "summary_evidence_level_e3":
            (
                summary.get(
                    "evidence_level"
                )
                ==
                EXPECTED_EVIDENCE_LEVEL
            ),

        "summary_integrity_checks_present":
            bool(
                summary_checks
            ),

        "summary_integrity_checks_pass":
            (
                bool(
                    summary_checks
                )
                and
                all(
                    summary_checks.values()
                )
            ),

        "summary_probe_count_16":
            (
                summary.get(
                    "probe_count"
                )
                ==
                16
            ),

        "summary_head_count_4":
            (
                summary.get(
                    "head_count"
                )
                ==
                4
            ),

        "summary_plan_id_matches":
            (
                summary.get(
                    "plan_id"
                )
                ==
                payload.get(
                    "plan_id"
                )
            ),

        "summary_spec_id_matches":
            (
                summary.get(
                    "execution_spec_id"
                )
                ==
                spec.get(
                    "execution_spec_id"
                )
            ),

        "summary_spec_sha_matches":
            (
                summary.get(
                    "execution_spec_sha256"
                )
                ==
                spec_sha256
            ),

        "summary_checkpoint_matches":
            (
                summary.get(
                    "checkpoint_sha256"
                )
                ==
                expected_checkpoint_sha
            ),

        "checkpoint_file_sha_matches":
            (
                actual_checkpoint_sha
                ==
                expected_checkpoint_sha
            ),

        "plan_version":
            (
                plan_artifact.get(
                    "engine_version"
                )
                ==
                EXPECTED_PLAN_VERSION
            ),

        "plan_id_matches":
            (
                plan_package.get(
                    "plan_id"
                )
                ==
                payload.get(
                    "plan_id"
                )
            ),

        "plan_sha_matches_spec":
            (
                plan_sha256
                ==
                source_plan.get(
                    "sha256"
                )
            ),

        "plan_experiment_id":
            (
                plan.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "plan_hypothesis_id":
            (
                plan.get(
                    "hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "spec_version":
            (
                spec_artifact.get(
                    "engine_version"
                )
                ==
                EXPECTED_SPEC_VERSION
            ),

        "spec_checks_pass":
            (
                bool(
                    spec_artifact.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    spec_artifact[
                        "overall_checks"
                    ].values()
                )
            ),

        "approval_version":
            (
                approval.get(
                    "approval_version"
                )
                ==
                EXPECTED_APPROVAL_VERSION
            ),

        "approval_decision":
            (
                approval.get(
                    "decision"
                )
                ==
                "approved"
            ),

        "approval_permission":
            (
                approval.get(
                    "permission"
                )
                ==
                "P3"
            ),

        "approval_by_human":
            (
                approval.get(
                    "approved_by"
                )
                ==
                "human"
            ),

        "summary_approval_id_matches":
            (
                summary.get(
                    "approval_id"
                )
                ==
                approval.get(
                    "approval_id"
                )
            ),

        "approval_binding_experiment":
            (
                binding.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "approval_binding_hypothesis":
            (
                binding.get(
                    "hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "approval_binding_plan":
            (
                binding.get(
                    "plan_id"
                )
                ==
                payload.get(
                    "plan_id"
                )
            ),

        "approval_binding_spec_id":
            (
                binding.get(
                    "execution_spec_id"
                )
                ==
                spec.get(
                    "execution_spec_id"
                )
            ),

        "approval_binding_spec_sha":
            (
                binding.get(
                    "execution_spec_sha256"
                )
                ==
                spec_sha256
            ),

        "approval_binding_plan_sha":
            (
                binding.get(
                    "source_plan_sha256"
                )
                ==
                plan_sha256
            ),

        "approval_binding_checkpoint":
            (
                binding.get(
                    "checkpoint_sha256"
                )
                ==
                expected_checkpoint_sha
            ),

        "positive_control_present":
            bool(
                positive_control
            ),

        "positive_control_pass":
            (
                positive_control.get(
                    "pass"
                )
                is True
            ),

        "primary_tests_present":
            (
                isinstance(
                    preregistered.get(
                        "primary_tests"
                    ),
                    dict,
                )
                and
                {
                    "L1H0",
                    "L3H3",
                    "L4H0",
                }
                <=
                set(
                    preregistered[
                        "primary_tests"
                    ].keys()
                )
            ),

        "weak_control_present":
            (
                isinstance(
                    preregistered.get(
                        "weak_control"
                    ),
                    dict,
                )
            ),

        "token_bias_diagnostics_present":
            bool(
                token_bias
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
            "L10F frozen-result verification "
            f"failed: {failed}"
        )

    return {
        "checks":
            checks,

        "summary_sha256":
            summary_sha256,

        "approval_sha256":
            approval_sha256,

        "execution_spec_sha256":
            spec_sha256,

        "plan_sha256":
            plan_sha256,

        "checkpoint_sha256":
            actual_checkpoint_sha,
    }


# ============================================================
# Finding construction
# ============================================================

def build_findings(
    *,
    summary,
    summary_path,
    experiment_record,
    created_at,
):

    source_artifact = (
        relative_project_path(
            summary_path
        )
    )

    model_scope = copy.deepcopy(
        experiment_record[
            "model_scope"
        ]
    )

    tests = summary[
        "preregistered_tests"
    ]

    diagnostics = summary[
        "hypothesis_diagnostics"
    ]

    token_bias = summary[
        "token_bias_diagnostics"
    ]

    outcome = tests[
        "primary_outcome_label"
    ]

    require(
        outcome
        ==
        (
            "further_weakens_general_"
            "arithmetic_competing_head_circuit"
        ),
        (
            "L10F expected the completed "
            "EXP-0006 preregistered outcome to "
            "further weaken HYP-V03-001, but "
            f"received {outcome!r}."
        ),
    )

    primary_tests = tests[
        "primary_tests"
    ]

    main_finding = {
        "finding_id":
            FINDING_MAIN,

        "title":
            (
                "EXP-0006 cross-pair arithmetic "
                "correctness hypothesis failure"
            ),

        "statement":
            (
                "With the P45 four-versus-five "
                "positive control passing, none "
                "of the three preregistered "
                "primary heads met both the "
                "global and cross-pair criteria "
                "for the proposed general "
                "arithmetic competing-head "
                "circuit. EXP-0006 therefore "
                "further weakens HYP-V03-001."
            ),

        "finding_type":
            (
                "preregistered_causal_"
                "disconfirmation"
            ),

        "model_scope":
            copy.deepcopy(
                model_scope
            ),

        "evidence_level":
            "E3",

        "source_experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "source_artifacts": [
            source_artifact,
        ],

        "metrics": {
            "positive_control_pass":
                bool(
                    summary[
                        "positive_control"
                    ][
                        "pass"
                    ]
                ),

            "primary_pass_count":
                int(
                    tests[
                        "primary_pass_count"
                    ]
                ),

            "L1H0_global_consistent_count":
                int(
                    primary_tests[
                        "L1H0"
                    ][
                        "observed_global_consistent_count"
                    ]
                ),

            "L1H0_pair_sign_pass_count":
                int(
                    primary_tests[
                        "L1H0"
                    ][
                        "pair_sign_pass_count"
                    ]
                ),

            "L3H3_global_consistent_count":
                int(
                    primary_tests[
                        "L3H3"
                    ][
                        "observed_global_consistent_count"
                    ]
                ),

            "L3H3_pair_sign_pass_count":
                int(
                    primary_tests[
                        "L3H3"
                    ][
                        "pair_sign_pass_count"
                    ]
                ),

            "L4H0_global_consistent_count":
                int(
                    primary_tests[
                        "L4H0"
                    ][
                        "observed_global_consistent_count"
                    ]
                ),

            "L4H0_pair_sign_pass_count":
                int(
                    primary_tests[
                        "L4H0"
                    ][
                        "pair_sign_pass_count"
                    ]
                ),

            "outcome_label":
                outcome,
        },

        "supports_hypotheses":
            [],

        "weakens_hypotheses": [
            EXPECTED_HYPOTHESIS_ID,
        ],

        "tags": [
            "EXP-0006",
            "arithmetic-correctness",
            "causal-disconfirmation",
            "cross-pair",
            "mature-v0.3",
        ],

        "created_at_utc":
            created_at,
    }

    broader_heads = list(
        token_bias.get(
            "broader_token_bias_signal_heads",
            [],
        )
    )

    token_bias_finding = {
        "finding_id":
            FINDING_TOKEN_BIAS,

        "title":
            (
                "Broader fixed-pair token-bias "
                "signals in L1H0 and L4H0"
            ),

        "statement":
            (
                "EXP-0006 found broader fixed "
                "candidate-pair causal direction "
                "signals for L1H0 and L4H0 "
                "across multiple non-control "
                "number pairs, indicating that "
                "their effects are better "
                "described by candidate-token or "
                "pair identity than by a stable "
                "arithmetic-correctness role."
            ),

        "finding_type":
            (
                "derived_causal_pattern"
            ),

        "model_scope":
            copy.deepcopy(
                model_scope
            ),

        "evidence_level":
            "E3",

        "source_experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "source_artifacts": [
            source_artifact,
        ],

        "metrics": {
            "broader_token_bias_signal_heads":
                broader_heads,

            "L1H0_non_control_fixed_pair_count":
                int(
                    token_bias[
                        "by_head"
                    ][
                        "L1H0"
                    ][
                        "non_control_fixed_pair_count"
                    ]
                ),

            "L4H0_non_control_fixed_pair_count":
                int(
                    token_bias[
                        "by_head"
                    ][
                        "L4H0"
                    ][
                        "non_control_fixed_pair_count"
                    ]
                ),

            "minimum_non_control_pairs_for_signal":
                2,
        },

        "supports_hypotheses":
            [],

        "weakens_hypotheses": [
            EXPECTED_HYPOTHESIS_ID,
        ],

        "tags": [
            "EXP-0006",
            "token-bias",
            "candidate-pair",
            "L1H0",
            "L4H0",
            "mature-v0.3",
        ],

        "created_at_utc":
            created_at,
    }

    specific_heads = list(
        token_bias.get(
            "four_five_specific_signal_heads",
            [],
        )
    )

    l3h3_finding = {
        "finding_id":
            FINDING_L3H3,

        "title":
            (
                "L3H3 four-versus-five-specific "
                "fixed-pair signal"
            ),

        "statement":
            (
                "EXP-0006 identified L3H3 as a "
                "four-versus-five-specific "
                "fixed-pair causal signal rather "
                "than a broadly generalizing "
                "arithmetic-correctness head."
            ),

        "finding_type":
            (
                "derived_causal_pattern"
            ),

        "model_scope":
            copy.deepcopy(
                model_scope
            ),

        "evidence_level":
            "E3",

        "source_experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "source_artifacts": [
            source_artifact,
        ],

        "metrics": {
            "four_five_specific_signal_heads":
                specific_heads,

            "L3H3_non_control_fixed_pair_count":
                int(
                    token_bias[
                        "by_head"
                    ][
                        "L3H3"
                    ][
                        "non_control_fixed_pair_count"
                    ]
                ),

            "L3H3_P45_fixed_pair_direction":
                copy.deepcopy(
                    token_bias[
                        "by_head"
                    ][
                        "L3H3"
                    ][
                        "pair_directions"
                    ][
                        "P45"
                    ]
                ),
        },

        "supports_hypotheses":
            [],

        "weakens_hypotheses": [
            EXPECTED_HYPOTHESIS_ID,
        ],

        "tags": [
            "EXP-0006",
            "four-vs-five",
            "L3H3",
            "pair-specific",
            "mature-v0.3",
        ],

        "created_at_utc":
            created_at,
    }

    return [
        main_finding,
        token_bias_finding,
        l3h3_finding,
    ]


# ============================================================
# Evidence record
# ============================================================

def build_evidence_record(
    *,
    summary,
    summary_path,
    summary_sha256,
    created_at,
):

    return {
        "evidence_record_id":
            EVIDENCE_RECORD_ID,

        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "evidence_level":
            "E3",

        "artifact_path":
            relative_project_path(
                summary_path
            ),

        "artifact_sha256":
            summary_sha256,

        "checkpoint_sha256":
            summary[
                "checkpoint_sha256"
            ],

        "source_created_at_utc":
            summary.get(
                "created_at_utc"
            ),

        "ingested_at_utc":
            created_at,

        "extraction_adapter":
            EXTRACTION_ADAPTER,

        "facts": {
            "hypothesis_id":
                summary.get(
                    "hypothesis_id"
                ),

            "plan_id":
                summary.get(
                    "plan_id"
                ),

            "execution_spec_id":
                summary.get(
                    "execution_spec_id"
                ),

            "execution_spec_sha256":
                summary.get(
                    "execution_spec_sha256"
                ),

            "approval_id":
                summary.get(
                    "approval_id"
                ),

            "probe_count":
                summary.get(
                    "probe_count"
                ),

            "head_count":
                summary.get(
                    "head_count"
                ),

            "candidate_pairs":
                summary.get(
                    "candidate_pairs"
                ),

            "heads":
                summary.get(
                    "heads"
                ),

            "head_summaries":
                summary.get(
                    "head_summaries"
                ),

            "pair_summaries":
                summary.get(
                    "pair_summaries"
                ),

            "preregistered_tests":
                summary.get(
                    "preregistered_tests"
                ),

            "positive_control":
                summary.get(
                    "positive_control"
                ),

            "token_bias_diagnostics":
                summary.get(
                    "token_bias_diagnostics"
                ),

            "hypothesis_diagnostics":
                summary.get(
                    "hypothesis_diagnostics"
                ),

            "interpretation_rule":
                summary.get(
                    "interpretation_rule"
                ),

            "integrity_checks":
                summary.get(
                    "checks"
                ),
        },
    }


# ============================================================
# Prepare L7 changes
# ============================================================

def prepare_integrated_collections(
    *,
    store,
    summary,
    summary_path,
    verification,
    overwrite,
):

    data = store.load_all()

    hypotheses = copy.deepcopy(
        data[
            "hypotheses"
        ]
    )

    findings = copy.deepcopy(
        data[
            "findings"
        ]
    )

    experiments = copy.deepcopy(
        data[
            "experiments"
        ]
    )

    research_state = copy.deepcopy(
        data[
            "research_state"
        ]
    )

    evidence_ledger = copy.deepcopy(
        data[
            "evidence_ledger"
        ]
    )

    hypothesis_index = index_by(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    experiment_index = index_by(
        experiments[
            "records"
        ],
        "experiment_id",
    )

    finding_index = index_by(
        findings[
            "records"
        ],
        "finding_id",
    )

    evidence_index = index_by(
        evidence_ledger[
            "records"
        ],
        "evidence_record_id",
    )

    require(
        EXPECTED_HYPOTHESIS_ID
        in
        hypothesis_index,
        (
            "L10F cannot find hypothesis "
            f"{EXPECTED_HYPOTHESIS_ID}."
        ),
    )

    require(
        EXPECTED_EXPERIMENT_ID
        in
        experiment_index,
        (
            "L10F cannot find experiment "
            f"{EXPECTED_EXPERIMENT_ID}."
        ),
    )

    hypothesis = hypothesis_index[
        EXPECTED_HYPOTHESIS_ID
    ]

    experiment = experiment_index[
        EXPECTED_EXPERIMENT_ID
    ]

    require(
        hypothesis.get(
            "status"
        )
        ==
        "weakened",
        (
            "L10F expected HYP-V03-001 to "
            "remain weakened before L8 "
            "re-evaluation."
        ),
    )

    created_at = utc_now()

    new_findings = build_findings(
        summary=summary,
        summary_path=summary_path,
        experiment_record=experiment,
        created_at=created_at,
    )

    evidence_record = (
        build_evidence_record(
            summary=summary,
            summary_path=summary_path,
            summary_sha256=(
                verification[
                    "summary_sha256"
                ]
            ),
            created_at=created_at,
        )
    )

    existing_evidence = evidence_index.get(
        EVIDENCE_RECORD_ID
    )

    expected_finding_ids = [
        FINDING_MAIN,
        FINDING_TOKEN_BIAS,
        FINDING_L3H3,
    ]

    already_integrated = (
        experiment.get(
            "status"
        )
        ==
        "completed"
        and
        existing_evidence
        is not None
        and
        existing_evidence.get(
            "artifact_sha256"
        )
        ==
        verification[
            "summary_sha256"
        ]
        and
        all(
            finding_id
            in
            finding_index

            for finding_id
            in expected_finding_ids
        )
        and
        all(
            finding_id
            in
            experiment.get(
                "finding_ids",
                []
            )

            for finding_id
            in expected_finding_ids
        )
    )

    if (
        already_integrated
        and
        not overwrite
    ):

        return {
            "already_integrated":
                True,

            "hypotheses":
                hypotheses,

            "findings":
                findings,

            "experiments":
                experiments,

            "research_state":
                research_state,

            "evidence_ledger":
                evidence_ledger,
        }

    conflicts = []

    for finding_id in expected_finding_ids:

        if (
            finding_id
            in
            finding_index
            and
            not overwrite
        ):

            conflicts.append(
                finding_id
            )

    if (
        existing_evidence
        is not None
        and
        not overwrite
    ):

        conflicts.append(
            EVIDENCE_RECORD_ID
        )

    if conflicts:

        raise RuntimeError(
            "L10F found existing integration "
            "records that require review before "
            "--overwrite: "
            f"{conflicts}"
        )

    # --------------------------------------------------------
    # EXP-0006: planned -> completed
    # --------------------------------------------------------

    experiment[
        "status"
    ] = "completed"

    experiment[
        "evidence_level"
    ] = "E3"

    experiment[
        "summary_artifact"
    ] = relative_project_path(
        summary_path
    )

    experiment[
        "finding_ids"
    ] = expected_finding_ids

    experiment[
        "artifact_ingestion"
    ] = {
        "status":
            "verified",

        "evidence_record_id":
            EVIDENCE_RECORD_ID,

        "artifact_sha256":
            verification[
                "summary_sha256"
            ],

        "adapter":
            EXTRACTION_ADAPTER,

        "ingested_at_utc":
            created_at,
    }

    # --------------------------------------------------------
    # Findings
    # --------------------------------------------------------

    for new_finding in new_findings:

        finding_id = new_finding[
            "finding_id"
        ]

        existing = finding_index.get(
            finding_id
        )

        if existing is None:

            findings[
                "records"
            ].append(
                new_finding
            )

        else:

            position = (
                findings[
                    "records"
                ].index(
                    existing
                )
            )

            findings[
                "records"
            ][
                position
            ] = new_finding

    # --------------------------------------------------------
    # Evidence ledger
    # --------------------------------------------------------

    if existing_evidence is None:

        evidence_ledger[
            "records"
        ].append(
            evidence_record
        )

    else:

        position = (
            evidence_ledger[
                "records"
            ].index(
                existing_evidence
            )
        )

        evidence_ledger[
            "records"
        ][
            position
        ] = evidence_record

    # --------------------------------------------------------
    # HYP-V03-001
    #
    # Add evidence relationships only.
    # Status remains weakened until L8B.
    # --------------------------------------------------------

    evidence_against = list(
        hypothesis.get(
            "evidence_against",
            [],
        )
    )

    for finding_id in expected_finding_ids:

        if finding_id not in evidence_against:

            evidence_against.append(
                finding_id
            )

    hypothesis[
        "evidence_against"
    ] = evidence_against

    hypothesis[
        "updated_at_utc"
    ] = created_at

    # --------------------------------------------------------
    # Research state
    # --------------------------------------------------------

    state = research_state[
        "state"
    ]

    state[
        "last_completed_experiment"
    ] = EXPECTED_EXPERIMENT_ID

    state[
        "next_recommended_experiment"
    ] = None

    ingested = set(
        state.get(
            "ingested_experiments",
            []
        )
    )

    ingested.add(
        EXPECTED_EXPERIMENT_ID
    )

    state[
        "ingested_experiments"
    ] = sorted(
        ingested
    )

    state[
        "artifact_ingestion"
    ] = {
        "status":
            "active",

        "last_ingested_at_utc":
            created_at,

        "verified_experiment_count":
            len(
                state[
                    "ingested_experiments"
                ]
            ),

        "evidence_record_count":
            len(
                evidence_ledger[
                    "records"
                ]
            ),
    }

    # --------------------------------------------------------
    # Collection timestamps
    # --------------------------------------------------------

    for collection in (
        hypotheses,
        findings,
        experiments,
        research_state,
        evidence_ledger,
    ):

        collection[
            "updated_at_utc"
        ] = created_at

    return {
        "already_integrated":
            False,

        "hypotheses":
            hypotheses,

        "findings":
            findings,

        "experiments":
            experiments,

        "research_state":
            research_state,

        "evidence_ledger":
            evidence_ledger,
    }


# ============================================================
# Integrated state validation
# ============================================================

def validate_integrated_state(
    *,
    store,
    summary_sha256,
):

    validation = store.validate()

    require(
        validation[
            "passed"
        ],
        (
            "L10F integrated knowledge store "
            "failed validation."
        ),
    )

    data = store.load_all()

    hypotheses = index_by(
        data[
            "hypotheses"
        ][
            "records"
        ],
        "hypothesis_id",
    )

    findings = index_by(
        data[
            "findings"
        ][
            "records"
        ],
        "finding_id",
    )

    experiments = index_by(
        data[
            "experiments"
        ][
            "records"
        ],
        "experiment_id",
    )

    evidence = index_by(
        data[
            "evidence_ledger"
        ][
            "records"
        ],
        "evidence_record_id",
    )

    state = data[
        "research_state"
    ][
        "state"
    ]

    hypothesis = hypotheses[
        EXPECTED_HYPOTHESIS_ID
    ]

    experiment = experiments[
        EXPECTED_EXPERIMENT_ID
    ]

    evidence_record = evidence[
        EVIDENCE_RECORD_ID
    ]

    expected_finding_ids = [
        FINDING_MAIN,
        FINDING_TOKEN_BIAS,
        FINDING_L3H3,
    ]

    checks = {
        "knowledge_store_valid":
            bool(
                validation[
                    "passed"
                ]
            ),

        "exp_0006_completed":
            (
                experiment.get(
                    "status"
                )
                ==
                "completed"
            ),

        "exp_0006_evidence_level_e3":
            (
                experiment.get(
                    "evidence_level"
                )
                ==
                "E3"
            ),

        "all_findings_registered":
            all(
                finding_id
                in
                experiment.get(
                    "finding_ids",
                    []
                )
                and
                finding_id
                in
                findings

                for finding_id
                in expected_finding_ids
            ),

        "all_findings_weaken_hypothesis":
            all(
                EXPECTED_HYPOTHESIS_ID
                in
                findings[
                    finding_id
                ].get(
                    "weakens_hypotheses",
                    []
                )

                for finding_id
                in expected_finding_ids
            ),

        "all_hypothesis_links_present":
            all(
                finding_id
                in
                hypothesis.get(
                    "evidence_against",
                    []
                )

                for finding_id
                in expected_finding_ids
            ),

        "evidence_record_registered":
            (
                experiment.get(
                    "artifact_ingestion",
                    {},
                ).get(
                    "evidence_record_id"
                )
                ==
                EVIDENCE_RECORD_ID
            ),

        "evidence_sha_matches_result":
            (
                evidence_record.get(
                    "artifact_sha256"
                )
                ==
                summary_sha256
            ),

        "hypothesis_status_unchanged_weakened":
            (
                hypothesis.get(
                    "status"
                )
                ==
                "weakened"
            ),

        "last_completed_experiment_exp_0006":
            (
                state.get(
                    "last_completed_experiment"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "next_recommendation_cleared":
            (
                state.get(
                    "next_recommended_experiment"
                )
                is None
            ),

        "exp_0006_in_ingested_experiments":
            (
                EXPECTED_EXPERIMENT_ID
                in
                state.get(
                    "ingested_experiments",
                    []
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
            "L10F integrated-state validation "
            f"failed: {failed}"
        )

    return {
        "validation":
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
            "L10F regenerated scientist "
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

    saved_validation = validate_snapshot(
        saved
    )

    require(
        saved_validation[
            "passed"
        ],
        (
            "L10F saved scientist snapshot "
            "failed validation."
        ),
    )

    snapshot_sha256 = file_sha256(
        snapshot_path
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

        "snapshot_sha256":
            snapshot_sha256,

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

        "weakened_hypothesis_count":
            len(
                snapshot[
                    "hypotheses"
                ][
                    "weakened"
                ]
            ),

        "planned_experiment_count":
            len(
                snapshot[
                    "experiments"
                ][
                    "planned"
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

        "snapshot_sha256":
            snapshot_sha256,

        "validation":
            saved_validation,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    summary_path = project_path(
        args.summary
    )

    approval_path = project_path(
        args.approval
    )

    spec_path = project_path(
        args.execution_spec
    )

    plan_path = project_path(
        args.plan
    )

    snapshot_path = project_path(
        args.snapshot
    )

    snapshot_report_path = project_path(
        args.snapshot_report
    )

    intake_report_path = project_path(
        args.report
    )

    print()
    print(
        "=" * 84
    )

    print(
        "STAGE L10F — EXP-0006 RESULT "
        "INTAKE + L7 REINTEGRATION"
    )

    print(
        "=" * 84
    )

    summary = load_json(
        summary_path
    )

    approval = load_json(
        approval_path
    )

    spec_artifact = load_json(
        spec_path
    )

    plan_artifact = load_json(
        plan_path
    )

    print()
    print(
        "Verifying frozen experiment "
        "identity and completed E3 result..."
    )

    verification = validate_frozen_inputs(
        summary=summary,
        summary_path=summary_path,
        approval=approval,
        approval_path=approval_path,
        spec_artifact=spec_artifact,
        spec_path=spec_path,
        plan_artifact=plan_artifact,
        plan_path=plan_path,
    )

    print()
    print(
        "Frozen-result checks"
    )

    print(
        "--------------------"
    )

    for name, passed in (
        verification[
            "checks"
        ].items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    store = KnowledgeStore()

    prepared = (
        prepare_integrated_collections(
            store=store,
            summary=summary,
            summary_path=summary_path,
            verification=verification,
            overwrite=args.overwrite,
        )
    )

    print()
    print(
        "Curated scientific findings"
    )

    print(
        "---------------------------"
    )

    print(
        f"{FINDING_MAIN}: "
        "preregistered result weakens "
        "HYP-V03-001"
    )

    print(
        f"{FINDING_TOKEN_BIAS}: "
        "broader token-bias signal "
        "(L1H0, L4H0)"
    )

    print(
        f"{FINDING_L3H3}: "
        "L3H3 four/five-specific signal"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  L10F does NOT change the "
        "hypothesis status."
    )

    print(
        "  HYP-V03-001 remains weakened "
        "until L8B re-evaluates it."
    )

    if args.dry_run:

        print()
        print(
            "Proposed L7 changes"
        )

        print(
            "-------------------"
        )

        print(
            "  EXP-0006: planned -> completed"
        )

        print(
            "  FND-0008/FND-0009/FND-0010: "
            "add verified E3 findings"
        )

        print(
            "  EVR-EXP-0006: add verified "
            "E3 evidence record"
        )

        print(
            "  HYP-V03-001: append evidence "
            "relationships only"
        )

        print(
            "  research_state: last completed "
            "-> EXP-0006"
        )

        print(
            "  research_state: next "
            "recommendation -> null"
        )

        print(
            "  scientist_context.json: "
            "regenerate"
        )

        print()
        print(
            "Knowledge files modified: False"
        )

        print()
        print(
            "=" * 84
        )

        print(
            "SUCCESS: L10F DRY RUN PASSED."
        )

        print(
            "=" * 84
        )

        return

    paths = [
        store.collection_path(
            "hypotheses"
        ),
        store.collection_path(
            "findings"
        ),
        store.collection_path(
            "experiments"
        ),
        store.collection_path(
            "research_state"
        ),
        store.collection_path(
            "evidence_ledger"
        ),
        snapshot_path,
        snapshot_report_path,
        intake_report_path,
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
            "findings",
            prepared[
                "findings"
            ],
        )

        store.save(
            "experiments",
            prepared[
                "experiments"
            ],
        )

        store.save(
            "research_state",
            prepared[
                "research_state"
            ],
        )

        store.save(
            "evidence_ledger",
            prepared[
                "evidence_ledger"
            ],
        )

        integrated = (
            validate_integrated_state(
                store=store,
                summary_sha256=(
                    verification[
                        "summary_sha256"
                    ]
                ),
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

        report = {
            "stage":
                "L10F",

            "intake_version":
                INTAKE_VERSION,

            "created_at_utc":
                utc_now(),

            "experiment_id":
                EXPECTED_EXPERIMENT_ID,

            "finding_ids": [
                FINDING_MAIN,
                FINDING_TOKEN_BIAS,
                FINDING_L3H3,
            ],

            "evidence_record_id":
                EVIDENCE_RECORD_ID,

            "source_result": {
                "path":
                    relative_project_path(
                        summary_path
                    ),

                "sha256":
                    verification[
                        "summary_sha256"
                    ],

                "checkpoint_sha256":
                    verification[
                        "checkpoint_sha256"
                    ],
            },

            "frozen_identity": {
                "plan_sha256":
                    verification[
                        "plan_sha256"
                    ],

                "execution_spec_sha256":
                    verification[
                        "execution_spec_sha256"
                    ],

                "approval_sha256":
                    verification[
                        "approval_sha256"
                    ],
            },

            "knowledge_integration": {
                "experiment_status":
                    "completed",

                "evidence_level":
                    "E3",

                "hypothesis_id":
                    EXPECTED_HYPOTHESIS_ID,

                "hypothesis_status_mutated":
                    False,

                "last_completed_experiment":
                    EXPECTED_EXPERIMENT_ID,

                "next_recommended_experiment":
                    None,
            },

            "integrated_state_checks":
                integrated[
                    "checks"
                ],

            "scientist_context": {
                "path":
                    relative_project_path(
                        snapshot_path
                    ),

                "sha256":
                    snapshot_result[
                        "snapshot_sha256"
                    ],

                "knowledge_state_sha256":
                    snapshot_result[
                        "snapshot"
                    ][
                        "knowledge_state_sha256"
                    ],

                "validation":
                    snapshot_result[
                        "validation"
                    ],
            },

            "passed":
                True,
        }

        atomic_write_json(
            intake_report_path,
            report,
        )

    except Exception:

        restore_backups(
            backups
        )

        raise

    counts = (
        snapshot_result[
            "snapshot"
        ][
            "knowledge_counts"
        ]
    )

    print()
    print(
        "=" * 84
    )

    print(
        "L10F RESULT INTAKE SUMMARY"
    )

    print(
        "=" * 84
    )

    print()
    print(
        "Experiment:"
    )

    print(
        "  EXP-0006 -> completed"
    )

    print()
    print(
        "New findings:"
    )

    print(
        "  FND-0008 — preregistered "
        "arithmetic-circuit weakening"
    )

    print(
        "  FND-0009 — broader token-bias "
        "signal"
    )

    print(
        "  FND-0010 — L3H3 four/five-"
        "specific signal"
    )

    print()
    print(
        "New evidence record:"
    )

    print(
        "  EVR-EXP-0006 [E3]"
    )

    print()
    print(
        "Hypothesis:"
    )

    print(
        "  HYP-V03-001 remains WEAKENED"
    )

    print(
        "  status mutation deferred to L8B"
    )

    print()
    print(
        "Research state:"
    )

    print(
        "  last completed: EXP-0006"
    )

    print(
        "  next recommended: None"
    )

    print()
    print(
        "Updated knowledge counts:"
    )

    print(
        f"  hypotheses: "
        f"{counts['hypotheses']}"
    )

    print(
        f"  findings: "
        f"{counts['findings']}"
    )

    print(
        f"  experiments: "
        f"{counts['experiments']}"
    )

    print(
        f"  evidence records: "
        f"{counts['evidence_records']}"
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
        "Scientist context:"
    )

    print(
        f"  {snapshot_path}"
    )

    print()
    print(
        "Intake report:"
    )

    print(
        f"  {intake_report_path}"
    )

    print()
    print(
        "=" * 84
    )

    print(
        "SUCCESS: STAGE L10F EXP-0006 "
        "RESULT INTAKE + L7 "
        "REINTEGRATION COMPLETE."
    )

    print(
        "=" * 84
    )


if __name__ == "__main__":
    main()
