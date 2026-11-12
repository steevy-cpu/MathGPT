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
    SCHEMA_VERSION,
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

INTAKE_VERSION = "L9E.1"

EXPECTED_EXPERIMENT_ID = "EXP-0005"
EXPECTED_HYPOTHESIS_ID = "HYP-V03-002"
EXPECTED_EVIDENCE_LEVEL = "E3"

EXPECTED_PLAN_VERSION = "L9A.1"
EXPECTED_SPEC_VERSION = "L9B.1"
EXPECTED_APPROVAL_VERSION = "P3A.1"

FINDING_ID = "FND-0007"
EVIDENCE_RECORD_ID = "EVR-EXP-0005"
EXTRACTION_ADAPTER = "exp_0005_v1"


# ============================================================
# Default paths
# ============================================================

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0005_fixed_four_five_summary.json"
)

DEFAULT_APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0005.json"
)

DEFAULT_SPEC_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec.json"
)

DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan.json"
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
    / "exp_0005_result_intake_report.json"
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


def deep_index(
    records,
    field,
):
    return {
        record[field]:
            record

        for record
        in records
    }


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L9E — verify the completed "
            "EXP-0005 E3 result and reintegrate "
            "it into the Stage-L7 scientific "
            "knowledge layer."
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
            "Validate the complete EXP-0005 "
            "result and show the proposed L7 "
            "changes without writing them."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permit replacement of an existing "
            "EXP-0005 integration only when the "
            "same frozen experiment identities "
            "still verify."
        ),
    )

    return parser.parse_args()


# ============================================================
# Frozen experiment verification
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

    spec = (
        spec_artifact.get(
            "execution_spec",
            {},
        )
    )

    payload = (
        spec.get(
            "experiment_payload",
            {},
        )
    )

    source_plan = (
        spec.get(
            "source_plan",
            {},
        )
    )

    plan_package = (
        plan_artifact.get(
            "experiment_plan",
            {},
        )
    )

    plan = (
        plan_package.get(
            "plan",
            {},
        )
    )

    summary_checks = (
        summary.get(
            "checks",
            {},
        )
    )

    preregistered = (
        summary.get(
            "preregistered_tests",
            {},
        )
    )

    binding = (
        approval.get(
            "binding",
            {},
        )
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
            "EXP-0005 checkpoint does not "
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

        "summary_probe_count_12":
            (
                summary.get(
                    "probe_count"
                )
                ==
                12
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

        "preregistered_tests_present":
            bool(
                preregistered
            ),

        "L1H0_test_present":
            (
                "L1H0_primary_four_direction"
                in
                preregistered
            ),

        "L4H0_test_present":
            (
                "L4H0_primary_five_direction"
                in
                preregistered
            ),

        "weak_control_test_present":
            (
                "L4H1_weak_control"
                in
                preregistered
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
            "L9E frozen-result verification "
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

        "checkpoint_path":
            checkpoint_path,
    }


# ============================================================
# Result interpretation
#
# This does NOT change hypothesis status.
#
# It only converts the preregistered EXP-0005
# outcome into a curated finding relationship.
# ============================================================

def classify_result(
    summary,
):

    tests = (
        summary[
            "preregistered_tests"
        ]
    )

    outcome = (
        tests[
            "primary_outcome_label"
        ]
    )

    if (
        outcome
        ==
        (
            "supports_fixed_four_five_"
            "direction_hypothesis"
        )
    ):

        return {
            "relationship":
                "supports",

            "supports_hypotheses": [
                EXPECTED_HYPOTHESIS_ID,
            ],

            "weakens_hypotheses": [],

            "title":
                (
                    "EXP-0005 preregistered "
                    "four-versus-five causal "
                    "replication"
                ),

            "statement":
                (
                    "Across 12 preregistered "
                    "fixed four-versus-five "
                    "contexts, intact L1H0 "
                    "causally favored four in "
                    "all 12 probes while intact "
                    "L4H0 causally favored five "
                    "in all 12 probes, including "
                    "both four-correct and "
                    "five-correct contexts. "
                    "L4H1 also passed the "
                    "preregistered weak-control "
                    "criterion."
                ),
        }

    if (
        outcome
        ==
        (
            "partial_support_fixed_"
            "direction_hypothesis"
        )
    ):

        return {
            "relationship":
                "neutral_partial",

            "supports_hypotheses": [],

            "weakens_hypotheses": [],

            "title":
                (
                    "EXP-0005 partial "
                    "four-versus-five causal "
                    "replication"
                ),

            "statement":
                (
                    "EXP-0005 produced partial "
                    "support for the fixed "
                    "four-versus-five causal "
                    "direction hypothesis: only "
                    "one preregistered primary "
                    "head criterion passed."
                ),
        }

    if (
        outcome
        ==
        (
            "does_not_support_fixed_"
            "direction_hypothesis"
        )
    ):

        return {
            "relationship":
                "weakens",

            "supports_hypotheses": [],

            "weakens_hypotheses": [
                EXPECTED_HYPOTHESIS_ID,
            ],

            "title":
                (
                    "EXP-0005 failed fixed "
                    "four-versus-five causal "
                    "replication"
                ),

            "statement":
                (
                    "EXP-0005 did not reproduce "
                    "the preregistered fixed "
                    "four-versus-five causal "
                    "directions for either "
                    "primary attention head."
                ),
        }

    raise RuntimeError(
        "Unknown EXP-0005 preregistered "
        f"outcome: {outcome!r}"
    )


# ============================================================
# Curated finding
# ============================================================

def build_finding(
    *,
    summary,
    summary_path,
    experiment_record,
    result_classification,
    created_at,
):

    head_summaries = (
        summary[
            "head_summaries"
        ]
    )

    tests = (
        summary[
            "preregistered_tests"
        ]
    )

    return {
        "finding_id":
            FINDING_ID,

        "title":
            result_classification[
                "title"
            ],

        "statement":
            result_classification[
                "statement"
            ],

        "finding_type":
            (
                "preregistered_causal_"
                "replication"
            ),

        "model_scope":
            copy.deepcopy(
                experiment_record[
                    "model_scope"
                ]
            ),

        "evidence_level":
            "E3",

        "source_experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "source_artifacts": [
            relative_project_path(
                summary_path
            ),
        ],

        "metrics": {
            "probe_count":
                int(
                    summary[
                        "probe_count"
                    ]
                ),

            "L1H0_four_support_count":
                int(
                    head_summaries[
                        "L1H0"
                    ][
                        "all"
                    ][
                        "four_support_count"
                    ]
                ),

            "L1H0_four_support_rate":
                float(
                    head_summaries[
                        "L1H0"
                    ][
                        "all"
                    ][
                        "four_support_rate"
                    ]
                ),

            "L4H0_five_support_count":
                int(
                    head_summaries[
                        "L4H0"
                    ][
                        "all"
                    ][
                        "five_support_count"
                    ]
                ),

            "L4H0_five_support_rate":
                float(
                    head_summaries[
                        "L4H0"
                    ][
                        "all"
                    ][
                        "five_support_rate"
                    ]
                ),

            "L1H0_mean_delta":
                float(
                    head_summaries[
                        "L1H0"
                    ][
                        "all"
                    ][
                        "mean_delta"
                    ]
                ),

            "L4H0_mean_delta":
                float(
                    head_summaries[
                        "L4H0"
                    ][
                        "all"
                    ][
                        "mean_delta"
                    ]
                ),

            "L4H1_mean_absolute_delta":
                float(
                    head_summaries[
                        "L4H1"
                    ][
                        "all"
                    ][
                        "mean_absolute_delta"
                    ]
                ),

            "L1H0_primary_test_pass":
                bool(
                    tests[
                        "L1H0_primary_four_direction"
                    ][
                        "pass"
                    ]
                ),

            "L4H0_primary_test_pass":
                bool(
                    tests[
                        "L4H0_primary_five_direction"
                    ][
                        "pass"
                    ]
                ),

            "L4H1_weak_control_pass":
                bool(
                    tests[
                        "L4H1_weak_control"
                    ][
                        "pass"
                    ]
                ),

            "primary_outcome_label":
                tests[
                    "primary_outcome_label"
                ],
        },

        "supports_hypotheses":
            list(
                result_classification[
                    "supports_hypotheses"
                ]
            ),

        "weakens_hypotheses":
            list(
                result_classification[
                    "weakens_hypotheses"
                ]
            ),

        "tags": [
            "four-vs-five",
            "causal-replication",
            "preregistered",
            "attention-heads",
            "mature-v0.3",
            "EXP-0005",
        ],

        "created_at_utc":
            created_at,
    }


# ============================================================
# Evidence ledger record
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

            "heads":
                summary.get(
                    "heads"
                ),

            "head_summaries":
                summary.get(
                    "head_summaries"
                ),

            "subset_summaries":
                summary.get(
                    "subset_summaries"
                ),

            "preregistered_tests":
                summary.get(
                    "preregistered_tests"
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
# Knowledge mutation
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

    hypothesis_index = deep_index(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    finding_index = deep_index(
        findings[
            "records"
        ],
        "finding_id",
    )

    experiment_index = deep_index(
        experiments[
            "records"
        ],
        "experiment_id",
    )

    evidence_index = deep_index(
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
            "L9E cannot find hypothesis "
            f"{EXPECTED_HYPOTHESIS_ID}."
        ),
    )

    require(
        EXPECTED_EXPERIMENT_ID
        in
        experiment_index,
        (
            "L9E cannot find experiment "
            f"{EXPECTED_EXPERIMENT_ID}."
        ),
    )

    hypothesis = (
        hypothesis_index[
            EXPECTED_HYPOTHESIS_ID
        ]
    )

    experiment = (
        experiment_index[
            EXPECTED_EXPERIMENT_ID
        ]
    )

    require(
        hypothesis.get(
            "status"
        )
        ==
        "active",
        (
            "L9E expected HYP-V03-002 to "
            "remain active before L8 "
            "re-evaluation."
        ),
    )

    existing_evidence = (
        evidence_index.get(
            EVIDENCE_RECORD_ID
        )
    )

    existing_finding = (
        finding_index.get(
            FINDING_ID
        )
    )

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
        existing_finding
        is not None
        and
        FINDING_ID
        in
        experiment.get(
            "finding_ids",
            []
        )
        and
        FINDING_ID
        in
        hypothesis.get(
            "evidence_for",
            []
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

            "result_classification":
                classify_result(
                    summary
                ),
        }

    if (
        existing_evidence
        is not None
        and
        not overwrite
    ):
        raise RuntimeError(
            "EVR-EXP-0005 already exists but "
            "does not match the current result. "
            "Review the change before using "
            "--overwrite."
        )

    if (
        existing_finding
        is not None
        and
        not overwrite
    ):
        raise RuntimeError(
            "FND-0007 already exists but the "
            "EXP-0005 integration is incomplete. "
            "Review before using --overwrite."
        )

    created_at = utc_now()

    result_classification = (
        classify_result(
            summary
        )
    )

    finding = build_finding(
        summary=summary,
        summary_path=summary_path,
        experiment_record=experiment,
        result_classification=(
            result_classification
        ),
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

    # --------------------------------------------------------
    # EXP-0005: planned -> completed
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
    ] = [
        FINDING_ID,
    ]

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
    # Finding
    # --------------------------------------------------------

    if existing_finding is None:
        findings[
            "records"
        ].append(
            finding
        )

    else:
        position = (
            findings[
                "records"
            ].index(
                existing_finding
            )
        )

        findings[
            "records"
        ][
            position
        ] = finding

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
    # HYP-V03-002:
    # add evidence relation ONLY.
    #
    # Status remains active. L8B decides whether
    # the new replicated evidence promotes it.
    # --------------------------------------------------------

    if (
        FINDING_ID
        not in
        hypothesis.get(
            "evidence_for",
            []
        )
        and
        FINDING_ID
        not in
        hypothesis.get(
            "evidence_against",
            []
        )
    ):

        if (
            result_classification[
                "relationship"
            ]
            ==
            "supports"
        ):

            hypothesis.setdefault(
                "evidence_for",
                [],
            ).append(
                FINDING_ID
            )

        elif (
            result_classification[
                "relationship"
            ]
            ==
            "weakens"
        ):

            hypothesis.setdefault(
                "evidence_against",
                [],
            ).append(
                FINDING_ID
            )

    hypothesis[
        "updated_at_utc"
    ] = created_at

    # --------------------------------------------------------
    # Research state
    # --------------------------------------------------------

    state = (
        research_state[
            "state"
        ]
    )

    state[
        "last_completed_experiment"
    ] = EXPECTED_EXPERIMENT_ID

    # EXP-0005 is no longer planned. We intentionally
    # clear the recommendation and let L8C choose again.
    state[
        "next_recommended_experiment"
    ] = None

    ingested_experiments = set(
        state.get(
            "ingested_experiments",
            []
        )
    )

    ingested_experiments.add(
        EXPECTED_EXPERIMENT_ID
    )

    state[
        "ingested_experiments"
    ] = sorted(
        ingested_experiments
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
    # Updated timestamps
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

        "result_classification":
            result_classification,
    }


# ============================================================
# Transaction backup / rollback
# ============================================================

def capture_backups(
    paths,
):

    backups = {}

    for path in paths:
        path = Path(path)

        backups[
            path
        ] = (
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
            "L9E regenerated scientist "
            "snapshot did not validate."
        ),
    )

    atomic_write_json(
        snapshot_path,
        snapshot,
    )

    saved_snapshot = load_json(
        snapshot_path
    )

    saved_validation = validate_snapshot(
        saved_snapshot
    )

    require(
        saved_validation[
            "passed"
        ],
        (
            "L9E saved scientist snapshot "
            "did not validate."
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

        "snapshot_sha256":
            snapshot_sha256,

        "snapshot_validation":
            saved_validation,
    }


# ============================================================
# Final integration validation
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
            "L9E integrated knowledge "
            "store failed validation."
        ),
    )

    data = store.load_all()

    hypotheses = deep_index(
        data[
            "hypotheses"
        ][
            "records"
        ],
        "hypothesis_id",
    )

    findings = deep_index(
        data[
            "findings"
        ][
            "records"
        ],
        "finding_id",
    )

    experiments = deep_index(
        data[
            "experiments"
        ][
            "records"
        ],
        "experiment_id",
    )

    evidence = deep_index(
        data[
            "evidence_ledger"
        ][
            "records"
        ],
        "evidence_record_id",
    )

    state = (
        data[
            "research_state"
        ][
            "state"
        ]
    )

    hypothesis = (
        hypotheses[
            EXPECTED_HYPOTHESIS_ID
        ]
    )

    finding = findings[
        FINDING_ID
    ]

    experiment = (
        experiments[
            EXPECTED_EXPERIMENT_ID
        ]
    )

    evidence_record = (
        evidence[
            EVIDENCE_RECORD_ID
        ]
    )

    relation_valid = (
        (
            FINDING_ID
            in
            hypothesis.get(
                "evidence_for",
                []
            )
            and
            EXPECTED_HYPOTHESIS_ID
            in
            finding.get(
                "supports_hypotheses",
                []
            )
        )
        or
        (
            FINDING_ID
            in
            hypothesis.get(
                "evidence_against",
                []
            )
            and
            EXPECTED_HYPOTHESIS_ID
            in
            finding.get(
                "weakens_hypotheses",
                []
            )
        )
        or
        (
            EXPECTED_HYPOTHESIS_ID
            not in
            finding.get(
                "supports_hypotheses",
                []
            )
            and
            EXPECTED_HYPOTHESIS_ID
            not in
            finding.get(
                "weakens_hypotheses",
                []
            )
        )
    )

    checks = {
        "knowledge_store_valid":
            bool(
                validation[
                    "passed"
                ]
            ),

        "exp_0005_completed":
            (
                experiment.get(
                    "status"
                )
                ==
                "completed"
            ),

        "exp_0005_evidence_level_e3":
            (
                experiment.get(
                    "evidence_level"
                )
                ==
                "E3"
            ),

        "finding_registered":
            (
                FINDING_ID
                in
                experiment.get(
                    "finding_ids",
                    []
                )
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

        "finding_relation_reciprocal":
            relation_valid,

        "hypothesis_status_unchanged_active":
            (
                hypothesis.get(
                    "status"
                )
                ==
                "active"
            ),

        "last_completed_experiment_exp_0005":
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

        "exp_0005_in_ingested_experiments":
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
            "L9E integrated-state validation "
            f"failed: {failed}"
        )

    return {
        "store_validation":
            validation,

        "checks":
            checks,
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
        "=" * 80
    )

    print(
        "STAGE L9E — EXP-0005 RESULT "
        "INTAKE + L7 REINTEGRATION"
    )

    print(
        "=" * 80
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

    verification = (
        validate_frozen_inputs(
            summary=summary,
            summary_path=summary_path,
            approval=approval,
            approval_path=approval_path,
            spec_artifact=spec_artifact,
            spec_path=spec_path,
            plan_artifact=plan_artifact,
            plan_path=plan_path,
        )
    )

    print()
    print(
        "Frozen-result checks"
    )

    print(
        "--------------------"
    )

    for (
        name,
        passed,
    ) in verification[
        "checks"
    ].items():

        print(
            f"{name:<44} "
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

    classification = (
        prepared[
            "result_classification"
        ]
    )

    print()
    print(
        "Curated scientific result"
    )

    print(
        "-------------------------"
    )

    print(
        f"Finding ID: "
        f"{FINDING_ID}"
    )

    print(
        f"Relationship: "
        f"{classification['relationship']}"
    )

    print(
        f"Hypothesis: "
        f"{EXPECTED_HYPOTHESIS_ID}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  L9E does NOT change the "
        "hypothesis status."
    )

    print(
        "  HYP-V03-002 remains active "
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
            "  EXP-0005: "
            "planned -> completed"
        )

        print(
            "  FND-0007: add curated "
            "EXP-0005 finding"
        )

        print(
            "  EVR-EXP-0005: add verified "
            "E3 evidence record"
        )

        print(
            "  HYP-V03-002: append evidence "
            "relationship only"
        )

        print(
            "  research_state: last completed "
            "-> EXP-0005"
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
            "=" * 80
        )

        print(
            "SUCCESS: L9E DRY RUN PASSED."
        )

        print(
            "No L7 files were modified."
        )

        print(
            "=" * 80
        )

        return

    # ========================================================
    # Transaction
    # ========================================================

    knowledge_paths = [
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
        knowledge_paths
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

        integrated_validation = (
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

        intake_report = {
            "stage":
                "L9E",

            "intake_version":
                INTAKE_VERSION,

            "created_at_utc":
                utc_now(),

            "experiment_id":
                EXPECTED_EXPERIMENT_ID,

            "finding_id":
                FINDING_ID,

            "evidence_record_id":
                EVIDENCE_RECORD_ID,

            "result_relationship":
                classification[
                    "relationship"
                ],

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
                integrated_validation[
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
                        "snapshot_validation"
                    ],
            },

            "passed":
                True,
        }

        atomic_write_json(
            intake_report_path,
            intake_report,
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
        "L9E RESULT INTAKE SUMMARY"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Experiment:"
    )

    print(
        "  EXP-0005 -> completed"
    )

    print()

    print(
        "New finding:"
    )

    print(
        f"  {FINDING_ID}"
    )

    print(
        f"  relationship: "
        f"{classification['relationship']}"
    )

    print()

    print(
        "New evidence record:"
    )

    print(
        f"  {EVIDENCE_RECORD_ID} [E3]"
    )

    print()

    print(
        "Hypothesis:"
    )

    print(
        "  HYP-V03-002 remains ACTIVE"
    )

    print(
        "  status mutation deferred to L8B"
    )

    print()

    print(
        "Research state:"
    )

    print(
        "  last completed: EXP-0005"
    )

    print(
        "  next recommended: None"
    )

    print()

    print(
        "Updated knowledge counts:"
    )

    counts = (
        snapshot_result[
            "snapshot"
        ][
            "knowledge_counts"
        ]
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
        "=" * 80
    )

    print(
        "SUCCESS: STAGE L9E RESULT "
        "INTAKE + L7 REINTEGRATION COMPLETE."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
