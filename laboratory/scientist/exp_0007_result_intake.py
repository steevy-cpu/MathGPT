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

INTAKE_VERSION = "L12F.1"

EXPECTED_EXPERIMENT_ID = "EXP-0007"
EXPECTED_HYPOTHESIS_ID = "HYP-V03-003"
EXPECTED_EVIDENCE_LEVEL = "E3"

EXPECTED_RESULT_VERSION = "EXP-0007.1"
EXPECTED_RUNNER_VERSION = "L12D.1"
EXPECTED_PLAN_ID = "PLAN-5825AD67CA0D2EC0"
EXPECTED_SPEC_ID = "SPEC-F220BBDF2ABECA3E"
EXPECTED_APPROVAL_ID = "APR-BD41561B679379A4"

EXPECTED_RESULT_SHA256 = (
    "69cfde879a548ecfc3f56991db111c7b5aef36824875faed13a67d602cd7304c"
)

EXPECTED_PLAN_SHA256 = (
    "3bb6eebe7da92546170e97decc9768388b483d08b06b07fb922ca3a3e30ef8cb"
)

EXPECTED_SPEC_SHA256 = (
    "062b41f1bb535cf0427d9bc4464a27455fd250d0b90ae47d988d29ac887303c9"
)

EXPECTED_APPROVAL_SHA256 = (
    "1e29bb8979bac3af1383e0396c3d13bc19d3df03fabccfa37a9c133b94e5b4a3"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "4cd9b8f1c68e4079449519fd601b573f8ea48137bc74172c7b37e108c9707cf3"
)

FINDING_ID = "FND-0011"
EVIDENCE_RECORD_ID = "EVR-EXP-0007"
EXTRACTION_ADAPTER = "exp_0007_v1"


# ============================================================
# Paths
# ============================================================

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0007_cross_pair_replication_summary.json"
)

DEFAULT_APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0007.json"
)

DEFAULT_SPEC_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec_exp_0007.json"
)

DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0007.json"
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
    / "exp_0007_result_intake_report.json"
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
    path = project_path(path)

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


def capture_backups(paths):
    result = {}

    for path in paths:

        path = Path(path)

        result[path] = (
            path.read_bytes()
            if path.exists()
            else None
        )

    return result


def restore_backups(backups):

    for path, content in backups.items():

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
            "Stage L12F — verify the completed "
            "EXP-0007 E3 artifact and reintegrate "
            "it into the Stage-L7 knowledge layer "
            "without changing hypothesis status."
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
            "Verify the entire frozen result and "
            "show the proposed L7 integration "
            "without writing knowledge files."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permit replacement only if an "
            "EXP-0007 intake already exists and "
            "the caller intentionally wants to "
            "rebuild it."
        ),
    )

    return parser.parse_args()


# ============================================================
# Frozen-result verification
# ============================================================

def validate_frozen_result(
    *,
    summary,
    summary_path,
    approval,
    approval_path,
    spec_document,
    spec_path,
    plan_document,
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

    spec = spec_document.get(
        "execution_spec",
        {},
    )

    plan = plan_document.get(
        "plan",
        {},
    )

    bindings = summary.get(
        "bindings",
        {},
    )

    analysis = summary.get(
        "analysis",
        {},
    )

    integrity = summary.get(
        "integrity_checks",
        {},
    )

    execution = summary.get(
        "execution",
        {},
    )

    model = summary.get(
        "model",
        {},
    )

    approval_bindings = approval.get(
        "bindings",
        {},
    )

    checks = {
        "result_sha256":
            (
                summary_sha256
                ==
                EXPECTED_RESULT_SHA256
            ),

        "result_version":
            (
                summary.get(
                    "result_version"
                )
                ==
                EXPECTED_RESULT_VERSION
            ),

        "runner_version":
            (
                summary.get(
                    "runner_version"
                )
                ==
                EXPECTED_RUNNER_VERSION
            ),

        "experiment_id":
            (
                summary.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "hypothesis_id":
            (
                summary.get(
                    "target_hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "execution_status":
            (
                summary.get(
                    "status"
                )
                ==
                "completed_execution"
            ),

        "evidence_level_e3":
            (
                summary.get(
                    "evidence_level"
                )
                ==
                EXPECTED_EVIDENCE_LEVEL
            ),

        "plan_file_sha":
            (
                plan_sha256
                ==
                EXPECTED_PLAN_SHA256
            ),

        "spec_file_sha":
            (
                spec_sha256
                ==
                EXPECTED_SPEC_SHA256
            ),

        "approval_file_sha":
            (
                approval_sha256
                ==
                EXPECTED_APPROVAL_SHA256
            ),

        "summary_plan_binding":
            (
                bindings.get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
                and
                bindings.get(
                    "plan_sha256"
                )
                ==
                EXPECTED_PLAN_SHA256
            ),

        "summary_spec_binding":
            (
                bindings.get(
                    "execution_spec_id"
                )
                ==
                EXPECTED_SPEC_ID
                and
                bindings.get(
                    "execution_spec_sha256"
                )
                ==
                EXPECTED_SPEC_SHA256
            ),

        "summary_approval_binding":
            (
                bindings.get(
                    "approval_id"
                )
                ==
                EXPECTED_APPROVAL_ID
                and
                bindings.get(
                    "approval_sha256"
                )
                ==
                EXPECTED_APPROVAL_SHA256
            ),

        "summary_checkpoint_binding":
            (
                bindings.get(
                    "checkpoint_sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
                and
                model.get(
                    "checkpoint_sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        "plan_identity":
            (
                plan_document.get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
                and
                plan.get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
            ),

        "spec_identity":
            (
                spec_document.get(
                    "execution_spec_id"
                )
                ==
                EXPECTED_SPEC_ID
                and
                spec.get(
                    "execution_spec_id"
                )
                ==
                EXPECTED_SPEC_ID
            ),

        "approval_identity":
            (
                approval.get(
                    "approval_id"
                )
                ==
                EXPECTED_APPROVAL_ID
            ),

        "approval_explicit_human":
            (
                approval.get(
                    "approval_type"
                )
                ==
                "explicit_human"
                and
                approval.get(
                    "permission"
                )
                ==
                "P3"
                and
                approval.get(
                    "execution_authorized"
                )
                is True
            ),

        "approval_spec_binding":
            (
                approval_bindings.get(
                    "execution_spec_id"
                )
                ==
                EXPECTED_SPEC_ID
                and
                approval_bindings.get(
                    "execution_spec_sha256"
                )
                ==
                EXPECTED_SPEC_SHA256
            ),

        "approval_plan_binding":
            (
                approval_bindings.get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
                and
                approval_bindings.get(
                    "plan_sha256"
                )
                ==
                EXPECTED_PLAN_SHA256
            ),

        "approval_checkpoint_binding":
            (
                approval_bindings.get(
                    "checkpoint_sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        "probe_count_20":
            (
                execution.get(
                    "probe_count"
                )
                ==
                20
                and
                len(
                    summary.get(
                        "probe_results",
                        [],
                    )
                )
                ==
                20
            ),

        "primary_probe_count_16":
            (
                execution.get(
                    "primary_probe_count"
                )
                ==
                16
            ),

        "effect_count_60":
            (
                execution.get(
                    "causal_effect_count"
                )
                ==
                60
                and
                len(
                    summary.get(
                        "effects",
                        [],
                    )
                )
                ==
                60
            ),

        "head_order":
            (
                execution.get(
                    "head_order"
                )
                ==
                [
                    "L1H0",
                    "L4H0",
                    "L4H1",
                ]
            ),

        "head_scale_zero":
            (
                float(
                    execution.get(
                        "head_scale"
                    )
                )
                ==
                0.0
            ),

        "no_persistent_weight_modification":
            (
                execution.get(
                    "persistent_weight_modification"
                )
                is False
            ),

        "integrity_present":
            bool(
                integrity
            ),

        "integrity_all_passed":
            (
                bool(
                    integrity
                )
                and
                all(
                    integrity.values()
                )
            ),

        "model_state_unchanged":
            (
                summary.get(
                    "model_state_sha256_before"
                )
                ==
                summary.get(
                    "model_state_sha256_after"
                )
            ),

        "knowledge_state_unchanged_by_runner":
            (
                summary.get(
                    "knowledge_hashes_before"
                )
                ==
                summary.get(
                    "knowledge_hashes_after"
                )
            ),

        "positive_control_pass":
            (
                analysis.get(
                    "positive_control",
                    {},
                ).get(
                    "overall_pass"
                )
                is True
            ),

        "L1H0_cross_pair_4_of_4":
            (
                analysis.get(
                    "cross_pair_generalization",
                    {},
                ).get(
                    "L1H0",
                    {},
                ).get(
                    "stable_pair_count"
                )
                ==
                4
                and
                analysis[
                    "cross_pair_generalization"
                ][
                    "L1H0"
                ].get(
                    "pass"
                )
                is True
            ),

        "L4H0_cross_pair_4_of_4":
            (
                analysis.get(
                    "cross_pair_generalization",
                    {},
                ).get(
                    "L4H0",
                    {},
                ).get(
                    "stable_pair_count"
                )
                ==
                4
                and
                analysis[
                    "cross_pair_generalization"
                ][
                    "L4H0"
                ].get(
                    "pass"
                )
                is True
            ),

        "L1H0_not_strong_correctness":
            (
                analysis.get(
                    "strong_correctness_tracking",
                    {},
                ).get(
                    "L1H0",
                    {},
                ).get(
                    "positive_count"
                )
                ==
                10
                and
                analysis[
                    "strong_correctness_tracking"
                ][
                    "L1H0"
                ].get(
                    "strong_correctness_tracking"
                )
                is False
            ),

        "L4H0_not_strong_correctness":
            (
                analysis.get(
                    "strong_correctness_tracking",
                    {},
                ).get(
                    "L4H0",
                    {},
                ).get(
                    "positive_count"
                )
                ==
                8
                and
                analysis[
                    "strong_correctness_tracking"
                ][
                    "L4H0"
                ].get(
                    "strong_correctness_tracking"
                )
                is False
            ),

        "weak_control_pass":
            (
                analysis.get(
                    "weak_control",
                    {},
                ).get(
                    "pass"
                )
                is True
            ),

        "L1H0_pair_dependent":
            (
                analysis.get(
                    "orientation_diagnostic",
                    {},
                ).get(
                    "L1H0",
                    {},
                ).get(
                    "interpretation"
                )
                ==
                "pair_dependent_signal"
            ),

        "L4H0_pair_dependent":
            (
                analysis.get(
                    "orientation_diagnostic",
                    {},
                ).get(
                    "L4H0",
                    {},
                ).get(
                    "interpretation"
                )
                ==
                "pair_dependent_signal"
            ),

        "preregistered_outcome":
            (
                analysis.get(
                    "preregistered_outcome"
                )
                ==
                (
                    "replicates_cross_pair_"
                    "candidate_selection_bias"
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
            "L12F frozen-result "
            f"verification failed: {failed}"
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
            EXPECTED_CHECKPOINT_SHA256,
    }


# ============================================================
# Scientific classification
#
# Result intake adds evidence relationships only.
# It does NOT change hypothesis status.
# ============================================================

def classify_result(
    summary,
):

    outcome = (
        summary[
            "analysis"
        ][
            "preregistered_outcome"
        ]
    )

    if (
        outcome
        ==
        (
            "replicates_cross_pair_"
            "candidate_selection_bias"
        )
    ):

        return {
            "relationship":
                "supports",

            "supports_hypotheses": [
                EXPECTED_HYPOTHESIS_ID,
            ],

            "weakens_hypotheses":
                [],

            "title":
                (
                    "EXP-0007 independent "
                    "cross-pair candidate-selection "
                    "replication"
                ),

            "statement":
                (
                    "In mature MathGPT v0.3, "
                    "EXP-0007 independently "
                    "replicated cross-pair causal "
                    "candidate-selection behavior "
                    "for L1H0 and L4H0 on four "
                    "previously untested numerical "
                    "candidate pairs. Both heads "
                    "passed within-pair directional "
                    "stability on 4/4 new pairs, "
                    "while strong arithmetic-"
                    "correctness tracking did not "
                    "pass (L1H0 10/16; L4H0 8/16). "
                    "The P45 positive control and "
                    "L4H1 weak-control criterion "
                    "both passed, and both primary "
                    "heads showed pair-dependent "
                    "rather than uniform lower/"
                    "upper orientation."
                ),
        }

    if outcome == "partial_replication":

        return {
            "relationship":
                "neutral_partial",

            "supports_hypotheses":
                [],

            "weakens_hypotheses":
                [],

            "title":
                (
                    "EXP-0007 partial cross-pair "
                    "candidate-selection replication"
                ),

            "statement":
                (
                    "EXP-0007 produced a partial "
                    "replication under its frozen "
                    "cross-pair causal criteria."
                ),
        }

    if (
        outcome
        ==
        "fails_cross_pair_replication"
    ):

        return {
            "relationship":
                "weakens",

            "supports_hypotheses":
                [],

            "weakens_hypotheses": [
                EXPECTED_HYPOTHESIS_ID,
            ],

            "title":
                (
                    "EXP-0007 failed cross-pair "
                    "candidate-selection replication"
                ),

            "statement":
                (
                    "EXP-0007 failed the frozen "
                    "cross-pair replication criteria "
                    "for HYP-V03-003 despite a "
                    "valid causal execution."
                ),
        }

    if (
        outcome
        ==
        (
            "inconclusive_positive_"
            "control_failure"
        )
    ):

        return {
            "relationship":
                "neutral_inconclusive",

            "supports_hypotheses":
                [],

            "weakens_hypotheses":
                [],

            "title":
                (
                    "EXP-0007 inconclusive due "
                    "to positive-control failure"
                ),

            "statement":
                (
                    "EXP-0007 was causally "
                    "executed but its positive "
                    "control failed, so the result "
                    "is not used to support or "
                    "weaken HYP-V03-003."
                ),
        }

    raise RuntimeError(
        "Unknown EXP-0007 outcome: "
        f"{outcome!r}"
    )


# ============================================================
# Curated finding
# ============================================================

def build_finding(
    *,
    summary,
    summary_path,
    experiment_record,
    classification,
    created_at,
):

    analysis = summary[
        "analysis"
    ]

    return {
        "finding_id":
            FINDING_ID,

        "title":
            classification[
                "title"
            ],

        "statement":
            classification[
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
                        "execution"
                    ][
                        "probe_count"
                    ]
                ),

            "primary_probe_count":
                int(
                    summary[
                        "execution"
                    ][
                        "primary_probe_count"
                    ]
                ),

            "causal_effect_count":
                int(
                    summary[
                        "execution"
                    ][
                        "causal_effect_count"
                    ]
                ),

            "positive_control_pass":
                bool(
                    analysis[
                        "positive_control"
                    ][
                        "overall_pass"
                    ]
                ),

            "L1H0_stable_pair_count":
                int(
                    analysis[
                        "cross_pair_generalization"
                    ][
                        "L1H0"
                    ][
                        "stable_pair_count"
                    ]
                ),

            "L4H0_stable_pair_count":
                int(
                    analysis[
                        "cross_pair_generalization"
                    ][
                        "L4H0"
                    ][
                        "stable_pair_count"
                    ]
                ),

            "L1H0_cross_pair_pass":
                bool(
                    analysis[
                        "cross_pair_generalization"
                    ][
                        "L1H0"
                    ][
                        "pass"
                    ]
                ),

            "L4H0_cross_pair_pass":
                bool(
                    analysis[
                        "cross_pair_generalization"
                    ][
                        "L4H0"
                    ][
                        "pass"
                    ]
                ),

            "L1H0_correctness_positive_count":
                int(
                    analysis[
                        "strong_correctness_tracking"
                    ][
                        "L1H0"
                    ][
                        "positive_count"
                    ]
                ),

            "L4H0_correctness_positive_count":
                int(
                    analysis[
                        "strong_correctness_tracking"
                    ][
                        "L4H0"
                    ][
                        "positive_count"
                    ]
                ),

            "L1H0_strong_correctness_pass":
                bool(
                    analysis[
                        "strong_correctness_tracking"
                    ][
                        "L1H0"
                    ][
                        "strong_correctness_tracking"
                    ]
                ),

            "L4H0_strong_correctness_pass":
                bool(
                    analysis[
                        "strong_correctness_tracking"
                    ][
                        "L4H0"
                    ][
                        "strong_correctness_tracking"
                    ]
                ),

            "weak_control_pass":
                bool(
                    analysis[
                        "weak_control"
                    ][
                        "pass"
                    ]
                ),

            "L1H0_orientation":
                analysis[
                    "orientation_diagnostic"
                ][
                    "L1H0"
                ][
                    "interpretation"
                ],

            "L4H0_orientation":
                analysis[
                    "orientation_diagnostic"
                ][
                    "L4H0"
                ][
                    "interpretation"
                ],

            "preregistered_outcome":
                analysis[
                    "preregistered_outcome"
                ],
        },

        "supports_hypotheses":
            list(
                classification[
                    "supports_hypotheses"
                ]
            ),

        "weakens_hypotheses":
            list(
                classification[
                    "weakens_hypotheses"
                ]
            ),

        "tags": [
            "cross-pair",
            "candidate-selection",
            "numerical-candidates",
            "causal-replication",
            "preregistered",
            "attention-heads",
            "mature-v0.3",
            "EXP-0007",
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
            EXPECTED_CHECKPOINT_SHA256,

        "source_created_at_utc":
            None,

        "ingested_at_utc":
            created_at,

        "extraction_adapter":
            EXTRACTION_ADAPTER,

        "facts": {
            "result_version":
                summary.get(
                    "result_version"
                ),

            "runner_version":
                summary.get(
                    "runner_version"
                ),

            "bindings":
                copy.deepcopy(
                    summary.get(
                        "bindings"
                    )
                ),

            "execution":
                copy.deepcopy(
                    summary.get(
                        "execution"
                    )
                ),

            "analysis":
                copy.deepcopy(
                    summary.get(
                        "analysis"
                    )
                ),

            "integrity_checks":
                copy.deepcopy(
                    summary.get(
                        "integrity_checks"
                    )
                ),

            "model_state_sha256_before":
                summary.get(
                    "model_state_sha256_before"
                ),

            "model_state_sha256_after":
                summary.get(
                    "model_state_sha256_after"
                ),
        },
    }


# ============================================================
# Knowledge preparation
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
            "L12F cannot find hypothesis "
            f"{EXPECTED_HYPOTHESIS_ID}."
        ),
    )

    require(
        EXPECTED_EXPERIMENT_ID
        in
        experiment_index,
        (
            "L12F cannot find experiment "
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
        "active",
        (
            "L12F expects HYP-V03-003 to "
            "remain active before the next "
            "L8 evaluation cycle."
        ),
    )

    require(
        experiment.get(
            "status"
        )
        ==
        "planned",
        (
            "L12F expects EXP-0007 to be "
            "planned before result intake."
        ),
    )

    existing_finding = finding_index.get(
        FINDING_ID
    )

    existing_evidence = evidence_index.get(
        EVIDENCE_RECORD_ID
    )

    already_integrated = (
        experiment.get(
            "status"
        )
        ==
        "completed"
        and
        existing_finding
        is not None
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

            "classification":
                classify_result(
                    summary
                ),
        }

    if (
        existing_finding
        is not None
        and
        not overwrite
    ):

        raise RuntimeError(
            f"{FINDING_ID} already exists. "
            "Review before using --overwrite."
        )

    if (
        existing_evidence
        is not None
        and
        not overwrite
    ):

        raise RuntimeError(
            f"{EVIDENCE_RECORD_ID} already "
            "exists. Review before using "
            "--overwrite."
        )

    created_at = utc_now()

    classification = classify_result(
        summary
    )

    finding = build_finding(
        summary=summary,
        summary_path=summary_path,
        experiment_record=experiment,
        classification=classification,
        created_at=created_at,
    )

    evidence_record = build_evidence_record(
        summary=summary,
        summary_path=summary_path,
        summary_sha256=(
            verification[
                "summary_sha256"
            ]
        ),
        created_at=created_at,
    )

    # --------------------------------------------------------
    # EXP-0007: planned -> completed
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

        position = findings[
            "records"
        ].index(
            existing_finding
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

        position = evidence_ledger[
            "records"
        ].index(
            existing_evidence
        )

        evidence_ledger[
            "records"
        ][
            position
        ] = evidence_record

    # --------------------------------------------------------
    # Hypothesis relationship ONLY.
    #
    # Do not mutate status here.
    # --------------------------------------------------------

    if (
        EXPECTED_HYPOTHESIS_ID
        in
        finding[
            "supports_hypotheses"
        ]
    ):

        evidence_for = list(
            hypothesis.get(
                "evidence_for",
                []
            )
        )

        if FINDING_ID not in evidence_for:
            evidence_for.append(
                FINDING_ID
            )

        hypothesis[
            "evidence_for"
        ] = evidence_for

    if (
        EXPECTED_HYPOTHESIS_ID
        in
        finding[
            "weakens_hypotheses"
        ]
    ):

        evidence_against = list(
            hypothesis.get(
                "evidence_against",
                []
            )
        )

        if FINDING_ID not in evidence_against:
            evidence_against.append(
                FINDING_ID
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

    ingested = list(
        state.get(
            "ingested_experiments",
            []
        )
    )

    if (
        EXPECTED_EXPERIMENT_ID
        not in
        ingested
    ):

        ingested.append(
            EXPECTED_EXPERIMENT_ID
        )

    state[
        "ingested_experiments"
    ] = ingested

    state[
        "experiment_planner"
    ] = "idle"

    state[
        "last_result_intake"
    ] = {
        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "finding_id":
            FINDING_ID,

        "evidence_record_id":
            EVIDENCE_RECORD_ID,

        "relationship":
            classification[
                "relationship"
            ],

        "updated_at_utc":
            created_at,
    }

    now = created_at

    hypotheses[
        "updated_at_utc"
    ] = now

    findings[
        "updated_at_utc"
    ] = now

    experiments[
        "updated_at_utc"
    ] = now

    research_state[
        "updated_at_utc"
    ] = now

    evidence_ledger[
        "updated_at_utc"
    ] = now

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

        "classification":
            classification,
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
            "L12F regenerated scientist "
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
            "L12F saved scientist snapshot "
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
            saved[
                "knowledge_state_sha256"
            ],

        "knowledge_counts":
            saved[
                "knowledge_counts"
            ],

        "next_recommended_experiment":
            saved[
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
            saved,

        "snapshot_sha256":
            snapshot_sha256,

        "validation":
            saved_validation,
    }


# ============================================================
# Integrated-state validation
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
            "L12F integrated knowledge "
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

    state = data[
        "research_state"
    ][
        "state"
    ]

    hypothesis = hypotheses[
        EXPECTED_HYPOTHESIS_ID
    ]

    finding = findings[
        FINDING_ID
    ]

    experiment = experiments[
        EXPECTED_EXPERIMENT_ID
    ]

    evidence_record = evidence[
        EVIDENCE_RECORD_ID
    ]

    checks = {
        "knowledge_store_valid":
            bool(
                validation[
                    "passed"
                ]
            ),

        "exp_0007_completed":
            (
                experiment.get(
                    "status"
                )
                ==
                "completed"
            ),

        "exp_0007_evidence_level_e3":
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

        "support_relation_reciprocal":
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
            ),

        "hypothesis_status_unchanged_active":
            (
                hypothesis.get(
                    "status"
                )
                ==
                "active"
            ),

        "last_completed_experiment_exp_0007":
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

        "exp_0007_in_ingested_experiments":
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
            "L12F integrated-state "
            f"validation failed: {failed}"
        )

    return {
        "validation":
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
    print("=" * 88)
    print(
        "STAGE L12F — EXP-0007 RESULT "
        "INTAKE + L7 REINTEGRATION"
    )
    print("=" * 88)

    summary = load_json(
        summary_path
    )

    approval = load_json(
        approval_path
    )

    spec_document = load_json(
        spec_path
    )

    plan_document = load_json(
        plan_path
    )

    print()
    print(
        "Verifying frozen EXP-0007 "
        "execution and E3 result..."
    )

    verification = validate_frozen_result(
        summary=summary,
        summary_path=summary_path,
        approval=approval,
        approval_path=approval_path,
        spec_document=spec_document,
        spec_path=spec_path,
        plan_document=plan_document,
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
            f"{name:<52} "
            f"{passed}"
        )

    store = KnowledgeStore()

    prepared = prepare_integrated_collections(
        store=store,
        summary=summary,
        summary_path=summary_path,
        verification=verification,
        overwrite=args.overwrite,
    )

    classification = prepared[
        "classification"
    ]

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
        "  L12F does NOT change "
        "hypothesis status."
    )
    print(
        "  HYP-V03-003 remains active "
        "until the next L8B evaluation."
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
            "  EXP-0007: planned -> completed"
        )
        print(
            "  FND-0011: add supporting "
            "preregistered E3 finding"
        )
        print(
            "  EVR-EXP-0007: add verified "
            "E3 evidence record"
        )
        print(
            "  HYP-V03-003: append FND-0011 "
            "to evidence_for only"
        )
        print(
            "  research_state: last completed "
            "-> EXP-0007"
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
        print("=" * 88)
        print(
            "SUCCESS: L12F DRY RUN PASSED."
        )
        print(
            "No L7 files were modified."
        )
        print("=" * 88)

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

        integrated = validate_integrated_state(
            store=store,
            summary_sha256=(
                verification[
                    "summary_sha256"
                ]
            ),
        )

        snapshot_result = regenerate_snapshot(
            store=store,
            snapshot_path=snapshot_path,
            snapshot_report_path=(
                snapshot_report_path
            ),
        )

        report = {
            "stage":
                "L12F",

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
                "plan_id":
                    EXPECTED_PLAN_ID,

                "plan_sha256":
                    verification[
                        "plan_sha256"
                    ],

                "execution_spec_id":
                    EXPECTED_SPEC_ID,

                "execution_spec_sha256":
                    verification[
                        "execution_spec_sha256"
                    ],

                "approval_id":
                    EXPECTED_APPROVAL_ID,

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

    # ========================================================
    # Display
    # ========================================================

    final_snapshot = snapshot_result[
        "snapshot"
    ]

    print()
    print("=" * 88)
    print(
        "L12F EXP-0007 RESULT INTAKE SUMMARY"
    )
    print("=" * 88)

    print()
    print(
        "Experiment:"
    )
    print(
        "  EXP-0007 -> completed"
    )

    print()
    print(
        "New finding:"
    )
    print(
        f"  {FINDING_ID}"
    )

    print(
        "  relationship: supports "
        "HYP-V03-003"
    )

    print()
    print(
        "New evidence record:"
    )
    print(
        f"  {EVIDENCE_RECORD_ID}"
    )

    print()
    print(
        "Hypothesis:"
    )
    print(
        "  HYP-V03-003 remains active"
    )
    print(
        "  status mutation: False"
    )

    print()
    print(
        "Knowledge counts:"
    )

    counts = final_snapshot[
        "knowledge_counts"
    ]

    for key, value in counts.items():

        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "New knowledge fingerprint:"
    )
    print(
        "  "
        +
        final_snapshot[
            "knowledge_state_sha256"
        ]
    )

    print()
    print(
        "Scientist context SHA-256:"
    )
    print(
        "  "
        +
        snapshot_result[
            "snapshot_sha256"
        ]
    )

    print()
    print(
        "Integrated-state checks"
    )
    print(
        "-----------------------"
    )

    for name, passed in (
        integrated[
            "checks"
        ].items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    print()
    print(
        "Intake report:"
    )
    print(
        intake_report_path
    )

    print()
    print("=" * 88)
    print(
        "SUCCESS: STAGE L12F EXP-0007 "
        "RESULT INTAKE COMPLETE."
    )
    print(
        "Evidence was integrated; hypothesis "
        "status was intentionally not changed."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
