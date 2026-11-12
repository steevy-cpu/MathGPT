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

ENGINE_VERSION = "L12G.1"

TARGET_HYPOTHESIS_ID = "HYP-V03-003"
CURRENT_STATUS = "active"
TARGET_STATUS = "supported"

EXPECTED_REASON_CODE = (
    "REPLICATED_CAUSAL_SUPPORT"
)

EXPECTED_DECISION = "promote"
EXPECTED_NEXT_ACTION = "monitor"

EXPECTED_KNOWLEDGE_STATE = (
    "49de0303b4ec7cd3d7d00348cdf680a51b360866a434728069cc75227f2c472f"
)

EXPECTED_SNAPSHOT_SHA256 = (
    "724198190a408071d39d562e4972ff3883f66471f462caa5be112e6e41bfd4cc"
)

EXPECTED_EVALUATION_SHA256 = (
    "eadb4f4eae04eb8e0413f150605ececeed530325a7968cfffb790136855ef38e"
)

EXPECTED_SUPPORT_SOURCES = [
    "EXP-0006",
    "EXP-0007",
]

EXPECTED_CONFIRMATION = (
    "COMMIT HYP-V03-003 SUPPORTED"
)


# ============================================================
# Paths
# ============================================================

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
    / "hypothesis_status_commit_hyp_v03_003.json"
)

DEFAULT_SNAPSHOT_REPORT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "knowledge_snapshot_report.json"
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


def canonical_json_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


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


def collection_hashes(
    store,
):
    result = {}

    for collection in (
        "hypotheses",
        "findings",
        "experiments",
        "research_state",
        "evidence_ledger",
    ):

        path = store.collection_path(
            collection
        )

        result[
            collection
        ] = file_sha256(
            path
        )

    return result


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L12G — controlled lifecycle "
            "commit for HYP-V03-003 after "
            "replicated E3 causal support."
        )
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
        ),
    )

    parser.add_argument(
        "--evaluation",
        default=str(
            DEFAULT_EVALUATION_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--snapshot-report",
        default=str(
            DEFAULT_SNAPSHOT_REPORT_PATH
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the validated active -> "
            "supported lifecycle transition."
        ),
    )

    parser.add_argument(
        "--confirmation",
        default=None,
        help=(
            "Required with --apply. Must equal "
            f"{EXPECTED_CONFIRMATION!r}."
        ),
    )

    return parser.parse_args()


# ============================================================
# Evaluation extraction
# ============================================================

def find_target_evaluation(
    evaluation_document,
):

    matches = [
        item
        for item
        in evaluation_document.get(
            "hypothesis_evaluations",
            [],
        )
        if item.get(
            "hypothesis_id"
        )
        ==
        TARGET_HYPOTHESIS_ID
    ]

    require(
        len(
            matches
        )
        ==
        1,
        (
            "Expected exactly one evaluation "
            f"for {TARGET_HYPOTHESIS_ID}."
        ),
    )

    return matches[
        0
    ]


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    snapshot,
    snapshot_path,
    evaluation_document,
    evaluation_path,
):

    snapshot_sha256 = file_sha256(
        snapshot_path
    )

    evaluation_sha256 = file_sha256(
        evaluation_path
    )

    target = find_target_evaluation(
        evaluation_document
    )

    evidence = target.get(
        "evidence_summary",
        {},
    )

    decision = target.get(
        "evaluation",
        {},
    )

    next_action = target.get(
        "next_action",
        {},
    )

    support_sources = sorted(
        evidence.get(
            "support_source_experiments",
            []
        )
    )

    checks = {
        "snapshot_sha256":
            (
                snapshot_sha256
                ==
                EXPECTED_SNAPSHOT_SHA256
            ),

        "evaluation_sha256":
            (
                evaluation_sha256
                ==
                EXPECTED_EVALUATION_SHA256
            ),

        "knowledge_state":
            (
                snapshot.get(
                    "knowledge_state_sha256"
                )
                ==
                EXPECTED_KNOWLEDGE_STATE
                ==
                evaluation_document.get(
                    "inputs",
                    {},
                ).get(
                    "knowledge_state_sha256"
                )
            ),

        "evaluation_checks_pass":
            (
                bool(
                    evaluation_document.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    evaluation_document[
                        "overall_checks"
                    ].values()
                )
            ),

        "target_hypothesis":
            (
                target.get(
                    "hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
            ),

        "current_status_active":
            (
                target.get(
                    "current_status"
                )
                ==
                CURRENT_STATUS
            ),

        "evidence_support_only":
            (
                evidence.get(
                    "classification"
                )
                ==
                "support_only"
            ),

        "support_sources_exact":
            (
                support_sources
                ==
                sorted(
                    EXPECTED_SUPPORT_SOURCES
                )
            ),

        "support_source_count_2":
            (
                len(
                    support_sources
                )
                ==
                2
            ),

        "weakening_source_count_0":
            (
                len(
                    evidence.get(
                        "weakening_source_experiments",
                        []
                    )
                )
                ==
                0
            ),

        "proposed_status_supported":
            (
                decision.get(
                    "proposed_status"
                )
                ==
                TARGET_STATUS
            ),

        "decision_promote":
            (
                decision.get(
                    "decision"
                )
                ==
                EXPECTED_DECISION
            ),

        "reason_replicated_causal_support":
            (
                decision.get(
                    "reason_code"
                )
                ==
                EXPECTED_REASON_CODE
            ),

        "next_action_monitor":
            (
                next_action.get(
                    "action"
                )
                ==
                EXPECTED_NEXT_ACTION
            ),

        "no_planned_experiment_for_target":
            (
                next_action.get(
                    "experiment_id"
                )
                is None
            ),

        "evaluation_did_not_mutate_store":
            (
                evaluation_document.get(
                    "overall_checks",
                    {},
                ).get(
                    "knowledge_store_not_mutated"
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
            "L12G input validation failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "target_evaluation":
            target,

        "snapshot_sha256":
            snapshot_sha256,

        "evaluation_sha256":
            evaluation_sha256,
    }


# ============================================================
# Candidate-state preparation
# ============================================================

def prepare_candidate_state(
    *,
    store,
):

    data = store.load_all()

    hypotheses = copy.deepcopy(
        data[
            "hypotheses"
        ]
    )

    research_state = copy.deepcopy(
        data[
            "research_state"
        ]
    )

    hypothesis_index = deep_index(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    require(
        TARGET_HYPOTHESIS_ID
        in
        hypothesis_index,
        (
            "Target hypothesis missing from "
            "knowledge store."
        ),
    )

    hypothesis = hypothesis_index[
        TARGET_HYPOTHESIS_ID
    ]

    require(
        hypothesis.get(
            "status"
        )
        ==
        CURRENT_STATUS,
        (
            "Canonical HYP-V03-003 is not "
            "currently active."
        ),
    )

    require(
        "FND-0009"
        in
        hypothesis.get(
            "evidence_for",
            []
        ),
        (
            "HYP-V03-003 is missing "
            "EXP-0006/FND-0009 support."
        ),
    )

    require(
        "FND-0011"
        in
        hypothesis.get(
            "evidence_for",
            []
        ),
        (
            "HYP-V03-003 is missing "
            "EXP-0007/FND-0011 support."
        ),
    )

    require(
        hypothesis.get(
            "evidence_against",
            []
        )
        ==
        [],
        (
            "HYP-V03-003 unexpectedly has "
            "weakening evidence."
        ),
    )

    now = utc_now()

    hypothesis[
        "status"
    ] = TARGET_STATUS

    hypothesis[
        "updated_at_utc"
    ] = now

    hypothesis[
        "lifecycle_commit"
    ] = {
        "engine_version":
            ENGINE_VERSION,

        "transition":
            (
                f"{CURRENT_STATUS}"
                "->"
                f"{TARGET_STATUS}"
            ),

        "reason_code":
            EXPECTED_REASON_CODE,

        "supporting_experiments":
            list(
                EXPECTED_SUPPORT_SOURCES
            ),

        "supporting_findings": [
            "FND-0009",
            "FND-0011",
        ],

        "committed_at_utc":
            now,
    }

    state = research_state[
        "state"
    ]

    active = list(
        state.get(
            "active_hypotheses",
            []
        )
    )

    if (
        TARGET_HYPOTHESIS_ID
        in
        active
    ):
        active.remove(
            TARGET_HYPOTHESIS_ID
        )

    state[
        "active_hypotheses"
    ] = active

    supported = list(
        state.get(
            "supported_hypotheses",
            []
        )
    )

    if (
        TARGET_HYPOTHESIS_ID
        not in
        supported
    ):
        supported.append(
            TARGET_HYPOTHESIS_ID
        )

    state[
        "supported_hypotheses"
    ] = supported

    state[
        "last_hypothesis_status_commit"
    ] = {
        "hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "from_status":
            CURRENT_STATUS,

        "to_status":
            TARGET_STATUS,

        "reason_code":
            EXPECTED_REASON_CODE,

        "updated_at_utc":
            now,
    }

    hypotheses[
        "updated_at_utc"
    ] = now

    research_state[
        "updated_at_utc"
    ] = now

    return {
        "hypotheses":
            hypotheses,

        "research_state":
            research_state,
    }


# ============================================================
# Candidate validation
# ============================================================

def validate_candidate_state(
    *,
    store,
    candidate,
):

    hypotheses_path = store.collection_path(
        "hypotheses"
    )

    research_state_path = store.collection_path(
        "research_state"
    )

    backups = capture_backups(
        [
            hypotheses_path,
            research_state_path,
        ]
    )

    try:

        store.save(
            "hypotheses",
            candidate[
                "hypotheses"
            ],
        )

        store.save(
            "research_state",
            candidate[
                "research_state"
            ],
        )

        validation = store.validate()

        require(
            validation[
                "passed"
            ],
            (
                "Candidate knowledge state "
                "failed store validation."
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

        target = hypotheses[
            TARGET_HYPOTHESIS_ID
        ]

        state = data[
            "research_state"
        ][
            "state"
        ]

        checks = {
            "store_validation":
                bool(
                    validation[
                        "passed"
                    ]
                ),

            "target_supported":
                (
                    target.get(
                        "status"
                    )
                    ==
                    TARGET_STATUS
                ),

            "evidence_for_preserved":
                (
                    "FND-0009"
                    in
                    target.get(
                        "evidence_for",
                        []
                    )
                    and
                    "FND-0011"
                    in
                    target.get(
                        "evidence_for",
                        []
                    )
                ),

            "evidence_against_preserved":
                (
                    target.get(
                        "evidence_against",
                        []
                    )
                    ==
                    []
                ),

            "removed_from_active_index":
                (
                    TARGET_HYPOTHESIS_ID
                    not in
                    state.get(
                        "active_hypotheses",
                        []
                    )
                ),

            "added_to_supported_index":
                (
                    TARGET_HYPOTHESIS_ID
                    in
                    state.get(
                        "supported_hypotheses",
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
                "L12G candidate-state "
                f"validation failed: {failed}"
            )

        return {
            "validation":
                validation,

            "checks":
                checks,
        }

    finally:

        restore_backups(
            backups
        )


# ============================================================
# Snapshot regeneration
# ============================================================

def regenerate_snapshot(
    *,
    store,
    snapshot_path,
    report_path,
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
            "Regenerated scientist context "
            "failed validation."
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
            "Saved scientist context "
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

        "snapshot_sha256":
            file_sha256(
                snapshot_path
            ),

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
        report_path,
        report,
    )

    return {
        "snapshot":
            saved,

        "validation":
            saved_validation,

        "snapshot_sha256":
            file_sha256(
                snapshot_path
            ),
    }


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

    snapshot_report_path = project_path(
        args.snapshot_report
    )

    print()
    print("=" * 88)
    print(
        "STAGE L12G — HYP-V03-003 "
        "CONTROLLED STATUS COMMIT"
    )
    print("=" * 88)

    snapshot = load_json(
        snapshot_path
    )

    evaluation_document = load_json(
        evaluation_path
    )

    validation = validate_inputs(
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        evaluation_document=(
            evaluation_document
        ),
        evaluation_path=evaluation_path,
    )

    print()
    print(
        "Evaluation binding"
    )
    print(
        "------------------"
    )

    print(
        f"Knowledge state: "
        f"{snapshot['knowledge_state_sha256']}"
    )

    print(
        f"Snapshot SHA-256: "
        f"{validation['snapshot_sha256']}"
    )

    print(
        f"Evaluation SHA-256: "
        f"{validation['evaluation_sha256']}"
    )

    print()
    print(
        "Input checks"
    )
    print(
        "------------"
    )

    for name, passed in (
        validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    store = KnowledgeStore()

    before_hashes = collection_hashes(
        store
    )

    candidate = prepare_candidate_state(
        store=store
    )

    candidate_validation = (
        validate_candidate_state(
            store=store,
            candidate=candidate,
        )
    )

    after_preview_hashes = (
        collection_hashes(
            store
        )
    )

    require(
        before_hashes
        ==
        after_preview_hashes,
        (
            "L12G preview unexpectedly "
            "mutated canonical L7 files."
        ),
    )

    print()
    print(
        "Proposed lifecycle transition"
    )
    print(
        "-----------------------------"
    )

    print(
        "HYP-V03-003:"
    )
    print(
        "  active -> supported"
    )
    print(
        "  reason: "
        "REPLICATED_CAUSAL_SUPPORT"
    )
    print(
        "  sources: EXP-0006, EXP-0007"
    )
    print(
        "  evidence preserved: "
        "FND-0009, FND-0011"
    )

    print()
    print(
        "Candidate-state checks"
    )
    print(
        "----------------------"
    )

    for name, passed in (
        candidate_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    if not args.apply:

        print()
        print("=" * 88)
        print(
            "SUCCESS: L12G PREVIEW PASSED."
        )
        print(
            "No knowledge files were modified."
        )
        print()
        print(
            "To commit, rerun with:"
        )
        print(
            '  --apply --confirmation '
            f'"{EXPECTED_CONFIRMATION}"'
        )
        print("=" * 88)

        return

    if (
        args.confirmation
        !=
        EXPECTED_CONFIRMATION
    ):

        raise PermissionError(
            "L12G commit blocked: exact "
            "confirmation phrase required."
        )

    if output_path.exists():

        raise FileExistsError(
            "HYP-V03-003 status commit report "
            f"already exists: {output_path}"
        )

    # ========================================================
    # Transaction
    # ========================================================

    transaction_paths = [
        store.collection_path(
            "hypotheses"
        ),
        store.collection_path(
            "research_state"
        ),
        snapshot_path,
        snapshot_report_path,
        output_path,
    ]

    backups = capture_backups(
        transaction_paths
    )

    old_knowledge_state = (
        snapshot[
            "knowledge_state_sha256"
        ]
    )

    try:

        # Preserve all non-status collections.
        immutable_before = {
            key:
                value
            for key, value
            in before_hashes.items()
            if key
            in {
                "findings",
                "experiments",
                "evidence_ledger",
            }
        }

        store.save(
            "hypotheses",
            candidate[
                "hypotheses"
            ],
        )

        store.save(
            "research_state",
            candidate[
                "research_state"
            ],
        )

        store_validation = store.validate()

        require(
            store_validation[
                "passed"
            ],
            (
                "Committed L12G state failed "
                "knowledge-store validation."
            ),
        )

        immutable_after_all = (
            collection_hashes(
                store
            )
        )

        immutable_after = {
            key:
                value
            for key, value
            in immutable_after_all.items()
            if key
            in {
                "findings",
                "experiments",
                "evidence_ledger",
            }
        }

        require(
            immutable_before
            ==
            immutable_after,
            (
                "L12G changed evidence/findings/"
                "experiment collections."
            ),
        )

        snapshot_result = (
            regenerate_snapshot(
                store=store,
                snapshot_path=snapshot_path,
                report_path=(
                    snapshot_report_path
                ),
            )
        )

        final_data = store.load_all()

        final_hypotheses = deep_index(
            final_data[
                "hypotheses"
            ][
                "records"
            ],
            "hypothesis_id",
        )

        final_target = final_hypotheses[
            TARGET_HYPOTHESIS_ID
        ]

        final_state = final_data[
            "research_state"
        ][
            "state"
        ]

        final_checks = {
            "store_validation":
                bool(
                    store_validation[
                        "passed"
                    ]
                ),

            "target_status_supported":
                (
                    final_target.get(
                        "status"
                    )
                    ==
                    TARGET_STATUS
                ),

            "support_evidence_preserved":
                (
                    "FND-0009"
                    in
                    final_target.get(
                        "evidence_for",
                        []
                    )
                    and
                    "FND-0011"
                    in
                    final_target.get(
                        "evidence_for",
                        []
                    )
                ),

            "no_weakening_evidence":
                (
                    final_target.get(
                        "evidence_against",
                        []
                    )
                    ==
                    []
                ),

            "active_index_updated":
                (
                    TARGET_HYPOTHESIS_ID
                    not in
                    final_state.get(
                        "active_hypotheses",
                        []
                    )
                ),

            "supported_index_updated":
                (
                    TARGET_HYPOTHESIS_ID
                    in
                    final_state.get(
                        "supported_hypotheses",
                        []
                    )
                ),

            "snapshot_validation":
                bool(
                    snapshot_result[
                        "validation"
                    ][
                        "passed"
                    ]
                ),

            "findings_unchanged":
                (
                    immutable_before[
                        "findings"
                    ]
                    ==
                    immutable_after[
                        "findings"
                    ]
                ),

            "experiments_unchanged":
                (
                    immutable_before[
                        "experiments"
                    ]
                    ==
                    immutable_after[
                        "experiments"
                    ]
                ),

            "evidence_ledger_unchanged":
                (
                    immutable_before[
                        "evidence_ledger"
                    ]
                    ==
                    immutable_after[
                        "evidence_ledger"
                    ]
                ),
        }

        failed = [
            name
            for name, passed
            in final_checks.items()
            if not passed
        ]

        if failed:
            raise RuntimeError(
                "L12G final validation "
                f"failed: {failed}"
            )

        commit_seed = {
            "hypothesis_id":
                TARGET_HYPOTHESIS_ID,

            "from_status":
                CURRENT_STATUS,

            "to_status":
                TARGET_STATUS,

            "reason_code":
                EXPECTED_REASON_CODE,

            "old_knowledge_state":
                old_knowledge_state,

            "new_knowledge_state":
                snapshot_result[
                    "snapshot"
                ][
                    "knowledge_state_sha256"
                ],

            "evaluation_sha256":
                validation[
                    "evaluation_sha256"
                ],
        }

        commit_id = (
            "HSC-"
            +
            canonical_json_sha256(
                commit_seed
            )[
                :16
            ].upper()
        )

        report = {
            "engine_version":
                ENGINE_VERSION,

            "commit_id":
                commit_id,

            "created_at_utc":
                utc_now(),

            "hypothesis_id":
                TARGET_HYPOTHESIS_ID,

            "transition": {
                "from":
                    CURRENT_STATUS,

                "to":
                    TARGET_STATUS,

                "reason_code":
                    EXPECTED_REASON_CODE,

                "decision":
                    EXPECTED_DECISION,
            },

            "evaluator_binding": {
                "knowledge_state_sha256":
                    EXPECTED_KNOWLEDGE_STATE,

                "scientist_context_sha256":
                    validation[
                        "snapshot_sha256"
                    ],

                "hypothesis_evaluation_sha256":
                    validation[
                        "evaluation_sha256"
                    ],

                "support_source_experiments":
                    list(
                        EXPECTED_SUPPORT_SOURCES
                    ),
            },

            "knowledge_state": {
                "old":
                    old_knowledge_state,

                "new":
                    snapshot_result[
                        "snapshot"
                    ][
                        "knowledge_state_sha256"
                    ],
            },

            "policy": {
                "evidence_mutated":
                    False,

                "findings_mutated":
                    False,

                "experiments_mutated":
                    False,

                "hypothesis_evidence_links_mutated":
                    False,

                "hypothesis_status_mutated":
                    True,

                "research_state_index_mutated":
                    True,
            },

            "final_checks":
                final_checks,

            "passed":
                True,
        }

        atomic_write_json(
            output_path,
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

    print()
    print("=" * 88)
    print(
        "L12G STATUS COMMIT SUMMARY"
    )
    print("=" * 88)

    print()
    print(
        "Commit ID:"
    )
    print(
        f"  {commit_id}"
    )

    print()
    print(
        "Hypothesis:"
    )
    print(
        "  HYP-V03-003"
    )

    print(
        "Transition:"
    )
    print(
        "  active -> supported"
    )

    print(
        "Reason:"
    )
    print(
        "  REPLICATED_CAUSAL_SUPPORT"
    )

    print()
    print(
        "Supporting experiments:"
    )
    print(
        "  EXP-0006"
    )
    print(
        "  EXP-0007"
    )

    print()
    print(
        "Old knowledge fingerprint:"
    )
    print(
        f"  {old_knowledge_state}"
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
        "Final checks"
    )
    print(
        "------------"
    )

    for name, passed in (
        final_checks.items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    print()
    print(
        "Commit report:"
    )
    print(
        output_path
    )

    print()
    print("=" * 88)
    print(
        "SUCCESS: STAGE L12G "
        "HYP-V03-003 STATUS COMMIT COMPLETE."
    )
    print(
        "Only lifecycle/index state changed; "
        "evidence, findings, and experiments "
        "were preserved."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
