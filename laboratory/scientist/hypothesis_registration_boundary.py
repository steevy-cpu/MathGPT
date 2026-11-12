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

ENGINE_VERSION = "L11B.1"

EXPECTED_SYNTHESIS_VERSION = "L11A.1"
EXPECTED_SOURCE_HYPOTHESIS = "HYP-V03-001"

EXPECTED_SUCCESSORS = {
    "HYP-V03-003",
    "HYP-V03-004",
}

SUPPORTING_FINDINGS = {
    "HYP-V03-003": "FND-0009",
    "HYP-V03-004": "FND-0010",
}

CONFIRMATION_PHRASE = (
    "REGISTER L11A SUCCESSORS AND RETIRE HYP-V03-001"
)


# ============================================================
# Default paths
# ============================================================

DEFAULT_PROPOSAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "hypothesis_synthesis_proposal.json"
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
    / "hypothesis_registration_commit.json"
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
            "Stage L11B — controlled registration "
            "of L11A successor hypotheses and "
            "retirement of the superseded source "
            "hypothesis. Preview-only by default."
        )
    )

    parser.add_argument(
        "--proposal",
        default=str(
            DEFAULT_PROPOSAL_PATH
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
        "--apply",
        action="store_true",
        help=(
            "Apply the exact previewed "
            "hypothesis registration transaction."
        ),
    )

    parser.add_argument(
        "--confirmation",
        default=None,
        help=(
            "Exact confirmation phrase required "
            "together with --apply."
        ),
    )

    return parser.parse_args()


# ============================================================
# Proposal validation
# ============================================================

def validate_proposal(
    *,
    proposal,
    proposal_path,
    snapshot,
    snapshot_path,
):

    synthesis = proposal.get(
        "hypothesis_synthesis",
        {},
    )

    candidates = synthesis.get(
        "candidates",
        [],
    )

    candidate_index = index_by(
        candidates,
        "proposed_hypothesis_id",
    )

    proposal_inputs = proposal.get(
        "inputs",
        {},
    )

    current_snapshot_sha256 = (
        file_sha256(
            snapshot_path
        )
    )

    checks = {
        "proposal_version":
            (
                proposal.get(
                    "engine_version"
                )
                ==
                EXPECTED_SYNTHESIS_VERSION
            ),

        "proposal_id_present":
            (
                isinstance(
                    proposal.get(
                        "proposal_id"
                    ),
                    str,
                )
                and
                proposal[
                    "proposal_id"
                ].startswith(
                    "HSP-"
                )
            ),

        "proposal_checks_present":
            bool(
                proposal.get(
                    "overall_checks"
                )
            ),

        "proposal_checks_passed":
            (
                bool(
                    proposal.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    proposal[
                        "overall_checks"
                    ].values()
                )
            ),

        "source_hypothesis":
            (
                synthesis.get(
                    "source_hypothesis_id"
                )
                ==
                EXPECTED_SOURCE_HYPOTHESIS
            ),

        "source_status_weakened":
            (
                synthesis.get(
                    "source_hypothesis_status"
                )
                ==
                "weakened"
            ),

        "exact_successor_ids":
            (
                set(
                    candidate_index.keys()
                )
                ==
                EXPECTED_SUCCESSORS
            ),

        "candidate_count_2":
            (
                len(
                    candidates
                )
                ==
                2
            ),

        "candidates_proposed_only":
            all(
                candidate.get(
                    "proposed_status"
                )
                ==
                "proposed"
                for candidate
                in candidates
            ),

        "proposal_did_not_register":
            (
                proposal.get(
                    "policy",
                    {},
                ).get(
                    "registers_hypotheses"
                )
                is False
            ),

        "proposal_did_not_mutate_store":
            (
                proposal.get(
                    "policy",
                    {},
                ).get(
                    "mutates_knowledge_store"
                )
                is False
            ),

        "proposal_knowledge_state_matches":
            (
                proposal_inputs.get(
                    "knowledge_state_sha256"
                )
                ==
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        "proposal_snapshot_sha_matches":
            (
                proposal_inputs.get(
                    "scientist_context_sha256"
                )
                ==
                current_snapshot_sha256
            ),

        "proposal_file_exists":
            project_path(
                proposal_path
            ).exists(),
    }

    for hypothesis_id, finding_id in (
        SUPPORTING_FINDINGS.items()
    ):

        candidate = candidate_index.get(
            hypothesis_id,
            {},
        )

        checks[
            f"{hypothesis_id}_primary_support"
        ] = (
            finding_id
            in
            candidate.get(
                "primary_supporting_finding_ids",
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
            "L11B synthesis-proposal "
            f"validation failed: {failed}"
        )

    return {
        "checks":
            checks,

        "candidates":
            candidates,

        "candidate_index":
            candidate_index,

        "proposal_sha256":
            file_sha256(
                proposal_path
            ),

        "snapshot_sha256":
            current_snapshot_sha256,
    }


# ============================================================
# Transaction preparation
# ============================================================

def build_successor_record(
    *,
    candidate,
    support_finding_id,
    created_at,
    proposal_id,
):

    return {
        "hypothesis_id":
            candidate[
                "proposed_hypothesis_id"
            ],

        "title":
            candidate[
                "title"
            ],

        "statement":
            candidate[
                "statement"
            ],

        "model_scope":
            copy.deepcopy(
                candidate[
                    "model_scope"
                ]
            ),

        # Registration activates the successor
        # because a verified E3 supporting
        # finding is already available.
        "status":
            "active",

        "confidence": {
            "score":
                None,

            "label":
                "unscored",

            "method":
                "L11B_registered_from_synthesis",

            "note":
                (
                    "Registered from an "
                    "evidence-grounded L11A "
                    "synthesis proposal. No "
                    "calibrated confidence score "
                    "has been assigned."
                ),
        },

        "created_from":
            proposal_id,

        # EXP-0006 motivated the hypothesis,
        # but was not preregistered as a test
        # of this successor hypothesis.
        "tested_by":
            [],

        "evidence_for": [
            support_finding_id,
        ],

        "evidence_against":
            [],

        "open_questions":
            copy.deepcopy(
                candidate.get(
                    "open_questions",
                    []
                )
            ),

        "tags":
            copy.deepcopy(
                candidate.get(
                    "tags",
                    []
                )
            ),

        "synthesis_provenance": {
            "proposal_id":
                proposal_id,

            "source_hypothesis_id":
                candidate.get(
                    "source_hypothesis_id"
                ),

            "relationship_to_source":
                candidate.get(
                    "relationship_to_source"
                ),

            "source_finding_ids":
                copy.deepcopy(
                    candidate.get(
                        "source_finding_ids",
                        []
                    )
                ),

            "primary_supporting_finding_ids":
                copy.deepcopy(
                    candidate.get(
                        "primary_supporting_finding_ids",
                        []
                    )
                ),

            "motivating_disconfirmation_ids":
                copy.deepcopy(
                    candidate.get(
                        "motivating_disconfirmation_ids",
                        []
                    )
                ),

            "candidate_fingerprint":
                candidate.get(
                    "candidate_fingerprint"
                ),
        },

        "created_at_utc":
            created_at,

        "updated_at_utc":
            created_at,
    }


def prepare_transaction(
    *,
    store,
    proposal,
    candidates,
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

    research_state = copy.deepcopy(
        data[
            "research_state"
        ]
    )

    hypothesis_idx = index_by(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    finding_idx = index_by(
        findings[
            "records"
        ],
        "finding_id",
    )

    require(
        EXPECTED_SOURCE_HYPOTHESIS
        in
        hypothesis_idx,
        (
            "L11B cannot find source "
            "hypothesis HYP-V03-001."
        ),
    )

    source = hypothesis_idx[
        EXPECTED_SOURCE_HYPOTHESIS
    ]

    require(
        source.get(
            "status"
        )
        ==
        "weakened",
        (
            "L11B expected HYP-V03-001 "
            "to remain weakened before "
            "registration."
        ),
    )

    for successor_id in (
        EXPECTED_SUCCESSORS
    ):

        require(
            successor_id
            not in
            hypothesis_idx,
            (
                "L11B refuses duplicate "
                f"registration of {successor_id}."
            ),
        )

    for finding_id in (
        SUPPORTING_FINDINGS.values()
    ):

        require(
            finding_id
            in
            finding_idx,
            (
                "L11B cannot find required "
                f"supporting finding {finding_id}."
            ),
        )

        require(
            finding_idx[
                finding_id
            ].get(
                "evidence_level"
            )
            ==
            "E3",
            (
                f"{finding_id} is not verified "
                "E3 evidence."
            ),
        )

    created_at = utc_now()

    candidate_idx = index_by(
        candidates,
        "proposed_hypothesis_id",
    )

    successor_records = []

    for successor_id in sorted(
        EXPECTED_SUCCESSORS
    ):

        successor_records.append(
            build_successor_record(
                candidate=(
                    candidate_idx[
                        successor_id
                    ]
                ),
                support_finding_id=(
                    SUPPORTING_FINDINGS[
                        successor_id
                    ]
                ),
                created_at=created_at,
                proposal_id=(
                    proposal[
                        "proposal_id"
                    ]
                ),
            )
        )

    hypotheses[
        "records"
    ].extend(
        successor_records
    )

    # --------------------------------------------------------
    # Retire, do not delete or falsify, HYP-001.
    # Its original evidence is preserved exactly.
    # --------------------------------------------------------

    source[
        "status"
    ] = "retired"

    source[
        "retirement"] = {
            "reason":
                (
                    "Superseded by more specific "
                    "candidate-selection hypotheses "
                    "after EXP-0006 directly "
                    "weakened the general "
                    "arithmetic-correctness "
                    "interpretation."
                ),

            "proposal_id":
                proposal[
                    "proposal_id"
                ],

            "successor_hypothesis_ids":
                sorted(
                    EXPECTED_SUCCESSORS
                ),

            "retired_at_utc":
                created_at,

            "evidence_preserved":
                True,

            "falsified":
                False,
        }

    source[
        "updated_at_utc"
    ] = created_at

    # --------------------------------------------------------
    # Reciprocal finding links
    # --------------------------------------------------------

    for successor_id, finding_id in (
        SUPPORTING_FINDINGS.items()
    ):

        finding = finding_idx[
            finding_id
        ]

        supports = list(
            finding.get(
                "supports_hypotheses",
                []
            )
        )

        if successor_id not in supports:
            supports.append(
                successor_id
            )

        finding[
            "supports_hypotheses"
        ] = supports

    # --------------------------------------------------------
    # Research state
    # --------------------------------------------------------

    state = research_state[
        "state"
    ]

    active_hypotheses = [
        hypothesis_id
        for hypothesis_id
        in state.get(
            "active_hypotheses",
            []
        )
        if hypothesis_id
        !=
        EXPECTED_SOURCE_HYPOTHESIS
    ]

    for successor_id in sorted(
        EXPECTED_SUCCESSORS
    ):

        if successor_id not in active_hypotheses:
            active_hypotheses.append(
                successor_id
            )

    state[
        "active_hypotheses"
    ] = active_hypotheses

    state[
        "weakened_hypotheses"
    ] = [
        hypothesis_id
        for hypothesis_id
        in state.get(
            "weakened_hypotheses",
            []
        )
        if hypothesis_id
        !=
        EXPECTED_SOURCE_HYPOTHESIS
    ]

    retired = list(
        state.get(
            "retired_hypotheses",
            []
        )
    )

    if (
        EXPECTED_SOURCE_HYPOTHESIS
        not in
        retired
    ):

        retired.append(
            EXPECTED_SOURCE_HYPOTHESIS
        )

    state[
        "retired_hypotheses"
    ] = retired

    state[
        "current_research_question"
    ] = (
        "What candidate-selection mechanisms "
        "govern numerical next-token decisions "
        "in mature MathGPT v0.3, and how broadly "
        "do L1H0, L4H0, and L3H3 generalize "
        "across candidate pairs and contexts?"
    )

    state[
        "next_recommended_experiment"
    ] = None

    state[
        "hypothesis_synthesis"
    ] = {
        "status":
            "active",

        "last_proposal_id":
            proposal[
                "proposal_id"
            ],

        "registered_successors":
            sorted(
                EXPECTED_SUCCESSORS
            ),

        "retired_source":
            EXPECTED_SOURCE_HYPOTHESIS,

        "updated_at_utc":
            created_at,
    }

    hypotheses[
        "updated_at_utc"
    ] = created_at

    findings[
        "updated_at_utc"
    ] = created_at

    research_state[
        "updated_at_utc"
    ] = created_at

    return {
        "created_at":
            created_at,

        "hypotheses":
            hypotheses,

        "findings":
            findings,

        "research_state":
            research_state,

        "successor_records":
            successor_records,
    }


# ============================================================
# Preview validation using temporary store
# ============================================================

def validate_candidate_state(
    *,
    store,
    prepared,
):

    data = store.load_all()

    original_hypotheses = data[
        "hypotheses"
    ]

    original_findings = data[
        "findings"
    ]

    original_research_state = data[
        "research_state"
    ]

    paths = {
        "hypotheses":
            store.collection_path(
                "hypotheses"
            ),

        "findings":
            store.collection_path(
                "findings"
            ),

        "research_state":
            store.collection_path(
                "research_state"
            ),
    }

    backups = capture_backups(
        paths.values()
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
            "research_state",
            prepared[
                "research_state"
            ],
        )

        validation = store.validate()

        require(
            validation[
                "passed"
            ],
            (
                "L11B candidate knowledge "
                "state failed validation."
            ),
        )

        candidate_data = store.load_all()

        hypothesis_idx = index_by(
            candidate_data[
                "hypotheses"
            ][
                "records"
            ],
            "hypothesis_id",
        )

        finding_idx = index_by(
            candidate_data[
                "findings"
            ][
                "records"
            ],
            "finding_id",
        )

        checks = {
            "store_validation":
                bool(
                    validation[
                        "passed"
                    ]
                ),

            "source_retired":
                (
                    hypothesis_idx[
                        EXPECTED_SOURCE_HYPOTHESIS
                    ][
                        "status"
                    ]
                    ==
                    "retired"
                ),

            "source_not_falsified":
                (
                    hypothesis_idx[
                        EXPECTED_SOURCE_HYPOTHESIS
                    ].get(
                        "retirement",
                        {},
                    ).get(
                        "falsified"
                    )
                    is False
                ),

            "source_evidence_preserved":
                (
                    hypothesis_idx[
                        EXPECTED_SOURCE_HYPOTHESIS
                    ].get(
                        "evidence_for"
                    )
                    ==
                    index_by(
                        original_hypotheses[
                            "records"
                        ],
                        "hypothesis_id",
                    )[
                        EXPECTED_SOURCE_HYPOTHESIS
                    ].get(
                        "evidence_for"
                    )
                    and
                    hypothesis_idx[
                        EXPECTED_SOURCE_HYPOTHESIS
                    ].get(
                        "evidence_against"
                    )
                    ==
                    index_by(
                        original_hypotheses[
                            "records"
                        ],
                        "hypothesis_id",
                    )[
                        EXPECTED_SOURCE_HYPOTHESIS
                    ].get(
                        "evidence_against"
                    )
                ),

            "hyp_003_active":
                (
                    hypothesis_idx[
                        "HYP-V03-003"
                    ][
                        "status"
                    ]
                    ==
                    "active"
                ),

            "hyp_004_active":
                (
                    hypothesis_idx[
                        "HYP-V03-004"
                    ][
                        "status"
                    ]
                    ==
                    "active"
                ),

            "hyp_003_support_link":
                (
                    hypothesis_idx[
                        "HYP-V03-003"
                    ][
                        "evidence_for"
                    ]
                    ==
                    [
                        "FND-0009",
                    ]
                    and
                    "HYP-V03-003"
                    in
                    finding_idx[
                        "FND-0009"
                    ].get(
                        "supports_hypotheses",
                        []
                    )
                ),

            "hyp_004_support_link":
                (
                    hypothesis_idx[
                        "HYP-V03-004"
                    ][
                        "evidence_for"
                    ]
                    ==
                    [
                        "FND-0010",
                    ]
                    and
                    "HYP-V03-004"
                    in
                    finding_idx[
                        "FND-0010"
                    ].get(
                        "supports_hypotheses",
                        []
                    )
                ),

            "new_hypotheses_have_no_test_claims":
                (
                    hypothesis_idx[
                        "HYP-V03-003"
                    ][
                        "tested_by"
                    ]
                    ==
                    []
                    and
                    hypothesis_idx[
                        "HYP-V03-004"
                    ][
                        "tested_by"
                    ]
                    ==
                    []
                ),

            "no_confidence_scores_generated":
                (
                    hypothesis_idx[
                        "HYP-V03-003"
                    ][
                        "confidence"
                    ][
                        "score"
                    ]
                    is None
                    and
                    hypothesis_idx[
                        "HYP-V03-004"
                    ][
                        "confidence"
                    ][
                        "score"
                    ]
                    is None
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
                "L11B candidate-state "
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
            "L11B regenerated snapshot "
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
            "L11B saved snapshot failed "
            "validation."
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

        "falsified_hypothesis_count":
            len(
                snapshot[
                    "hypotheses"
                ][
                    "falsified"
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

    proposal_path = project_path(
        args.proposal
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

    print()
    print("=" * 84)
    print(
        "STAGE L11B — HYPOTHESIS "
        "REGISTRATION & REVISION BOUNDARY"
    )
    print("=" * 84)

    proposal = load_json(
        proposal_path
    )

    snapshot = load_json(
        snapshot_path
    )

    proposal_validation = (
        validate_proposal(
            proposal=proposal,
            proposal_path=(
                proposal_path
            ),
            snapshot=snapshot,
            snapshot_path=(
                snapshot_path
            ),
        )
    )

    print()
    print(
        "Proposal:"
    )
    print(
        f"  {proposal['proposal_id']}"
    )

    print()
    print(
        "Current knowledge state:"
    )
    print(
        "  "
        +
        snapshot[
            "knowledge_state_sha256"
        ]
    )

    print()
    print(
        "Proposal checks"
    )
    print(
        "---------------"
    )

    for name, passed in (
        proposal_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    store = KnowledgeStore()

    prepared = prepare_transaction(
        store=store,
        proposal=proposal,
        candidates=(
            proposal_validation[
                "candidates"
            ]
        ),
    )

    candidate_validation = (
        validate_candidate_state(
            store=store,
            prepared=prepared,
        )
    )

    print()
    print(
        "Proposed state transition"
    )
    print(
        "-------------------------"
    )

    print(
        "HYP-V03-001:"
    )
    print(
        "  weakened -> retired"
    )
    print(
        "  evidence preserved: True"
    )
    print(
        "  falsified: False"
    )

    print()
    print(
        "HYP-V03-003:"
    )
    print(
        "  unregistered -> active"
    )
    print(
        "  E3 support: FND-0009"
    )
    print(
        "  tested_by: []"
    )

    print()
    print(
        "HYP-V03-004:"
    )
    print(
        "  unregistered -> active"
    )
    print(
        "  E3 support: FND-0010"
    )
    print(
        "  tested_by: []"
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
            f"{name:<44} "
            f"{passed}"
        )

    # ========================================================
    # Preview-only default
    # ========================================================

    if not args.apply:

        print()
        print(
            "Mutation status:"
        )
        print(
            "  PREVIEW ONLY"
        )

        print()
        print(
            "Knowledge store modified:"
        )
        print(
            "  False"
        )

        print()
        print(
            "Scientist snapshot modified:"
        )
        print(
            "  False"
        )

        print()
        print(
            "Model loaded:"
        )
        print(
            "  False"
        )

        print()
        print(
            "Experiment executed:"
        )
        print(
            "  False"
        )

        print()
        print(
            "Required confirmation phrase:"
        )
        print(
            f"  {CONFIRMATION_PHRASE}"
        )

        print()
        print("=" * 84)
        print(
            "SUCCESS: L11B REGISTRATION "
            "BOUNDARY PREVIEW PASSED."
        )
        print(
            "No scientific state was changed."
        )
        print("=" * 84)

        return

    # ========================================================
    # Explicit mutation boundary
    # ========================================================

    if (
        args.confirmation
        !=
        CONFIRMATION_PHRASE
    ):

        raise PermissionError(
            "L11B registration denied: "
            "exact confirmation phrase was "
            "not provided.\n"
            "Required phrase:\n"
            f"{CONFIRMATION_PHRASE}"
        )

    paths = [
        store.collection_path(
            "hypotheses"
        ),
        store.collection_path(
            "findings"
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

    old_knowledge_state = (
        snapshot[
            "knowledge_state_sha256"
        ]
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
            "research_state",
            prepared[
                "research_state"
            ],
        )

        validation = store.validate()

        require(
            validation[
                "passed"
            ],
            (
                "L11B committed knowledge "
                "store failed validation."
            ),
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

        new_snapshot = (
            snapshot_result[
                "snapshot"
            ]
        )

        active_ids = {
            item[
                "hypothesis_id"
            ]
            for item
            in new_snapshot[
                "hypotheses"
            ][
                "active"
            ]
        }

        weakened_ids = {
            item[
                "hypothesis_id"
            ]
            for item
            in new_snapshot[
                "hypotheses"
            ][
                "weakened"
            ]
        }

        require(
            EXPECTED_SUCCESSORS
            <=
            active_ids,
            (
                "L11B successor hypotheses "
                "are not visible as active in "
                "the regenerated snapshot."
            ),
        )

        require(
            EXPECTED_SOURCE_HYPOTHESIS
            not in
            weakened_ids,
            (
                "L11B retired source remains "
                "visible as weakened."
            ),
        )

        report = {
            "stage":
                "L11B",

            "engine_version":
                ENGINE_VERSION,

            "created_at_utc":
                utc_now(),

            "proposal_id":
                proposal[
                    "proposal_id"
                ],

            "proposal_sha256":
                proposal_validation[
                    "proposal_sha256"
                ],

            "confirmation_phrase_verified":
                True,

            "old_knowledge_state_sha256":
                old_knowledge_state,

            "new_knowledge_state_sha256":
                new_snapshot[
                    "knowledge_state_sha256"
                ],

            "registration": {
                "registered_active_hypotheses":
                    sorted(
                        EXPECTED_SUCCESSORS
                    ),

                "retired_hypothesis":
                    EXPECTED_SOURCE_HYPOTHESIS,

                "source_evidence_preserved":
                    True,

                "source_marked_falsified":
                    False,

                "new_hypotheses_tested_by":
                    {
                        "HYP-V03-003":
                            [],

                        "HYP-V03-004":
                            [],
                    },
            },

            "knowledge_counts":
                new_snapshot[
                    "knowledge_counts"
                ],

            "snapshot": {
                "path":
                    relative_project_path(
                        snapshot_path
                    ),

                "sha256":
                    snapshot_result[
                        "snapshot_sha256"
                    ],

                "validation":
                    snapshot_result[
                        "validation"
                    ],
            },

            "policy": {
                "model_loaded":
                    False,

                "experiment_executed":
                    False,

                "experiment_registry_mutated":
                    False,

                "confidence_scores_generated":
                    False,
            },

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

    print()
    print("=" * 84)
    print(
        "L11B REGISTRATION COMMIT SUMMARY"
    )
    print("=" * 84)

    print()
    print(
        "Registered active hypotheses:"
    )

    print(
        "  HYP-V03-003"
    )
    print(
        "  HYP-V03-004"
    )

    print()
    print(
        "Retired hypothesis:"
    )

    print(
        "  HYP-V03-001"
    )

    print()
    print(
        "Retirement semantics:"
    )

    print(
        "  superseded: True"
    )
    print(
        "  falsified: False"
    )
    print(
        "  historical evidence preserved: True"
    )

    print()
    print(
        "Old knowledge fingerprint:"
    )

    print(
        "  "
        +
        old_knowledge_state
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
        "Knowledge counts:"
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
        "Commit report:"
    )
    print(
        report_path
    )

    print()
    print("=" * 84)
    print(
        "SUCCESS: STAGE L11B HYPOTHESIS "
        "REGISTRATION + REVISION COMPLETE."
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
