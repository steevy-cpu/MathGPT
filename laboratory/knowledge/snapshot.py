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
# Imports
# ============================================================

from laboratory.knowledge.store import (
    COLLECTION_FILES,
    SCHEMA_VERSION,
    KnowledgeStore,
)

from laboratory.knowledge.query import (
    KnowledgeQuery,
)


# ============================================================
# Snapshot identity
# ============================================================

SNAPSHOT_VERSION = (
    "L7D.1"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)


DEFAULT_REPORT = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "knowledge_snapshot_report.json"
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


def sha256_file(
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


def sha256_json(
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


def atomic_write_json(
    path,
    value,
):

    path = Path(
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


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L7D — build one compact, "
            "validated scientist context "
            "snapshot from the MathGPT "
            "scientific knowledge store."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--report",
        type=str,
        default=str(
            DEFAULT_REPORT
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing derived "
            "scientist snapshot."
        ),
    )

    return parser.parse_args()


# ============================================================
# Compact records
# ============================================================

def compact_finding(
    finding,
):

    return {
        "finding_id":
            finding[
                "finding_id"
            ],

        "title":
            finding.get(
                "title"
            ),

        "statement":
            finding.get(
                "statement"
            ),

        "finding_type":
            finding.get(
                "finding_type"
            ),

        "evidence_level":
            finding.get(
                "evidence_level"
            ),

        "source_experiment_id":
            finding.get(
                "source_experiment_id"
            ),

        "supports_hypotheses":
            list(
                finding.get(
                    "supports_hypotheses",
                    [],
                )
            ),

        "weakens_hypotheses":
            list(
                finding.get(
                    "weakens_hypotheses",
                    [],
                )
            ),

        "tags":
            list(
                finding.get(
                    "tags",
                    [],
                )
            ),
    }


def compact_experiment(
    experiment,
    evidence_record=None,
):

    ingestion = experiment.get(
        "artifact_ingestion"
    )

    result = {
        "experiment_id":
            experiment[
                "experiment_id"
            ],

        "title":
            experiment.get(
                "title"
            ),

        "question":
            experiment.get(
                "question"
            ),

        "status":
            experiment.get(
                "status"
            ),

        "evidence_level":
            experiment.get(
                "evidence_level"
            ),

        "summary_artifact":
            experiment.get(
                "summary_artifact"
            ),

        "hypothesis_ids":
            list(
                experiment.get(
                    "hypothesis_ids",
                    [],
                )
            ),

        "finding_ids":
            list(
                experiment.get(
                    "finding_ids",
                    [],
                )
            ),

        "model_scope":
            experiment.get(
                "model_scope"
            ),

        "artifact_ingestion":
            ingestion,
    }

    if evidence_record is not None:

        result[
            "verified_evidence"
        ] = {
            "evidence_record_id":
                evidence_record.get(
                    "evidence_record_id"
                ),

            "artifact_sha256":
                evidence_record.get(
                    "artifact_sha256"
                ),

            "checkpoint_sha256":
                evidence_record.get(
                    "checkpoint_sha256"
                ),

            "extraction_adapter":
                evidence_record.get(
                    "extraction_adapter"
                ),
        }

    else:

        result[
            "verified_evidence"
        ] = None

    return result


def compact_hypothesis(
    knowledge,
    hypothesis,
):

    hypothesis_id = (
        hypothesis[
            "hypothesis_id"
        ]
    )

    detailed = (
        knowledge.get_hypothesis(
            hypothesis_id
        )
    )

    supporting = [
        compact_finding(
            finding
        )

        for finding
        in detailed[
            "supporting_findings"
        ]
    ]

    weakening = [
        compact_finding(
            finding
        )

        for finding
        in detailed[
            "weakening_findings"
        ]
    ]

    tested_by = []

    for experiment in detailed[
        "tested_by_experiments"
    ]:

        evidence = (
            knowledge.evidence_for_experiment(
                experiment[
                    "experiment_id"
                ]
            )
        )

        tested_by.append(
            compact_experiment(
                experiment,
                evidence,
            )
        )

    return {
        "hypothesis_id":
            hypothesis_id,

        "title":
            hypothesis.get(
                "title"
            ),

        "statement":
            hypothesis.get(
                "statement"
            ),

        "status":
            hypothesis.get(
                "status"
            ),

        "confidence":
            hypothesis.get(
                "confidence"
            ),

        "model_scope":
            hypothesis.get(
                "model_scope"
            ),

        "tags":
            list(
                hypothesis.get(
                    "tags",
                    [],
                )
            ),

        "open_questions":
            list(
                hypothesis.get(
                    "open_questions",
                    [],
                )
            ),

        "supporting_findings":
            supporting,

        "weakening_findings":
            weakening,

        "tested_by":
            tested_by,
    }


# ============================================================
# Model/checkpoint inventory
# ============================================================

def build_model_inventory(
    knowledge,
):

    inventory = {}

    candidates = []

    candidates.extend(
        knowledge.hypotheses
    )

    candidates.extend(
        knowledge.findings
    )

    candidates.extend(
        knowledge.experiments
    )

    for item in candidates:

        model_scope = item.get(
            "model_scope"
        )

        if not isinstance(
            model_scope,
            dict,
        ):

            continue

        checkpoint_sha256 = (
            model_scope.get(
                "checkpoint_sha256"
            )
        )

        if not checkpoint_sha256:

            continue

        if (
            checkpoint_sha256
            not in
            inventory
        ):

            inventory[
                checkpoint_sha256
            ] = {
                "checkpoint_sha256":
                    checkpoint_sha256,

                "model":
                    model_scope.get(
                        "model"
                    ),

                "stage":
                    model_scope.get(
                        "stage"
                    ),

                "checkpoint":
                    model_scope.get(
                        "checkpoint"
                    ),

                "training_step":
                    model_scope.get(
                        "training_step"
                    ),

                "tokens_seen":
                    model_scope.get(
                        "tokens_seen"
                    ),

                "experiment_ids":
                    [],

                "finding_ids":
                    [],

                "hypothesis_ids":
                    [],
            }

        record = (
            inventory[
                checkpoint_sha256
            ]
        )

        if "experiment_id" in item:

            experiment_id = item[
                "experiment_id"
            ]

            if (
                experiment_id
                not in
                record[
                    "experiment_ids"
                ]
            ):

                record[
                    "experiment_ids"
                ].append(
                    experiment_id
                )

        if "finding_id" in item:

            finding_id = item[
                "finding_id"
            ]

            if (
                finding_id
                not in
                record[
                    "finding_ids"
                ]
            ):

                record[
                    "finding_ids"
                ].append(
                    finding_id
                )

        if "hypothesis_id" in item:

            hypothesis_id = item[
                "hypothesis_id"
            ]

            if (
                hypothesis_id
                not in
                record[
                    "hypothesis_ids"
                ]
            ):

                record[
                    "hypothesis_ids"
                ].append(
                    hypothesis_id
                )

    records = list(
        inventory.values()
    )

    for record in records:

        record[
            "experiment_ids"
        ].sort()

        record[
            "finding_ids"
        ].sort()

        record[
            "hypothesis_ids"
        ].sort()

    records.sort(
        key=lambda item:
            (
                item.get(
                    "training_step"
                )
                if
                item.get(
                    "training_step"
                )
                is not None
                else -1
            )
    )

    return records


# ============================================================
# Source fingerprints
# ============================================================

def build_source_fingerprints(
    store,
):

    records = {}

    for collection in (
        COLLECTION_FILES
    ):

        path = store.collection_path(
            collection
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        records[
            collection
        ] = {
            "path":
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "size_bytes":
                int(
                    path.stat().st_size
                ),
        }

    return records


# ============================================================
# Build snapshot
# ============================================================

def build_snapshot(
    store,
    knowledge,
):

    generated_at = utc_now()

    validation = (
        store.validate()
    )

    if not validation[
        "passed"
    ]:

        raise RuntimeError(
            "Knowledge store is not valid."
        )

    # ========================================================
    # Exact aggregate fingerprint
    #
    # This fingerprints the logical knowledge state,
    # independent of JSON whitespace.
    # ========================================================

    full_data = (
        store.load_all()
    )

    knowledge_state_sha256 = (
        sha256_json(
            full_data
        )
    )

    # ========================================================
    # Hypothesis groups
    # ========================================================

    active_hypotheses = [
        compact_hypothesis(
            knowledge,
            hypothesis,
        )

        for hypothesis
        in knowledge.active_hypotheses()
    ]

    weakened_hypotheses = [
        compact_hypothesis(
            knowledge,
            hypothesis,
        )

        for hypothesis
        in knowledge.list_hypotheses(
            status="weakened"
        )
    ]

    supported_hypotheses = [
        compact_hypothesis(
            knowledge,
            hypothesis,
        )

        for hypothesis
        in knowledge.list_hypotheses(
            status="supported"
        )
    ]

    falsified_hypotheses = [
        compact_hypothesis(
            knowledge,
            hypothesis,
        )

        for hypothesis
        in knowledge.list_hypotheses(
            status="falsified"
        )
    ]

    # ========================================================
    # Experiments
    # ========================================================

    completed_experiments = []

    for experiment in (
        knowledge.list_experiments(
            status="completed"
        )
    ):

        evidence = (
            knowledge.evidence_for_experiment(
                experiment[
                    "experiment_id"
                ]
            )
        )

        completed_experiments.append(
            compact_experiment(
                experiment,
                evidence,
            )
        )

    planned_experiments = []

    for experiment in (
        knowledge.list_experiments(
            status="planned"
        )
    ):

        evidence = (
            knowledge.evidence_for_experiment(
                experiment[
                    "experiment_id"
                ]
            )
        )

        planned_experiments.append(
            compact_experiment(
                experiment,
                evidence,
            )
        )

    next_experiment = (
        knowledge.next_recommended_experiment()
    )

    if next_experiment is not None:

        next_experiment = (
            compact_experiment(
                next_experiment[
                    "experiment"
                ],
                next_experiment[
                    "evidence_record"
                ],
            )
        )

    # ========================================================
    # Evidence inventory
    # ========================================================

    evidence_inventory = []

    for record in (
        knowledge.evidence_records
    ):

        evidence_inventory.append(
            {
                "evidence_record_id":
                    record[
                        "evidence_record_id"
                    ],

                "experiment_id":
                    record[
                        "experiment_id"
                    ],

                "evidence_level":
                    record.get(
                        "evidence_level"
                    ),

                "artifact_path":
                    record.get(
                        "artifact_path"
                    ),

                "artifact_sha256":
                    record.get(
                        "artifact_sha256"
                    ),

                "checkpoint_sha256":
                    record.get(
                        "checkpoint_sha256"
                    ),

                "extraction_adapter":
                    record.get(
                        "extraction_adapter"
                    ),
            }
        )

    evidence_inventory.sort(
        key=lambda item:
            item[
                "evidence_record_id"
            ]
    )

    # ========================================================
    # Integrity / handoff checks
    # ========================================================

    completed_ids = {
        item[
            "experiment_id"
        ]

        for item
        in completed_experiments
    }

    evidence_experiment_ids = {
        item[
            "experiment_id"
        ]

        for item
        in evidence_inventory
    }

    completed_have_verified_evidence = (
        completed_ids
        <=
        evidence_experiment_ids
    )

    planned_have_no_evidence = all(
        item[
            "verified_evidence"
        ]
        is None

        for item
        in planned_experiments
    )

    active_have_supporting_evidence = all(
        len(
            hypothesis[
                "supporting_findings"
            ]
        )
        >
        0

        for hypothesis
        in active_hypotheses
    )

    next_experiment_is_planned = (
        next_experiment is None

        or

        next_experiment.get(
            "status"
        )
        ==
        "planned"
    )

    handoff_checks = {
        "knowledge_store_valid":
            bool(
                validation[
                    "passed"
                ]
            ),

        "completed_experiments_have_verified_evidence":
            completed_have_verified_evidence,

        "planned_experiments_have_no_evidence":
            planned_have_no_evidence,

        "active_hypotheses_have_supporting_findings":
            active_have_supporting_evidence,

        "next_recommended_experiment_is_planned":
            next_experiment_is_planned,
    }

    if not all(
        handoff_checks.values()
    ):

        failed = [
            name

            for name, passed
            in handoff_checks.items()

            if not passed
        ]

        raise RuntimeError(
            "Scientist handoff checks "
            f"failed: {failed}"
        )

    # ========================================================
    # Snapshot
    # ========================================================

    snapshot = {
        "snapshot_version":
            SNAPSHOT_VERSION,

        "knowledge_schema_version":
            SCHEMA_VERSION,

        "generated_at_utc":
            generated_at,

        "knowledge_state_sha256":
            knowledge_state_sha256,

        "program": {
            "name":
                knowledge.research_state.get(
                    "program"
                ),

            "stage":
                knowledge.research_state.get(
                    "stage"
                ),

            "stage_name":
                knowledge.research_state.get(
                    "stage_name"
                ),

            "scientist_status":
                knowledge.research_state.get(
                    "scientist_status"
                ),
        },

        "research_focus": {
            "current_question":
                knowledge.research_state.get(
                    "current_research_question"
                ),

            "last_completed_experiment":
                knowledge.research_state.get(
                    "last_completed_experiment"
                ),

            "next_recommended_experiment":
                knowledge.research_state.get(
                    "next_recommended_experiment"
                ),
        },

        "knowledge_counts": {
            "hypotheses":
                validation[
                    "hypotheses"
                ],

            "findings":
                validation[
                    "findings"
                ],

            "experiments":
                validation[
                    "experiments"
                ],

            "evidence_records":
                validation[
                    "evidence_records"
                ],
        },

        "hypotheses": {
            "active":
                active_hypotheses,

            "weakened":
                weakened_hypotheses,

            "supported":
                supported_hypotheses,

            "falsified":
                falsified_hypotheses,
        },

        "open_questions":
            knowledge.open_questions(),

        "experiments": {
            "completed":
                completed_experiments,

            "planned":
                planned_experiments,

            "next_recommended":
                next_experiment,
        },

        "evidence_inventory":
            evidence_inventory,

        "model_inventory":
            build_model_inventory(
                knowledge
            ),

        "source_fingerprints":
            build_source_fingerprints(
                store
            ),

        # ====================================================
        # Handoff contract for L8+
        # ====================================================

        "scientist_handoff_contract": {
            "facts_and_interpretations_are_distinct":
                True,

            "artifact_ingestion_is_factual_only":
                True,

            "planned_experiments_are_not_evidence":
                True,

            "confidence_scores_may_be_null":
                True,

            "confidence_engine_status":
                knowledge.research_state.get(
                    "confidence_engine"
                ),

            "experiment_planner_status":
                knowledge.research_state.get(
                    "experiment_planner"
                ),

            "evidence_reviewer_status":
                knowledge.research_state.get(
                    "evidence_reviewer"
                ),

            "requirements_for_future_reasoning": [
                (
                    "Distinguish verified raw "
                    "evidence from curated "
                    "scientific findings."
                ),
                (
                    "Do not treat planned "
                    "experiments as evidence."
                ),
                (
                    "Trace scientific claims "
                    "through findings to source "
                    "experiments and verified "
                    "artifact identities."
                ),
                (
                    "Do not invent confidence "
                    "scores when the confidence "
                    "engine has not evaluated "
                    "a hypothesis."
                ),
                (
                    "Preserve evidence level "
                    "and checkpoint identity "
                    "when comparing findings."
                ),
            ],
        },

        "handoff_checks":
            handoff_checks,
    }

    return snapshot


# ============================================================
# Validate generated snapshot
# ============================================================

def validate_snapshot(
    snapshot,
):

    checks = {
        "snapshot_version":
            (
                snapshot.get(
                    "snapshot_version"
                )
                ==
                SNAPSHOT_VERSION
            ),

        "knowledge_schema_version":
            (
                snapshot.get(
                    "knowledge_schema_version"
                )
                ==
                SCHEMA_VERSION
            ),

        "knowledge_fingerprint_present":
            (
                isinstance(
                    snapshot.get(
                        "knowledge_state_sha256"
                    ),
                    str,
                )
                and
                len(
                    snapshot[
                        "knowledge_state_sha256"
                    ]
                )
                ==
                64
            ),

        "research_question_present":
            bool(
                snapshot.get(
                    "research_focus",
                    {},
                ).get(
                    "current_question"
                )
            ),

        "hypothesis_section_present":
            isinstance(
                snapshot.get(
                    "hypotheses"
                ),
                dict,
            ),

        "experiment_section_present":
            isinstance(
                snapshot.get(
                    "experiments"
                ),
                dict,
            ),

        "evidence_inventory_present":
            isinstance(
                snapshot.get(
                    "evidence_inventory"
                ),
                list,
            ),

        "model_inventory_present":
            isinstance(
                snapshot.get(
                    "model_inventory"
                ),
                list,
            ),

        "source_fingerprints_present":
            isinstance(
                snapshot.get(
                    "source_fingerprints"
                ),
                dict,
            ),

        "handoff_checks_passed":
            all(
                snapshot.get(
                    "handoff_checks",
                    {},
                ).values()
            ),
    }

    return {
        "checks":
            checks,

        "passed":
            all(
                checks.values()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    output_path = project_path(
        args.output
    )

    report_path = project_path(
        args.report
    )

    for path in (
        output_path,
        report_path,
    ):

        if (
            path.exists()
            and
            not args.overwrite
        ):

            raise FileExistsError(
                f"Derived L7D artifact "
                f"already exists: {path}\n"
                "Use --overwrite after the "
                "knowledge state changes."
            )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L7D — SCIENTIST "
        "KNOWLEDGE SNAPSHOT"
    )

    print(
        "=" * 76
    )

    store = KnowledgeStore()

    knowledge = KnowledgeQuery(
        store=store
    )

    print()
    print(
        "Validating scientific "
        "knowledge store..."
    )

    store_validation = (
        store.validate()
    )

    print(
        f"Knowledge valid: "
        f"{store_validation['passed']}"
    )

    print()
    print(
        "Building scientist context..."
    )

    snapshot = build_snapshot(
        store,
        knowledge,
    )

    snapshot_validation = (
        validate_snapshot(
            snapshot
        )
    )

    if not snapshot_validation[
        "passed"
    ]:

        failed = [
            name

            for name, passed
            in snapshot_validation[
                "checks"
            ].items()

            if not passed
        ]

        raise RuntimeError(
            "Scientist snapshot validation "
            f"failed: {failed}"
        )

    # ========================================================
    # Write snapshot
    # ========================================================

    atomic_write_json(
        output_path,
        snapshot,
    )

    # ========================================================
    # Verify exact saved snapshot
    # ========================================================

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as file:

        saved_snapshot = json.load(
            file
        )

    saved_validation = (
        validate_snapshot(
            saved_snapshot
        )
    )

    if not saved_validation[
        "passed"
    ]:

        raise RuntimeError(
            "Saved scientist context "
            "did not validate."
        )

    snapshot_sha256 = (
        sha256_file(
            output_path
        )
    )

    # ========================================================
    # Report
    # ========================================================

    report = {
        "stage":
            "L7D",

        "snapshot_version":
            SNAPSHOT_VERSION,

        "created_at_utc":
            utc_now(),

        "snapshot_path":
            str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
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
            bool(
                saved_validation[
                    "passed"
                ]
            ),
    }

    atomic_write_json(
        report_path,
        report,
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Scientist context"
    )

    print(
        "-----------------"
    )

    print(
        f"Knowledge fingerprint: "
        f"{snapshot['knowledge_state_sha256']}"
    )

    print(
        f"Snapshot SHA-256: "
        f"{snapshot_sha256}"
    )

    print()

    print(
        f"Active hypotheses: "
        f"{len(snapshot['hypotheses']['active'])}"
    )

    print(
        f"Weakened hypotheses: "
        f"{len(snapshot['hypotheses']['weakened'])}"
    )

    print(
        f"Completed experiments: "
        f"{len(snapshot['experiments']['completed'])}"
    )

    print(
        f"Planned experiments: "
        f"{len(snapshot['experiments']['planned'])}"
    )

    print(
        f"Verified evidence records: "
        f"{len(snapshot['evidence_inventory'])}"
    )

    print(
        f"Open questions: "
        f"{len(snapshot['open_questions'])}"
    )

    print()

    print(
        f"Current question:"
    )

    print(
        snapshot[
            "research_focus"
        ][
            "current_question"
        ]
    )

    print()

    print(
        f"Next recommended experiment: "
        f"{snapshot['research_focus']['next_recommended_experiment']}"
    )

    print()

    print(
        "Snapshot:"
    )

    print(
        output_path
    )

    print()

    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Snapshot validation"
    )

    print(
        "-------------------"
    )

    for (
        name,
        passed,
    ) in saved_validation[
        "checks"
    ].items():

        print(
            f"{name:<38} "
            f"{passed}"
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L7D "
        "SCIENTIST HANDOFF COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()