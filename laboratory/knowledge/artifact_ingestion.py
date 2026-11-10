import argparse
import hashlib
import json
import sys

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
    SCHEMA_VERSION,
    KnowledgeStore,
    utc_now,
)


# ============================================================
# Report
# ============================================================

INGESTION_REPORT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "artifact_ingestion_report.json"
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L7B — verify laboratory "
            "experiment artifacts and ingest "
            "their factual measurements into "
            "the scientific evidence ledger."
        )
    )

    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help=(
            "Optional experiment ID. "
            "Repeat to ingest multiple. "
            "Default: all completed experiments."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing evidence "
            "record if the source artifact "
            "has changed."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verify and extract without "
            "changing the knowledge store."
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

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


def relative_project_path(
    path,
):

    path = Path(
        path
    ).resolve()

    try:

        return str(
            path.relative_to(
                PROJECT_ROOT.resolve()
            )
        )

    except ValueError:

        return str(
            path
        )


# ============================================================
# Checkpoint identity extraction
# ============================================================

def artifact_checkpoint_sha256(
    artifact,
):

    candidates = [
        artifact.get(
            "checkpoint_sha256"
        ),

        artifact.get(
            "expected_checkpoint_sha256"
        ),
    ]

    model = artifact.get(
        "model"
    )

    if isinstance(
        model,
        dict,
    ):

        candidates.append(
            model.get(
                "checkpoint_sha256"
            )
        )

    for value in candidates:

        if (
            isinstance(
                value,
                str,
            )
            and
            len(
                value
            )
            ==
            64
        ):

            return value

    return None


# ============================================================
# Artifact adapters
#
# Adapters extract measurements only.
#
# They DO NOT interpret what the measurements mean.
# ============================================================

def extract_exp_0002(
    artifact,
):

    return {
        "adapter":
            "exp_0002_v1",

        "facts": {
            "target_component":
                artifact.get(
                    "target_component"
                ),

            "probe_count":
                artifact.get(
                    "probe_count"
                ),

            "result_count":
                len(
                    artifact.get(
                        "results",
                        [],
                    )
                ),

            "group_summary":
                artifact.get(
                    "group_summary"
                ),
        },
    }


def extract_exp_0003(
    artifact,
):

    similarity = artifact.get(
        "similarity",
        {},
    )

    group_summary = (
        similarity.get(
            "group_summary"
        )
    )

    if group_summary is None:

        group_summary = artifact.get(
            "group_summary"
        )

    return {
        "adapter":
            "exp_0003_v1",

        "facts": {
            "target_component":
                artifact.get(
                    "target_component"
                ),

            "context_count":
                artifact.get(
                    "context_count"
                ),

            "group_summary":
                group_summary,
        },
    }


def extract_exp_0004(
    artifact,
):

    return {
        "adapter":
            "exp_0004_v1",

        "facts": {
            "anchor_probe_id":
                artifact.get(
                    "anchor_probe_id"
                ),

            "probe_count":
                artifact.get(
                    "probe_count"
                ),

            "head_count":
                artifact.get(
                    "head_count"
                ),

            "heads":
                artifact.get(
                    "heads"
                ),

            "head_summaries":
                artifact.get(
                    "head_summaries"
                ),

            "hypothesis_diagnostics":
                artifact.get(
                    "hypothesis_diagnostics"
                ),

            "interpretation_rule":
                artifact.get(
                    "interpretation_rule"
                ),

            "integrity_checks":
                artifact.get(
                    "checks"
                ),
        },
    }


def extract_generic(
    artifact,
):

    return {
        "adapter":
            "generic_v1",

        "facts": {
            "title":
                artifact.get(
                    "title"
                ),

            "status":
                artifact.get(
                    "status"
                ),

            "created_at_utc":
                artifact.get(
                    "created_at_utc"
                ),

            "available_top_level_keys":
                sorted(
                    artifact.keys()
                ),
        },
    }


ADAPTERS = {
    "EXP-0002":
        extract_exp_0002,

    "EXP-0003":
        extract_exp_0003,

    "EXP-0004":
        extract_exp_0004,
}


# ============================================================
# Validate one registered artifact
# ============================================================

def inspect_experiment_artifact(
    experiment,
):

    experiment_id = (
        experiment[
            "experiment_id"
        ]
    )

    summary_artifact = (
        experiment.get(
            "summary_artifact"
        )
    )

    if not summary_artifact:

        raise RuntimeError(
            f"{experiment_id} has no "
            "summary artifact."
        )

    artifact_path = project_path(
        summary_artifact
    )

    artifact = load_json(
        artifact_path
    )

    artifact_id = artifact.get(
        "experiment_id"
    )

    if (
        artifact_id
        !=
        experiment_id
    ):

        raise RuntimeError(
            f"{experiment_id}: artifact "
            "experiment identity mismatch. "
            f"Artifact says {artifact_id!r}."
        )

    expected_checkpoint_sha = (
        experiment.get(
            "model_scope",
            {},
        ).get(
            "checkpoint_sha256"
        )
    )

    observed_checkpoint_sha = (
        artifact_checkpoint_sha256(
            artifact
        )
    )

    if (
        expected_checkpoint_sha
        is not None
        and
        observed_checkpoint_sha
        !=
        expected_checkpoint_sha
    ):

        raise RuntimeError(
            f"{experiment_id}: checkpoint "
            "identity mismatch.\n"
            f"Expected: "
            f"{expected_checkpoint_sha}\n"
            f"Observed: "
            f"{observed_checkpoint_sha}"
        )

    artifact_hash = file_sha256(
        artifact_path
    )

    adapter = ADAPTERS.get(
        experiment_id,
        extract_generic,
    )

    extraction = adapter(
        artifact
    )

    return {
        "experiment_id":
            experiment_id,

        "artifact_path":
            relative_project_path(
                artifact_path
            ),

        "artifact_sha256":
            artifact_hash,

        "checkpoint_sha256":
            observed_checkpoint_sha,

        "source_created_at_utc":
            artifact.get(
                "created_at_utc"
            ),

        "adapter":
            extraction[
                "adapter"
            ],

        "facts":
            extraction[
                "facts"
            ],
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    store = KnowledgeStore()

    experiments = store.load(
        "experiments"
    )

    research_state = store.load(
        "research_state"
    )

    evidence_ledger = store.load(
        "evidence_ledger"
    )

    experiment_records = (
        experiments[
            "records"
        ]
    )

    # ========================================================
    # Select experiments
    # ========================================================

    requested = set(
        args.experiment_id
    )

    selected = []

    for experiment in experiment_records:

        experiment_id = (
            experiment[
                "experiment_id"
            ]
        )

        if requested:

            if (
                experiment_id
                in
                requested
            ):

                selected.append(
                    experiment
                )

            continue

        if (
            experiment.get(
                "status"
            )
            ==
            "completed"
            and
            experiment.get(
                "summary_artifact"
            )
        ):

            selected.append(
                experiment
            )

    if requested:

        found = {
            experiment[
                "experiment_id"
            ]

            for experiment
            in selected
        }

        missing = (
            requested
            -
            found
        )

        if missing:

            raise ValueError(
                f"Unknown or unavailable "
                f"experiments: "
                f"{sorted(missing)}"
            )

    if not selected:

        raise RuntimeError(
            "No experiment artifacts "
            "selected for ingestion."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L7B — LABORATORY "
        "ARTIFACT INGESTION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Selected experiments: "
        f"{len(selected)}"
    )

    # ========================================================
    # Current evidence index
    # ========================================================

    evidence_records = (
        evidence_ledger[
            "records"
        ]
    )

    evidence_index = {
        record[
            "evidence_record_id"
        ]:
            record

        for record
        in evidence_records
    }

    ingested = []

    skipped = []

    now = utc_now()

    # ========================================================
    # Ingest
    # ========================================================

    for experiment in selected:

        experiment_id = (
            experiment[
                "experiment_id"
            ]
        )

        print()
        print(
            "-" * 76
        )

        print(
            experiment_id
        )

        print(
            "-" * 76
        )

        inspected = (
            inspect_experiment_artifact(
                experiment
            )
        )

        evidence_record_id = (
            f"EVR-{experiment_id}"
        )

        existing = evidence_index.get(
            evidence_record_id
        )

        if existing is not None:

            same_artifact = (
                existing.get(
                    "artifact_sha256"
                )
                ==
                inspected[
                    "artifact_sha256"
                ]
            )

            if same_artifact:

                print(
                    "Existing evidence record "
                    "matches artifact SHA."
                )

                skipped.append(
                    experiment_id
                )

                # Ensure registry linkage is still
                # restored if necessary.
                experiment[
                    "artifact_ingestion"
                ] = {
                    "status":
                        "verified",

                    "evidence_record_id":
                        evidence_record_id,

                    "artifact_sha256":
                        inspected[
                            "artifact_sha256"
                        ],

                    "adapter":
                        inspected[
                            "adapter"
                        ],

                    "ingested_at_utc":
                        existing.get(
                            "ingested_at_utc"
                        ),
                }

                continue

            if not args.overwrite:

                raise RuntimeError(
                    f"{experiment_id}: evidence "
                    "record exists but artifact "
                    "SHA changed. Use --overwrite "
                    "only after reviewing the "
                    "artifact change."
                )

        evidence_record = {
            "evidence_record_id":
                evidence_record_id,

            "experiment_id":
                experiment_id,

            "evidence_level":
                experiment.get(
                    "evidence_level"
                ),

            "artifact_path":
                inspected[
                    "artifact_path"
                ],

            "artifact_sha256":
                inspected[
                    "artifact_sha256"
                ],

            "checkpoint_sha256":
                inspected[
                    "checkpoint_sha256"
                ],

            "source_created_at_utc":
                inspected[
                    "source_created_at_utc"
                ],

            "ingested_at_utc":
                now,

            "extraction_adapter":
                inspected[
                    "adapter"
                ],

            "facts":
                inspected[
                    "facts"
                ],
        }

        if existing is None:

            evidence_records.append(
                evidence_record
            )

        else:

            position = (
                evidence_records.index(
                    existing
                )
            )

            evidence_records[
                position
            ] = evidence_record

        evidence_index[
            evidence_record_id
        ] = evidence_record

        experiment[
            "artifact_ingestion"
        ] = {
            "status":
                "verified",

            "evidence_record_id":
                evidence_record_id,

            "artifact_sha256":
                inspected[
                    "artifact_sha256"
                ],

            "adapter":
                inspected[
                    "adapter"
                ],

            "ingested_at_utc":
                now,
        }

        print(
            f"Artifact: "
            f"{inspected['artifact_path']}"
        )

        print(
            f"SHA-256: "
            f"{inspected['artifact_sha256']}"
        )

        print(
            f"Checkpoint identity: "
            f"{inspected['checkpoint_sha256']}"
        )

        print(
            f"Adapter: "
            f"{inspected['adapter']}"
        )

        ingested.append(
            experiment_id
        )

    # ========================================================
    # Update collection metadata
    # ========================================================

    experiments[
        "updated_at_utc"
    ] = now

    evidence_ledger[
        "schema_version"
    ] = SCHEMA_VERSION

    evidence_ledger[
        "collection"
    ] = "evidence_ledger"

    evidence_ledger[
        "updated_at_utc"
    ] = now

    # ========================================================
    # Update research state
    # ========================================================

    state = (
        research_state[
            "state"
        ]
    )

    verified_experiments = sorted(
        {
            record[
                "experiment_id"
            ]

            for record
            in evidence_records
        }
    )

    state[
        "artifact_ingestion"
    ] = {
        "status":
            "active",

        "last_ingested_at_utc":
            now,

        "verified_experiment_count":
            len(
                verified_experiments
            ),

        "evidence_record_count":
            len(
                evidence_records
            ),
    }

    state[
        "ingested_experiments"
    ] = verified_experiments

    research_state[
        "updated_at_utc"
    ] = now

    # ========================================================
    # Dry run
    # ========================================================

    if args.dry_run:

        print()
        print(
            "=" * 76
        )

        print(
            "DRY RUN COMPLETE"
        )

        print(
            "=" * 76
        )

        print()

        print(
            f"Would ingest/update: "
            f"{len(ingested)}"
        )

        print(
            f"Already current: "
            f"{len(skipped)}"
        )

        print()

        print(
            "No knowledge files were changed."
        )

        return

    # ========================================================
    # Save atomically collection-by-collection
    # ========================================================

    store.save(
        "evidence_ledger",
        evidence_ledger,
    )

    store.save(
        "experiments",
        experiments,
    )

    store.save(
        "research_state",
        research_state,
    )

    # ========================================================
    # Validate whole knowledge graph
    # ========================================================

    validation = (
        store.validate()
    )

    # ========================================================
    # Ingestion report
    # ========================================================

    report = {
        "stage":
            "L7B",

        "created_at_utc":
            now,

        "selected_experiment_count":
            len(
                selected
            ),

        "ingested_or_updated":
            ingested,

        "already_current":
            skipped,

        "evidence_record_count":
            len(
                evidence_records
            ),

        "verified_experiments":
            verified_experiments,

        "knowledge_validation":
            validation,

        "passed":
            bool(
                validation[
                    "passed"
                ]
            ),
    }

    temporary_report = (
        INGESTION_REPORT_PATH
        .with_name(
            INGESTION_REPORT_PATH.name
            +
            ".tmp"
        )
    )

    with open(
        temporary_report,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

        file.write(
            "\n"
        )

    temporary_report.replace(
        INGESTION_REPORT_PATH
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "L7B INGESTION SUMMARY"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Ingested / updated: "
        f"{len(ingested)}"
    )

    for experiment_id in ingested:

        print(
            f"  {experiment_id}"
        )

    print()

    print(
        f"Already current: "
        f"{len(skipped)}"
    )

    for experiment_id in skipped:

        print(
            f"  {experiment_id}"
        )

    print()

    print(
        f"Evidence records: "
        f"{len(evidence_records)}"
    )

    print()

    store.print_summary()

    print()
    print(
        f"Ingestion report:"
    )

    print(
        INGESTION_REPORT_PATH
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L7B "
        "ARTIFACT INGESTION COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()