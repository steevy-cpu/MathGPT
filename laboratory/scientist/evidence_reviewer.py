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
    "L8A.1"
)


EXPECTED_SNAPSHOT_VERSION = (
    "L7D.1"
)


EXPECTED_KNOWLEDGE_SCHEMA = (
    "L7.1"
)


DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "evidence_review.json"
)


# ============================================================
# Evidence levels
#
# Ranking is used only for ordering evidence.
# It is NOT a confidence score.
# ============================================================

EVIDENCE_LEVEL_ORDER = {
    "E0": 0,
    "E1": 1,
    "E2": 2,
    "E3": 3,
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


def strongest_level(
    records,
):

    levels = [
        record[
            "finding"
        ].get(
            "evidence_level"
        )

        for record
        in records

        if (
            record[
                "finding"
            ].get(
                "evidence_level"
            )
            in
            EVIDENCE_LEVEL_ORDER
        )
    ]

    if not levels:

        return None

    return max(
        levels,
        key=lambda level:
            EVIDENCE_LEVEL_ORDER[
                level
            ],
    )


def level_counts(
    records,
):

    result = {
        "E0": 0,
        "E1": 0,
        "E2": 0,
        "E3": 0,
    }

    for record in records:

        level = (
            record[
                "finding"
            ].get(
                "evidence_level"
            )
        )

        if level in result:

            result[
                level
            ] += 1

    return result


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L8A — deterministic "
            "evidence reviewer for the "
            "MathGPT AI Scientist."
        )
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Snapshot validation
# ============================================================

def validate_snapshot(
    snapshot,
):

    handoff_checks = (
        snapshot.get(
            "handoff_checks",
            {},
        )
    )

    checks = {
        "snapshot_version":
            (
                snapshot.get(
                    "snapshot_version"
                )
                ==
                EXPECTED_SNAPSHOT_VERSION
            ),

        "knowledge_schema_version":
            (
                snapshot.get(
                    "knowledge_schema_version"
                )
                ==
                EXPECTED_KNOWLEDGE_SCHEMA
            ),

        "knowledge_fingerprint":
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

        "handoff_checks_present":
            bool(
                handoff_checks
            ),

        "handoff_checks_passed":
            (
                bool(
                    handoff_checks
                )
                and
                all(
                    handoff_checks.values()
                )
            ),

        "hypotheses_present":
            isinstance(
                snapshot.get(
                    "hypotheses"
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

        "experiments_present":
            isinstance(
                snapshot.get(
                    "experiments"
                ),
                dict,
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
            "Scientist context snapshot "
            f"failed L8A validation: "
            f"{failed}"
        )

    return checks


# ============================================================
# Index builders
# ============================================================

def build_indexes(
    snapshot,
):

    evidence_by_experiment = {}

    for evidence in (
        snapshot[
            "evidence_inventory"
        ]
    ):

        experiment_id = (
            evidence[
                "experiment_id"
            ]
        )

        if (
            experiment_id
            in
            evidence_by_experiment
        ):

            raise RuntimeError(
                "Multiple evidence records "
                "found for experiment "
                f"{experiment_id}."
            )

        evidence_by_experiment[
            experiment_id
        ] = evidence

    experiments = {}

    experiment_groups = (
        snapshot[
            "experiments"
        ]
    )

    for group_name in (
        "completed",
        "planned",
    ):

        for experiment in (
            experiment_groups.get(
                group_name,
                []
            )
        ):

            experiment_id = (
                experiment[
                    "experiment_id"
                ]
            )

            if experiment_id in experiments:

                raise RuntimeError(
                    "Duplicate experiment in "
                    "scientist context: "
                    f"{experiment_id}"
                )

            experiments[
                experiment_id
            ] = experiment

    return {
        "evidence_by_experiment":
            evidence_by_experiment,

        "experiments":
            experiments,
    }


# ============================================================
# Finding provenance
# ============================================================

def review_finding(
    *,
    finding,
    hypothesis_checkpoint_sha256,
    evidence_by_experiment,
    experiments,
):

    experiment_id = (
        finding.get(
            "source_experiment_id"
        )
    )

    experiment = experiments.get(
        experiment_id
    )

    evidence = (
        evidence_by_experiment.get(
            experiment_id
        )
    )

    finding_checkpoint = (
        finding.get(
            "model_scope",
            {},
        ).get(
            "checkpoint_sha256"
        )
    )

    experiment_checkpoint = None

    if experiment is not None:

        experiment_checkpoint = (
            experiment.get(
                "model_scope",
                {},
            ).get(
                "checkpoint_sha256"
            )
        )

    evidence_checkpoint = None

    if evidence is not None:

        evidence_checkpoint = (
            evidence.get(
                "checkpoint_sha256"
            )
        )

    experiment_completed = (
        experiment is not None
        and
        experiment.get(
            "status"
        )
        ==
        "completed"
    )

    evidence_verified = (
        evidence is not None
        and
        isinstance(
            evidence.get(
                "artifact_sha256"
            ),
            str,
        )
        and
        len(
            evidence[
                "artifact_sha256"
            ]
        )
        ==
        64
    )

    checkpoint_consistent = (
        finding_checkpoint
        ==
        hypothesis_checkpoint_sha256
        ==
        experiment_checkpoint
        ==
        evidence_checkpoint
    )

    provenance_complete = (
        experiment_completed
        and
        evidence_verified
        and
        checkpoint_consistent
    )

    return {
        "finding":
            finding,

        "provenance": {
            "experiment_id":
                experiment_id,

            "experiment_status":
                (
                    None
                    if experiment is None
                    else experiment.get(
                        "status"
                    )
                ),

            "evidence_record_id":
                (
                    None
                    if evidence is None
                    else evidence.get(
                        "evidence_record_id"
                    )
                ),

            "artifact_path":
                (
                    None
                    if evidence is None
                    else evidence.get(
                        "artifact_path"
                    )
                ),

            "artifact_sha256":
                (
                    None
                    if evidence is None
                    else evidence.get(
                        "artifact_sha256"
                    )
                ),

            "finding_checkpoint_sha256":
                finding_checkpoint,

            "experiment_checkpoint_sha256":
                experiment_checkpoint,

            "evidence_checkpoint_sha256":
                evidence_checkpoint,

            "hypothesis_checkpoint_sha256":
                hypothesis_checkpoint_sha256,

            "experiment_completed":
                experiment_completed,

            "evidence_verified":
                evidence_verified,

            "checkpoint_consistent":
                checkpoint_consistent,

            "provenance_complete":
                provenance_complete,
        },
    }


# ============================================================
# Evidence classification
# ============================================================

def classify_profile(
    support_count,
    weakening_count,
):

    if (
        support_count == 0
        and
        weakening_count == 0
    ):

        return "untested"

    if (
        support_count > 0
        and
        weakening_count == 0
    ):

        return "support_only"

    if (
        support_count == 0
        and
        weakening_count > 0
    ):

        return "weakening_only"

    return "mixed"


# ============================================================
# Review one hypothesis
# ============================================================

def review_hypothesis(
    *,
    hypothesis,
    evidence_by_experiment,
    experiments,
):

    hypothesis_id = (
        hypothesis[
            "hypothesis_id"
        ]
    )

    hypothesis_checkpoint = (
        hypothesis.get(
            "model_scope",
            {},
        ).get(
            "checkpoint_sha256"
        )
    )

    support_records = [
        review_finding(
            finding=finding,
            hypothesis_checkpoint_sha256=(
                hypothesis_checkpoint
            ),
            evidence_by_experiment=(
                evidence_by_experiment
            ),
            experiments=experiments,
        )

        for finding
        in hypothesis.get(
            "supporting_findings",
            []
        )
    ]

    weakening_records = [
        review_finding(
            finding=finding,
            hypothesis_checkpoint_sha256=(
                hypothesis_checkpoint
            ),
            evidence_by_experiment=(
                evidence_by_experiment
            ),
            experiments=experiments,
        )

        for finding
        in hypothesis.get(
            "weakening_findings",
            []
        )
    ]

    all_records = (
        support_records
        +
        weakening_records
    )

    provenance_complete = all(
        record[
            "provenance"
        ][
            "provenance_complete"
        ]

        for record
        in all_records
    )

    checkpoint_consistent = all(
        record[
            "provenance"
        ][
            "checkpoint_consistent"
        ]

        for record
        in all_records
    )

    evidence_levels_valid = all(
        record[
            "finding"
        ].get(
            "evidence_level"
        )
        in
        EVIDENCE_LEVEL_ORDER

        for record
        in all_records
    )

    support_count = len(
        support_records
    )

    weakening_count = len(
        weakening_records
    )

    review_class = classify_profile(
        support_count,
        weakening_count,
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

        "current_status":
            hypothesis.get(
                "status"
            ),

        "existing_confidence":
            hypothesis.get(
                "confidence"
            ),

        "model_scope":
            hypothesis.get(
                "model_scope"
            ),

        "evidence_profile": {
            "classification":
                review_class,

            "supporting_count":
                support_count,

            "weakening_count":
                weakening_count,

            "supporting_by_level":
                level_counts(
                    support_records
                ),

            "weakening_by_level":
                level_counts(
                    weakening_records
                ),

            "strongest_supporting_level":
                strongest_level(
                    support_records
                ),

            "strongest_weakening_level":
                strongest_level(
                    weakening_records
                ),
        },

        "supporting_evidence":
            support_records,

        "weakening_evidence":
            weakening_records,

        "open_questions":
            list(
                hypothesis.get(
                    "open_questions",
                    [],
                )
            ),

        "tested_by":
            [
                experiment.get(
                    "experiment_id"
                )

                for experiment
                in hypothesis.get(
                    "tested_by",
                    []
                )
            ],

        "checks": {
            "provenance_complete":
                provenance_complete,

            "checkpoint_consistent":
                checkpoint_consistent,

            "evidence_levels_valid":
                evidence_levels_valid,

            # L8A does not invent or update confidence.
            "no_generated_confidence_score":
                True,
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    snapshot_path = project_path(
        args.snapshot
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
            f"Evidence review already exists: "
            f"{output_path}\n"
            "Use --overwrite after the "
            "scientist context changes."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L8A — SCIENTIFIC "
        "EVIDENCE REVIEW"
    )

    print(
        "=" * 76
    )

    snapshot = load_json(
        snapshot_path
    )

    snapshot_checks = (
        validate_snapshot(
            snapshot
        )
    )

    snapshot_sha256 = (
        file_sha256(
            snapshot_path
        )
    )

    print()
    print(
        f"Scientist context:"
    )

    print(
        snapshot_path
    )

    print()

    print(
        f"Snapshot SHA-256:"
    )

    print(
        snapshot_sha256
    )

    print()

    print(
        "Snapshot checks"
    )

    print(
        "---------------"
    )

    for name, passed in (
        snapshot_checks.items()
    ):

        print(
            f"{name:<34} "
            f"{passed}"
        )

    indexes = build_indexes(
        snapshot
    )

    evidence_by_experiment = (
        indexes[
            "evidence_by_experiment"
        ]
    )

    experiments = (
        indexes[
            "experiments"
        ]
    )

    # ========================================================
    # Collect all hypotheses represented in snapshot
    # ========================================================

    hypothesis_groups = (
        snapshot[
            "hypotheses"
        ]
    )

    hypotheses = []

    for group_name in (
        "active",
        "weakened",
        "supported",
        "falsified",
    ):

        hypotheses.extend(
            hypothesis_groups.get(
                group_name,
                []
            )
        )

    seen_ids = set()

    reviews = []

    print()
    print(
        "Hypothesis reviews"
    )

    print(
        "------------------"
    )

    for hypothesis in hypotheses:

        hypothesis_id = (
            hypothesis[
                "hypothesis_id"
            ]
        )

        if hypothesis_id in seen_ids:

            raise RuntimeError(
                "Duplicate hypothesis in "
                "scientist snapshot: "
                f"{hypothesis_id}"
            )

        seen_ids.add(
            hypothesis_id
        )

        review = review_hypothesis(
            hypothesis=hypothesis,
            evidence_by_experiment=(
                evidence_by_experiment
            ),
            experiments=experiments,
        )

        reviews.append(
            review
        )

        profile = (
            review[
                "evidence_profile"
            ]
        )

        print()

        print(
            f"{hypothesis_id}"
        )

        print(
            f"  current status: "
            f"{review['current_status']}"
        )

        print(
            f"  evidence profile: "
            f"{profile['classification']}"
        )

        print(
            f"  support: "
            f"{profile['supporting_count']}"
        )

        print(
            f"  weakening: "
            f"{profile['weakening_count']}"
        )

        print(
            f"  strongest support: "
            f"{profile['strongest_supporting_level']}"
        )

        print(
            f"  strongest weakening: "
            f"{profile['strongest_weakening_level']}"
        )

        print(
            f"  provenance complete: "
            f"{review['checks']['provenance_complete']}"
        )

    # ========================================================
    # Overall checks
    # ========================================================

    all_provenance_complete = all(
        review[
            "checks"
        ][
            "provenance_complete"
        ]

        for review
        in reviews
    )

    all_checkpoint_consistent = all(
        review[
            "checks"
        ][
            "checkpoint_consistent"
        ]

        for review
        in reviews
    )

    all_levels_valid = all(
        review[
            "checks"
        ][
            "evidence_levels_valid"
        ]

        for review
        in reviews
    )

    all_no_generated_scores = all(
        review[
            "checks"
        ][
            "no_generated_confidence_score"
        ]

        for review
        in reviews
    )

    planned_experiment_ids = {
        experiment[
            "experiment_id"
        ]

        for experiment
        in snapshot[
            "experiments"
        ].get(
            "planned",
            []
        )
    }

    evidence_experiment_ids = set(
        evidence_by_experiment
    )

    planned_excluded_from_evidence = (
        planned_experiment_ids
        .isdisjoint(
            evidence_experiment_ids
        )
    )

    overall_checks = {
        "snapshot_valid":
            all(
                snapshot_checks.values()
            ),

        "at_least_one_hypothesis_reviewed":
            (
                len(
                    reviews
                )
                >
                0
            ),

        "all_provenance_complete":
            all_provenance_complete,

        "all_checkpoint_consistent":
            all_checkpoint_consistent,

        "all_evidence_levels_valid":
            all_levels_valid,

        "planned_experiments_excluded_from_evidence":
            planned_excluded_from_evidence,

        "no_confidence_scores_generated":
            all_no_generated_scores,
    }

    failed = [
        name

        for name, passed
        in overall_checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "L8A evidence review failed: "
            f"{failed}"
        )

    # ========================================================
    # Output
    # ========================================================

    output = {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "deterministic_qualitative_"
                "evidence_review"
            ),

        "scientist_context": {
            "path":
                str(
                    snapshot_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sha256":
                snapshot_sha256,

            "knowledge_state_sha256":
                snapshot[
                    "knowledge_state_sha256"
                ],

            "snapshot_version":
                snapshot[
                    "snapshot_version"
                ],
        },

        "policy": {
            "numerical_confidence_generated":
                False,

            "evidence_level_used_as_weight":
                False,

            "evidence_level_usage":
                (
                    "Evidence levels are retained "
                    "and ordered for provenance "
                    "only. L8A does not convert "
                    "E1/E2/E3 into numerical "
                    "confidence weights."
                ),

            "planned_experiments_count_as_evidence":
                False,

            "hypothesis_status_mutated":
                False,
        },

        "hypothesis_review_count":
            len(
                reviews
            ),

        "hypothesis_reviews":
            reviews,

        "overall_checks":
            overall_checks,
    }

    atomic_write_json(
        output_path,
        output,
    )

    # ========================================================
    # Verify saved output
    # ========================================================

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
            "Saved evidence-review identity "
            "did not validate."
        )

    if not all(
        saved[
            "overall_checks"
        ].values()
    ):

        raise RuntimeError(
            "Saved evidence review contains "
            "failed checks."
        )

    # ========================================================
    # Final display
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "L8A EVIDENCE REVIEW SUMMARY"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Hypotheses reviewed: "
        f"{len(reviews)}"
    )

    print()

    for review in reviews:

        print(
            f"{review['hypothesis_id']}: "
            f"{review['evidence_profile']['classification']}"
        )

    print()

    print(
        "Overall checks"
    )

    print(
        "--------------"
    )

    for name, passed in (
        overall_checks.items()
    ):

        print(
            f"{name:<45} "
            f"{passed}"
        )

    print()

    print(
        "Evidence review:"
    )

    print(
        output_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L8A "
        "EVIDENCE REVIEW COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()