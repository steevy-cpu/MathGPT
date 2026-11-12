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
    "L8B.1"
)


EXPECTED_REVIEW_VERSION = (
    "L8A.1"
)


EXPECTED_SNAPSHOT_VERSION = (
    "L7D.1"
)


DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)


DEFAULT_REVIEW_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "evidence_review.json"
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "hypothesis_evaluation.json"
)


# ============================================================
# Allowed hypothesis states
# ============================================================

ALLOWED_STATUSES = {
    "proposed",
    "active",
    "supported",
    "weakened",
    "falsified",
    "unresolved",
    "retired",
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


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L8B — deterministic, "
            "conservative hypothesis evaluator "
            "for the MathGPT AI Scientist."
        )
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
        ),
        help=(
            "Path to the L7D scientist context "
            "snapshot."
        ),
    )

    parser.add_argument(
        "--review",
        default=str(
            DEFAULT_REVIEW_PATH
        ),
        help=(
            "Path to the L8A evidence review."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Path for the L8B hypothesis "
            "evaluation artifact."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing hypothesis "
            "evaluation artifact."
        ),
    )

    return parser.parse_args()


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    snapshot,
    snapshot_sha256,
    review,
):

    review_context = (
        review.get(
            "scientist_context",
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

        "review_version":
            (
                review.get(
                    "engine_version"
                )
                ==
                EXPECTED_REVIEW_VERSION
            ),

        "snapshot_handoff_checks":
            (
                bool(
                    snapshot.get(
                        "handoff_checks"
                    )
                )
                and
                all(
                    snapshot[
                        "handoff_checks"
                    ].values()
                )
            ),

        "review_overall_checks":
            (
                bool(
                    review.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    review[
                        "overall_checks"
                    ].values()
                )
            ),

        "exact_snapshot_sha_matches_review":
            (
                review_context.get(
                    "sha256"
                )
                ==
                snapshot_sha256
            ),

        "knowledge_state_matches":
            (
                review_context.get(
                    "knowledge_state_sha256"
                )
                ==
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        "review_contains_hypotheses":
            isinstance(
                review.get(
                    "hypothesis_reviews"
                ),
                list,
            ),

        "no_review_confidence_generation":
            (
                review.get(
                    "policy",
                    {},
                ).get(
                    "numerical_confidence_generated"
                )
                is False
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
            "L8B input validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Experiment indexes
# ============================================================

def build_experiment_index(
    snapshot,
):

    index = {}

    for group in (
        "completed",
        "planned",
    ):

        for experiment in (
            snapshot.get(
                "experiments",
                {},
            ).get(
                group,
                [],
            )
        ):

            experiment_id = (
                experiment[
                    "experiment_id"
                ]
            )

            if experiment_id in index:

                raise RuntimeError(
                    "Duplicate experiment in "
                    "scientist context: "
                    f"{experiment_id}"
                )

            index[
                experiment_id
            ] = experiment

    return index


# ============================================================
# Evidence-source helpers
# ============================================================

def completed_source_experiments(
    evidence_records,
):

    result = set()

    for record in evidence_records:

        provenance = (
            record.get(
                "provenance",
                {},
            )
        )

        experiment_id = (
            provenance.get(
                "experiment_id"
            )
        )

        if (
            experiment_id
            and
            provenance.get(
                "experiment_completed"
            )
            is True
            and
            provenance.get(
                "provenance_complete"
            )
            is True
        ):

            result.add(
                experiment_id
            )

    return sorted(
        result
    )


def count_e3_findings(
    records,
):

    return sum(
        1

        for record
        in records

        if (
            record.get(
                "finding",
                {},
            ).get(
                "evidence_level"
            )
            ==
            "E3"
        )
    )


# ============================================================
# Planned test discovery
# ============================================================

def planned_tests_for_hypothesis(
    *,
    hypothesis_id,
    snapshot,
):

    result = []

    for experiment in (
        snapshot.get(
            "experiments",
            {},
        ).get(
            "planned",
            [],
        )
    ):

        if (
            hypothesis_id
            in
            experiment.get(
                "hypothesis_ids",
                [],
            )
        ):

            result.append(
                {
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
                }
            )

    result.sort(
        key=lambda item:
            item[
                "experiment_id"
            ]
    )

    return result


# ============================================================
# Deterministic status policy
# ============================================================

def evaluate_status(
    *,
    current_status,
    evidence_profile,
    support_source_count,
    weakening_source_count,
    support_e3_count,
    weakening_e3_count,
):

    classification = (
        evidence_profile[
            "classification"
        ]
    )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if classification == "untested":

        return {
            "proposed_status":
                current_status,

            "decision":
                "retain",

            "reason_code":
                "NO_VERIFIED_EVIDENCE",

            "reason":
                (
                    "No verified supporting or "
                    "weakening findings are "
                    "available. Preserve the "
                    "current hypothesis state."
                ),
        }

    # --------------------------------------------------------
    # Support only
    # --------------------------------------------------------

    if classification == "support_only":

        # Conservative promotion rule:
        #
        # At least TWO distinct completed,
        # provenance-complete supporting
        # experiments, with causal E3 support,
        # and no weakening evidence.
        if (
            support_source_count
            >=
            2
            and
            support_e3_count
            >=
            2
            and
            weakening_source_count
            ==
            0
        ):

            return {
                "proposed_status":
                    "supported",

                "decision":
                    "promote",

                "reason_code":
                    (
                        "REPLICATED_CAUSAL_"
                        "SUPPORT"
                    ),

                "reason":
                    (
                        "The hypothesis has "
                        "support-only evidence "
                        "from at least two "
                        "distinct completed "
                        "experiments, including "
                        "replicated E3 causal "
                        "support."
                    ),
            }

        # One experiment is not enough to
        # promote an active hypothesis.
        if current_status in {
            "active",
            "proposed",
            "unresolved",
        }:

            return {
                "proposed_status":
                    "active",

                "decision":
                    "retain_active",

                "reason_code":
                    (
                        "SUPPORT_PRESENT_"
                        "REPLICATION_NEEDED"
                    ),

                "reason":
                    (
                        "Verified support exists, "
                        "but it has not yet been "
                        "replicated across enough "
                        "independent completed "
                        "experiments to promote "
                        "the hypothesis."
                    ),
            }

        return {
            "proposed_status":
                current_status,

            "decision":
                "retain",

            "reason_code":
                (
                    "SUPPORT_PRESENT_"
                    "NO_AUTOMATIC_REVERSAL"
                ),

            "reason":
                (
                    "Supporting evidence exists, "
                    "but the evaluator will not "
                    "automatically reverse an "
                    "existing weakened or "
                    "falsified state without a "
                    "dedicated reconciliation "
                    "step."
                ),
        }

    # --------------------------------------------------------
    # Weakening only
    # --------------------------------------------------------

    if classification == "weakening_only":

        # Extremely conservative falsification
        # rule: replicated E3 weakening from at
        # least two distinct completed sources.
        if (
            weakening_source_count
            >=
            2
            and
            weakening_e3_count
            >=
            2
            and
            support_source_count
            ==
            0
        ):

            return {
                "proposed_status":
                    "falsified",

                "decision":
                    "demote",

                "reason_code":
                    (
                        "REPLICATED_CAUSAL_"
                        "CONTRADICTION"
                    ),

                "reason":
                    (
                        "The hypothesis has "
                        "weakening-only evidence "
                        "from at least two "
                        "distinct completed "
                        "experiments with "
                        "replicated E3 causal "
                        "contradiction."
                    ),
            }

        return {
            "proposed_status":
                "weakened",

            "decision":
                "weaken",

            "reason_code":
                (
                    "VERIFIED_WEAKENING_"
                    "EVIDENCE"
                ),

            "reason":
                (
                    "Verified weakening evidence "
                    "exists, but replication is "
                    "insufficient for automatic "
                    "falsification."
                ),
        }

    # --------------------------------------------------------
    # Mixed evidence
    # --------------------------------------------------------

    if classification == "mixed":

        return {
            "proposed_status":
                "weakened",

            "decision":
                (
                    "retain_weakened"
                    if current_status
                    ==
                    "weakened"
                    else
                    "weaken"
                ),

            "reason_code":
                "MIXED_VERIFIED_EVIDENCE",

            "reason":
                (
                    "Both supporting and "
                    "weakening verified evidence "
                    "exist. The hypothesis should "
                    "remain or become weakened "
                    "until a discriminating "
                    "experiment resolves the "
                    "conflict."
                ),
        }

    raise RuntimeError(
        "Unknown evidence classification: "
        f"{classification}"
    )


# ============================================================
# Action policy
# ============================================================

def determine_next_action(
    *,
    hypothesis_id,
    proposed_status,
    evidence_profile,
    planned_tests,
):

    classification = (
        evidence_profile[
            "classification"
        ]
    )

    if planned_tests:

        return {
            "action":
                (
                    "execute_registered_"
                    "discriminating_test"
                ),

            "priority":
                "high",

            "experiment_id":
                planned_tests[
                    0
                ][
                    "experiment_id"
                ],

            "reason":
                (
                    "A planned experiment is "
                    "already registered for this "
                    "hypothesis and should be "
                    "completed before stronger "
                    "epistemic claims are made."
                ),
        }

    if classification == "mixed":

        return {
            "action":
                (
                    "design_discriminating_"
                    "experiment"
                ),

            "priority":
                "high",

            "experiment_id":
                None,

            "reason":
                (
                    "Mixed evidence requires an "
                    "experiment designed to "
                    "distinguish competing "
                    "explanations."
                ),
        }

    if (
        classification
        ==
        "support_only"
        and
        proposed_status
        !=
        "supported"
    ):

        return {
            "action":
                (
                    "design_replication_"
                    "experiment"
                ),

            "priority":
                "medium",

            "experiment_id":
                None,

            "reason":
                (
                    "Support exists but has not "
                    "met the conservative "
                    "replication requirement."
                ),
        }

    if (
        classification
        ==
        "weakening_only"
        and
        proposed_status
        !=
        "falsified"
    ):

        return {
            "action":
                (
                    "design_replication_"
                    "or_falsification_test"
                ),

            "priority":
                "medium",

            "experiment_id":
                None,

            "reason":
                (
                    "Weakening evidence exists "
                    "but has not met the "
                    "replicated causal threshold "
                    "for falsification."
                ),
        }

    if classification == "untested":

        return {
            "action":
                "design_initial_test",

            "priority":
                "medium",

            "experiment_id":
                None,

            "reason":
                (
                    "The hypothesis has no "
                    "verified empirical test."
                ),
        }

    return {
        "action":
            "monitor",

        "priority":
            "low",

        "experiment_id":
            None,

        "reason":
            (
                "No immediate epistemic action "
                "is required under the current "
                "deterministic policy."
            ),
    }


# ============================================================
# Evaluate one hypothesis
# ============================================================

def evaluate_hypothesis(
    *,
    review,
    snapshot,
):

    hypothesis_id = (
        review[
            "hypothesis_id"
        ]
    )

    current_status = (
        review[
            "current_status"
        ]
    )

    if (
        current_status
        not in
        ALLOWED_STATUSES
    ):

        raise RuntimeError(
            f"{hypothesis_id}: invalid current "
            f"status {current_status!r}."
        )

    evidence_profile = (
        review[
            "evidence_profile"
        ]
    )

    support_records = (
        review.get(
            "supporting_evidence",
            [],
        )
    )

    weakening_records = (
        review.get(
            "weakening_evidence",
            [],
        )
    )

    support_sources = (
        completed_source_experiments(
            support_records
        )
    )

    weakening_sources = (
        completed_source_experiments(
            weakening_records
        )
    )

    support_e3_count = (
        count_e3_findings(
            support_records
        )
    )

    weakening_e3_count = (
        count_e3_findings(
            weakening_records
        )
    )

    decision = evaluate_status(
        current_status=current_status,
        evidence_profile=evidence_profile,
        support_source_count=(
            len(
                support_sources
            )
        ),
        weakening_source_count=(
            len(
                weakening_sources
            )
        ),
        support_e3_count=(
            support_e3_count
        ),
        weakening_e3_count=(
            weakening_e3_count
        ),
    )

    planned_tests = (
        planned_tests_for_hypothesis(
            hypothesis_id=hypothesis_id,
            snapshot=snapshot,
        )
    )

    next_action = determine_next_action(
        hypothesis_id=hypothesis_id,
        proposed_status=(
            decision[
                "proposed_status"
            ]
        ),
        evidence_profile=evidence_profile,
        planned_tests=planned_tests,
    )

    return {
        "hypothesis_id":
            hypothesis_id,

        "title":
            review.get(
                "title"
            ),

        "statement":
            review.get(
                "statement"
            ),

        "current_status":
            current_status,

        "evidence_summary": {
            "classification":
                evidence_profile[
                    "classification"
                ],

            "supporting_findings":
                evidence_profile[
                    "supporting_count"
                ],

            "weakening_findings":
                evidence_profile[
                    "weakening_count"
                ],

            "support_source_experiments":
                support_sources,

            "weakening_source_experiments":
                weakening_sources,

            "support_source_count":
                len(
                    support_sources
                ),

            "weakening_source_count":
                len(
                    weakening_sources
                ),

            "support_e3_finding_count":
                support_e3_count,

            "weakening_e3_finding_count":
                weakening_e3_count,

            "strongest_supporting_level":
                evidence_profile.get(
                    "strongest_supporting_level"
                ),

            "strongest_weakening_level":
                evidence_profile.get(
                    "strongest_weakening_level"
                ),
        },

        "evaluation": {
            "proposed_status":
                decision[
                    "proposed_status"
                ],

            "decision":
                decision[
                    "decision"
                ],

            "reason_code":
                decision[
                    "reason_code"
                ],

            "reason":
                decision[
                    "reason"
                ],

            # Explicitly no score.
            "generated_confidence_score":
                None,
        },

        "planned_tests":
            planned_tests,

        "next_action":
            next_action,

        "open_questions":
            list(
                review.get(
                    "open_questions",
                    [],
                )
            ),

        "checks": {
            "source_review_provenance_complete":
                review[
                    "checks"
                ][
                    "provenance_complete"
                ],

            "source_review_checkpoint_consistent":
                review[
                    "checks"
                ][
                    "checkpoint_consistent"
                ],

            "no_generated_confidence":
                True,

            "knowledge_not_mutated":
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

    review_path = project_path(
        args.review
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
            f"Hypothesis evaluation already "
            f"exists: {output_path}\n"
            "Use --overwrite after evidence "
            "or scientific state changes."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE L8B — HYPOTHESIS "
        "EVALUATOR"
    )

    print(
        "=" * 76
    )

    snapshot = load_json(
        snapshot_path
    )

    review = load_json(
        review_path
    )

    snapshot_sha256 = file_sha256(
        snapshot_path
    )

    review_sha256 = file_sha256(
        review_path
    )

    checks = validate_inputs(
        snapshot=snapshot,
        snapshot_sha256=(
            snapshot_sha256
        ),
        review=review,
    )

    print()
    print(
        "Input identity"
    )

    print(
        "--------------"
    )

    print(
        f"Knowledge state:"
    )

    print(
        snapshot[
            "knowledge_state_sha256"
        ]
    )

    print()

    print(
        f"Scientist context SHA:"
    )

    print(
        snapshot_sha256
    )

    print()

    print(
        f"Evidence review SHA:"
    )

    print(
        review_sha256
    )

    print()

    print(
        "Input checks"
    )

    print(
        "------------"
    )

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name:<42} "
            f"{passed}"
        )

    # ========================================================
    # Evaluate hypotheses
    # ========================================================

    evaluations = []

    print()
    print(
        "Hypothesis evaluations"
    )

    print(
        "----------------------"
    )

    for hypothesis_review in (
        review[
            "hypothesis_reviews"
        ]
    ):

        evaluation = (
            evaluate_hypothesis(
                review=hypothesis_review,
                snapshot=snapshot,
            )
        )

        evaluations.append(
            evaluation
        )

        print()

        print(
            evaluation[
                "hypothesis_id"
            ]
        )

        print(
            f"  current status: "
            f"{evaluation['current_status']}"
        )

        print(
            f"  evidence: "
            f"{evaluation['evidence_summary']['classification']}"
        )

        print(
            f"  support sources: "
            f"{evaluation['evidence_summary']['support_source_count']}"
        )

        print(
            f"  weakening sources: "
            f"{evaluation['evidence_summary']['weakening_source_count']}"
        )

        print(
            f"  proposed status: "
            f"{evaluation['evaluation']['proposed_status']}"
        )

        print(
            f"  decision: "
            f"{evaluation['evaluation']['decision']}"
        )

        print(
            f"  next action: "
            f"{evaluation['next_action']['action']}"
        )

        if (
            evaluation[
                "next_action"
            ][
                "experiment_id"
            ]
            is not None
        ):

            print(
                f"  next experiment: "
                f"{evaluation['next_action']['experiment_id']}"
            )

    # ========================================================
    # Overall checks
    # ========================================================

    hypothesis_ids = [
        evaluation[
            "hypothesis_id"
        ]

        for evaluation
        in evaluations
    ]

    source_review_ids = [
        item[
            "hypothesis_id"
        ]

        for item
        in review[
            "hypothesis_reviews"
        ]
    ]

    all_provenance_complete = all(
        evaluation[
            "checks"
        ][
            "source_review_provenance_complete"
        ]

        for evaluation
        in evaluations
    )

    all_checkpoint_consistent = all(
        evaluation[
            "checks"
        ][
            "source_review_checkpoint_consistent"
        ]

        for evaluation
        in evaluations
    )

    no_confidence_generated = all(
        evaluation[
            "evaluation"
        ][
            "generated_confidence_score"
        ]
        is None

        for evaluation
        in evaluations
    )

    statuses_valid = all(
        evaluation[
            "evaluation"
        ][
            "proposed_status"
        ]
        in
        ALLOWED_STATUSES

        for evaluation
        in evaluations
    )

    planned_experiments = {
        experiment[
            "experiment_id"
        ]

        for experiment
        in snapshot.get(
            "experiments",
            {},
        ).get(
            "planned",
            [],
        )
    }

    recommended_experiments_valid = all(
        (
            evaluation[
                "next_action"
            ][
                "experiment_id"
            ]
            is None

            or

            evaluation[
                "next_action"
            ][
                "experiment_id"
            ]
            in
            planned_experiments
        )

        for evaluation
        in evaluations
    )

    overall_checks = {
        "inputs_valid":
            all(
                checks.values()
            ),

        "all_hypotheses_evaluated":
            (
                sorted(
                    hypothesis_ids
                )
                ==
                sorted(
                    source_review_ids
                )
            ),

        "all_source_provenance_complete":
            all_provenance_complete,

        "all_checkpoint_consistent":
            all_checkpoint_consistent,

        "all_proposed_statuses_valid":
            statuses_valid,

        "planned_experiment_references_valid":
            recommended_experiments_valid,

        "no_confidence_scores_generated":
            no_confidence_generated,

        "knowledge_store_not_mutated":
            True,
    }

    failed = [
        name

        for name, passed
        in overall_checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "L8B hypothesis evaluation "
            f"failed: {failed}"
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
                "deterministic_conservative_"
                "hypothesis_evaluation"
            ),

        "inputs": {
            "scientist_context_path":
                str(
                    snapshot_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "scientist_context_sha256":
                snapshot_sha256,

            "knowledge_state_sha256":
                snapshot[
                    "knowledge_state_sha256"
                ],

            "evidence_review_path":
                str(
                    review_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "evidence_review_sha256":
                review_sha256,
        },

        "policy": {
            "mutates_knowledge_store":
                False,

            "generates_numerical_confidence":
                False,

            "single_experiment_support_is_sufficient_for_supported":
                False,

            "single_experiment_weakening_is_sufficient_for_falsified":
                False,

            "supported_requires_distinct_completed_support_sources":
                2,

            "falsified_requires_distinct_completed_weakening_sources":
                2,

            "replicated_causal_evidence_required_for_terminal_status":
                True,
        },

        "hypothesis_evaluation_count":
            len(
                evaluations
            ),

        "hypothesis_evaluations":
            evaluations,

        "overall_checks":
            overall_checks,
    }

    atomic_write_json(
        output_path,
        output,
    )

    # ========================================================
    # Saved output verification
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
            "Saved hypothesis evaluation "
            "identity mismatch."
        )

    if not all(
        saved[
            "overall_checks"
        ].values()
    ):

        raise RuntimeError(
            "Saved hypothesis evaluation "
            "contains failed checks."
        )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "L8B HYPOTHESIS EVALUATION SUMMARY"
    )

    print(
        "=" * 76
    )

    print()

    for evaluation in evaluations:

        print(
            f"{evaluation['hypothesis_id']}"
        )

        print(
            f"  {evaluation['current_status']} "
            f"→ "
            f"{evaluation['evaluation']['proposed_status']}"
        )

        print(
            f"  reason: "
            f"{evaluation['evaluation']['reason_code']}"
        )

        print(
            f"  action: "
            f"{evaluation['next_action']['action']}"
        )

        if (
            evaluation[
                "next_action"
            ][
                "experiment_id"
            ]
            is not None
        ):

            print(
                f"  experiment: "
                f"{evaluation['next_action']['experiment_id']}"
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
            f"{name:<46} "
            f"{passed}"
        )

    print()

    print(
        "Hypothesis evaluation:"
    )

    print(
        output_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE L8B "
        "HYPOTHESIS EVALUATION COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()
