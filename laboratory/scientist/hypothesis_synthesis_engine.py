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

ENGINE_VERSION = "L11A.1"

EXPECTED_SNAPSHOT_VERSION = "L7D.1"
EXPECTED_EVALUATOR_VERSION = "L8B.1"
EXPECTED_HANDOFF_VERSION = "L8D.1"

SOURCE_HYPOTHESIS_ID = "HYP-V03-001"

EXPECTED_LAST_EXPERIMENT = "EXP-0006"

EXPECTED_FINDINGS = {
    "FND-0008",
    "FND-0009",
    "FND-0010",
}

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

DEFAULT_HANDOFF_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "research_action_handoff.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "hypothesis_synthesis_proposal.json"
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
            "Stage L11A — evidence-grounded "
            "successor-hypothesis synthesis. "
            "Produces proposals only and never "
            "mutates the L7 knowledge store."
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
        "--handoff",
        default=str(
            DEFAULT_HANDOFF_PATH
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
        help=(
            "Replace the prior synthesis "
            "proposal after source scientific "
            "state changes."
        ),
    )

    return parser.parse_args()


# ============================================================
# Snapshot indexing
# ============================================================

def all_snapshot_hypotheses(snapshot):
    hypotheses = []

    for status in (
        "active",
        "supported",
        "weakened",
        "falsified",
    ):

        hypotheses.extend(
            snapshot.get(
                "hypotheses",
                {},
            ).get(
                status,
                [],
            )
        )

    return hypotheses


def hypothesis_index(snapshot):
    index = {}

    for record in all_snapshot_hypotheses(
        snapshot
    ):

        hypothesis_id = record[
            "hypothesis_id"
        ]

        if hypothesis_id in index:
            raise RuntimeError(
                "Duplicate hypothesis in "
                "scientist context: "
                f"{hypothesis_id}"
            )

        index[
            hypothesis_id
        ] = record

    return index


def finding_index(snapshot):
    index = {}

    for hypothesis in (
        all_snapshot_hypotheses(
            snapshot
        )
    ):

        for key in (
            "supporting_findings",
            "weakening_findings",
        ):

            for finding in hypothesis.get(
                key,
                [],
            ):

                finding_id = finding[
                    "finding_id"
                ]

                if (
                    finding_id in index
                    and
                    index[
                        finding_id
                    ]
                    !=
                    finding
                ):

                    raise RuntimeError(
                        "Conflicting finding "
                        "copies in snapshot: "
                        f"{finding_id}"
                    )

                index[
                    finding_id
                ] = finding

    return index


def completed_experiment_index(snapshot):
    index = {}

    experiments = snapshot.get(
        "experiments",
        {},
    )

    for status_key in (
        "completed",
        "planned",
    ):

        for experiment in experiments.get(
            status_key,
            [],
        ):

            experiment_id = experiment[
                "experiment_id"
            ]

            if experiment_id in index:
                raise RuntimeError(
                    "Duplicate experiment in "
                    "scientist context: "
                    f"{experiment_id}"
                )

            index[
                experiment_id
            ] = experiment

    # Snapshot hypothesis records also carry
    # completed tested_by experiment objects.
    for hypothesis in (
        all_snapshot_hypotheses(
            snapshot
        )
    ):

        for experiment in hypothesis.get(
            "tested_by",
            [],
        ):

            experiment_id = experiment[
                "experiment_id"
            ]

            if experiment_id not in index:
                index[
                    experiment_id
                ] = experiment

    return index


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    snapshot,
    snapshot_sha256,
    evaluation,
    evaluation_sha256,
    handoff,
    handoff_sha256,
):

    hypotheses = hypothesis_index(
        snapshot
    )

    findings = finding_index(
        snapshot
    )

    experiments = completed_experiment_index(
        snapshot
    )

    source_hypothesis = hypotheses.get(
        SOURCE_HYPOTHESIS_ID
    )

    source_evaluation = next(
        (
            item
            for item
            in evaluation.get(
                "hypothesis_evaluations",
                [],
            )
            if item.get(
                "hypothesis_id"
            )
            ==
            SOURCE_HYPOTHESIS_ID
        ),
        None,
    )

    # L8D output schema:
    #
    # {
    #   "handoff": {
    #     "hypothesis": {...},
    #     "research_action": {...},
    #     "source_decision": {
    #       "knowledge_state_sha256": ...
    #     }
    #   }
    # }
    #
    # Keep this explicit so L11A is bound to the
    # actual L8D contract rather than guessing at
    # top-level aliases.

    handoff_record = handoff.get(
        "handoff",
        {},
    )

    selected_hypothesis_id = (
        handoff_record.get(
            "hypothesis",
            {},
        ).get(
            "hypothesis_id"
        )
    )

    selected_action = (
        handoff_record.get(
            "research_action",
            {},
        ).get(
            "action"
        )
    )

    handoff_knowledge_state = (
        handoff_record.get(
            "source_decision",
            {},
        ).get(
            "knowledge_state_sha256"
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

        "evaluation_version":
            (
                evaluation.get(
                    "engine_version"
                )
                ==
                EXPECTED_EVALUATOR_VERSION
            ),

        "evaluation_checks":
            (
                bool(
                    evaluation.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    evaluation[
                        "overall_checks"
                    ].values()
                )
            ),

        "handoff_version":
            (
                handoff.get(
                    "engine_version"
                )
                ==
                EXPECTED_HANDOFF_VERSION
            ),

        "handoff_checks":
            (
                bool(
                    handoff.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    handoff[
                        "overall_checks"
                    ].values()
                )
            ),

        "knowledge_state_matches_evaluation":
            (
                evaluation.get(
                    "inputs",
                    {},
                ).get(
                    "knowledge_state_sha256"
                )
                ==
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        "knowledge_state_matches_handoff":
            (
                handoff_knowledge_state
                ==
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        "handoff_selected_hypothesis_matches":
            (
                selected_hypothesis_id
                ==
                SOURCE_HYPOTHESIS_ID
            ),

        "handoff_selected_action_matches":
            (
                selected_action
                ==
                "design_discriminating_experiment"
            ),

        "snapshot_sha_matches_evaluation":
            (
                evaluation.get(
                    "inputs",
                    {},
                ).get(
                    "scientist_context_sha256"
                )
                ==
                snapshot_sha256
            ),

        "source_hypothesis_exists":
            (
                source_hypothesis
                is not None
            ),

        "source_hypothesis_weakened":
            (
                source_hypothesis
                is not None
                and
                source_hypothesis.get(
                    "status"
                )
                ==
                "weakened"
            ),

        "source_evaluation_exists":
            (
                source_evaluation
                is not None
            ),

        "source_evidence_mixed":
            (
                source_evaluation
                is not None
                and
                source_evaluation.get(
                    "evidence_summary",
                    {},
                ).get(
                    "classification"
                )
                ==
                "mixed"
            ),

        "source_action_requests_discrimination":
            (
                source_evaluation
                is not None
                and
                source_evaluation.get(
                    "next_action",
                    {},
                ).get(
                    "action"
                )
                ==
                "design_discriminating_experiment"
            ),

        "exp_0006_completed":
            (
                EXPECTED_LAST_EXPERIMENT
                in
                experiments
                and
                experiments[
                    EXPECTED_LAST_EXPERIMENT
                ].get(
                    "status"
                )
                ==
                "completed"
            ),

        "exp_0006_e3":
            (
                EXPECTED_LAST_EXPERIMENT
                in
                experiments
                and
                experiments[
                    EXPECTED_LAST_EXPERIMENT
                ].get(
                    "evidence_level"
                )
                ==
                "E3"
            ),

        "expected_findings_present":
            (
                EXPECTED_FINDINGS
                <=
                set(
                    findings.keys()
                )
            ),

        "expected_findings_e3":
            all(
                findings[
                    finding_id
                ].get(
                    "evidence_level"
                )
                ==
                "E3"

                for finding_id
                in EXPECTED_FINDINGS
                if finding_id
                in findings
            ),

        "expected_findings_from_exp_0006":
            all(
                findings[
                    finding_id
                ].get(
                    "source_experiment_id"
                )
                ==
                EXPECTED_LAST_EXPERIMENT

                for finding_id
                in EXPECTED_FINDINGS
                if finding_id
                in findings
            ),

        "finding_0009_has_broader_signal":
            (
                "FND-0009"
                in
                findings
                and
                {
                    "L1H0",
                    "L4H0",
                }
                <=
                set(
                    findings[
                        "FND-0009"
                    ].get(
                        "metrics",
                        {},
                    ).get(
                        "broader_token_bias_signal_heads",
                        [],
                    )
                )
            ),

        "finding_0010_has_l3h3_specific_signal":
            (
                "FND-0010"
                in
                findings
                and
                "L3H3"
                in
                findings[
                    "FND-0010"
                ].get(
                    "metrics",
                    {},
                ).get(
                    "four_five_specific_signal_heads",
                    [],
                )
            ),

        "no_planned_successor_experiment":
            (
                snapshot.get(
                    "research_focus",
                    {},
                ).get(
                    "next_recommended_experiment"
                )
                is None
            ),
    }

    # Handoff's exact nesting is less important
    # than the verified L8B action, but record
    # what was observed for transparency.
    observations = {
        "handoff_selected_hypothesis_id":
            selected_hypothesis_id,

        "handoff_selected_action":
            selected_action,

        "handoff_knowledge_state_sha256":
            handoff_knowledge_state,

        "snapshot_sha256":
            snapshot_sha256,

        "evaluation_sha256":
            evaluation_sha256,

        "handoff_sha256":
            handoff_sha256,
    }

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "L11A input validation failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "observations":
            observations,

        "source_hypothesis":
            source_hypothesis,

        "source_evaluation":
            source_evaluation,

        "findings":
            findings,

        "experiments":
            experiments,
    }


# ============================================================
# Successor-hypothesis synthesis
# ============================================================

def build_candidates(
    *,
    source_hypothesis,
    findings,
):

    model_scope = source_hypothesis[
        "model_scope"
    ]

    candidate_1_core = {
        "proposed_hypothesis_id":
            "HYP-V03-003",

        "title":
            (
                "Cross-pair numerical "
                "candidate-selection bias"
            ),

        "statement":
            (
                "In mature MathGPT v0.3, L1H0 "
                "and L4H0 participate in "
                "candidate-token or candidate-pair "
                "selection mechanisms whose causal "
                "directions generalize across "
                "multiple numerical candidate pairs "
                "more reliably than they track "
                "arithmetic correctness."
            ),

        "model_scope":
            model_scope,

        "proposed_status":
            "proposed",

        "relationship_to_source":
            (
                "successor_revision_of_"
                "HYP-V03-001"
            ),

        "source_hypothesis_id":
            SOURCE_HYPOTHESIS_ID,

        "source_finding_ids": [
            "FND-0004",
            "FND-0008",
            "FND-0009",
        ],

        "primary_supporting_finding_ids": [
            "FND-0009",
        ],

        "motivating_disconfirmation_ids": [
            "FND-0004",
            "FND-0008",
        ],

        "mechanistic_targets": [
            "L1H0",
            "L4H0",
        ],

        "open_questions": [
            (
                "Does each head preserve a "
                "consistent lower/upper or "
                "token-identity direction across "
                "number pairs outside P23/P34/"
                "P45/P56?"
            ),
            (
                "Are the effects tied to lexical "
                "number tokens, numerical "
                "magnitude, candidate ordering, "
                "or pair identity?"
            ),
            (
                "Do the same candidate-selection "
                "biases appear in non-arithmetic "
                "contexts using the same number "
                "tokens?"
            ),
        ],

        "tags": [
            "numerical",
            "candidate-selection",
            "token-bias",
            "attention-heads",
            "L1H0",
            "L4H0",
            "causal",
            "mature-v0.3",
        ],

        "confidence": {
            "score":
                None,

            "label":
                "unscored",

            "method":
                "proposal_only",

            "note":
                (
                    "L11A does not assign a "
                    "confidence score. Candidate "
                    "requires registration and "
                    "dedicated testing."
                ),
        },
    }

    candidate_2_core = {
        "proposed_hypothesis_id":
            "HYP-V03-004",

        "title":
            (
                "L3H3 four-versus-five "
                "specialized causal bias"
            ),

        "statement":
            (
                "In mature MathGPT v0.3, L3H3 "
                "exhibits a comparatively "
                "four-versus-five-specific causal "
                "candidate-selection effect that "
                "does not generalize as broadly "
                "across adjacent numerical "
                "candidate pairs."
            ),

        "model_scope":
            model_scope,

        "proposed_status":
            "proposed",

        "relationship_to_source":
            (
                "mechanistic_branch_from_"
                "HYP-V03-001"
            ),

        "source_hypothesis_id":
            SOURCE_HYPOTHESIS_ID,

        "source_finding_ids": [
            "FND-0003",
            "FND-0008",
            "FND-0010",
        ],

        "primary_supporting_finding_ids": [
            "FND-0010",
        ],

        "motivating_disconfirmation_ids": [
            "FND-0008",
        ],

        "mechanistic_targets": [
            "L3H3",
        ],

        "open_questions": [
            (
                "Does the L3H3 four/five effect "
                "replicate across new prompt "
                "families while four and five are "
                "held fixed?"
            ),
            (
                "Does L3H3 respond similarly to "
                "other semantically or lexically "
                "related token pairs?"
            ),
            (
                "Is the apparent specialization "
                "caused by token identity, pair "
                "contrast, or prompt context?"
            ),
        ],

        "tags": [
            "four-vs-five",
            "pair-specific",
            "L3H3",
            "candidate-selection",
            "attention-heads",
            "causal",
            "mature-v0.3",
        ],

        "confidence": {
            "score":
                None,

            "label":
                "unscored",

            "method":
                "proposal_only",

            "note":
                (
                    "L11A does not assign a "
                    "confidence score. Candidate "
                    "requires registration and "
                    "dedicated testing."
                ),
        },
    }

    candidates = []

    for priority, core in enumerate(
        [
            candidate_1_core,
            candidate_2_core,
        ],
        start=1,
    ):

        missing_findings = [
            finding_id
            for finding_id
            in core[
                "source_finding_ids"
            ]
            if finding_id
            not in findings
        ]

        require(
            not missing_findings,
            (
                "L11A candidate references "
                "missing findings: "
                f"{missing_findings}"
            ),
        )

        source_experiments = sorted(
            {
                findings[
                    finding_id
                ][
                    "source_experiment_id"
                ]
                for finding_id
                in core[
                    "source_finding_ids"
                ]
            }
        )

        candidate_fingerprint = (
            "SYN-"
            +
            canonical_json_sha256(
                core
            )[
                :16
            ].upper()
        )

        candidates.append(
            {
                **core,

                "candidate_rank":
                    priority,

                "candidate_fingerprint":
                    candidate_fingerprint,

                "source_experiment_ids":
                    source_experiments,

                "registration_authorized":
                    False,

                "experiment_design_authorized":
                    False,

                "execution_authorized":
                    False,
            }
        )

    return candidates


# ============================================================
# Proposal validation
# ============================================================

def validate_candidates(
    *,
    candidates,
    source_hypothesis,
    findings,
):

    candidate_ids = [
        candidate[
            "proposed_hypothesis_id"
        ]
        for candidate
        in candidates
    ]

    fingerprints = [
        candidate[
            "candidate_fingerprint"
        ]
        for candidate
        in candidates
    ]

    checks = {
        "two_candidates":
            (
                len(
                    candidates
                )
                ==
                2
            ),

        "candidate_ids_unique":
            (
                len(
                    candidate_ids
                )
                ==
                len(
                    set(
                        candidate_ids
                    )
                )
            ),

        "candidate_fingerprints_unique":
            (
                len(
                    fingerprints
                )
                ==
                len(
                    set(
                        fingerprints
                    )
                )
            ),

        "source_hypothesis_is_hyp_001":
            (
                source_hypothesis.get(
                    "hypothesis_id"
                )
                ==
                SOURCE_HYPOTHESIS_ID
            ),

        "all_candidates_proposed_only":
            all(
                candidate[
                    "proposed_status"
                ]
                ==
                "proposed"
                for candidate
                in candidates
            ),

        "all_source_findings_exist":
            all(
                all(
                    finding_id
                    in
                    findings
                    for finding_id
                    in candidate[
                        "source_finding_ids"
                    ]
                )
                for candidate
                in candidates
            ),

        "all_primary_support_is_e3":
            all(
                all(
                    findings[
                        finding_id
                    ].get(
                        "evidence_level"
                    )
                    ==
                    "E3"
                    for finding_id
                    in candidate[
                        "primary_supporting_finding_ids"
                    ]
                )
                for candidate
                in candidates
            ),

        "all_model_scopes_match_source":
            all(
                candidate[
                    "model_scope"
                ]
                ==
                source_hypothesis[
                    "model_scope"
                ]
                for candidate
                in candidates
            ),

        "no_confidence_scores":
            all(
                candidate[
                    "confidence"
                ][
                    "score"
                ]
                is None
                for candidate
                in candidates
            ),

        "registration_not_authorized":
            all(
                candidate[
                    "registration_authorized"
                ]
                is False
                for candidate
                in candidates
            ),

        "experiment_design_not_authorized":
            all(
                candidate[
                    "experiment_design_authorized"
                ]
                is False
                for candidate
                in candidates
            ),

        "execution_not_authorized":
            all(
                candidate[
                    "execution_authorized"
                ]
                is False
                for candidate
                in candidates
            ),

        "candidate_003_targets_l1h0_l4h0":
            (
                set(
                    candidates[
                        0
                    ][
                        "mechanistic_targets"
                    ]
                )
                ==
                {
                    "L1H0",
                    "L4H0",
                }
            ),

        "candidate_004_targets_l3h3":
            (
                candidates[
                    1
                ][
                    "mechanistic_targets"
                ]
                ==
                [
                    "L3H3",
                ]
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
            "L11A candidate validation "
            f"failed: {failed}"
        )

    return checks


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

    handoff_path = project_path(
        args.handoff
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
            "Hypothesis synthesis proposal "
            f"already exists: {output_path}\n"
            "Use --overwrite only after the "
            "scientific source state changes."
        )

    print()
    print("=" * 82)
    print(
        "STAGE L11A — HYPOTHESIS "
        "REVISION & SYNTHESIS PROPOSAL"
    )
    print("=" * 82)

    snapshot = load_json(
        snapshot_path
    )

    evaluation = load_json(
        evaluation_path
    )

    handoff = load_json(
        handoff_path
    )

    snapshot_sha256 = file_sha256(
        snapshot_path
    )

    evaluation_sha256 = file_sha256(
        evaluation_path
    )

    handoff_sha256 = file_sha256(
        handoff_path
    )

    validation = validate_inputs(
        snapshot=snapshot,
        snapshot_sha256=(
            snapshot_sha256
        ),
        evaluation=evaluation,
        evaluation_sha256=(
            evaluation_sha256
        ),
        handoff=handoff,
        handoff_sha256=(
            handoff_sha256
        ),
    )

    input_checks = validation[
        "checks"
    ]

    findings = validation[
        "findings"
    ]

    source_hypothesis = (
        validation[
            "source_hypothesis"
        ]
    )

    print()
    print(
        "Input identity"
    )
    print(
        "--------------"
    )

    print(
        "Knowledge state:"
    )
    print(
        snapshot[
            "knowledge_state_sha256"
        ]
    )

    print()
    print(
        "Scientist context SHA:"
    )
    print(
        snapshot_sha256
    )

    print()
    print(
        "Hypothesis evaluation SHA:"
    )
    print(
        evaluation_sha256
    )

    print()
    print(
        "Research handoff SHA:"
    )
    print(
        handoff_sha256
    )

    print()
    print(
        "Input checks"
    )
    print(
        "------------"
    )

    for name, passed in (
        input_checks.items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    candidates = build_candidates(
        source_hypothesis=(
            source_hypothesis
        ),
        findings=findings,
    )

    candidate_checks = (
        validate_candidates(
            candidates=candidates,
            source_hypothesis=(
                source_hypothesis
            ),
            findings=findings,
        )
    )

    proposal_core = {
        "source_hypothesis_id":
            SOURCE_HYPOTHESIS_ID,

        "source_hypothesis_status":
            source_hypothesis[
                "status"
            ],

        "trigger": {
            "reason":
                (
                    "Existing hypothesis remains "
                    "mixed/weakened after a "
                    "completed discriminating E3 "
                    "experiment, while EXP-0006 "
                    "produced alternative "
                    "mechanistic findings."
                ),

            "last_completed_experiment":
                EXPECTED_LAST_EXPERIMENT,

            "trigger_finding_ids":
                sorted(
                    EXPECTED_FINDINGS
                ),

            "current_l8_action":
                (
                    "design_discriminating_"
                    "experiment"
                ),
        },

        "revision_recommendation": {
            "source_hypothesis":
                SOURCE_HYPOTHESIS_ID,

            "recommended_future_lifecycle_action":
                (
                    "consider_retirement_after_"
                    "successor_registration"
                ),

            "status_change_authorized":
                False,

            "rationale":
                (
                    "EXP-0006 directly tested the "
                    "general arithmetic-correctness "
                    "interpretation and produced "
                    "stronger alternative "
                    "candidate-selection patterns."
                ),
        },

        "candidate_count":
            len(
                candidates
            ),

        "candidates":
            candidates,
    }

    proposal_id = (
        "HSP-"
        +
        canonical_json_sha256(
            proposal_core
        )[
            :16
        ].upper()
    )

    overall_checks = {
        "inputs_valid":
            all(
                input_checks.values()
            ),

        "candidate_design_valid":
            all(
                candidate_checks.values()
            ),

        "source_status_not_mutated":
            True,

        "hypotheses_not_registered":
            True,

        "experiment_registry_not_mutated":
            True,

        "knowledge_store_not_mutated":
            True,

        "confidence_scores_not_generated":
            True,

        "model_not_loaded":
            True,

        "intervention_not_performed":
            True,

        "execution_not_authorized":
            True,
    }

    require(
        all(
            overall_checks.values()
        ),
        (
            "L11A overall validation failed."
        ),
    )

    output = {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "deterministic_evidence_grounded_"
                "successor_hypothesis_proposal"
            ),

        "proposal_id":
            proposal_id,

        "inputs": {
            "scientist_context_path":
                relative_project_path(
                    snapshot_path
                ),

            "scientist_context_sha256":
                snapshot_sha256,

            "knowledge_state_sha256":
                snapshot[
                    "knowledge_state_sha256"
                ],

            "hypothesis_evaluation_path":
                relative_project_path(
                    evaluation_path
                ),

            "hypothesis_evaluation_sha256":
                evaluation_sha256,

            "research_action_handoff_path":
                relative_project_path(
                    handoff_path
                ),

            "research_action_handoff_sha256":
                handoff_sha256,
        },

        "policy": {
            "mutates_knowledge_store":
                False,

            "registers_hypotheses":
                False,

            "changes_hypothesis_status":
                False,

            "registers_experiments":
                False,

            "executes_experiments":
                False,

            "generates_confidence_score":
                False,

            "proposal_requires_future_registration":
                True,
        },

        "input_checks":
            input_checks,

        "candidate_checks":
            candidate_checks,

        "hypothesis_synthesis":
            {
                "proposal_id":
                    proposal_id,

                **proposal_core,
            },

        "overall_checks":
            overall_checks,
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
            "Saved L11A engine identity "
            "mismatch."
        ),
    )

    require(
        saved.get(
            "proposal_id"
        )
        ==
        proposal_id,
        (
            "Saved L11A proposal identity "
            "mismatch."
        ),
    )

    require(
        all(
            saved[
                "overall_checks"
            ].values()
        ),
        (
            "Saved L11A proposal contains "
            "failed checks."
        ),
    )

    print()
    print("=" * 82)
    print(
        "L11A HYPOTHESIS SYNTHESIS SUMMARY"
    )
    print("=" * 82)

    print()
    print(
        "Proposal ID:"
    )
    print(
        f"  {proposal_id}"
    )

    print()
    print(
        "Source hypothesis:"
    )
    print(
        "  HYP-V03-001 — General arithmetic "
        "competing-head circuit"
    )

    print()
    print(
        "Revision recommendation:"
    )
    print(
        "  consider retirement only after "
        "successor hypothesis registration"
    )

    print()
    print(
        "Candidate successor hypotheses"
    )
    print(
        "------------------------------"
    )

    for candidate in candidates:

        print()
        print(
            f"{candidate['candidate_rank']}. "
            f"{candidate['proposed_hypothesis_id']}"
        )

        print(
            f"   {candidate['title']}"
        )

        print(
            f"   targets: "
            f"{', '.join(candidate['mechanistic_targets'])}"
        )

        print(
            f"   primary evidence: "
            f"{', '.join(candidate['primary_supporting_finding_ids'])}"
        )

        print(
            f"   status: "
            f"{candidate['proposed_status']}"
        )

        print(
            f"   registered: False"
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
            f"{name:<42} "
            f"{passed}"
        )

    print()
    print(
        "Proposal artifact:"
    )
    print(
        output_path
    )

    print()
    print("=" * 82)
    print(
        "SUCCESS: STAGE L11A HYPOTHESIS "
        "SYNTHESIS PROPOSAL COMPLETE."
    )
    print(
        "No hypothesis was registered or "
        "retired."
    )
    print("=" * 82)


if __name__ == "__main__":
    main()
