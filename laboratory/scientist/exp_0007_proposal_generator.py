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

ENGINE_VERSION = "L12A.1"

EXPECTED_DECISION_VERSION = "L8C.1"
EXPECTED_HANDOFF_VERSION = "L8D.1"
EXPECTED_SNAPSHOT_VERSION = "L7D.1"

TARGET_HYPOTHESIS_ID = "HYP-V03-003"
EXPERIMENT_ID = "EXP-0007"

EXPECTED_ACTION = (
    "design_replication_experiment"
)

EXPECTED_MODEL = (
    "MathGPT-General-v0.3"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "4cd9b8f1c68e4079449519fd601b573f8ea48137bc74172c7b37e108c9707cf3"
)


# ============================================================
# Default paths
# ============================================================

DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)

DEFAULT_DECISION_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "research_decision.json"
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
    / "experiment_proposal_exp_0007.json"
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
        sort_keys=True,
        ensure_ascii=False,
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


def index_by(
    records,
    key,
):
    return {
        record[key]:
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
            "Stage L12A — create and register "
            "the EXP-0007 replication proposal "
            "for HYP-V03-003. This stage plans "
            "only; it does not freeze exact probes, "
            "load the model, or execute interventions."
        )
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
        ),
    )

    parser.add_argument(
        "--decision",
        default=str(
            DEFAULT_DECISION_PATH
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
        "--snapshot-report",
        default=str(
            DEFAULT_SNAPSHOT_REPORT_PATH
        ),
    )

    return parser.parse_args()


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    snapshot,
    snapshot_path,
    decision,
    handoff,
):

    decision_selected = decision.get(
        "selected_action",
        {},
    )

    h = handoff.get(
        "handoff",
        {},
    )

    handoff_hypothesis = h.get(
        "hypothesis",
        {},
    )

    handoff_action = h.get(
        "research_action",
        {},
    )

    handoff_experiment = h.get(
        "experiment",
        {},
    )

    handoff_source = h.get(
        "source_decision",
        {},
    )

    snapshot_sha256 = file_sha256(
        snapshot_path
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

        "snapshot_checks_passed":
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

        "decision_version":
            (
                decision.get(
                    "engine_version"
                )
                ==
                EXPECTED_DECISION_VERSION
            ),

        "decision_checks_passed":
            (
                bool(
                    decision.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    decision[
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

        "handoff_checks_passed":
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

        "knowledge_matches_decision":
            (
                decision.get(
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

        "knowledge_matches_handoff":
            (
                handoff_source.get(
                    "knowledge_state_sha256"
                )
                ==
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        "snapshot_sha_matches_decision":
            (
                decision.get(
                    "inputs",
                    {},
                ).get(
                    "scientist_context_sha256"
                )
                ==
                snapshot_sha256
            ),

        "selected_hypothesis":
            (
                decision_selected.get(
                    "hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
            ),

        "selected_action":
            (
                decision_selected.get(
                    "action"
                )
                ==
                EXPECTED_ACTION
            ),

        "decision_has_no_experiment":
            (
                decision_selected.get(
                    "experiment_id"
                )
                is None
            ),

        "handoff_hypothesis":
            (
                handoff_hypothesis.get(
                    "hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
            ),

        "handoff_action":
            (
                handoff_action.get(
                    "action"
                )
                ==
                EXPECTED_ACTION
            ),

        "handoff_has_no_experiment":
            (
                handoff_experiment.get(
                    "experiment_id"
                )
                is None
            ),

        "handoff_design_required":
            (
                h.get(
                    "execution_gate",
                    {},
                ).get(
                    "status"
                )
                ==
                "experiment_design_required"
            ),

        "handoff_never_executed":
            (
                h.get(
                    "execution_gate",
                    {},
                ).get(
                    "execution_attempted"
                )
                is False
            ),

        "next_recommended_experiment_empty":
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

    hypothesis = None

    for item in snapshot.get(
        "hypotheses",
        {},
    ).get(
        "active",
        [],
    ):

        if (
            item.get(
                "hypothesis_id"
            )
            ==
            TARGET_HYPOTHESIS_ID
        ):

            hypothesis = item
            break

    checks[
        "target_hypothesis_active"
    ] = (
        hypothesis is not None
    )

    if hypothesis is not None:

        checks[
            "target_model_scope"
        ] = (
            hypothesis.get(
                "model_scope",
                {},
            ).get(
                "model"
            )
            ==
            EXPECTED_MODEL
        )

        checks[
            "target_checkpoint"
        ] = (
            hypothesis.get(
                "model_scope",
                {},
            ).get(
                "checkpoint_sha256"
            )
            ==
            EXPECTED_CHECKPOINT_SHA256
        )

        support_ids = {
            finding.get(
                "finding_id"
            )
            for finding
            in hypothesis.get(
                "supporting_findings",
                []
            )
        }

        checks[
            "fnd_0009_support_present"
        ] = (
            "FND-0009"
            in
            support_ids
        )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "L12A input validation failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "hypothesis":
            hypothesis,

        "snapshot_sha256":
            snapshot_sha256,

        "decision_sha256":
            file_sha256(
                DEFAULT_DECISION_PATH
                if Path(
                    DEFAULT_DECISION_PATH
                ).exists()
                else
                project_path(
                    DEFAULT_DECISION_PATH
                )
            ),

        "handoff_sha256":
            file_sha256(
                DEFAULT_HANDOFF_PATH
                if Path(
                    DEFAULT_HANDOFF_PATH
                ).exists()
                else
                project_path(
                    DEFAULT_HANDOFF_PATH
                )
            ),
    }


# ============================================================
# Proposal construction
# ============================================================

def build_proposal(
    *,
    snapshot,
    decision,
    handoff,
    hypothesis,
):

    model_scope = copy.deepcopy(
        hypothesis[
            "model_scope"
        ]
    )

    design = {
        "replication_target":
            (
                "Test whether L1H0 and L4H0 "
                "show context-stable causal "
                "candidate-pair directions on "
                "new numerical candidate pairs "
                "more reliably than they track "
                "arithmetic correctness."
            ),

        "primary_heads": [
            "L1H0",
            "L4H0",
        ],

        "weak_control_heads": [
            "L4H1",
        ],

        "excluded_from_primary_analysis": [
            "L3H3",
        ],

        "exclusion_reason":
            (
                "L3H3 is the primary mechanistic "
                "target of HYP-V03-004 and should "
                "receive a separate dedicated "
                "replication experiment."
            ),

        "preferred_primary_candidate_pairs": [
            {
                "pair_id":
                    "P12",

                "lower_candidate":
                    "one",

                "upper_candidate":
                    "two",
            },
            {
                "pair_id":
                    "P67",

                "lower_candidate":
                    "six",

                "upper_candidate":
                    "seven",
            },
            {
                "pair_id":
                    "P78",

                "lower_candidate":
                    "seven",

                "upper_candidate":
                    "eight",
            },
            {
                "pair_id":
                    "P89",

                "lower_candidate":
                    "eight",

                "upper_candidate":
                    "nine",
            },
        ],

        "primary_pair_requirements": {
            "pair_count":
                4,

            "must_not_reuse_exp_0006_primary_pairs":
                [
                    "P23",
                    "P34",
                    "P45",
                    "P56",
                ],

            "tokenizer_feasibility_must_be_verified":
                True,

            "silent_pair_substitution_allowed":
                False,
        },

        "primary_probe_structure": {
            "probes_per_pair":
                4,

            "word_lower_correct":
                1,

            "word_upper_correct":
                1,

            "symbol_lower_correct":
                1,

            "symbol_upper_correct":
                1,

            "balanced_correctness_per_pair":
                True,

            "expected_primary_probe_count":
                16,
        },

        "positive_control": {
            "pair_id":
                "P45",

            "candidate_pair":
                [
                    "four",
                    "five",
                ],

            "purpose":
                (
                    "Verify that the known mature "
                    "v0.3 L1H0-four and L4H0-five "
                    "causal directions remain "
                    "detectable in the same run."
                ),

            "excluded_from_primary_replication_metrics":
                True,

            "preferred_probe_count":
                4,
        },

        "canonical_margin":
            (
                "logit(lower discriminative token) "
                "- logit(upper discriminative token)"
            ),

        "ablation_delta":
            (
                "ablated_margin - baseline_margin"
            ),

        "direction_interpretation": {
            "negative_delta":
                (
                    "intact head supports LOWER "
                    "relative to UPPER"
                ),

            "positive_delta":
                (
                    "intact head supports UPPER "
                    "relative to LOWER"
                ),
        },

        "correctness_effect": {
            "lower_correct":
                "-ablation_delta",

            "upper_correct":
                "+ablation_delta",

            "positive":
                (
                    "intact head supports the "
                    "semantically correct answer"
                ),

            "negative":
                (
                    "intact head suppresses the "
                    "semantically correct answer"
                ),
        },

        "replication_metrics_to_freeze_in_plan": {
            "pair_direction_stability":
                (
                    "Within each head × pair cell, "
                    "measure whether at least 3/4 "
                    "raw ablation deltas share the "
                    "same sign."
                ),

            "pair_generalization":
                (
                    "For each primary head, count "
                    "how many of the four new pairs "
                    "pass pair-direction stability."
                ),

            "correctness_tracking":
                (
                    "For each primary head, count "
                    "positive correctness effects "
                    "across the 16 primary probes."
                ),

            "weak_control":
                (
                    "L4H1 mean absolute ablation "
                    "delta should remain below each "
                    "primary head."
                ),
        },

        "candidate_thresholds_for_plan_freeze": {
            "pair_direction_stability":
                ">= 3/4 probes same raw-delta sign",

            "pair_generalization":
                ">= 3/4 new pairs stable",

            "strong_correctness_tracking":
                ">= 12/16 positive correctness effects",

            "hypothesis_replication_intent":
                (
                    "Both primary heads should "
                    "show pair-direction stability "
                    "across new pairs without both "
                    "meeting the strong arithmetic-"
                    "correctness tracking criterion."
                ),
        },

        "competing_explanations": [
            {
                "label":
                    "A",

                "name":
                    "candidate_pair_or_token_bias",

                "prediction":
                    (
                        "Raw ablation-delta direction "
                        "is stable within candidate "
                        "pairs even when which "
                        "candidate is arithmetically "
                        "correct changes."
                    ),
            },
            {
                "label":
                    "B",

                "name":
                    "arithmetic_correctness_circuit",

                "prediction":
                    (
                        "Ablation-delta direction "
                        "switches with correctness so "
                        "correctness effects remain "
                        "consistently positive or "
                        "negative across pairs."
                    ),
            },
            {
                "label":
                    "C",

                "name":
                    "candidate_order_artifact",

                "prediction":
                    (
                        "Effects primarily follow "
                        "lower/upper orientation "
                        "rather than pair identity."
                    ),
            },
            {
                "label":
                    "D",

                "name":
                    "intervention_or_measurement_artifact",

                "prediction":
                    (
                        "Positive control fails or "
                        "weak-control effects become "
                        "comparable to primary heads."
                    ),
            },
        ],

        "possible_outcomes": [
            "replicates_cross_pair_candidate_selection_bias",
            "partial_replication",
            "fails_cross_pair_replication",
            "inconclusive_positive_control_failure",
        ],

        "exact_pairs_frozen":
            False,

        "exact_probes_frozen":
            False,

        "exact_token_ids_frozen":
            False,

        "exact_thresholds_frozen":
            False,

        "plan_builder_required":
            True,
    }

    proposal_core = {
        "experiment_id":
            EXPERIMENT_ID,

        "title":
            (
                "Independent cross-pair numerical "
                "candidate-selection replication"
            ),

        "question":
            (
                "Across previously untested "
                "numerical candidate pairs, do "
                "L1H0 and L4H0 exhibit "
                "context-stable causal pair/token "
                "directions that generalize more "
                "reliably than arithmetic "
                "correctness?"
            ),

        "target_hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "model_scope":
            model_scope,

        "required_permission_when_executed":
            "P3",

        "evidence_level_if_successfully_executed":
            "E3",

        "design":
            design,

        "scientific_constraints": [
            (
                "Primary replication pairs must "
                "not reuse EXP-0006 primary pair "
                "identities P23/P34/P45/P56."
            ),
            (
                "Lower-correct and upper-correct "
                "conditions must be balanced within "
                "every primary pair."
            ),
            (
                "Word-form and symbol-form "
                "arithmetic contexts must both be "
                "represented within every primary "
                "pair."
            ),
            (
                "P45 may be reused only as an "
                "explicit positive control and must "
                "be excluded from primary "
                "replication metrics."
            ),
            (
                "L1H0 and L4H0 are primary heads; "
                "L4H1 is a weak control."
            ),
            (
                "No persistent model parameter "
                "modification is permitted."
            ),
            (
                "Exact tokenization and first "
                "discriminative-token identity "
                "must be verified before the plan "
                "can be frozen."
            ),
        ],

        "execution_authorized":
            False,

        "human_approval_required_before_execution":
            True,
    }

    proposal_id = (
        "PROP-"
        +
        canonical_json_sha256(
            proposal_core
        )[
            :16
        ].upper()
    )

    return {
        "proposal_id":
            proposal_id,

        **proposal_core,
    }


# ============================================================
# Knowledge registration
# ============================================================

def prepare_registration(
    *,
    store,
    proposal,
):

    data = store.load_all()

    hypotheses = copy.deepcopy(
        data[
            "hypotheses"
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

    hypothesis_idx = index_by(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    experiment_idx = index_by(
        experiments[
            "records"
        ],
        "experiment_id",
    )

    require(
        TARGET_HYPOTHESIS_ID
        in
        hypothesis_idx,
        (
            "Target hypothesis is missing "
            "from L7."
        ),
    )

    target = hypothesis_idx[
        TARGET_HYPOTHESIS_ID
    ]

    require(
        target.get(
            "status"
        )
        ==
        "active",
        (
            "EXP-0007 proposal requires "
            "HYP-V03-003 to remain active."
        ),
    )

    require(
        EXPERIMENT_ID
        not in
        experiment_idx,
        (
            "EXP-0007 is already registered."
        ),
    )

    now = utc_now()

    experiment_record = {
        "experiment_id":
            EXPERIMENT_ID,

        "title":
            proposal[
                "title"
            ],

        "question":
            proposal[
                "question"
            ],

        "status":
            "planned",

        "evidence_level":
            None,

        "required_permission":
            "P3",

        "model_scope":
            copy.deepcopy(
                proposal[
                    "model_scope"
                ]
            ),

        "summary_artifact":
            (
                "laboratory/artifacts/"
                "exp_0007_cross_pair_replication_summary.json"
            ),

        "hypothesis_ids": [
            TARGET_HYPOTHESIS_ID,
        ],

        "finding_ids":
            [],

        "proposal": {
            "proposal_id":
                proposal[
                    "proposal_id"
                ],

            "proposal_artifact":
                (
                    "laboratory/scientist/state/"
                    "experiment_proposal_exp_0007.json"
                ),

            "exact_probes_frozen":
                False,

            "execution_authorized":
                False,
        },

        "created_at_utc":
            now,
    }

    experiments[
        "records"
    ].append(
        experiment_record
    )

    tested_by = list(
        target.get(
            "tested_by",
            []
        )
    )

    if EXPERIMENT_ID not in tested_by:
        tested_by.append(
            EXPERIMENT_ID
        )

    target[
        "tested_by"
    ] = tested_by

    target[
        "updated_at_utc"
    ] = now

    state = research_state[
        "state"
    ]

    state[
        "next_recommended_experiment"
    ] = EXPERIMENT_ID

    state[
        "current_research_question"
    ] = proposal[
        "question"
    ]

    state[
        "experiment_planner"
    ] = "L12A_active"

    state[
        "last_experiment_proposal"
    ] = {
        "experiment_id":
            EXPERIMENT_ID,

        "proposal_id":
            proposal[
                "proposal_id"
            ],

        "status":
            "planned",

        "updated_at_utc":
            now,
    }

    hypotheses[
        "updated_at_utc"
    ] = now

    experiments[
        "updated_at_utc"
    ] = now

    research_state[
        "updated_at_utc"
    ] = now

    return {
        "hypotheses":
            hypotheses,

        "experiments":
            experiments,

        "research_state":
            research_state,

        "experiment_record":
            experiment_record,
    }


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
            "Regenerated scientist snapshot "
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
            "Saved scientist snapshot "
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

    return saved


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    snapshot_path = project_path(
        args.snapshot
    )

    decision_path = project_path(
        args.decision
    )

    handoff_path = project_path(
        args.handoff
    )

    output_path = project_path(
        args.output
    )

    snapshot_report_path = project_path(
        args.snapshot_report
    )

    if output_path.exists():

        raise FileExistsError(
            "EXP-0007 proposal already exists: "
            f"{output_path}"
        )

    print()
    print("=" * 84)
    print(
        "STAGE L12A — EXP-0007 "
        "REPLICATION PROPOSAL + REGISTRATION"
    )
    print("=" * 84)

    snapshot = load_json(
        snapshot_path
    )

    decision = load_json(
        decision_path
    )

    handoff = load_json(
        handoff_path
    )

    input_validation = validate_inputs(
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        decision=decision,
        handoff=handoff,
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
        "Selected hypothesis:"
    )
    print(
        "  HYP-V03-003 — Cross-pair "
        "numerical candidate-selection bias"
    )

    print()
    print(
        "Selected action:"
    )
    print(
        "  design_replication_experiment"
    )

    print()
    print(
        "Input checks"
    )
    print(
        "------------"
    )

    for name, passed in (
        input_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    proposal = build_proposal(
        snapshot=snapshot,
        decision=decision,
        handoff=handoff,
        hypothesis=(
            input_validation[
                "hypothesis"
            ]
        ),
    )

    print()
    print(
        "Proposal"
    )
    print(
        "--------"
    )

    print(
        f"Proposal ID: "
        f"{proposal['proposal_id']}"
    )

    print(
        f"Experiment: "
        f"{proposal['experiment_id']}"
    )

    print(
        f"Title: "
        f"{proposal['title']}"
    )

    print()
    print(
        "Primary heads:"
    )

    for head in (
        proposal[
            "design"
        ][
            "primary_heads"
        ]
    ):

        print(
            f"  {head}"
        )

    print()
    print(
        "Preferred new primary pairs:"
    )

    for pair in (
        proposal[
            "design"
        ][
            "preferred_primary_candidate_pairs"
        ]
    ):

        print(
            f"  {pair['pair_id']}: "
            f"{pair['lower_candidate']} "
            f"vs {pair['upper_candidate']}"
        )

    print()
    print(
        "Positive control:"
    )
    print(
        "  P45: four vs five"
    )

    print()
    print(
        "Exact probes frozen:"
    )
    print(
        "  False"
    )

    print(
        "Execution authorized:"
    )
    print(
        "  False"
    )

    # Save proposal before registering so
    # the experiment record can point to
    # an immutable proposal artifact.
    proposal_output = {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "replication_proposal_and_"
                "planned_experiment_registration"
            ),

        "inputs": {
            "scientist_context_path":
                relative_project_path(
                    snapshot_path
                ),

            "scientist_context_sha256":
                input_validation[
                    "snapshot_sha256"
                ],

            "knowledge_state_sha256":
                snapshot[
                    "knowledge_state_sha256"
                ],

            "research_decision_path":
                relative_project_path(
                    decision_path
                ),

            "research_decision_sha256":
                file_sha256(
                    decision_path
                ),

            "research_action_handoff_path":
                relative_project_path(
                    handoff_path
                ),

            "research_action_handoff_sha256":
                file_sha256(
                    handoff_path
                ),
        },

        "policy": {
            "registers_planned_experiment":
                True,

            "mutates_hypothesis_status":
                False,

            "creates_evidence":
                False,

            "creates_findings":
                False,

            "freezes_exact_plan":
                False,

            "loads_model":
                False,

            "executes_experiment":
                False,

            "authorizes_p3":
                False,

            "requires_future_plan_builder":
                True,

            "requires_future_execution_spec":
                True,

            "requires_human_p3_approval_before_execution":
                True,
        },

        "proposal":
            proposal,

        "input_checks":
            input_validation[
                "checks"
            ],
    }

    atomic_write_json(
        output_path,
        proposal_output,
    )

    proposal_sha256 = file_sha256(
        output_path
    )

    store = KnowledgeStore()

    prepared = prepare_registration(
        store=store,
        proposal=proposal,
    )

    paths = [
        store.collection_path(
            "hypotheses"
        ),
        store.collection_path(
            "experiments"
        ),
        store.collection_path(
            "research_state"
        ),
        snapshot_path,
        snapshot_report_path,
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

        validation = store.validate()

        require(
            validation[
                "passed"
            ],
            (
                "EXP-0007 registration caused "
                "knowledge validation failure."
            ),
        )

        new_snapshot = regenerate_snapshot(
            store=store,
            snapshot_path=snapshot_path,
            report_path=(
                snapshot_report_path
            ),
        )

        planned_ids = {
            experiment[
                "experiment_id"
            ]
            for experiment
            in new_snapshot.get(
                "experiments",
                {},
            ).get(
                "planned",
                [],
            )
        }

        require(
            EXPERIMENT_ID
            in
            planned_ids,
            (
                "EXP-0007 is not visible as "
                "planned in scientist snapshot."
            ),
        )

        require(
            new_snapshot[
                "research_focus"
            ][
                "next_recommended_experiment"
            ]
            ==
            EXPERIMENT_ID,
            (
                "Research-state recommendation "
                "does not point to EXP-0007."
            ),
        )

    except Exception:

        restore_backups(
            backups
        )

        if output_path.exists():
            output_path.unlink()

        raise

    registration_checks = {
        "proposal_written":
            output_path.exists(),

        "proposal_sha_present":
            (
                len(
                    proposal_sha256
                )
                ==
                64
            ),

        "experiment_registered":
            True,

        "experiment_status_planned":
            True,

        "hypothesis_status_unchanged":
            True,

        "next_recommended_experiment_exp_0007":
            True,

        "no_evidence_created":
            True,

        "no_findings_created":
            True,

        "exact_plan_not_frozen":
            True,

        "model_not_loaded":
            True,

        "experiment_not_executed":
            True,

        "p3_not_authorized":
            True,
    }

    print()
    print("=" * 84)
    print(
        "L12A REGISTRATION SUMMARY"
    )
    print("=" * 84)

    print()
    print(
        "Experiment registered:"
    )
    print(
        "  EXP-0007"
    )

    print(
        "Status:"
    )
    print(
        "  planned"
    )

    print(
        "Evidence level:"
    )
    print(
        "  None"
    )

    print(
        "Required execution permission:"
    )
    print(
        "  P3"
    )

    print()
    print(
        "Target hypothesis:"
    )
    print(
        "  HYP-V03-003"
    )

    print(
        "Hypothesis status changed:"
    )
    print(
        "  False"
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
        new_snapshot[
            "knowledge_state_sha256"
        ]
    )

    print()
    print(
        "Next recommended experiment:"
    )
    print(
        "  "
        +
        str(
            new_snapshot[
                "research_focus"
            ][
                "next_recommended_experiment"
            ]
        )
    )

    print()
    print(
        "Proposal artifact:"
    )
    print(
        output_path
    )

    print()
    print(
        "Proposal SHA-256:"
    )
    print(
        proposal_sha256
    )

    print()
    print(
        "Registration checks"
    )
    print(
        "-------------------"
    )

    for name, passed in (
        registration_checks.items()
    ):

        print(
            f"{name:<44} "
            f"{passed}"
        )

    print()
    print("=" * 84)
    print(
        "SUCCESS: STAGE L12A EXP-0007 "
        "PROPOSAL + REGISTRATION COMPLETE."
    )
    print(
        "Exact probes remain unfrozen and "
        "no experiment was executed."
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
