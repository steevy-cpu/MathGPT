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

ENGINE_VERSION = "L10B.1"
EXPECTED_PROPOSAL_VERSION = "L10A.1"
EXPECTED_EXPERIMENT_ID = "EXP-0006"
EXPECTED_HYPOTHESIS_ID = "HYP-V03-001"

DEFAULT_PROPOSAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_proposal.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0006.json"
)

DEFAULT_EXPERIMENT_REGISTRY_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "experiment_registry.json"
)

DEFAULT_HYPOTHESES_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "hypotheses.json"
)

DEFAULT_RESEARCH_STATE_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "research_state.json"
)


# ============================================================
# Frozen target heads
# ============================================================

TARGET_HEADS = [
    {
        "layer": 1,
        "head": 0,
        "head_id": "L1H0",
        "hypothesis_role": "predicted_correct_answer_supporter",
    },
    {
        "layer": 3,
        "head": 3,
        "head_id": "L3H3",
        "hypothesis_role": "predicted_correct_answer_suppressor",
    },
    {
        "layer": 4,
        "head": 0,
        "head_id": "L4H0",
        "hypothesis_role": "predicted_correct_answer_suppressor",
    },
    {
        "layer": 4,
        "head": 1,
        "head_id": "L4H1",
        "hypothesis_role": "weak_control",
    },
]


# ============================================================
# Frozen candidate-pair families
#
# Each pair has:
#   lower candidate
#   upper candidate
#
# Every pair receives four probes:
#   1) words, lower correct
#   2) words, upper correct
#   3) symbols, lower correct
#   4) symbols, upper correct
#
# This orthogonalizes:
#   candidate-token identity
#   semantic correctness
#   prompt format
# ============================================================

PAIR_FAMILIES = [
    {
        "pair_id": "P23",
        "label": "two_vs_three",
        "lower_label": "two",
        "upper_label": "three",
        "lower_candidate": " two.",
        "upper_candidate": " three.",
        "role": "non_control_pair",
        "word_lower_prompt": "One plus one equals",
        "word_upper_prompt": "One plus two equals",
        "symbol_lower_prompt": "1 + 1 =",
        "symbol_upper_prompt": "1 + 2 =",
    },
    {
        "pair_id": "P34",
        "label": "three_vs_four",
        "lower_label": "three",
        "upper_label": "four",
        "lower_candidate": " three.",
        "upper_candidate": " four.",
        "role": "non_control_pair",
        "word_lower_prompt": "One plus two equals",
        "word_upper_prompt": "Two plus two equals",
        "symbol_lower_prompt": "1 + 2 =",
        "symbol_upper_prompt": "2 + 2 =",
    },
    {
        "pair_id": "P45",
        "label": "four_vs_five",
        "lower_label": "four",
        "upper_label": "five",
        "lower_candidate": " four.",
        "upper_candidate": " five.",
        "role": "positive_control_pair",
        "word_lower_prompt": "Two plus two equals",
        "word_upper_prompt": "Two plus three equals",
        "symbol_lower_prompt": "2 + 2 =",
        "symbol_upper_prompt": "2 + 3 =",
    },
    {
        "pair_id": "P56",
        "label": "five_vs_six",
        "lower_label": "five",
        "upper_label": "six",
        "lower_candidate": " five.",
        "upper_candidate": " six.",
        "role": "non_control_pair",
        "word_lower_prompt": "Two plus three equals",
        "word_upper_prompt": "Three plus three equals",
        "symbol_lower_prompt": "2 + 3 =",
        "symbol_upper_prompt": "3 + 3 =",
    },
]


def build_probes():
    probes = []

    for pair in PAIR_FAMILIES:

        pair_id = pair["pair_id"]

        probes.extend(
            [
                {
                    "probe_id": f"{pair_id}-W-L",
                    "pair_id": pair_id,
                    "pair_label": pair["label"],
                    "pair_role": pair["role"],
                    "format": "words",
                    "prompt": pair["word_lower_prompt"],
                    "lower_candidate": pair["lower_candidate"],
                    "upper_candidate": pair["upper_candidate"],
                    "semantically_correct": pair["lower_label"],
                    "correct_side": "lower",
                },
                {
                    "probe_id": f"{pair_id}-W-U",
                    "pair_id": pair_id,
                    "pair_label": pair["label"],
                    "pair_role": pair["role"],
                    "format": "symbols",
                    "prompt": pair["word_upper_prompt"],
                    "lower_candidate": pair["lower_candidate"],
                    "upper_candidate": pair["upper_candidate"],
                    "semantically_correct": pair["upper_label"],
                    "correct_side": "upper",
                },
                {
                    "probe_id": f"{pair_id}-S-L",
                    "pair_id": pair_id,
                    "pair_label": pair["label"],
                    "pair_role": pair["role"],
                    "format": "symbols",
                    "prompt": pair["symbol_lower_prompt"],
                    "lower_candidate": pair["lower_candidate"],
                    "upper_candidate": pair["upper_candidate"],
                    "semantically_correct": pair["lower_label"],
                    "correct_side": "lower",
                },
                {
                    "probe_id": f"{pair_id}-S-U",
                    "pair_id": pair_id,
                    "pair_label": pair["label"],
                    "pair_role": pair["role"],
                    "format": "symbols",
                    "prompt": pair["symbol_upper_prompt"],
                    "lower_candidate": pair["lower_candidate"],
                    "upper_candidate": pair["upper_candidate"],
                    "semantically_correct": pair["upper_label"],
                    "correct_side": "upper",
                },
            ]
        )

    # Correct the one intentionally generated field above:
    # word upper probes are word-format probes.
    for probe in probes:
        if probe["probe_id"].endswith("-W-U"):
            probe["format"] = "words"

    return probes


PROBES = build_probes()


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


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L10B — freeze the exact "
            "preregistered EXP-0006 design "
            "without executing the experiment "
            "or granting P3 approval."
        )
    )

    parser.add_argument(
        "--proposal",
        default=str(
            DEFAULT_PROPOSAL_PATH
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--experiment-registry",
        default=str(
            DEFAULT_EXPERIMENT_REGISTRY_PATH
        ),
    )

    parser.add_argument(
        "--hypotheses",
        default=str(
            DEFAULT_HYPOTHESES_PATH
        ),
    )

    parser.add_argument(
        "--research-state",
        default=str(
            DEFAULT_RESEARCH_STATE_PATH
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing derived "
            "EXP-0006 preregistration plan."
        ),
    )

    return parser.parse_args()


# ============================================================
# Source validation
# ============================================================

def validate_sources(
    *,
    proposal_artifact,
    experiment_registry,
    hypotheses,
    research_state,
):

    proposal = proposal_artifact.get(
        "proposal",
        {},
    )

    registration = proposal_artifact.get(
        "registration",
        {},
    )

    experiments = index_by(
        experiment_registry[
            "records"
        ],
        "experiment_id",
    )

    hypothesis_index = index_by(
        hypotheses[
            "records"
        ],
        "hypothesis_id",
    )

    experiment = experiments.get(
        EXPECTED_EXPERIMENT_ID
    )

    hypothesis = hypothesis_index.get(
        EXPECTED_HYPOTHESIS_ID
    )

    state = research_state[
        "state"
    ]

    checks = {
        "proposal_version":
            (
                proposal_artifact.get(
                    "engine_version"
                )
                ==
                EXPECTED_PROPOSAL_VERSION
            ),

        "proposal_checks_present":
            bool(
                proposal_artifact.get(
                    "overall_checks"
                )
            ),

        "proposal_checks_passed":
            (
                bool(
                    proposal_artifact.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    proposal_artifact[
                        "overall_checks"
                    ].values()
                )
            ),

        "proposal_is_exp_0006":
            (
                proposal.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "proposal_targets_hyp_001":
            (
                proposal.get(
                    "target_hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "proposal_requires_p3":
            (
                proposal.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "proposal_does_not_authorize_execution":
            (
                proposal.get(
                    "execution_authorized"
                )
                is False
            ),

        "proposal_was_registered":
            (
                registration.get(
                    "registered"
                )
                is True
            ),

        "experiment_exists":
            isinstance(
                experiment,
                dict,
            ),

        "experiment_planned":
            (
                isinstance(
                    experiment,
                    dict,
                )
                and
                experiment.get(
                    "status"
                )
                ==
                "planned"
            ),

        "experiment_has_no_evidence":
            (
                isinstance(
                    experiment,
                    dict,
                )
                and
                experiment.get(
                    "evidence_level"
                )
                is None
                and
                experiment.get(
                    "summary_artifact"
                )
                is None
                and
                experiment.get(
                    "finding_ids"
                )
                ==
                []
            ),

        "experiment_proposal_binding":
            (
                isinstance(
                    experiment,
                    dict,
                )
                and
                experiment.get(
                    "proposal_id"
                )
                ==
                proposal_artifact.get(
                    "proposal_id"
                )
            ),

        "hypothesis_exists":
            isinstance(
                hypothesis,
                dict,
            ),

        "hypothesis_still_weakened":
            (
                isinstance(
                    hypothesis,
                    dict,
                )
                and
                hypothesis.get(
                    "status"
                )
                ==
                "weakened"
            ),

        "hypothesis_tested_by_exp_0006":
            (
                isinstance(
                    hypothesis,
                    dict,
                )
                and
                EXPECTED_EXPERIMENT_ID
                in
                hypothesis.get(
                    "tested_by",
                    []
                )
            ),

        "research_state_recommends_exp_0006":
            (
                state.get(
                    "next_recommended_experiment"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "model_scope_match":
            (
                isinstance(
                    experiment,
                    dict,
                )
                and
                proposal.get(
                    "model_scope"
                )
                ==
                experiment.get(
                    "model_scope"
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
            "L10B source validation failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "experiment":
            experiment,

        "hypothesis":
            hypothesis,
    }


# ============================================================
# Design validation
# ============================================================

def validate_design():

    pair_ids = [
        pair[
            "pair_id"
        ]
        for pair
        in PAIR_FAMILIES
    ]

    probe_ids = [
        probe[
            "probe_id"
        ]
        for probe
        in PROBES
    ]

    pair_counts = {
        pair_id:
            sum(
                1
                for probe
                in PROBES
                if probe[
                    "pair_id"
                ]
                ==
                pair_id
            )
        for pair_id
        in pair_ids
    }

    lower_correct = sum(
        1
        for probe
        in PROBES
        if probe[
            "correct_side"
        ]
        ==
        "lower"
    )

    upper_correct = sum(
        1
        for probe
        in PROBES
        if probe[
            "correct_side"
        ]
        ==
        "upper"
    )

    word_count = sum(
        1
        for probe
        in PROBES
        if probe[
            "format"
        ]
        ==
        "words"
    )

    symbol_count = sum(
        1
        for probe
        in PROBES
        if probe[
            "format"
        ]
        ==
        "symbols"
    )

    control_pairs = [
        pair
        for pair
        in PAIR_FAMILIES
        if pair[
            "role"
        ]
        ==
        "positive_control_pair"
    ]

    non_control_pairs = [
        pair
        for pair
        in PAIR_FAMILIES
        if pair[
            "role"
        ]
        ==
        "non_control_pair"
    ]

    checks = {
        "pair_ids_unique":
            (
                len(
                    pair_ids
                )
                ==
                len(
                    set(
                        pair_ids
                    )
                )
            ),

        "probe_ids_unique":
            (
                len(
                    probe_ids
                )
                ==
                len(
                    set(
                        probe_ids
                    )
                )
            ),

        "pair_count_is_4":
            (
                len(
                    PAIR_FAMILIES
                )
                ==
                4
            ),

        "probe_count_is_16":
            (
                len(
                    PROBES
                )
                ==
                16
            ),

        "four_probes_per_pair":
            all(
                count
                ==
                4

                for count
                in pair_counts.values()
            ),

        "lower_correct_count_is_8":
            (
                lower_correct
                ==
                8
            ),

        "upper_correct_count_is_8":
            (
                upper_correct
                ==
                8
            ),

        "word_count_is_8":
            (
                word_count
                ==
                8
            ),

        "symbol_count_is_8":
            (
                symbol_count
                ==
                8
            ),

        "exactly_one_positive_control_pair":
            (
                len(
                    control_pairs
                )
                ==
                1
                and
                control_pairs[
                    0
                ][
                    "pair_id"
                ]
                ==
                "P45"
            ),

        "three_non_control_pairs":
            (
                len(
                    non_control_pairs
                )
                ==
                3
            ),

        "all_candidates_nonempty":
            all(
                bool(
                    pair[
                        "lower_candidate"
                    ]
                )
                and
                bool(
                    pair[
                        "upper_candidate"
                    ]
                )

                for pair
                in PAIR_FAMILIES
            ),

        "all_prompts_nonempty":
            all(
                bool(
                    probe[
                        "prompt"
                    ]
                )

                for probe
                in PROBES
            ),

        "all_correct_sides_valid":
            all(
                probe[
                    "correct_side"
                ]
                in
                {
                    "lower",
                    "upper",
                }

                for probe
                in PROBES
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
            "L10B design validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Preregistered measurement definitions
# ============================================================

MEASUREMENT_DEFINITION = {
    "canonical_pair_margin":
        (
            "logit(lower_candidate_"
            "discriminative_token) - "
            "logit(upper_candidate_"
            "discriminative_token)"
        ),

    "ablation_delta":
        (
            "ablated_canonical_pair_margin "
            "- baseline_canonical_pair_margin"
        ),

    "negative_delta_interpretation":
        (
            "The intact head causally supports "
            "the lower candidate relative to "
            "the upper candidate."
        ),

    "positive_delta_interpretation":
        (
            "The intact head causally supports "
            "the upper candidate relative to "
            "the lower candidate."
        ),

    "correctness_effect":
        (
            "If lower is correct: -delta. "
            "If upper is correct: +delta."
        ),

    "positive_correctness_effect_interpretation":
        (
            "The intact head causally supports "
            "the semantically correct candidate."
        ),

    "negative_correctness_effect_interpretation":
        (
            "The intact head causally suppresses "
            "the semantically correct candidate."
        ),

    "fixed_pair_direction":
        (
            "Within one candidate pair, a head "
            "shows fixed-pair direction when its "
            "delta sign remains the same even "
            "when semantic correctness flips."
        ),

    "tokenization_rule":
        (
            "For each candidate pair, measure "
            "the first tokenizer position where "
            "the lower and upper candidate "
            "continuations differ. The execution "
            "stage must verify that both "
            "candidates have a valid "
            "discriminative position before "
            "running interventions."
        ),
}


# ============================================================
# Preregistered decision rules
# ============================================================

DECISION_RULES = {
    "global_probe_count":
        16,

    "global_directional_threshold":
        0.75,

    "minimum_global_consistent_probes":
        12,

    "per_pair_probe_count":
        4,

    "per_pair_directional_threshold":
        0.75,

    "minimum_per_pair_consistent_probes":
        3,

    "positive_control": {
        "pair_id":
            "P45",

        "L1H0_expected_delta_sign":
            "negative",

        "L4H0_expected_delta_sign":
            "positive",

        "criterion":
            (
                "L1H0 must have negative delta "
                "on at least 3 of 4 P45 probes "
                "and L4H0 must have positive "
                "delta on at least 3 of 4 P45 "
                "probes."
            ),

        "failure_outcome":
            (
                "inconclusive_positive_control_"
                "failure"
            ),
    },

    "hypothesis_primary_tests": {
        "L1H0": {
            "prediction":
                (
                    "Supports the semantically "
                    "correct arithmetic answer."
                ),

            "passing_correctness_effect":
                "positive",

            "global_criterion":
                (
                    "Positive correctness_effect "
                    "on at least 12 of 16 probes."
                ),

            "pair_generalization_criterion":
                (
                    "Mean correctness_effect is "
                    "positive in at least 3 of 4 "
                    "candidate pairs."
                ),
        },

        "L3H3": {
            "prediction":
                (
                    "Suppresses the semantically "
                    "correct arithmetic answer."
                ),

            "passing_correctness_effect":
                "negative",

            "global_criterion":
                (
                    "Negative correctness_effect "
                    "on at least 12 of 16 probes."
                ),

            "pair_generalization_criterion":
                (
                    "Mean correctness_effect is "
                    "negative in at least 3 of 4 "
                    "candidate pairs."
                ),
        },

        "L4H0": {
            "prediction":
                (
                    "Suppresses the semantically "
                    "correct arithmetic answer."
                ),

            "passing_correctness_effect":
                "negative",

            "global_criterion":
                (
                    "Negative correctness_effect "
                    "on at least 12 of 16 probes."
                ),

            "pair_generalization_criterion":
                (
                    "Mean correctness_effect is "
                    "negative in at least 3 of 4 "
                    "candidate pairs."
                ),
        },
    },

    "primary_head_pass_rule":
        (
            "A primary head passes only if both "
            "its global criterion and its "
            "pair-generalization criterion pass."
        ),

    "hypothesis_outcomes": {
        "all_three_primary_heads_pass":
            (
                "supports_general_arithmetic_"
                "competing_head_circuit"
            ),

        "one_or_two_primary_heads_pass":
            (
                "partial_support_general_"
                "arithmetic_competing_head_circuit"
            ),

        "zero_primary_heads_pass":
            (
                "further_weakens_general_"
                "arithmetic_competing_head_circuit"
            ),
    },

    "pair_specific_token_bias_diagnostic": {
        "criterion":
            (
                "For a given head and candidate "
                "pair, at least 3 of 4 probes "
                "share the same delta sign even "
                "though correctness is balanced "
                "2 lower / 2 upper."
            ),

        "broader_token_bias_signal":
            (
                "At least two of the three "
                "non-control pairs show a "
                "fixed-pair direction for the "
                "same primary head."
            ),

        "four_five_specific_signal":
            (
                "The P45 positive control shows "
                "its preregistered fixed "
                "directions while fewer than two "
                "non-control pairs show analogous "
                "fixed-pair direction."
            ),
    },

    "weak_control": {
        "head_id":
            "L4H1",

        "criterion":
            (
                "Global mean absolute delta for "
                "L4H1 must be smaller than the "
                "global mean absolute delta of "
                "each primary head: L1H0, L3H3, "
                "and L4H0."
            ),

        "role":
            (
                "Quality-control diagnostic only; "
                "it does not independently decide "
                "the HYP-V03-001 outcome."
            ),
    },
}


# ============================================================
# Build preregistered plan
# ============================================================

def build_plan(
    *,
    proposal_artifact,
    proposal_sha256,
    experiment_record,
    hypothesis_record,
):

    model_scope = experiment_record[
        "model_scope"
    ]

    checkpoint_sha256 = model_scope.get(
        "checkpoint_sha256"
    )

    require(
        isinstance(
            checkpoint_sha256,
            str,
        )
        and
        len(
            checkpoint_sha256
        )
        ==
        64,
        (
            "EXP-0006 model checkpoint SHA-256 "
            "is invalid."
        ),
    )

    plan_core = {
        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "title":
            experiment_record.get(
                "title"
            ),

        "question":
            experiment_record.get(
                "question"
            ),

        "hypothesis_id":
            EXPECTED_HYPOTHESIS_ID,

        "hypothesis_statement":
            hypothesis_record.get(
                "statement"
            ),

        "hypothesis_status_at_preregistration":
            hypothesis_record.get(
                "status"
            ),

        "model_scope":
            model_scope,

        "required_permission":
            "P3",

        "target_heads":
            TARGET_HEADS,

        "candidate_pair_design": {
            "pair_count":
                len(
                    PAIR_FAMILIES
                ),

            "positive_control_pair":
                "P45",

            "non_control_pair_count":
                3,

            "pairs":
                PAIR_FAMILIES,
        },

        "probe_design": {
            "design_type":
                (
                    "balanced_cross_pair_"
                    "correctness_vs_token_identity"
                ),

            "probe_count":
                len(
                    PROBES
                ),

            "lower_correct_count":
                8,

            "upper_correct_count":
                8,

            "word_probe_count":
                8,

            "symbol_probe_count":
                8,

            "probes":
                PROBES,
        },

        "measurement_definition":
            MEASUREMENT_DEFINITION,

        "intervention": {
            "type":
                "attention_head_ablation",

            "baseline_scale":
                1.0,

            "ablation_scale":
                0.0,

            "scale_one_exactness_control":
                True,

            "persistent_weight_modification":
                False,

            "model_reload_between_heads":
                False,

            "model_reload_between_probes":
                False,
        },

        "decision_rules":
            DECISION_RULES,

        "scientific_interpretation_guardrails": [
            (
                "Do not interpret EXP-0006 if "
                "the P45 positive control fails."
            ),
            (
                "Do not infer a general arithmetic "
                "circuit from one candidate pair."
            ),
            (
                "Do not treat fixed-pair direction "
                "as arithmetic correctness support."
            ),
            (
                "Keep candidate-pair orientation "
                "fixed as lower-minus-upper for "
                "every baseline and intervention."
            ),
            (
                "Do not change prompts, candidate "
                "strings, head order, thresholds, "
                "or outcome rules after execution "
                "begins."
            ),
            (
                "L4H1 is a weak-control diagnostic "
                "and is not a primary hypothesis "
                "head."
            ),
        ],

        "execution_policy": {
            "execution_authorized":
                False,

            "human_approval_required":
                True,

            "approval_must_bind_exact_plan":
                True,

            "approval_must_bind_exact_execution_spec":
                True,

            "plan_does_not_execute":
                True,

            "plan_does_not_mutate_knowledge_store":
                True,

            "plan_does_not_create_evidence":
                True,
        },
    }

    plan_fingerprint = canonical_json_sha256(
        plan_core
    )

    plan_id = (
        "PLAN-"
        +
        plan_fingerprint[
            :16
        ].upper()
    )

    return {
        "engine_version":
            ENGINE_VERSION,

        "created_at_utc":
            utc_now(),

        "mode":
            (
                "preregistered_discriminating_"
                "experiment_plan"
            ),

        "source_proposal": {
            "path":
                relative_project_path(
                    DEFAULT_PROPOSAL_PATH
                ),

            "sha256":
                proposal_sha256,

            "proposal_id":
                proposal_artifact[
                    "proposal_id"
                ],

            "source_knowledge_state_sha256":
                proposal_artifact[
                    "registration"
                ][
                    "new_knowledge_state_sha256"
                ],
        },

        "experiment_plan": {
            "plan_id":
                plan_id,

            "plan_fingerprint_sha256":
                plan_fingerprint,

            "plan":
                plan_core,
        },
    }


# ============================================================
# Plan validation
# ============================================================

def validate_plan(
    plan_artifact,
):

    package = plan_artifact.get(
        "experiment_plan",
        {},
    )

    plan = package.get(
        "plan",
        {},
    )

    decision_rules = plan.get(
        "decision_rules",
        {},
    )

    execution_policy = plan.get(
        "execution_policy",
        {},
    )

    checks = {
        "engine_version":
            (
                plan_artifact.get(
                    "engine_version"
                )
                ==
                ENGINE_VERSION
            ),

        "plan_id_present":
            (
                isinstance(
                    package.get(
                        "plan_id"
                    ),
                    str,
                )
                and
                package[
                    "plan_id"
                ].startswith(
                    "PLAN-"
                )
            ),

        "plan_fingerprint_present":
            (
                isinstance(
                    package.get(
                        "plan_fingerprint_sha256"
                    ),
                    str,
                )
                and
                len(
                    package[
                        "plan_fingerprint_sha256"
                    ]
                )
                ==
                64
            ),

        "experiment_is_exp_0006":
            (
                plan.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "hypothesis_is_hyp_v03_001":
            (
                plan.get(
                    "hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "permission_is_p3":
            (
                plan.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "probe_count_16":
            (
                plan.get(
                    "probe_design",
                    {},
                ).get(
                    "probe_count"
                )
                ==
                16
            ),

        "pair_count_4":
            (
                plan.get(
                    "candidate_pair_design",
                    {},
                ).get(
                    "pair_count"
                )
                ==
                4
            ),

        "positive_control_p45":
            (
                decision_rules.get(
                    "positive_control",
                    {},
                ).get(
                    "pair_id"
                )
                ==
                "P45"
            ),

        "threshold_is_75_percent":
            (
                decision_rules.get(
                    "global_directional_threshold"
                )
                ==
                0.75
            ),

        "global_minimum_is_12":
            (
                decision_rules.get(
                    "minimum_global_consistent_probes"
                )
                ==
                12
            ),

        "per_pair_minimum_is_3":
            (
                decision_rules.get(
                    "minimum_per_pair_consistent_probes"
                )
                ==
                3
            ),

        "execution_not_authorized":
            (
                execution_policy.get(
                    "execution_authorized"
                )
                is False
            ),

        "human_approval_required":
            (
                execution_policy.get(
                    "human_approval_required"
                )
                is True
            ),

        "plan_does_not_execute":
            (
                execution_policy.get(
                    "plan_does_not_execute"
                )
                is True
            ),

        "plan_does_not_mutate_knowledge":
            (
                execution_policy.get(
                    "plan_does_not_mutate_knowledge_store"
                )
                is True
            ),

        "plan_does_not_create_evidence":
            (
                execution_policy.get(
                    "plan_does_not_create_evidence"
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
            "L10B plan validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    proposal_path = project_path(
        args.proposal
    )

    output_path = project_path(
        args.output
    )

    experiment_registry_path = project_path(
        args.experiment_registry
    )

    hypotheses_path = project_path(
        args.hypotheses
    )

    research_state_path = project_path(
        args.research_state
    )

    if (
        output_path.exists()
        and
        not args.overwrite
    ):
        raise FileExistsError(
            "EXP-0006 preregistration plan "
            f"already exists: {output_path}\n"
            "Use --overwrite only before "
            "execution begins."
        )

    print()
    print(
        "=" * 80
    )

    print(
        "STAGE L10B — EXP-0006 "
        "PREREGISTERED PLAN BUILDER"
    )

    print(
        "=" * 80
    )

    proposal_artifact = load_json(
        proposal_path
    )

    experiment_registry = load_json(
        experiment_registry_path
    )

    hypotheses = load_json(
        hypotheses_path
    )

    research_state = load_json(
        research_state_path
    )

    proposal_sha256 = file_sha256(
        proposal_path
    )

    source_validation = (
        validate_sources(
            proposal_artifact=(
                proposal_artifact
            ),
            experiment_registry=(
                experiment_registry
            ),
            hypotheses=hypotheses,
            research_state=research_state,
        )
    )

    design_checks = validate_design()

    print()
    print(
        "Source checks"
    )

    print(
        "-------------"
    )

    for name, passed in (
        source_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    print()
    print(
        "Design checks"
    )

    print(
        "-------------"
    )

    for name, passed in (
        design_checks.items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    plan_artifact = build_plan(
        proposal_artifact=(
            proposal_artifact
        ),
        proposal_sha256=(
            proposal_sha256
        ),
        experiment_record=(
            source_validation[
                "experiment"
            ]
        ),
        hypothesis_record=(
            source_validation[
                "hypothesis"
            ]
        ),
    )

    plan_checks = validate_plan(
        plan_artifact
    )

    plan_artifact[
        "source_checks"
    ] = source_validation[
        "checks"
    ]

    plan_artifact[
        "design_checks"
    ] = design_checks

    plan_artifact[
        "overall_checks"
    ] = {
        "sources_valid":
            all(
                source_validation[
                    "checks"
                ].values()
            ),

        "design_valid":
            all(
                design_checks.values()
            ),

        "plan_valid":
            all(
                plan_checks.values()
            ),

        "execution_not_authorized":
            True,

        "model_not_loaded":
            True,

        "intervention_not_performed":
            True,

        "knowledge_store_not_mutated":
            True,

        "evidence_not_created":
            True,
    }

    if not all(
        plan_artifact[
            "overall_checks"
        ].values()
    ):
        raise RuntimeError(
            "L10B overall checks failed."
        )

    atomic_write_json(
        output_path,
        plan_artifact,
    )

    saved = load_json(
        output_path
    )

    saved_checks = validate_plan(
        saved
    )

    require(
        all(
            saved_checks.values()
        ),
        (
            "Saved EXP-0006 plan failed "
            "validation."
        ),
    )

    package = (
        saved[
            "experiment_plan"
        ]
    )

    plan = package[
        "plan"
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "L10B EXP-0006 PREREGISTRATION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Plan ID:"
    )

    print(
        f"  {package['plan_id']}"
    )

    print()

    print(
        "Experiment:"
    )

    print(
        "  EXP-0006"
    )

    print()

    print(
        "Hypothesis:"
    )

    print(
        "  HYP-V03-001"
    )

    print()

    print(
        "Candidate pairs:"
    )

    for pair in (
        plan[
            "candidate_pair_design"
        ][
            "pairs"
        ]
    ):

        print(
            f"  {pair['pair_id']}: "
            f"{pair['label']} "
            f"({pair['role']})"
        )

    print()

    print(
        "Probe balance:"
    )

    print(
        "  total: 16"
    )

    print(
        "  lower correct: 8"
    )

    print(
        "  upper correct: 8"
    )

    print(
        "  words: 8"
    )

    print(
        "  symbols: 8"
    )

    print()

    print(
        "Primary correctness tests:"
    )

    print(
        "  L1H0: supports correct answer"
    )

    print(
        "  L3H3: suppresses correct answer"
    )

    print(
        "  L4H0: suppresses correct answer"
    )

    print()

    print(
        "Positive control:"
    )

    print(
        "  P45 four-vs-five"
    )

    print(
        "  L1H0 negative delta >= 3/4"
    )

    print(
        "  L4H0 positive delta >= 3/4"
    )

    print()

    print(
        "Global threshold:"
    )

    print(
        "  >= 12/16 = 75%"
    )

    print()

    print(
        "Required permission:"
    )

    print(
        "  P3"
    )

    print()

    print(
        "Execution authorized:"
    )

    print(
        "  False"
    )

    print()

    print(
        "Plan artifact:"
    )

    print(
        f"  {output_path}"
    )

    print()

    print(
        "Overall checks"
    )

    print(
        "--------------"
    )

    for name, passed in (
        saved[
            "overall_checks"
        ].items()
    ):

        print(
            f"{name:<40} "
            f"{passed}"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "SUCCESS: STAGE L10B EXP-0006 "
        "PREREGISTRATION COMPLETE."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
