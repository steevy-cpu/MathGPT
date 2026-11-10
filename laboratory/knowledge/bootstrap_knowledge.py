import argparse
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
    COLLECTION_FILES,
    KNOWLEDGE_DIR,
    SCHEMA_VERSION,
    KnowledgeStore,
    utc_now,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap the MathGPT "
            "Stage-L7 scientific "
            "knowledge store."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing "
            "knowledge store."
        ),
    )

    return parser.parse_args()


# ============================================================
# Model records
# ============================================================

STEP_1000_MODEL = {
    "model":
        "MathGPT",

    "stage":
        "step_1000_instrument_checkpoint",

    "checkpoint":
        (
            "checkpoints/general_pretraining/"
            "general_pretraining_short_v0.1_"
            "20261105_112710/"
            "mathgpt_step_001000.npz"
        ),

    "checkpoint_sha256":
        (
            "d6563949b9e3a8b429d0d916ab0544e"
            "a7dc11cd03a6e4debcd40f125410d00bb"
        ),

    "training_step":
        1000,
}


V0_3_MODEL = {
    "model":
        "MathGPT-General-v0.3",

    "stage":
        "general_pretraining_v0.3_300m",

    "checkpoint":
        (
            "checkpoints/general_pretraining_v0.3/"
            "general_pretraining_v0_3_300m_full/"
            "mathgpt_step_292969.npz"
        ),

    "checkpoint_sha256":
        (
            "4cd9b8f1c68e4079449519fd601b573f"
            "8ea48137bc74172c7b37e108c9707cf3"
        ),

    "training_step":
        292969,

    "tokens_seen":
        300000256,
}


# ============================================================
# Hypotheses
# ============================================================

def build_hypotheses(
    created_at,
):

    return {
        "schema_version":
            SCHEMA_VERSION,

        "collection":
            "hypotheses",

        "updated_at_utc":
            created_at,

        "records": [
            {
                "hypothesis_id":
                    "HYP-V03-001",

                "title":
                    (
                        "General arithmetic "
                        "competing-head circuit"
                    ),

                "statement":
                    (
                        "Across addition contexts, "
                        "L1H0 generally supports "
                        "the preferred arithmetic "
                        "answer while L3H3 and "
                        "L4H0 generally suppress "
                        "the preferred arithmetic "
                        "answer."
                    ),

                "model_scope":
                    V0_3_MODEL,

                "status":
                    "weakened",

                "confidence": {
                    "score":
                        None,

                    "label":
                        "unscored",

                    "method":
                        "L8_pending",

                    "note":
                        (
                            "The knowledge layer "
                            "does not assign "
                            "calibrated confidence "
                            "scores yet."
                        ),
                },

                "created_from":
                    "EXP-0004",

                "tested_by": [
                    "EXP-0004",
                ],

                "evidence_for": [
                    "FND-0003",
                ],

                "evidence_against": [
                    "FND-0004",
                ],

                "open_questions": [
                    (
                        "Are the observed head "
                        "effects tied to candidate "
                        "token identity rather than "
                        "arithmetic correctness?"
                    ),
                ],

                "tags": [
                    "arithmetic",
                    "attention-heads",
                    "causal",
                    "mature-v0.3",
                ],

                "created_at_utc":
                    created_at,

                "updated_at_utc":
                    created_at,
            },

            {
                "hypothesis_id":
                    "HYP-V03-002",

                "title":
                    (
                        "Four-versus-five "
                        "causal token preference"
                    ),

                "statement":
                    (
                        "In mature MathGPT v0.3, "
                        "L1H0 may causally bias "
                        "the four-versus-five "
                        "decision toward 'four', "
                        "while L4H0 may bias the "
                        "same decision toward "
                        "'five', across multiple "
                        "numerical contexts."
                    ),

                "model_scope":
                    V0_3_MODEL,

                "status":
                    "active",

                "confidence": {
                    "score":
                        None,

                    "label":
                        "preliminary",

                    "method":
                        "L8_pending",

                    "note":
                        (
                            "Pattern observed in "
                            "four fixed four/five "
                            "comparisons. Requires "
                            "a dedicated controlled "
                            "experiment."
                        ),
                },

                "created_from":
                    "EXP-0004",

                "tested_by": [
                    "EXP-0004",
                    "EXP-0005",
                ],

                "evidence_for": [
                    "FND-0005",
                ],

                "evidence_against": [],

                "open_questions": [
                    (
                        "Does L1H0 continue to "
                        "favor four when prompt "
                        "semantics change while "
                        "four/five candidates "
                        "remain fixed?"
                    ),
                    (
                        "Does L4H0 continue to "
                        "favor five under the "
                        "same controlled changes?"
                    ),
                    (
                        "Is the effect lexical, "
                        "numerical-semantic, or "
                        "task-dependent?"
                    ),
                ],

                "tags": [
                    "four-vs-five",
                    "numerical",
                    "attention-heads",
                    "causal",
                    "mature-v0.3",
                ],

                "created_at_utc":
                    created_at,

                "updated_at_utc":
                    created_at,
            },
        ],
    }


# ============================================================
# Findings
# ============================================================

def build_findings(
    created_at,
):

    return {
        "schema_version":
            SCHEMA_VERSION,

        "collection":
            "findings",

        "updated_at_utc":
            created_at,

        "records": [
            {
                "finding_id":
                    "FND-0001",

                "title":
                    (
                        "Step-1000 L0H1 "
                        "context-sensitive "
                        "four influence"
                    ),

                "statement":
                    (
                        "At the step-1000 "
                        "checkpoint, systematic "
                        "L0H1 ablations showed "
                        "non-uniform causal effects "
                        "on four-related decisions "
                        "across arithmetic and "
                        "control contexts."
                    ),

                "finding_type":
                    "causal",

                "model_scope":
                    STEP_1000_MODEL,

                "evidence_level":
                    "E3",

                "source_experiment_id":
                    "EXP-0002",

                "source_artifacts": [
                    (
                        "laboratory/artifacts/"
                        "exp_0002_systematic_summary.json"
                    ),
                ],

                "metrics": {
                    "mean_four_support_effect_with_two":
                        0.37745501411457855,

                    "mean_four_support_effect_without_two":
                        0.24299733340740204,
                },

                "supports_hypotheses": [],

                "weakens_hypotheses": [],

                "tags": [
                    "L0H1",
                    "step1000",
                    "causal-ablation",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "finding_id":
                    "FND-0002",

                "title":
                    (
                        "Step-1000 L0H1 "
                        "addition signature "
                        "stability"
                    ),

                "statement":
                    (
                        "The full 16,000-token "
                        "causal effect signature "
                        "of L0H1 was highly "
                        "similar across addition "
                        "contexts at step 1000."
                    ),

                "finding_type":
                    "causal_synthesis",

                "model_scope":
                    STEP_1000_MODEL,

                "evidence_level":
                    "E3",

                "source_experiment_id":
                    "EXP-0003",

                "source_artifacts": [
                    (
                        "laboratory/artifacts/"
                        "exp_0003_signature_summary.json"
                    ),
                ],

                "metrics": {
                    "addition_vs_addition_centered_cosine_mean":
                        0.9930459251016914,

                    "control_vs_control_centered_cosine_mean":
                        0.7536754643668724,

                    "addition_vs_control_centered_cosine_mean":
                        0.8369198320153511,
                },

                "supports_hypotheses": [],

                "weakens_hypotheses": [],

                "tags": [
                    "L0H1",
                    "vocabulary-profile",
                    "causal-signature",
                    "step1000",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "finding_id":
                    "FND-0003",

                "title":
                    (
                        "Mature v0.3 2+2 "
                        "competing-head anchor"
                    ),

                "statement":
                    (
                        "For 'Two plus two equals' "
                        "with four versus five at "
                        "the first discriminative "
                        "token, L1H0 supported "
                        "four while L3H3 and L4H0 "
                        "suppressed four. L4H1 had "
                        "a much smaller effect."
                    ),

                "finding_type":
                    "causal",

                "model_scope":
                    V0_3_MODEL,

                "evidence_level":
                    "E3",

                "source_experiment_id":
                    "EXP-0004",

                "source_artifacts": [
                    (
                        "laboratory/artifacts/"
                        "exp_0004_competing_heads/"
                        "a03_competing_heads.json"
                    ),
                    (
                        "laboratory/artifacts/"
                        "exp_0004_competing_heads_summary.json"
                    ),
                ],

                "metrics": {
                    "baseline_margin":
                        -0.5697602033615112,

                    "L1H0_ablation_margin_delta":
                        -0.5050597786903381,

                    "L3H3_ablation_margin_delta":
                        0.49808311462402344,

                    "L4H0_ablation_margin_delta":
                        0.4733119010925293,

                    "L4H1_ablation_margin_delta":
                        0.04903137683868408,
                },

                "supports_hypotheses": [
                    "HYP-V03-001",
                ],

                "weakens_hypotheses": [],

                "tags": [
                    "2+2",
                    "four-vs-five",
                    "attention-heads",
                    "mature-v0.3",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "finding_id":
                    "FND-0004",

                "title":
                    (
                        "Competing-head roles "
                        "do not generalize by "
                        "arithmetic correctness"
                    ),

                "statement":
                    (
                        "Across six addition "
                        "probes, the A03 head-role "
                        "pattern did not generalize "
                        "consistently. L1H0 "
                        "supported the preferred "
                        "answer in only 2 of 6 "
                        "addition probes, L3H3 in "
                        "3 of 6, and L4H0 in "
                        "4 of 6."
                    ),

                "finding_type":
                    "causal_synthesis",

                "model_scope":
                    V0_3_MODEL,

                "evidence_level":
                    "E3",

                "source_experiment_id":
                    "EXP-0004",

                "source_artifacts": [
                    (
                        "laboratory/artifacts/"
                        "exp_0004_competing_heads_summary.json"
                    ),
                ],

                "metrics": {
                    "L1H0_addition_support_rate":
                        0.3333333333333333,

                    "L3H3_addition_support_rate":
                        0.5,

                    "L4H0_addition_support_rate":
                        0.6666666666666666,

                    "L1H0_addition_mean_delta":
                        0.17256971914321184,

                    "L3H3_addition_mean_delta":
                        -0.06815508908281724,

                    "L4H0_addition_mean_delta":
                        -0.12598871967444816,
                },

                "supports_hypotheses": [],

                "weakens_hypotheses": [
                    "HYP-V03-001",
                ],

                "tags": [
                    "generalization",
                    "arithmetic",
                    "hypothesis-revision",
                    "mature-v0.3",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "finding_id":
                    "FND-0005",

                "title":
                    (
                        "Four/five direction "
                        "is consistent for "
                        "L1H0 and L4H0"
                    ),

                "statement":
                    (
                        "Across the four EXP-0004 "
                        "contexts that directly "
                        "contrasted four and five "
                        "(A03, A04, A05, C03), "
                        "the causal signs were "
                        "consistent with intact "
                        "L1H0 favoring four over "
                        "five and intact L4H0 "
                        "favoring five over four "
                        "in all four contexts."
                    ),

                "finding_type":
                    "derived_causal_pattern",

                "model_scope":
                    V0_3_MODEL,

                "evidence_level":
                    "E3",

                "source_experiment_id":
                    "EXP-0004",

                "source_artifacts": [
                    (
                        "laboratory/artifacts/"
                        "exp_0004_competing_heads_summary.json"
                    ),
                ],

                "metrics": {
                    "contexts": [
                        "A03",
                        "A04",
                        "A05",
                        "C03",
                    ],

                    "L1H0_four_direction_consistency":
                        "4/4",

                    "L4H0_five_direction_consistency":
                        "4/4",

                    "L1H0_deltas": {
                        "A03":
                            -0.5050597786903381,

                        "A04":
                            0.9667401313781738,

                        "A05":
                            0.6731274351477623,

                        "C03":
                            -1.1588538885116577,
                    },

                    "L4H0_deltas": {
                        "A03":
                            0.4733119010925293,

                        "A04":
                            -0.48275113105773926,

                        "A05":
                            -0.4758336544036865,

                        "C03":
                            0.5218524262309074,
                    },
                },

                "derivation_note":
                    (
                        "The sign is interpreted "
                        "relative to candidate "
                        "orientation. In A04/A05 "
                        "five is preferred and "
                        "four is comparison, while "
                        "A03/C03 use four as "
                        "preferred and five as "
                        "comparison."
                    ),

                "supports_hypotheses": [
                    "HYP-V03-002",
                ],

                "weakens_hypotheses": [],

                "tags": [
                    "four-vs-five",
                    "token-identity",
                    "causal-pattern",
                    "mature-v0.3",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "finding_id":
                    "FND-0006",

                "title":
                    (
                        "L4H1 is a weak "
                        "addition-head control"
                    ),

                "statement":
                    (
                        "Across the six addition "
                        "contexts in EXP-0004, "
                        "L4H1 had substantially "
                        "smaller absolute causal "
                        "margin effects than L4H0."
                    ),

                "finding_type":
                    "causal_synthesis",

                "model_scope":
                    V0_3_MODEL,

                "evidence_level":
                    "E3",

                "source_experiment_id":
                    "EXP-0004",

                "source_artifacts": [
                    (
                        "laboratory/artifacts/"
                        "exp_0004_competing_heads_summary.json"
                    ),
                ],

                "metrics": {
                    "L4H1_addition_mean_absolute_delta":
                        0.04277985139439503,

                    "L4H0_addition_mean_absolute_delta":
                        0.2945943254356583,
                },

                "supports_hypotheses": [],

                "weakens_hypotheses": [],

                "tags": [
                    "control",
                    "L4H1",
                    "L4H0",
                    "mature-v0.3",
                ],

                "created_at_utc":
                    created_at,
            },
        ],
    }


# ============================================================
# Experiments
# ============================================================

def build_experiments(
    created_at,
):

    return {
        "schema_version":
            SCHEMA_VERSION,

        "collection":
            "experiments",

        "updated_at_utc":
            created_at,

        "records": [
            {
                "experiment_id":
                    "EXP-0002",

                "title":
                    (
                        "L0H1 arithmetic and "
                        "lexical generalization"
                    ),

                "question":
                    (
                        "Does the step-1000 L0H1 "
                        "causal effect generalize "
                        "across arithmetic and "
                        "control contexts?"
                    ),

                "status":
                    "completed",

                "evidence_level":
                    "E3",

                "model_scope":
                    STEP_1000_MODEL,

                "summary_artifact":
                    (
                        "laboratory/artifacts/"
                        "exp_0002_systematic_summary.json"
                    ),

                "hypothesis_ids": [],

                "finding_ids": [
                    "FND-0001",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "experiment_id":
                    "EXP-0003",

                "title":
                    (
                        "L0H1 causal vocabulary "
                        "signature similarity"
                    ),

                "question":
                    (
                        "How similar is the "
                        "full vocabulary-wide "
                        "causal effect direction "
                        "of L0H1 across contexts?"
                    ),

                "status":
                    "completed",

                "evidence_level":
                    "E3",

                "model_scope":
                    STEP_1000_MODEL,

                "summary_artifact":
                    (
                        "laboratory/artifacts/"
                        "exp_0003_signature_summary.json"
                    ),

                "hypothesis_ids": [],

                "finding_ids": [
                    "FND-0002",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "experiment_id":
                    "EXP-0004",

                "title":
                    (
                        "Mature v0.3 competing "
                        "attention-head "
                        "generalization"
                    ),

                "question":
                    (
                        "Do the competing causal "
                        "head directions observed "
                        "on Two plus two equals "
                        "generalize across "
                        "arithmetic and control "
                        "contexts?"
                    ),

                "status":
                    "completed",

                "evidence_level":
                    "E3",

                "model_scope":
                    V0_3_MODEL,

                "summary_artifact":
                    (
                        "laboratory/artifacts/"
                        "exp_0004_competing_heads_summary.json"
                    ),

                "hypothesis_ids": [
                    "HYP-V03-001",
                    "HYP-V03-002",
                ],

                "finding_ids": [
                    "FND-0003",
                    "FND-0004",
                    "FND-0005",
                    "FND-0006",
                ],

                "created_at_utc":
                    created_at,
            },

            {
                "experiment_id":
                    "EXP-0005",

                "title":
                    (
                        "Fixed four-vs-five "
                        "causal context transfer"
                    ),

                "question":
                    (
                        "When the candidate "
                        "tokens four and five "
                        "are held fixed while "
                        "prompt semantics change, "
                        "do L1H0 and L4H0 retain "
                        "opposing causal "
                        "directions?"
                    ),

                "status":
                    "planned",

                "evidence_level":
                    None,

                "model_scope":
                    V0_3_MODEL,

                "summary_artifact":
                    None,

                "hypothesis_ids": [
                    "HYP-V03-002",
                ],

                "finding_ids": [],

                "created_at_utc":
                    created_at,
            },
        ],
    }


# ============================================================
# Research state
# ============================================================

def build_research_state(
    created_at,
):

    return {
        "schema_version":
            SCHEMA_VERSION,

        "collection":
            "research_state",

        "updated_at_utc":
            created_at,

        "state": {
            "program":
                "MathGPT AI Scientist Laboratory",

            "stage":
                "L7",

            "stage_name":
                "Scientific Knowledge Layer",

            "active_model":
                V0_3_MODEL,

            "current_research_question":
                (
                    "What internal causal "
                    "mechanisms govern numerical "
                    "next-token decisions in "
                    "mature MathGPT v0.3?"
                ),

            "last_completed_experiment":
                "EXP-0004",

            "next_recommended_experiment":
                "EXP-0005",

            "active_hypotheses": [
                "HYP-V03-002",
            ],

            "weakened_hypotheses": [
                "HYP-V03-001",
            ],

            "open_questions": [
                (
                    "Is the L1H0/L4H0 "
                    "four-versus-five pattern "
                    "lexical or numerical-semantic?"
                ),
                (
                    "Does the pattern survive "
                    "fixed candidate tokens across "
                    "different prompt semantics?"
                ),
                (
                    "How do these mechanisms "
                    "change after General-v0.4 "
                    "finishes training?"
                ),
            ],

            "scientist_status":
                "manual_orchestration",

            "confidence_engine":
                "not_built",

            "experiment_planner":
                "not_built",

            "evidence_reviewer":
                "not_built",

            "created_at_utc":
                created_at,
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    store = KnowledgeStore()

    existing = [
        store.collection_path(
            collection
        )

        for collection
        in COLLECTION_FILES

        if store.collection_path(
            collection
        ).exists()
    ]

    if (
        existing
        and
        not args.overwrite
    ):

        print()
        print(
            "Knowledge store already exists:"
        )

        for path in existing:

            print(
                f"  {path}"
            )

        print()
        print(
            "Use --overwrite only if you "
            "intend to rebuild the initial "
            "Stage-L7 seed."
        )

        raise SystemExit(
            0
        )

    created_at = utc_now()

    hypotheses = build_hypotheses(
        created_at
    )

    findings = build_findings(
        created_at
    )

    experiments = build_experiments(
        created_at
    )

    research_state = (
        build_research_state(
            created_at
        )
    )

    store.save(
        "hypotheses",
        hypotheses,
    )

    store.save(
        "findings",
        findings,
    )

    store.save(
        "experiments",
        experiments,
    )

    store.save(
        "research_state",
        research_state,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE L7A — SCIENTIFIC "
        "KNOWLEDGE STORE BOOTSTRAP"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Knowledge directory:"
    )

    print(
        KNOWLEDGE_DIR
    )

    print()

    validation = (
        store.print_summary()
    )

    if not validation[
        "passed"
    ]:

        raise RuntimeError(
            "Knowledge store validation "
            "did not pass."
        )

    print()
    print(
        "=" * 72
    )

    print(
        "SUCCESS: STAGE L7A BOOTSTRAP COMPLETE."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()