import argparse
import hashlib
import json
import sys

from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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
# Laboratory imports
# ============================================================

from laboratory.core.runtime import (
    LAB_TOOL_VERSION,
    MARS_PROTOCOL_VERSION,
    LaboratoryRuntime,
)

from laboratory.tools.continuation_scorer import (
    score_continuation,
    build_discriminative_analysis,
)

from laboratory.tools.attention_head_ablation import (
    forward_with_head_scale,
    measure_targets,
    distribution_metrics,
)


# ============================================================
# Identity
# ============================================================

RUNNER_VERSION = "L10D.1"
EXPECTED_SPEC_VERSION = "L10C.1"
EXPECTED_EXPERIMENT_ID = "EXP-0006"
EXPECTED_HYPOTHESIS_ID = "HYP-V03-001"
EXPECTED_APPROVAL_VERSION = "P3A.1"

DEFAULT_EXECUTION_SPEC = (
    "laboratory/scientist/state/"
    "experiment_execution_spec_exp_0006.json"
)

DEFAULT_APPROVAL = (
    "laboratory/scientist/state/"
    "p3_approval_exp_0006.json"
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "EXP-0006 controlled runner — "
            "cross-pair numerical candidate "
            "identity versus arithmetic "
            "correctness."
        )
    )

    parser.add_argument(
        "--execution-spec",
        default=DEFAULT_EXECUTION_SPEC,
        type=str,
    )

    parser.add_argument(
        "--approval",
        default=DEFAULT_APPROVAL,
        type=str,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate the frozen execution "
            "specification without hashing the "
            "checkpoint, loading the model, "
            "writing result artifacts, or "
            "running interventions. Missing "
            "approval is reported as expected."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of EXP-0006 "
            "result artifacts after valid P3 "
            "approval."
        ),
    )

    return parser.parse_args()


# ============================================================
# Paths / JSON / SHA
# ============================================================

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


def save_json(
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

    if temporary.exists():
        temporary.unlink()

    try:

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

    finally:

        if temporary.exists():
            temporary.unlink()


def sha256_file(path):
    path = Path(path)

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


# ============================================================
# Formatting
# ============================================================

def format_signed(value):

    if value is None:
        return "N/A"

    return f"{float(value):+.6f}"


def format_percentage(value):

    if value is None:
        return "N/A"

    return f"{float(value):.2%}"


# ============================================================
# Frozen execution-spec validation
# ============================================================

def validate_execution_spec(
    spec_artifact,
):

    spec = spec_artifact.get(
        "execution_spec",
        {},
    )

    payload = spec.get(
        "experiment_payload",
        {},
    )

    approval_contract = spec.get(
        "approval_contract",
        {},
    )

    runtime = spec.get(
        "runtime_constraints",
        {},
    )

    artifacts = spec.get(
        "artifact_contract",
        {},
    )

    heads = payload.get(
        "target_heads",
        [],
    )

    probes = payload.get(
        "probes",
        [],
    )

    pairs = payload.get(
        "candidate_pairs",
        [],
    )

    decision_rules = payload.get(
        "decision_rules",
        {},
    )

    head_ids = [
        item.get(
            "head_id"
        )
        for item
        in heads
    ]

    probe_ids = [
        item.get(
            "probe_id"
        )
        for item
        in probes
    ]

    pair_ids = [
        item.get(
            "pair_id"
        )
        for item
        in pairs
    ]

    checks = {
        "spec_version":
            (
                spec_artifact.get(
                    "engine_version"
                )
                ==
                EXPECTED_SPEC_VERSION
            ),

        "spec_checks_present":
            bool(
                spec_artifact.get(
                    "overall_checks"
                )
            ),

        "spec_checks_passed":
            (
                bool(
                    spec_artifact.get(
                        "overall_checks"
                    )
                )
                and
                all(
                    spec_artifact[
                        "overall_checks"
                    ].values()
                )
            ),

        "experiment_id":
            (
                payload.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "hypothesis_id":
            (
                payload.get(
                    "hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "execution_spec_id_present":
            (
                isinstance(
                    spec.get(
                        "execution_spec_id"
                    ),
                    str,
                )
                and
                spec[
                    "execution_spec_id"
                ].startswith(
                    "SPEC-"
                )
            ),

        "plan_id_present":
            (
                isinstance(
                    payload.get(
                        "plan_id"
                    ),
                    str,
                )
                and
                payload[
                    "plan_id"
                ].startswith(
                    "PLAN-"
                )
            ),

        "checkpoint_path_present":
            bool(
                payload.get(
                    "checkpoint",
                    {},
                ).get(
                    "path"
                )
            ),

        "checkpoint_sha_valid":
            (
                isinstance(
                    payload.get(
                        "checkpoint",
                        {},
                    ).get(
                        "sha256"
                    ),
                    str,
                )
                and
                len(
                    payload[
                        "checkpoint"
                    ][
                        "sha256"
                    ]
                )
                ==
                64
            ),

        "four_heads":
            (
                len(
                    heads
                )
                ==
                4
                and
                len(
                    set(
                        head_ids
                    )
                )
                ==
                4
            ),

        "expected_primary_heads":
            (
                {
                    "L1H0",
                    "L3H3",
                    "L4H0",
                }
                <=
                set(
                    head_ids
                )
            ),

        "weak_control":
            (
                "L4H1"
                in
                head_ids
            ),

        "four_pairs":
            (
                len(
                    pairs
                )
                ==
                4
                and
                set(
                    pair_ids
                )
                ==
                {
                    "P23",
                    "P34",
                    "P45",
                    "P56",
                }
            ),

        "sixteen_probes":
            (
                len(
                    probes
                )
                ==
                16
                and
                len(
                    set(
                        probe_ids
                    )
                )
                ==
                16
            ),

        "pair_order_frozen":
            (
                runtime.get(
                    "deterministic_pair_order"
                )
                ==
                pair_ids
            ),

        "probe_order_frozen":
            (
                runtime.get(
                    "deterministic_probe_order"
                )
                ==
                probe_ids
            ),

        "head_order_frozen":
            (
                runtime.get(
                    "deterministic_head_order"
                )
                ==
                head_ids
            ),

        "effect_count_64":
            (
                runtime.get(
                    "expected_effect_record_count"
                )
                ==
                64
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

        "global_threshold_12_of_16":
            (
                decision_rules.get(
                    "minimum_global_consistent_probes"
                )
                ==
                12
            ),

        "per_pair_threshold_3_of_4":
            (
                decision_rules.get(
                    "minimum_per_pair_consistent_probes"
                )
                ==
                3
            ),

        "p3_required":
            (
                approval_contract.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "human_approval_required":
            (
                approval_contract.get(
                    "human_approval_required"
                )
                is True
            ),

        "spec_itself_does_not_authorize":
            (
                approval_contract.get(
                    "execution_authorized"
                )
                is False
                and
                approval_contract.get(
                    "may_execute_now"
                )
                is False
            ),

        "evidence_level_e3":
            (
                artifacts.get(
                    "required_evidence_level"
                )
                ==
                "E3"
            ),

        "artifact_paths_present":
            (
                bool(
                    artifacts.get(
                        "artifact_directory"
                    )
                )
                and
                bool(
                    artifacts.get(
                        "summary_artifact"
                    )
                )
            ),

        "persistent_weights_unchanged":
            (
                runtime.get(
                    "persistent_weight_modification"
                )
                is False
            ),

        "scale_one_control_required":
            (
                runtime.get(
                    "scale_one_exactness_required"
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
            "EXP-0006 execution specification "
            f"failed validation: {failed}"
        )

    return checks


# ============================================================
# P3 approval validation
# ============================================================

def expected_approval_binding(
    *,
    spec_artifact,
    spec_sha256,
):

    spec = spec_artifact[
        "execution_spec"
    ]

    payload = spec[
        "experiment_payload"
    ]

    source_plan = spec[
        "source_plan"
    ]

    return {
        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "hypothesis_id":
            EXPECTED_HYPOTHESIS_ID,

        "plan_id":
            payload[
                "plan_id"
            ],

        "execution_spec_id":
            spec[
                "execution_spec_id"
            ],

        "execution_spec_sha256":
            spec_sha256,

        "source_plan_sha256":
            source_plan[
                "sha256"
            ],

        "checkpoint_sha256":
            payload[
                "checkpoint"
            ][
                "sha256"
            ],
    }


def validate_approval(
    *,
    approval,
    expected_binding,
):

    checks = {
        "approval_version":
            (
                approval.get(
                    "approval_version"
                )
                ==
                EXPECTED_APPROVAL_VERSION
            ),

        "approval_id_present":
            (
                isinstance(
                    approval.get(
                        "approval_id"
                    ),
                    str,
                )
                and
                approval[
                    "approval_id"
                ].startswith(
                    "APR-"
                )
            ),

        "decision_approved":
            (
                approval.get(
                    "decision"
                )
                ==
                "approved"
            ),

        "permission_p3":
            (
                approval.get(
                    "permission"
                )
                ==
                "P3"
            ),

        "approved_by_human":
            (
                approval.get(
                    "approved_by"
                )
                ==
                "human"
            ),

        "explicit_human_approval":
            (
                approval.get(
                    "checks",
                    {},
                ).get(
                    "explicit_human_approval"
                )
                is True
            ),

        "temporary_intervention_scope":
            (
                approval.get(
                    "scope"
                )
                ==
                (
                    "temporary_causal_"
                    "intervention_only"
                )
            ),
    }

    for field, expected_value in (
        expected_binding.items()
    ):

        checks[
            f"binding_{field}"
        ] = (
            approval.get(
                "binding",
                {},
            ).get(
                field
            )
            ==
            expected_value
        )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "EXP-0006 P3 approval validation "
            f"failed: {failed}"
        )

    return checks


# ============================================================
# Statistics
# ============================================================

def summarize_effects(
    records,
):

    if not records:

        return {
            "count":
                0,

            "mean_delta":
                None,

            "median_delta":
                None,

            "mean_absolute_delta":
                None,

            "negative_delta_count":
                0,

            "positive_delta_count":
                0,

            "zero_delta_count":
                0,

            "negative_delta_rate":
                None,

            "positive_delta_rate":
                None,

            "mean_correctness_effect":
                None,

            "median_correctness_effect":
                None,

            "supports_correct_count":
                0,

            "suppresses_correct_count":
                0,

            "zero_correctness_count":
                0,

            "supports_correct_rate":
                None,

            "suppresses_correct_rate":
                None,
        }

    deltas = np.asarray(
        [
            record[
                "ablation_delta"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    correctness = np.asarray(
        [
            record[
                "correctness_effect"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    negative_delta_count = int(
        np.sum(
            deltas
            <
            0.0
        )
    )

    positive_delta_count = int(
        np.sum(
            deltas
            >
            0.0
        )
    )

    zero_delta_count = int(
        len(
            deltas
        )
        -
        negative_delta_count
        -
        positive_delta_count
    )

    supports_correct_count = int(
        np.sum(
            correctness
            >
            0.0
        )
    )

    suppresses_correct_count = int(
        np.sum(
            correctness
            <
            0.0
        )
    )

    zero_correctness_count = int(
        len(
            correctness
        )
        -
        supports_correct_count
        -
        suppresses_correct_count
    )

    return {
        "count":
            int(
                len(
                    deltas
                )
            ),

        "mean_delta":
            float(
                np.mean(
                    deltas
                )
            ),

        "median_delta":
            float(
                np.median(
                    deltas
                )
            ),

        "mean_absolute_delta":
            float(
                np.mean(
                    np.abs(
                        deltas
                    )
                )
            ),

        "negative_delta_count":
            negative_delta_count,

        "positive_delta_count":
            positive_delta_count,

        "zero_delta_count":
            zero_delta_count,

        "negative_delta_rate":
            float(
                negative_delta_count
                /
                len(
                    deltas
                )
            ),

        "positive_delta_rate":
            float(
                positive_delta_count
                /
                len(
                    deltas
                )
            ),

        "mean_correctness_effect":
            float(
                np.mean(
                    correctness
                )
            ),

        "median_correctness_effect":
            float(
                np.median(
                    correctness
                )
            ),

        "supports_correct_count":
            supports_correct_count,

        "suppresses_correct_count":
            suppresses_correct_count,

        "zero_correctness_count":
            zero_correctness_count,

        "supports_correct_rate":
            float(
                supports_correct_count
                /
                len(
                    correctness
                )
            ),

        "suppresses_correct_rate":
            float(
                suppresses_correct_count
                /
                len(
                    correctness
                )
            ),
    }


def fixed_pair_direction(
    records,
    minimum_consistent,
):

    summary = summarize_effects(
        records
    )

    negative_count = (
        summary[
            "negative_delta_count"
        ]
    )

    positive_count = (
        summary[
            "positive_delta_count"
        ]
    )

    if (
        negative_count
        >=
        minimum_consistent
    ):

        direction = (
            "lower"
        )

        passed = True

    elif (
        positive_count
        >=
        minimum_consistent
    ):

        direction = (
            "upper"
        )

        passed = True

    else:

        direction = (
            "none"
        )

        passed = False

    return {
        "direction":
            direction,

        "pass":
            passed,

        "negative_delta_count":
            negative_count,

        "positive_delta_count":
            positive_count,

        "minimum_consistent":
            int(
                minimum_consistent
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    execution_spec_path = project_path(
        args.execution_spec
    )

    approval_path = project_path(
        args.approval
    )

    # ========================================================
    # Load and verify frozen execution specification
    # ========================================================

    if not execution_spec_path.exists():
        raise FileNotFoundError(
            execution_spec_path
        )

    spec_sha256 = sha256_file(
        execution_spec_path
    )

    spec_artifact = load_json(
        execution_spec_path
    )

    spec_checks = validate_execution_spec(
        spec_artifact
    )

    spec = spec_artifact[
        "execution_spec"
    ]

    payload = spec[
        "experiment_payload"
    ]

    artifacts = spec[
        "artifact_contract"
    ]

    runtime_constraints = spec[
        "runtime_constraints"
    ]

    checkpoint_path = project_path(
        payload[
            "checkpoint"
        ][
            "path"
        ]
    )

    expected_checkpoint_sha256 = (
        payload[
            "checkpoint"
        ][
            "sha256"
        ]
    )

    output_dir = project_path(
        artifacts[
            "artifact_directory"
        ]
    )

    summary_path = project_path(
        artifacts[
            "summary_artifact"
        ]
    )

    heads = list(
        payload[
            "target_heads"
        ]
    )

    probes = list(
        payload[
            "probes"
        ]
    )

    pairs = list(
        payload[
            "candidate_pairs"
        ]
    )

    decision_rules = payload[
        "decision_rules"
    ]

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "=" * 88
    )

    print(
        "EXP-0006 — CROSS-PAIR NUMERICAL "
        "CANDIDATE IDENTITY vs ARITHMETIC "
        "CORRECTNESS"
    )

    print(
        "=" * 88
    )

    print()
    print(
        "Execution specification:"
    )

    print(
        execution_spec_path
    )

    print()
    print(
        "Execution Spec ID:"
    )

    print(
        spec[
            "execution_spec_id"
        ]
    )

    print()
    print(
        "Plan ID:"
    )

    print(
        payload[
            "plan_id"
        ]
    )

    print()
    print(
        "Hypothesis:"
    )

    print(
        payload[
            "hypothesis_id"
        ]
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        checkpoint_path
    )

    print()
    print(
        f"Pairs: {len(pairs)}"
    )

    print(
        f"Probes: {len(probes)}"
    )

    print(
        f"Heads: {len(heads)}"
    )

    print(
        "Expected causal effects: "
        f"{runtime_constraints['expected_effect_record_count']}"
    )

    # ========================================================
    # PRE-FLIGHT MODE
    #
    # No approval required.
    # No checkpoint hashing.
    # No model load.
    # No intervention.
    # No result artifact writes.
    # ========================================================

    if args.preflight_only:

        print()
        print(
            "Preflight checks"
        )

        print(
            "----------------"
        )

        for name, passed in (
            spec_checks.items()
        ):

            print(
                f"{name}: {passed}"
            )

        print()
        print(
            "Approval artifact:"
        )

        if approval_path.exists():

            print(
                f"  present: {approval_path}"
            )

            print(
                "  NOTE: preflight does not "
                "grant, validate for execution, "
                "or consume approval."
            )

        else:

            print(
                "  not present"
            )

            print(
                "  status: expected / "
                "awaiting human approval"
            )

        print()
        print(
            "Checkpoint exists:"
        )

        print(
            f"  {checkpoint_path.exists()}"
        )

        print()
        print(
            "Frozen design:"
        )

        print(
            "  pair order: "
            +
            ", ".join(
                runtime_constraints[
                    "deterministic_pair_order"
                ]
            )
        )

        print(
            "  probes: "
            f"{len(probes)}"
        )

        print(
            "  heads: "
            +
            ", ".join(
                runtime_constraints[
                    "deterministic_head_order"
                ]
            )
        )

        print(
            "  positive control: P45"
        )

        print()
        print(
            "Runner safety:"
        )

        print(
            "  checkpoint hashed: False"
        )

        print(
            "  model loaded: False"
        )

        print(
            "  interventions executed: False"
        )

        print(
            "  result artifacts written: False"
        )

        print(
            "  knowledge store mutated: False"
        )

        print()
        print(
            "=" * 88
        )

        print(
            "SUCCESS: EXP-0006 RUNNER "
            "PREFLIGHT PASSED."
        )

        print(
            "Execution remains blocked "
            "pending exact P3 approval."
        )

        print(
            "=" * 88
        )

        return

    # ========================================================
    # REAL EXECUTION PATH
    #
    # Approval MUST validate before:
    #   - checkpoint hashing
    #   - output artifact creation
    #   - model loading
    #   - interventions
    # ========================================================

    if not approval_path.exists():

        raise PermissionError(
            "EXP-0006 execution blocked: "
            "P3 approval artifact does not "
            "exist.\n"
            f"Expected: {approval_path}"
        )

    approval = load_json(
        approval_path
    )

    expected_binding = (
        expected_approval_binding(
            spec_artifact=(
                spec_artifact
            ),
            spec_sha256=(
                spec_sha256
            ),
        )
    )

    approval_checks = validate_approval(
        approval=approval,
        expected_binding=(
            expected_binding
        ),
    )

    print()
    print(
        "P3 approval verified."
    )

    print(
        "Approval ID: "
        f"{approval['approval_id']}"
    )

    # ========================================================
    # Checkpoint verification before model load
    # ========================================================

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    actual_checkpoint_sha256 = (
        sha256_file(
            checkpoint_path
        )
    )

    if (
        actual_checkpoint_sha256
        !=
        expected_checkpoint_sha256
    ):

        raise RuntimeError(
            "EXP-0006 checkpoint SHA-256 "
            "mismatch.\n"
            f"Expected: "
            f"{expected_checkpoint_sha256}\n"
            f"Actual:   "
            f"{actual_checkpoint_sha256}"
        )

    # ========================================================
    # Output protection
    # ========================================================

    if (
        summary_path.exists()
        and
        not args.overwrite
    ):

        raise FileExistsError(
            "EXP-0006 summary already exists: "
            f"{summary_path}\n"
            "Use --overwrite only for an "
            "explicit approved rerun."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load model ONCE
    # ========================================================

    print()
    print(
        "Loading mature v0.3 checkpoint once..."
    )

    runtime = LaboratoryRuntime(
        checkpoint_path=(
            checkpoint_path
        ),
        expected_checkpoint_sha256=(
            expected_checkpoint_sha256
        ),
    )

    # ========================================================
    # Head bounds / identity
    # ========================================================

    for head_spec in heads:

        layer = int(
            head_spec[
                "layer"
            ]
        )

        head = int(
            head_spec[
                "head"
            ]
        )

        expected_head_id = (
            f"L{layer}H{head}"
        )

        if (
            head_spec[
                "head_id"
            ]
            !=
            expected_head_id
        ):

            raise RuntimeError(
                "EXP-0006 head identity "
                f"mismatch: {head_spec}"
            )

        if not (
            0
            <=
            layer
            <
            runtime.model.num_layers
        ):

            raise ValueError(
                f"Invalid layer {layer}."
            )

        if not (
            0
            <=
            head
            <
            runtime.model.num_heads
        ):

            raise ValueError(
                f"Invalid head {head}."
            )

    # ========================================================
    # Run frozen probes
    # ========================================================

    experiment_records = []
    all_effect_records = []

    all_scale_1_exact = True
    all_heads_zeroed = True

    for probe_index, probe in enumerate(
        probes,
        start=1,
    ):

        probe_id = probe[
            "probe_id"
        ]

        prompt = probe[
            "prompt"
        ]

        pair_id = probe[
            "pair_id"
        ]

        pair_label = probe[
            "pair_label"
        ]

        pair_role = probe[
            "pair_role"
        ]

        prompt_format = probe[
            "format"
        ]

        lower_text = probe[
            "lower_candidate"
        ]

        upper_text = probe[
            "upper_candidate"
        ]

        semantically_correct = probe[
            "semantically_correct"
        ]

        correct_side = probe[
            "correct_side"
        ]

        probe_output = (
            output_dir
            /
            (
                f"{probe_id.lower()}"
                "_cross_pair_heads.json"
            )
        )

        if (
            probe_output.exists()
            and
            not args.overwrite
        ):

            raise FileExistsError(
                "EXP-0006 probe artifact "
                f"already exists: "
                f"{probe_output}"
            )

        print()
        print(
            "=" * 88
        )

        print(
            f"[{probe_index}/{len(probes)}] "
            f"{probe_id}"
        )

        print(
            "=" * 88
        )

        print(
            f"Pair: {pair_id} "
            f"({pair_label})"
        )

        print(
            f"Role: {pair_role}"
        )

        print(
            f"Format: {prompt_format}"
        )

        print(
            f"Prompt: {prompt}"
        )

        print(
            "Fixed candidates: "
            f"{repr(lower_text)} vs "
            f"{repr(upper_text)}"
        )

        print(
            "Semantically correct: "
            f"{semantically_correct} "
            f"({correct_side})"
        )

        # ====================================================
        # Candidate scoring
        #
        # Candidate index 0 is ALWAYS LOWER.
        # Candidate index 1 is ALWAYS UPPER.
        # This orientation never changes.
        # ====================================================

        candidate_results = []

        for candidate_index, continuation in (
            enumerate(
                [
                    lower_text,
                    upper_text,
                ]
            )
        ):

            candidate = score_continuation(
                runtime=runtime,
                prompt=prompt,
                continuation=continuation,
            )

            candidate[
                "candidate_index"
            ] = int(
                candidate_index
            )

            candidate_results.append(
                candidate
            )

        discriminative = (
            build_discriminative_analysis(
                runtime=runtime,
                prompt=prompt,
                candidate_results=(
                    candidate_results
                ),
            )
        )

        candidates_by_index = {
            int(
                item[
                    "candidate_index"
                ]
            ):
                item

            for item
            in discriminative[
                "candidates"
            ]

            if item.get(
                "has_discriminative_token",
                False,
            )
        }

        if (
            0
            not in candidates_by_index
            or
            1
            not in candidates_by_index
        ):

            raise RuntimeError(
                f"{probe_id}: both fixed "
                "pair candidates must have "
                "discriminative tokens."
            )

        lower_token_id = int(
            candidates_by_index[
                0
            ][
                "token_id"
            ]
        )

        upper_token_id = int(
            candidates_by_index[
                1
            ][
                "token_id"
            ]
        )

        if (
            lower_token_id
            ==
            upper_token_id
        ):

            raise RuntimeError(
                f"{probe_id}: lower and upper "
                "candidates resolved to the "
                "same discriminative token."
            )

        context_ids = [
            int(
                token_id
            )

            for token_id
            in discriminative[
                "scoring_context_token_ids"
            ]
        ]

        input_ids = np.asarray(
            [
                context_ids
            ],
            dtype=np.int64,
        )

        # ====================================================
        # Baseline canonical LOWER-minus-UPPER margin
        # ====================================================

        baseline_logits = (
            runtime
            .model
            .forward(
                input_ids
            )
        )

        baseline_targets = measure_targets(
            runtime=runtime,
            logits=baseline_logits,
            preferred_token_id=(
                lower_token_id
            ),
            comparison_token_id=(
                upper_token_id
            ),
        )

        baseline_margin = float(
            baseline_targets[
                "logit_margin"
            ]
        )

        print()
        print(
            "Baseline LOWER-UPPER margin: "
            f"{baseline_margin:+.6f}"
        )

        print(
            "Baseline direction: "
            +
            (
                "LOWER"
                if baseline_margin > 0.0
                else
                "UPPER"
                if baseline_margin < 0.0
                else
                "TIE"
            )
        )

        # ====================================================
        # Head interventions
        # ====================================================

        head_results = []

        for head_spec in heads:

            layer = int(
                head_spec[
                    "layer"
                ]
            )

            head = int(
                head_spec[
                    "head"
                ]
            )

            head_id = head_spec[
                "head_id"
            ]

            role = head_spec[
                "hypothesis_role"
            ]

            # ------------------------------------------------
            # Scale=1 exactness control
            # ------------------------------------------------

            (
                control_logits,
                _,
            ) = forward_with_head_scale(
                runtime=runtime,
                token_ids=input_ids,
                target_layer=layer,
                target_head=head,
                head_scale=1.0,
            )

            control_exact = bool(
                np.array_equal(
                    baseline_logits,
                    control_logits,
                )
            )

            control_max_difference = float(
                np.max(
                    np.abs(
                        baseline_logits.astype(
                            np.float64
                        )
                        -
                        control_logits.astype(
                            np.float64
                        )
                    )
                )
            )

            all_scale_1_exact = (
                all_scale_1_exact
                and
                control_exact
            )

            # ------------------------------------------------
            # Scale=0 causal ablation
            # ------------------------------------------------

            (
                ablated_logits,
                intervention_record,
            ) = forward_with_head_scale(
                runtime=runtime,
                token_ids=input_ids,
                target_layer=layer,
                target_head=head,
                head_scale=0.0,
            )

            ablated_targets = measure_targets(
                runtime=runtime,
                logits=ablated_logits,
                preferred_token_id=(
                    lower_token_id
                ),
                comparison_token_id=(
                    upper_token_id
                ),
            )

            ablated_margin = float(
                ablated_targets[
                    "logit_margin"
                ]
            )

            ablation_delta = float(
                ablated_margin
                -
                baseline_margin
            )

            # ----------------------------------------------
            # Fixed candidate-pair direction
            # ----------------------------------------------

            if ablation_delta < 0.0:

                causal_direction = (
                    "intact_head_supports_lower"
                )

            elif ablation_delta > 0.0:

                causal_direction = (
                    "intact_head_supports_upper"
                )

            else:

                causal_direction = (
                    "no_detected_pair_direction"
                )

            # ----------------------------------------------
            # Correctness-normalized effect
            #
            # lower correct:
            #   negative delta means intact head
            #   supports correct answer.
            #
            # upper correct:
            #   positive delta means intact head
            #   supports correct answer.
            # ----------------------------------------------

            if correct_side == "lower":

                correctness_effect = float(
                    -ablation_delta
                )

            elif correct_side == "upper":

                correctness_effect = float(
                    ablation_delta
                )

            else:

                raise RuntimeError(
                    f"{probe_id}: invalid "
                    f"correct_side "
                    f"{correct_side!r}."
                )

            if correctness_effect > 0.0:

                correctness_direction = (
                    "intact_head_supports_correct"
                )

            elif correctness_effect < 0.0:

                correctness_direction = (
                    "intact_head_suppresses_correct"
                )

            else:

                correctness_direction = (
                    "no_detected_correctness_effect"
                )

            distribution = (
                distribution_metrics(
                    baseline_logits[
                        0,
                        -1,
                        :
                    ],
                    ablated_logits[
                        0,
                        -1,
                        :
                    ],
                )
            )

            lower_rank_delta = int(
                ablated_targets[
                    "preferred"
                ][
                    "rank"
                ]
                -
                baseline_targets[
                    "preferred"
                ][
                    "rank"
                ]
            )

            upper_rank_delta = int(
                ablated_targets[
                    "comparison"
                ][
                    "rank"
                ]
                -
                baseline_targets[
                    "comparison"
                ][
                    "rank"
                ]
            )

            head_zeroed = bool(
                np.isclose(
                    intervention_record[
                        "intervened_last_token_l2"
                    ],
                    0.0,
                )
            )

            all_heads_zeroed = (
                all_heads_zeroed
                and
                head_zeroed
            )

            result = {
                "layer":
                    layer,

                "head":
                    head,

                "head_id":
                    head_id,

                "hypothesis_role":
                    role,

                "control_scale_1_exact":
                    control_exact,

                "control_scale_1_max_difference":
                    control_max_difference,

                "intervention":
                    intervention_record,

                "ablated":
                    ablated_targets,

                "effect": {
                    "baseline_lower_minus_upper_margin":
                        baseline_margin,

                    "ablated_lower_minus_upper_margin":
                        ablated_margin,

                    "ablation_delta":
                        ablation_delta,

                    "absolute_ablation_delta":
                        float(
                            abs(
                                ablation_delta
                            )
                        ),

                    "pair_causal_direction":
                        causal_direction,

                    "correctness_effect":
                        correctness_effect,

                    "correctness_direction":
                        correctness_direction,

                    "lower_rank_delta":
                        lower_rank_delta,

                    "upper_rank_delta":
                        upper_rank_delta,

                    "distribution_change":
                        distribution,
                },
            }

            head_results.append(
                result
            )

            effect_record = {
                "probe_id":
                    probe_id,

                "pair_id":
                    pair_id,

                "pair_label":
                    pair_label,

                "pair_role":
                    pair_role,

                "format":
                    prompt_format,

                "semantically_correct":
                    semantically_correct,

                "correct_side":
                    correct_side,

                "head_id":
                    head_id,

                "hypothesis_role":
                    role,

                "ablation_delta":
                    ablation_delta,

                "absolute_ablation_delta":
                    float(
                        abs(
                            ablation_delta
                        )
                    ),

                "pair_causal_direction":
                    causal_direction,

                "correctness_effect":
                    correctness_effect,

                "correctness_direction":
                    correctness_direction,
            }

            all_effect_records.append(
                effect_record
            )

            print(
                f"{head_id:<6} "
                f"delta={ablation_delta:+.6f} "
                f"correctness="
                f"{correctness_effect:+.6f} "
                f"{correctness_direction}"
            )

        # ====================================================
        # Per-probe integrity
        # ====================================================

        probe_checks = {
            "runtime_identity":
                all(
                    runtime
                    .identity_checks
                    .values()
                ),

            "two_discriminative_candidates":
                (
                    0
                    in candidates_by_index
                    and
                    1
                    in candidates_by_index
                ),

            "fixed_candidate_orientation":
                (
                    candidate_results[
                        0
                    ][
                        "continuation"
                    ]
                    ==
                    lower_text
                    and
                    candidate_results[
                        1
                    ][
                        "continuation"
                    ]
                    ==
                    upper_text
                ),

            "correct_side_valid":
                (
                    correct_side
                    in
                    {
                        "lower",
                        "upper",
                    }
                ),

            "all_scale_1_controls_exact":
                all(
                    item[
                        "control_scale_1_exact"
                    ]

                    for item
                    in head_results
                ),

            "all_effects_finite":
                all(
                    np.isfinite(
                        item[
                            "effect"
                        ][
                            "ablation_delta"
                        ]
                    )
                    and
                    np.isfinite(
                        item[
                            "effect"
                        ][
                            "correctness_effect"
                        ]
                    )

                    for item
                    in head_results
                ),

            "all_heads_zeroed":
                all(
                    np.isclose(
                        item[
                            "intervention"
                        ][
                            "intervened_last_token_l2"
                        ],
                        0.0,
                    )

                    for item
                    in head_results
                ),
        }

        if not all(
            probe_checks.values()
        ):

            raise RuntimeError(
                f"{probe_id} failed EXP-0006 "
                f"integrity checks: "
                f"{probe_checks}"
            )

        # ====================================================
        # Per-probe artifact
        # ====================================================

        probe_report = {
            "mars_protocol":
                MARS_PROTOCOL_VERSION,

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "experiment_id":
                EXPECTED_EXPERIMENT_ID,

            "hypothesis_id":
                EXPECTED_HYPOTHESIS_ID,

            "plan_id":
                payload[
                    "plan_id"
                ],

            "execution_spec_id":
                spec[
                    "execution_spec_id"
                ],

            "approval_id":
                approval[
                    "approval_id"
                ],

            "status":
                "success",

            "evidence_level":
                "E3",

            "model":
                runtime.model_record(),

            "probe":
                probe,

            "candidate_pair": {
                "pair_id":
                    pair_id,

                "lower":
                    lower_text,

                "upper":
                    upper_text,
            },

            "continuation_scoring": {
                "candidates":
                    candidate_results,

                "discriminative_analysis":
                    discriminative,
            },

            "baseline":
                baseline_targets,

            "canonical_lower_minus_upper_margin":
                baseline_margin,

            "head_results":
                head_results,

            "checks":
                probe_checks,
        }

        save_json(
            probe_output,
            probe_report,
        )

        experiment_records.append(
            {
                "probe_id":
                    probe_id,

                "pair_id":
                    pair_id,

                "pair_label":
                    pair_label,

                "pair_role":
                    pair_role,

                "format":
                    prompt_format,

                "prompt":
                    prompt,

                "semantically_correct":
                    semantically_correct,

                "correct_side":
                    correct_side,

                "candidate_lower":
                    lower_text,

                "candidate_upper":
                    upper_text,

                "lower_token_id":
                    lower_token_id,

                "upper_token_id":
                    upper_token_id,

                "baseline_lower_minus_upper_margin":
                    baseline_margin,

                "artifact":
                    relative_project_path(
                        probe_output
                    ),

                "head_effects": {
                    item[
                        "head_id"
                    ]: {
                        "ablation_delta":
                            float(
                                item[
                                    "effect"
                                ][
                                    "ablation_delta"
                                ]
                            ),

                        "correctness_effect":
                            float(
                                item[
                                    "effect"
                                ][
                                    "correctness_effect"
                                ]
                            ),
                    }

                    for item
                    in head_results
                },
            }
        )

    # ========================================================
    # Aggregate head summaries
    # ========================================================

    head_summaries = {}

    pair_ids = runtime_constraints[
        "deterministic_pair_order"
    ]

    for head_spec in heads:

        head_id = head_spec[
            "head_id"
        ]

        all_records = [
            record
            for record
            in all_effect_records
            if record[
                "head_id"
            ]
            ==
            head_id
        ]

        by_pair = {}

        for pair_id in pair_ids:

            pair_records = [
                record
                for record
                in all_records
                if record[
                    "pair_id"
                ]
                ==
                pair_id
            ]

            by_pair[
                pair_id
            ] = {
                "summary":
                    summarize_effects(
                        pair_records
                    ),

                "fixed_pair_direction":
                    fixed_pair_direction(
                        pair_records,
                        decision_rules[
                            "minimum_per_pair_consistent_probes"
                        ],
                    ),
            }

        by_format = {}

        for prompt_format in (
            "words",
            "symbols",
        ):

            by_format[
                prompt_format
            ] = summarize_effects(
                [
                    record
                    for record
                    in all_records
                    if record[
                        "format"
                    ]
                    ==
                    prompt_format
                ]
            )

        lower_correct = summarize_effects(
            [
                record
                for record
                in all_records
                if record[
                    "correct_side"
                ]
                ==
                "lower"
            ]
        )

        upper_correct = summarize_effects(
            [
                record
                for record
                in all_records
                if record[
                    "correct_side"
                ]
                ==
                "upper"
            ]
        )

        head_summaries[
            head_id
        ] = {
            "hypothesis_role":
                head_spec[
                    "hypothesis_role"
                ],

            "all":
                summarize_effects(
                    all_records
                ),

            "lower_correct":
                lower_correct,

            "upper_correct":
                upper_correct,

            "by_pair":
                by_pair,

            "by_format":
                by_format,
        }

    # ========================================================
    # Top-level pair / correctness / format summaries
    # ========================================================

    pair_summaries = {}

    for pair_id in pair_ids:

        pair_summaries[
            pair_id
        ] = {
            "probe_ids": [
                record[
                    "probe_id"
                ]
                for record
                in experiment_records
                if record[
                    "pair_id"
                ]
                ==
                pair_id
            ],

            "head_summaries": {
                head_id:
                    head_summaries[
                        head_id
                    ][
                        "by_pair"
                    ][
                        pair_id
                    ]

                for head_id
                in runtime_constraints[
                    "deterministic_head_order"
                ]
            },
        }

    correctness_summaries = {
        "lower_correct_probe_ids": [
            record[
                "probe_id"
            ]
            for record
            in experiment_records
            if record[
                "correct_side"
            ]
            ==
            "lower"
        ],

        "upper_correct_probe_ids": [
            record[
                "probe_id"
            ]
            for record
            in experiment_records
            if record[
                "correct_side"
            ]
            ==
            "upper"
        ],
    }

    format_summaries = {
        "word_probe_ids": [
            record[
                "probe_id"
            ]
            for record
            in experiment_records
            if record[
                "format"
            ]
            ==
            "words"
        ],

        "symbol_probe_ids": [
            record[
                "probe_id"
            ]
            for record
            in experiment_records
            if record[
                "format"
            ]
            ==
            "symbols"
        ],
    }

    # ========================================================
    # Positive control
    # ========================================================

    positive_control_pair_id = (
        decision_rules[
            "positive_control"
        ][
            "pair_id"
        ]
    )

    minimum_per_pair = int(
        decision_rules[
            "minimum_per_pair_consistent_probes"
        ]
    )

    l1_p45 = head_summaries[
        "L1H0"
    ][
        "by_pair"
    ][
        positive_control_pair_id
    ][
        "summary"
    ]

    l4_p45 = head_summaries[
        "L4H0"
    ][
        "by_pair"
    ][
        positive_control_pair_id
    ][
        "summary"
    ]

    l1_positive_control_pass = bool(
        l1_p45[
            "negative_delta_count"
        ]
        >=
        minimum_per_pair
    )

    l4_positive_control_pass = bool(
        l4_p45[
            "positive_delta_count"
        ]
        >=
        minimum_per_pair
    )

    positive_control_pass = bool(
        l1_positive_control_pass
        and
        l4_positive_control_pass
    )

    positive_control = {
        "pair_id":
            positive_control_pair_id,

        "minimum_consistent":
            minimum_per_pair,

        "L1H0": {
            "expected_delta_sign":
                "negative",

            "observed_negative_count":
                int(
                    l1_p45[
                        "negative_delta_count"
                    ]
                ),

            "pass":
                l1_positive_control_pass,
        },

        "L4H0": {
            "expected_delta_sign":
                "positive",

            "observed_positive_count":
                int(
                    l4_p45[
                        "positive_delta_count"
                    ]
                ),

            "pass":
                l4_positive_control_pass,
        },

        "pass":
            positive_control_pass,
    }

    # ========================================================
    # Primary correctness tests
    # ========================================================

    minimum_global = int(
        decision_rules[
            "minimum_global_consistent_probes"
        ]
    )

    primary_tests = {}

    primary_head_ids = [
        "L1H0",
        "L3H3",
        "L4H0",
    ]

    for head_id in primary_head_ids:

        overall = head_summaries[
            head_id
        ][
            "all"
        ]

        primary_rule = (
            decision_rules[
                "hypothesis_primary_tests"
            ][
                head_id
            ]
        )

        expected_sign = (
            primary_rule[
                "passing_correctness_effect"
            ]
        )

        if expected_sign == "positive":

            observed_count = int(
                overall[
                    "supports_correct_count"
                ]
            )

            global_pass = bool(
                observed_count
                >=
                minimum_global
            )

            pair_sign_pass_count = int(
                sum(
                    1
                    for pair_id
                    in pair_ids
                    if (
                        head_summaries[
                            head_id
                        ][
                            "by_pair"
                        ][
                            pair_id
                        ][
                            "summary"
                        ][
                            "mean_correctness_effect"
                        ]
                        >
                        0.0
                    )
                )
            )

        elif expected_sign == "negative":

            observed_count = int(
                overall[
                    "suppresses_correct_count"
                ]
            )

            global_pass = bool(
                observed_count
                >=
                minimum_global
            )

            pair_sign_pass_count = int(
                sum(
                    1
                    for pair_id
                    in pair_ids
                    if (
                        head_summaries[
                            head_id
                        ][
                            "by_pair"
                        ][
                            pair_id
                        ][
                            "summary"
                        ][
                            "mean_correctness_effect"
                        ]
                        <
                        0.0
                    )
                )
            )

        else:

            raise RuntimeError(
                f"{head_id}: unsupported "
                f"correctness sign rule "
                f"{expected_sign!r}."
            )

        pair_generalization_pass = bool(
            pair_sign_pass_count
            >=
            3
        )

        head_pass = bool(
            global_pass
            and
            pair_generalization_pass
        )

        primary_tests[
            head_id
        ] = {
            "expected_correctness_effect_sign":
                expected_sign,

            "observed_global_consistent_count":
                observed_count,

            "minimum_global_consistent":
                minimum_global,

            "global_pass":
                global_pass,

            "pair_sign_pass_count":
                pair_sign_pass_count,

            "minimum_pairs_consistent":
                3,

            "pair_generalization_pass":
                pair_generalization_pass,

            "pass":
                head_pass,
        }

    primary_pass_count = int(
        sum(
            1
            for result
            in primary_tests.values()
            if result[
                "pass"
            ]
        )
    )

    # ========================================================
    # Weak control
    # ========================================================

    l4h1_mean_abs = float(
        head_summaries[
            "L4H1"
        ][
            "all"
        ][
            "mean_absolute_delta"
        ]
    )

    primary_mean_abs = {
        head_id:
            float(
                head_summaries[
                    head_id
                ][
                    "all"
                ][
                    "mean_absolute_delta"
                ]
            )

        for head_id
        in primary_head_ids
    }

    weak_control_pass = bool(
        all(
            l4h1_mean_abs
            <
            value

            for value
            in primary_mean_abs.values()
        )
    )

    weak_control = {
        "head_id":
            "L4H1",

        "mean_absolute_delta":
            l4h1_mean_abs,

        "primary_mean_absolute_deltas":
            primary_mean_abs,

        "pass":
            weak_control_pass,
    }

    # ========================================================
    # Token-bias diagnostics
    # ========================================================

    non_control_pair_ids = [
        pair[
            "pair_id"
        ]
        for pair
        in pairs
        if pair[
            "role"
        ]
        ==
        "non_control_pair"
    ]

    token_bias_diagnostics = {
        "by_head":
            {},
    }

    broader_signal_heads = []

    four_five_specific_signal_heads = []

    for head_id in primary_head_ids:

        pair_directions = {
            pair_id:
                head_summaries[
                    head_id
                ][
                    "by_pair"
                ][
                    pair_id
                ][
                    "fixed_pair_direction"
                ]

            for pair_id
            in pair_ids
        }

        non_control_fixed_count = int(
            sum(
                1
                for pair_id
                in non_control_pair_ids
                if pair_directions[
                    pair_id
                ][
                    "pass"
                ]
            )
        )

        broader_signal = bool(
            non_control_fixed_count
            >=
            2
        )

        p45_fixed = bool(
            pair_directions[
                "P45"
            ][
                "pass"
            ]
        )

        four_five_specific = bool(
            p45_fixed
            and
            non_control_fixed_count
            <
            2
        )

        if broader_signal:
            broader_signal_heads.append(
                head_id
            )

        if four_five_specific:
            four_five_specific_signal_heads.append(
                head_id
            )

        token_bias_diagnostics[
            "by_head"
        ][
            head_id
        ] = {
            "pair_directions":
                pair_directions,

            "non_control_fixed_pair_count":
                non_control_fixed_count,

            "broader_token_bias_signal":
                broader_signal,

            "four_five_specific_signal":
                four_five_specific,
        }

    token_bias_diagnostics[
        "broader_token_bias_signal_heads"
    ] = broader_signal_heads

    token_bias_diagnostics[
        "four_five_specific_signal_heads"
    ] = four_five_specific_signal_heads

    # ========================================================
    # Preregistered outcome
    # ========================================================

    outcome_labels = (
        decision_rules[
            "hypothesis_outcomes"
        ]
    )

    if not positive_control_pass:

        outcome_label = (
            decision_rules[
                "positive_control"
            ][
                "failure_outcome"
            ]
        )

    elif primary_pass_count == 3:

        outcome_label = (
            outcome_labels[
                "all_three_primary_heads_pass"
            ]
        )

    elif primary_pass_count in (
        1,
        2,
    ):

        outcome_label = (
            outcome_labels[
                "one_or_two_primary_heads_pass"
            ]
        )

    else:

        outcome_label = (
            outcome_labels[
                "zero_primary_heads_pass"
            ]
        )

    preregistered_tests = {
        "positive_control":
            positive_control,

        "primary_tests":
            primary_tests,

        "primary_pass_count":
            primary_pass_count,

        "weak_control":
            weak_control,

        "primary_outcome_label":
            outcome_label,
    }

    # ========================================================
    # Diagnostics
    # ========================================================

    hypothesis_diagnostics = {
        "positive_control_pass":
            positive_control_pass,

        "primary_pass_count":
            primary_pass_count,

        "all_three_primary_heads_pass":
            bool(
                primary_pass_count
                ==
                3
            ),

        "no_primary_heads_pass":
            bool(
                primary_pass_count
                ==
                0
            ),

        "weak_control_pass":
            weak_control_pass,

        "broader_token_bias_signal_heads":
            broader_signal_heads,

        "four_five_specific_signal_heads":
            four_five_specific_signal_heads,

        "outcome_label":
            outcome_label,
    }

    # ========================================================
    # Final integrity
    # ========================================================

    expected_probe_order = (
        runtime_constraints[
            "deterministic_probe_order"
        ]
    )

    expected_head_order = (
        runtime_constraints[
            "deterministic_head_order"
        ]
    )

    expected_pair_order = (
        runtime_constraints[
            "deterministic_pair_order"
        ]
    )

    observed_pair_order = []

    for record in experiment_records:

        pair_id = record[
            "pair_id"
        ]

        if (
            not observed_pair_order
            or
            observed_pair_order[
                -1
            ]
            !=
            pair_id
        ):

            observed_pair_order.append(
                pair_id
            )

    final_checks = {
        "execution_spec_verified":
            all(
                spec_checks.values()
            ),

        "p3_approval_verified":
            all(
                approval_checks.values()
            ),

        "checkpoint_sha256_verified":
            (
                actual_checkpoint_sha256
                ==
                expected_checkpoint_sha256
            ),

        "runtime_identity":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "frozen_pair_order_preserved":
            (
                observed_pair_order
                ==
                expected_pair_order
            ),

        "frozen_probe_order_preserved":
            (
                [
                    record[
                        "probe_id"
                    ]
                    for record
                    in experiment_records
                ]
                ==
                expected_probe_order
            ),

        "frozen_head_order_preserved":
            (
                [
                    head[
                        "head_id"
                    ]
                    for head
                    in heads
                ]
                ==
                expected_head_order
            ),

        "all_probes_completed":
            (
                len(
                    experiment_records
                )
                ==
                16
            ),

        "expected_effect_record_count":
            (
                len(
                    all_effect_records
                )
                ==
                64
            ),

        "all_scale_1_controls_exact":
            bool(
                all_scale_1_exact
            ),

        "all_effects_finite":
            all(
                np.isfinite(
                    record[
                        "ablation_delta"
                    ]
                )
                and
                np.isfinite(
                    record[
                        "correctness_effect"
                    ]
                )
                for record
                in all_effect_records
            ),

        "all_heads_zeroed":
            bool(
                all_heads_zeroed
            ),

        "fixed_candidate_orientation_preserved":
            all(
                record[
                    "candidate_lower"
                ]
                ==
                next(
                    pair[
                        "lower_candidate"
                    ]
                    for pair
                    in pairs
                    if pair[
                        "pair_id"
                    ]
                    ==
                    record[
                        "pair_id"
                    ]
                )
                and
                record[
                    "candidate_upper"
                ]
                ==
                next(
                    pair[
                        "upper_candidate"
                    ]
                    for pair
                    in pairs
                    if pair[
                        "pair_id"
                    ]
                    ==
                    record[
                        "pair_id"
                    ]
                )
                for record
                in experiment_records
            ),

        "positive_control_evaluated":
            isinstance(
                positive_control_pass,
                bool,
            ),

        "no_persistent_weight_modification":
            True,

        "knowledge_store_not_mutated":
            True,
    }

    if not all(
        final_checks.values()
    ):

        raise RuntimeError(
            "EXP-0006 final integrity "
            f"checks failed: {final_checks}"
        )

    # ========================================================
    # Summary artifact
    # ========================================================

    summary = {
        "mars_protocol":
            MARS_PROTOCOL_VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "hypothesis_id":
            EXPECTED_HYPOTHESIS_ID,

        "title":
            (
                "Cross-pair numerical candidate "
                "identity versus arithmetic "
                "correctness"
            ),

        "question":
            (
                "Across multiple fixed numerical "
                "candidate pairs, do the causal "
                "effects of L1H0, L3H3, and L4H0 "
                "track the semantically correct "
                "arithmetic answer, or do they "
                "track particular candidate-token "
                "identities?"
            ),

        "status":
            "success",

        "tool_version":
            LAB_TOOL_VERSION,

        "runner_version":
            RUNNER_VERSION,

        "evidence_level":
            "E3",

        "checkpoint":
            relative_project_path(
                checkpoint_path
            ),

        "checkpoint_sha256":
            actual_checkpoint_sha256,

        "model":
            runtime.model_record(),

        "plan_id":
            payload[
                "plan_id"
            ],

        "execution_spec_id":
            spec[
                "execution_spec_id"
            ],

        "execution_spec_sha256":
            spec_sha256,

        "approval_id":
            approval[
                "approval_id"
            ],

        "candidate_pairs":
            pairs,

        "selected_probe_ids": [
            record[
                "probe_id"
            ]
            for record
            in experiment_records
        ],

        "probe_count":
            len(
                experiment_records
            ),

        "head_count":
            len(
                heads
            ),

        "heads":
            heads,

        "probes":
            experiment_records,

        "head_summaries":
            head_summaries,

        "pair_summaries":
            pair_summaries,

        "correctness_summaries":
            correctness_summaries,

        "format_summaries":
            format_summaries,

        "preregistered_tests":
            preregistered_tests,

        "positive_control":
            positive_control,

        "token_bias_diagnostics":
            token_bias_diagnostics,

        "hypothesis_diagnostics":
            hypothesis_diagnostics,

        "interpretation_rule": {
            "canonical_margin":
                (
                    "LOWER candidate logit minus "
                    "UPPER candidate logit at the "
                    "first discriminative token."
                ),

            "negative_ablation_delta":
                (
                    "The intact head causally "
                    "supports the LOWER candidate "
                    "relative to the UPPER "
                    "candidate."
                ),

            "positive_ablation_delta":
                (
                    "The intact head causally "
                    "supports the UPPER candidate "
                    "relative to the LOWER "
                    "candidate."
                ),

            "positive_correctness_effect":
                (
                    "The intact head causally "
                    "supports the semantically "
                    "correct candidate."
                ),

            "negative_correctness_effect":
                (
                    "The intact head causally "
                    "suppresses the semantically "
                    "correct candidate."
                ),
        },

        "checks":
            final_checks,
    }

    save_json(
        summary_path,
        summary,
    )

    # ========================================================
    # Display summary
    # ========================================================

    print()
    print(
        "=" * 88
    )

    print(
        "EXP-0006 SUMMARY"
    )

    print(
        "=" * 88
    )

    print()
    print(
        "Positive control"
    )

    print(
        "----------------"
    )

    print(
        "P45 L1H0 negative delta >= 3/4: "
        f"{l1_positive_control_pass}"
    )

    print(
        "P45 L4H0 positive delta >= 3/4: "
        f"{l4_positive_control_pass}"
    )

    print(
        "Positive control overall: "
        f"{positive_control_pass}"
    )

    print()
    print(
        "Primary correctness tests"
    )

    print(
        "-------------------------"
    )

    for head_id in primary_head_ids:

        result = primary_tests[
            head_id
        ]

        print(
            f"{head_id}: "
            f"{result['pass']} "
            f"(global "
            f"{result['observed_global_consistent_count']}/16, "
            f"pairs "
            f"{result['pair_sign_pass_count']}/4)"
        )

    print()
    print(
        "Weak control"
    )

    print(
        "------------"
    )

    print(
        f"L4H1 weak control: "
        f"{weak_control_pass}"
    )

    print()
    print(
        "Token-bias diagnostics"
    )

    print(
        "----------------------"
    )

    print(
        "Broader token-bias signal heads: "
        +
        (
            ", ".join(
                broader_signal_heads
            )
            if broader_signal_heads
            else
            "None"
        )
    )

    print(
        "Four/five-specific signal heads: "
        +
        (
            ", ".join(
                four_five_specific_signal_heads
            )
            if four_five_specific_signal_heads
            else
            "None"
        )
    )

    print()
    print(
        "Preregistered outcome:"
    )

    print(
        f"  {outcome_label}"
    )

    print()
    print(
        "Integrity checks"
    )

    print(
        "----------------"
    )

    for name, passed in (
        final_checks.items()
    ):

        print(
            f"{name}: {passed}"
        )

    print()
    print(
        "Summary artifact:"
    )

    print(
        summary_path
    )

    print()
    print(
        "=" * 88
    )

    print(
        "SUCCESS: EXP-0006 COMPLETE."
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
