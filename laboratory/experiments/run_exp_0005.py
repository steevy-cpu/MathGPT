import argparse
import hashlib
import json
import sys

from datetime import (
    datetime,
    timezone,
)

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

RUNNER_VERSION = (
    "L9C.1"
)

EXPECTED_SPEC_VERSION = (
    "L9B.1"
)

EXPECTED_EXPERIMENT_ID = (
    "EXP-0005"
)

EXPECTED_APPROVAL_VERSION = (
    "P3A.1"
)


DEFAULT_EXECUTION_SPEC = (
    "laboratory/scientist/state/"
    "experiment_execution_spec.json"
)

DEFAULT_APPROVAL = (
    "laboratory/scientist/state/"
    "p3_approval_exp_0005.json"
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "EXP-0005 controlled runner — "
            "fixed four-vs-five causal "
            "context-transfer experiment."
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
            "specification without loading the "
            "model or running interventions. "
            "A missing approval artifact is "
            "reported as pending rather than "
            "treated as a preflight failure."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of EXP-0005 "
            "result artifacts after approval."
        ),
    )

    return parser.parse_args()


# ============================================================
# Paths / JSON / SHA
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


def save_json(
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

    temporary = path.with_name(
        path.name
        +
        ".tmp"
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

            file.write(
                "\n"
            )

        temporary.replace(
            path
        )

    finally:

        if temporary.exists():

            temporary.unlink()


def sha256_file(
    path,
):

    path = Path(
        path
    )

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
# Formatting
# ============================================================

def format_signed(
    value,
):

    if value is None:

        return "N/A"

    return (
        f"{float(value):+.6f}"
    )


def format_percentage(
    value,
):

    if value is None:

        return "N/A"

    return (
        f"{float(value):.2%}"
    )


# ============================================================
# Frozen execution-spec validation
# ============================================================

def validate_execution_spec(
    spec_artifact,
):

    spec = (
        spec_artifact.get(
            "execution_spec",
            {},
        )
    )

    payload = (
        spec.get(
            "experiment_payload",
            {},
        )
    )

    approval_contract = (
        spec.get(
            "approval_contract",
            {},
        )
    )

    runtime = (
        spec.get(
            "runtime_constraints",
            {},
        )
    )

    artifacts = (
        spec.get(
            "artifact_contract",
            {},
        )
    )

    heads = (
        payload.get(
            "target_heads",
            [],
        )
    )

    probes = (
        payload.get(
            "probes",
            [],
        )
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

        "twelve_probes":
            (
                len(
                    probes
                )
                ==
                12
                and
                len(
                    set(
                        probe_ids
                    )
                )
                ==
                12
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

        "fixed_candidates":
            (
                payload.get(
                    "fixed_candidates",
                    {},
                ).get(
                    "four"
                )
                ==
                " four."
                and
                payload.get(
                    "fixed_candidates",
                    {},
                ).get(
                    "five"
                )
                ==
                " five."
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
    }

    failed = [
        name

        for name, passed
        in checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "EXP-0005 execution specification "
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

    spec = (
        spec_artifact[
            "execution_spec"
        ]
    )

    payload = (
        spec[
            "experiment_payload"
        ]
    )

    source_plan = (
        spec[
            "source_plan"
        ]
    )

    return {
        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

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

    for (
        field,
        expected_value,
    ) in expected_binding.items():

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
            "EXP-0005 P3 approval validation "
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

            "four_support_count":
                0,

            "five_support_count":
                0,

            "zero_effect_count":
                0,

            "four_support_rate":
                None,

            "five_support_rate":
                None,
        }

    values = np.asarray(
        [
            record[
                "ablation_delta"
            ]

            for record
            in records
        ],
        dtype=np.float64,
    )

    four_support = int(
        np.sum(
            values
            <
            0.0
        )
    )

    five_support = int(
        np.sum(
            values
            >
            0.0
        )
    )

    zero_effect = int(
        len(
            values
        )
        -
        four_support
        -
        five_support
    )

    return {
        "count":
            int(
                len(
                    values
                )
            ),

        "mean_delta":
            float(
                np.mean(
                    values
                )
            ),

        "median_delta":
            float(
                np.median(
                    values
                )
            ),

        "mean_absolute_delta":
            float(
                np.mean(
                    np.abs(
                        values
                    )
                )
            ),

        "four_support_count":
            four_support,

        "five_support_count":
            five_support,

        "zero_effect_count":
            zero_effect,

        "four_support_rate":
            float(
                four_support
                /
                len(
                    values
                )
            ),

        "five_support_rate":
            float(
                five_support
                /
                len(
                    values
                )
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

    spec = (
        spec_artifact[
            "execution_spec"
        ]
    )

    payload = (
        spec[
            "experiment_payload"
        ]
    )

    artifacts = (
        spec[
            "artifact_contract"
        ]
    )

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

    four_text = (
        payload[
            "fixed_candidates"
        ][
            "four"
        ]
    )

    five_text = (
        payload[
            "fixed_candidates"
        ][
            "five"
        ]
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "=" * 84
    )

    print(
        "EXP-0005 — FIXED FOUR-vs-FIVE "
        "CAUSAL CONTEXT TRANSFER"
    )

    print(
        "=" * 84
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
        "Checkpoint:"
    )

    print(
        checkpoint_path
    )

    print()
    print(
        f"Probes: {len(probes)}"
    )

    print(
        f"Heads: {len(heads)}"
    )

    # ========================================================
    # PRE-FLIGHT MODE
    #
    # No approval required.
    # No checkpoint hashing.
    # No model load.
    # No intervention.
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
                "grant or consume approval."
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
            "Runner safety:"
        )

        print(
            "  model loaded: False"
        )

        print(
            "  interventions executed: False"
        )

        print(
            "  artifacts written: False"
        )

        print()
        print(
            "=" * 84
        )

        print(
            "SUCCESS: EXP-0005 RUNNER "
            "PREFLIGHT PASSED."
        )

        print(
            "Execution remains blocked "
            "pending valid P3 approval."
        )

        print(
            "=" * 84
        )

        return

    # ========================================================
    # REAL EXECUTION PATH
    #
    # Approval MUST be validated before:
    #   - checkpoint hashing
    #   - model loading
    #   - causal interventions
    # ========================================================

    if not approval_path.exists():

        raise PermissionError(
            "EXP-0005 execution blocked: "
            "P3 approval artifact does not exist.\n"
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
        f"Approval ID: "
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
            "EXP-0005 checkpoint SHA-256 "
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
            f"EXP-0005 summary already exists: "
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
                "EXP-0005 head identity "
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

    for (
        probe_index,
        probe,
    ) in enumerate(
        probes,
        start=1,
    ):

        probe_id = (
            probe[
                "probe_id"
            ]
        )

        prompt = (
            probe[
                "prompt"
            ]
        )

        category = (
            probe[
                "category"
            ]
        )

        semantically_correct = (
            probe[
                "semantically_correct"
            ]
        )

        probe_output = (
            output_dir
            /
            (
                f"{probe_id.lower()}"
                "_fixed_four_five_heads.json"
            )
        )

        if (
            probe_output.exists()
            and
            not args.overwrite
        ):

            raise FileExistsError(
                f"EXP-0005 probe artifact "
                f"already exists: {probe_output}"
            )

        print()
        print(
            "=" * 84
        )

        print(
            f"[{probe_index}/{len(probes)}] "
            f"{probe_id}"
        )

        print(
            "=" * 84
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Prompt: {prompt}"
        )

        print(
            "Fixed candidates: "
            f"{repr(four_text)} vs "
            f"{repr(five_text)}"
        )

        print(
            f"Semantically correct: "
            f"{semantically_correct}"
        )

        # ====================================================
        # Candidate scoring
        #
        # Candidate index 0 is ALWAYS FOUR.
        # Candidate index 1 is ALWAYS FIVE.
        # This orientation never changes.
        # ====================================================

        candidate_results = []

        for (
            candidate_index,
            continuation,
        ) in enumerate(
            [
                four_text,
                five_text,
            ]
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
                "candidates must have "
                "discriminative tokens."
            )

        four_token_id = int(
            candidates_by_index[
                0
            ][
                "token_id"
            ]
        )

        five_token_id = int(
            candidates_by_index[
                1
            ][
                "token_id"
            ]
        )

        if (
            four_token_id
            ==
            five_token_id
        ):

            raise RuntimeError(
                f"{probe_id}: FOUR and FIVE "
                "resolved to the same "
                "discriminative token."
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
        # Baseline canonical FOUR-minus-FIVE margin
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
                four_token_id
            ),
            comparison_token_id=(
                five_token_id
            ),
        )

        baseline_margin = float(
            baseline_targets[
                "logit_margin"
            ]
        )

        print()
        print(
            "Baseline FOUR-FIVE margin: "
            f"{baseline_margin:+.6f}"
        )

        print(
            "Baseline direction: "
            +
            (
                "FOUR"
                if baseline_margin > 0.0
                else
                "FIVE"
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

            head_id = (
                head_spec[
                    "head_id"
                ]
            )

            role = (
                head_spec[
                    "role"
                ]
            )

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
                    four_token_id
                ),
                comparison_token_id=(
                    five_token_id
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

            if ablation_delta < 0.0:

                causal_direction = (
                    "intact_head_supports_four"
                )

            elif ablation_delta > 0.0:

                causal_direction = (
                    "intact_head_supports_five"
                )

            else:

                causal_direction = (
                    "no_detected_direction"
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

            four_rank_delta = int(
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

            five_rank_delta = int(
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

                "role":
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
                    "baseline_four_minus_five_margin":
                        baseline_margin,

                    "ablated_four_minus_five_margin":
                        ablated_margin,

                    "ablation_delta":
                        ablation_delta,

                    "absolute_ablation_delta":
                        float(
                            abs(
                                ablation_delta
                            )
                        ),

                    "causal_direction":
                        causal_direction,

                    "four_rank_delta":
                        four_rank_delta,

                    "five_rank_delta":
                        five_rank_delta,

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

                "category":
                    category,

                "semantically_correct":
                    semantically_correct,

                "head_id":
                    head_id,

                "role":
                    role,

                "ablation_delta":
                    ablation_delta,

                "absolute_ablation_delta":
                    float(
                        abs(
                            ablation_delta
                        )
                    ),

                "causal_direction":
                    causal_direction,
            }

            all_effect_records.append(
                effect_record
            )

            print(
                f"{head_id:<6} "
                f"delta={ablation_delta:+.6f} "
                f"{causal_direction}"
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
                    four_text
                    and
                    candidate_results[
                        1
                    ][
                        "continuation"
                    ]
                    ==
                    five_text
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
                f"{probe_id} failed EXP-0005 "
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

            "fixed_candidates": {
                "four":
                    four_text,

                "five":
                    five_text,
            },

            "continuation_scoring": {
                "candidates":
                    candidate_results,

                "discriminative_analysis":
                    discriminative,
            },

            "baseline":
                baseline_targets,

            "canonical_margin":
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

                "category":
                    category,

                "prompt":
                    prompt,

                "semantically_correct":
                    semantically_correct,

                "candidate_four":
                    four_text,

                "candidate_five":
                    five_text,

                "four_token_id":
                    four_token_id,

                "five_token_id":
                    five_token_id,

                "baseline_four_minus_five_margin":
                    baseline_margin,

                "artifact":
                    relative_project_path(
                        probe_output
                    ),

                "head_effects": {
                    item[
                        "head_id"
                    ]:
                        float(
                            item[
                                "effect"
                            ][
                                "ablation_delta"
                            ]
                        )

                    for item
                    in head_results
                },
            }
        )

    # ========================================================
    # Aggregate head summaries
    # ========================================================

    head_summaries = {}

    for head_spec in heads:

        head_id = (
            head_spec[
                "head_id"
            ]
        )

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

        four_correct_records = [
            record

            for record
            in all_records

            if record[
                "semantically_correct"
            ]
            ==
            "four"
        ]

        five_correct_records = [
            record

            for record
            in all_records

            if record[
                "semantically_correct"
            ]
            ==
            "five"
        ]

        categories = {}

        for category in sorted(
            {
                record[
                    "category"
                ]

                for record
                in all_records
            }
        ):

            categories[
                category
            ] = summarize_effects(
                [
                    record

                    for record
                    in all_records

                    if record[
                        "category"
                    ]
                    ==
                    category
                ]
            )

        head_summaries[
            head_id
        ] = {
            "role":
                head_spec[
                    "role"
                ],

            "all":
                summarize_effects(
                    all_records
                ),

            "four_correct":
                summarize_effects(
                    four_correct_records
                ),

            "five_correct":
                summarize_effects(
                    five_correct_records
                ),

            "by_category":
                categories,
        }

    # ========================================================
    # Top-level subset summaries
    # ========================================================

    subset_summaries = {
        "four_correct_probe_ids": [
            record[
                "probe_id"
            ]

            for record
            in experiment_records

            if record[
                "semantically_correct"
            ]
            ==
            "four"
        ],

        "five_correct_probe_ids": [
            record[
                "probe_id"
            ]

            for record
            in experiment_records

            if record[
                "semantically_correct"
            ]
            ==
            "five"
        ],

        "categories": {
            category:
                [
                    record[
                        "probe_id"
                    ]

                    for record
                    in experiment_records

                    if record[
                        "category"
                    ]
                    ==
                    category
                ]

            for category
            in sorted(
                {
                    record[
                        "category"
                    ]

                    for record
                    in experiment_records
                }
            )
        },
    }

    # ========================================================
    # Preregistered hypothesis tests
    # ========================================================

    rules = (
        payload[
            "decision_rules"
        ]
    )

    minimum_consistent = int(
        rules[
            "minimum_consistent_probes"
        ]
    )

    l1h0 = (
        head_summaries[
            "L1H0"
        ][
            "all"
        ]
    )

    l4h0 = (
        head_summaries[
            "L4H0"
        ][
            "all"
        ]
    )

    l4h1 = (
        head_summaries[
            "L4H1"
        ][
            "all"
        ]
    )

    l1h0_pass = bool(
        l1h0[
            "four_support_count"
        ]
        >=
        minimum_consistent
    )

    l4h0_pass = bool(
        l4h0[
            "five_support_count"
        ]
        >=
        minimum_consistent
    )

    weak_control_pass = bool(
        l4h1[
            "mean_absolute_delta"
        ]
        <
        l1h0[
            "mean_absolute_delta"
        ]
        and
        l4h1[
            "mean_absolute_delta"
        ]
        <
        l4h0[
            "mean_absolute_delta"
        ]
    )

    if (
        l1h0_pass
        and
        l4h0_pass
    ):

        outcome_label = (
            rules[
                "outcome_labels"
            ][
                "both_primary_heads_pass"
            ]
        )

    elif (
        l1h0_pass
        or
        l4h0_pass
    ):

        outcome_label = (
            rules[
                "outcome_labels"
            ][
                "exactly_one_primary_head_passes"
            ]
        )

    else:

        outcome_label = (
            rules[
                "outcome_labels"
            ][
                "neither_primary_head_passes"
            ]
        )

    preregistered_tests = {
        "minimum_consistent_probes":
            minimum_consistent,

        "total_probe_count":
            int(
                rules[
                    "total_probe_count"
                ]
            ),

        "directional_consistency_threshold":
            float(
                rules[
                    "directional_consistency_threshold"
                ]
            ),

        "L1H0_primary_four_direction": {
            "observed_four_support_count":
                int(
                    l1h0[
                        "four_support_count"
                    ]
                ),

            "observed_four_support_rate":
                float(
                    l1h0[
                        "four_support_rate"
                    ]
                ),

            "pass":
                l1h0_pass,
        },

        "L4H0_primary_five_direction": {
            "observed_five_support_count":
                int(
                    l4h0[
                        "five_support_count"
                    ]
                ),

            "observed_five_support_rate":
                float(
                    l4h0[
                        "five_support_rate"
                    ]
                ),

            "pass":
                l4h0_pass,
        },

        "L4H1_weak_control": {
            "L4H1_mean_absolute_delta":
                float(
                    l4h1[
                        "mean_absolute_delta"
                    ]
                ),

            "L1H0_mean_absolute_delta":
                float(
                    l1h0[
                        "mean_absolute_delta"
                    ]
                ),

            "L4H0_mean_absolute_delta":
                float(
                    l4h0[
                        "mean_absolute_delta"
                    ]
                ),

            "pass":
                weak_control_pass,
        },

        "primary_outcome_label":
            outcome_label,
    }

    # ========================================================
    # Diagnostics
    # ========================================================

    hypothesis_diagnostics = {
        "L1H0_fixed_four_direction_pass":
            l1h0_pass,

        "L4H0_fixed_five_direction_pass":
            l4h0_pass,

        "both_primary_heads_pass":
            bool(
                l1h0_pass
                and
                l4h0_pass
            ),

        "weak_control_pass":
            weak_control_pass,

        "outcome_label":
            outcome_label,

        "L1H0_four_support_rate_four_correct":
            (
                head_summaries[
                    "L1H0"
                ][
                    "four_correct"
                ][
                    "four_support_rate"
                ]
            ),

        "L1H0_four_support_rate_five_correct":
            (
                head_summaries[
                    "L1H0"
                ][
                    "five_correct"
                ][
                    "four_support_rate"
                ]
            ),

        "L4H0_five_support_rate_four_correct":
            (
                head_summaries[
                    "L4H0"
                ][
                    "four_correct"
                ][
                    "five_support_rate"
                ]
            ),

        "L4H0_five_support_rate_five_correct":
            (
                head_summaries[
                    "L4H0"
                ][
                    "five_correct"
                ][
                    "five_support_rate"
                ]
            ),
    }

    # ========================================================
    # Final integrity
    # ========================================================

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
                [
                    probe[
                        "probe_id"
                    ]

                    for probe
                    in probes
                ]
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
                spec[
                    "runtime_constraints"
                ][
                    "deterministic_head_order"
                ]
            ),

        "all_probes_completed":
            (
                len(
                    experiment_records
                )
                ==
                12
            ),

        "expected_effect_record_count":
            (
                len(
                    all_effect_records
                )
                ==
                48
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
                    "candidate_four"
                ]
                ==
                four_text
                and
                record[
                    "candidate_five"
                ]
                ==
                five_text

                for record
                in experiment_records
            ),

        "no_persistent_weight_modification":
            True,
    }

    if not all(
        final_checks.values()
    ):

        raise RuntimeError(
            "EXP-0005 final integrity "
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

        "title":
            (
                "Fixed four-vs-five causal "
                "context transfer"
            ),

        "question":
            (
                "When the candidate tokens four "
                "and five are held fixed while "
                "prompt semantics change, do "
                "L1H0 and L4H0 retain opposing "
                "causal directions?"
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

        "subset_summaries":
            subset_summaries,

        "preregistered_tests":
            preregistered_tests,

        "hypothesis_diagnostics":
            hypothesis_diagnostics,

        "interpretation_rule": {
            "canonical_margin":
                (
                    "FOUR logit minus FIVE logit "
                    "at the first discriminative "
                    "token."
                ),

            "negative_ablation_delta":
                (
                    "The intact head causally "
                    "supports FOUR relative to "
                    "FIVE."
                ),

            "positive_ablation_delta":
                (
                    "The intact head causally "
                    "supports FIVE relative to "
                    "FOUR."
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
        "=" * 84
    )

    print(
        "EXP-0005 SUMMARY"
    )

    print(
        "=" * 84
    )

    print()

    for head_id in (
        spec[
            "runtime_constraints"
        ][
            "deterministic_head_order"
        ]
    ):

        result = (
            head_summaries[
                head_id
            ]
        )

        overall = (
            result[
                "all"
            ]
        )

        four_correct = (
            result[
                "four_correct"
            ]
        )

        five_correct = (
            result[
                "five_correct"
            ]
        )

        print(
            f"{head_id}"
        )

        print(
            f"  mean delta: "
            f"{format_signed(overall['mean_delta'])}"
        )

        print(
            f"  mean |delta|: "
            f"{format_signed(overall['mean_absolute_delta'])}"
        )

        print(
            f"  FOUR-support rate: "
            f"{format_percentage(overall['four_support_rate'])}"
        )

        print(
            f"  FIVE-support rate: "
            f"{format_percentage(overall['five_support_rate'])}"
        )

        print(
            f"  FOUR-support rate "
            f"(four correct): "
            f"{format_percentage(four_correct['four_support_rate'])}"
        )

        print(
            f"  FOUR-support rate "
            f"(five correct): "
            f"{format_percentage(five_correct['four_support_rate'])}"
        )

        print()

    print(
        "Preregistered tests"
    )

    print(
        "-------------------"
    )

    print(
        "L1H0 fixed FOUR direction: "
        f"{l1h0_pass}"
    )

    print(
        "L4H0 fixed FIVE direction: "
        f"{l4h0_pass}"
    )

    print(
        "L4H1 weak control: "
        f"{weak_control_pass}"
    )

    print(
        "Outcome: "
        f"{outcome_label}"
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
        "=" * 84
    )

    print(
        "SUCCESS: EXP-0005 COMPLETE."
    )

    print(
        "=" * 84
    )


if __name__ == "__main__":

    main()
