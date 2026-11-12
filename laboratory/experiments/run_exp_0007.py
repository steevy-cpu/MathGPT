import argparse
import hashlib
import json
import math
import sys

from collections import defaultdict
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
# Imports
#
# IMPORTANT:
# Importing these modules does NOT instantiate LaboratoryRuntime.
# The model is loaded only after the approval gate passes.
# ============================================================

from laboratory.core.runtime import (
    LaboratoryRuntime,
)

from laboratory.tools.attention_head_ablation import (
    forward_with_head_scale,
)

from training.parameters import (
    named_parameters,
)


# ============================================================
# Identity
# ============================================================

RUNNER_VERSION = "L12D.1"

EXPECTED_SPEC_VERSION = "L12C.1"

EXPECTED_SPEC_ID = (
    "SPEC-F220BBDF2ABECA3E"
)

EXPECTED_SPEC_SHA256 = (
    "062b41f1bb535cf0427d9bc4464a27455fd250d0b90ae47d988d29ac887303c9"
)

EXPECTED_PLAN_ID = (
    "PLAN-5825AD67CA0D2EC0"
)

EXPECTED_PLAN_SHA256 = (
    "3bb6eebe7da92546170e97decc9768388b483d08b06b07fb922ca3a3e30ef8cb"
)

EXPECTED_EXPERIMENT_ID = "EXP-0007"
EXPECTED_HYPOTHESIS_ID = "HYP-V03-003"

EXPECTED_CHECKPOINT_SHA256 = (
    "4cd9b8f1c68e4079449519fd601b573f8ea48137bc74172c7b37e108c9707cf3"
)

EXPECTED_HEAD_ORDER = [
    "L1H0",
    "L4H0",
    "L4H1",
]

EXPECTED_PROBE_COUNT = 20
EXPECTED_PRIMARY_PROBE_COUNT = 16
EXPECTED_EFFECT_COUNT = 60

EXPECTED_APPROVAL_VERSION = "L12E.1"


# ============================================================
# Default paths
# ============================================================

DEFAULT_SPEC_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec_exp_0007.json"
)

DEFAULT_APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0007.json"
)

KNOWLEDGE_PATHS = [
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "hypotheses.json",

    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "findings.json",

    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "experiment_registry.json",

    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "research_state.json",

    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "evidence_ledger.json",
]


# ============================================================
# Helpers
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
    path = project_path(path)

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


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


def knowledge_hashes():
    values = {}

    for path in KNOWLEDGE_PATHS:

        require(
            path.exists(),
            (
                "Knowledge file missing: "
                f"{path}"
            ),
        )

        values[
            relative_project_path(
                path
            )
        ] = file_sha256(
            path
        )

    return values


def array_digest(
    digest,
    name,
    array,
):
    array = np.asarray(
        array
    )

    digest.update(
        name.encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            array.dtype
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        json.dumps(
            list(
                array.shape
            )
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        np.ascontiguousarray(
            array
        ).tobytes(
            order="C"
        )
    )


def model_state_sha256(
    model,
):
    digest = hashlib.sha256()

    for (
        name,
        parameter,
        _,
    ) in named_parameters(
        model
    ):

        array_digest(
            digest,
            name,
            parameter,
        )

    return digest.hexdigest()


def sign_label(value):
    if value > 0.0:
        return "positive"

    if value < 0.0:
        return "negative"

    return "zero"


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L12D — controlled EXP-0007 "
            "runner. Validates the frozen execution "
            "spec and P3 approval before loading "
            "the mature-v0.3 model."
        )
    )

    parser.add_argument(
        "--execution-spec",
        default=str(
            DEFAULT_SPEC_PATH
        ),
    )

    parser.add_argument(
        "--approval",
        default=str(
            DEFAULT_APPROVAL_PATH
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate frozen spec, checkpoint "
            "bytes, source bindings, task order, "
            "and approval absence/presence without "
            "loading the model."
        ),
    )

    parser.add_argument(
        "--overwrite-result",
        action="store_true",
        help=(
            "Allow replacement of an existing "
            "EXP-0007 result artifact. Normally "
            "disabled to protect frozen evidence."
        ),
    )

    return parser.parse_args()


# ============================================================
# Spec validation
# ============================================================

def validate_spec(
    *,
    spec_document,
    spec_path,
):

    spec_sha256 = file_sha256(
        spec_path
    )

    spec = spec_document.get(
        "execution_spec",
        {},
    )

    tasks = spec.get(
        "execution_tasks",
        [],
    )

    protocol = spec.get(
        "execution_protocol",
        {},
    )

    integrity = spec.get(
        "integrity_requirements",
        {},
    )

    checks = {
        "spec_version":
            (
                spec_document.get(
                    "spec_version"
                )
                ==
                EXPECTED_SPEC_VERSION
            ),

        "spec_id":
            (
                spec_document.get(
                    "execution_spec_id"
                )
                ==
                EXPECTED_SPEC_ID
                and
                spec.get(
                    "execution_spec_id"
                )
                ==
                EXPECTED_SPEC_ID
            ),

        "spec_sha256":
            (
                spec_sha256
                ==
                EXPECTED_SPEC_SHA256
            ),

        "experiment_id":
            (
                spec.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "hypothesis_id":
            (
                spec.get(
                    "target_hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "plan_id":
            (
                spec.get(
                    "plan_binding",
                    {},
                ).get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
            ),

        "plan_sha":
            (
                spec.get(
                    "plan_binding",
                    {},
                ).get(
                    "plan_sha256"
                )
                ==
                EXPECTED_PLAN_SHA256
            ),

        "checkpoint_sha":
            (
                spec.get(
                    "checkpoint_binding",
                    {},
                ).get(
                    "sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        "permission_p3":
            (
                spec.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "human_approval_required":
            (
                spec.get(
                    "human_approval_required"
                )
                is True
            ),

        "execution_not_authorized_in_spec":
            (
                spec.get(
                    "execution_authorized"
                )
                is False
            ),

        "approval_absent_in_spec":
            (
                spec.get(
                    "approval_present"
                )
                is False
            ),

        "task_count":
            (
                len(
                    tasks
                )
                ==
                EXPECTED_EFFECT_COUNT
            ),

        "effect_indexes":
            (
                [
                    int(
                        task[
                            "effect_index"
                        ]
                    )
                    for task
                    in tasks
                ]
                ==
                list(
                    range(
                        1,
                        EXPECTED_EFFECT_COUNT
                        +
                        1,
                    )
                )
            ),

        "all_scale_zero":
            all(
                float(
                    task.get(
                        "head_scale"
                    )
                )
                ==
                0.0
                for task
                in tasks
            ),

        "head_order":
            all(
                [
                    tasks[
                        index
                    ][
                        "head_id"
                    ],
                    tasks[
                        index
                        +
                        1
                    ][
                        "head_id"
                    ],
                    tasks[
                        index
                        +
                        2
                    ][
                        "head_id"
                    ],
                ]
                ==
                EXPECTED_HEAD_ORDER

                for index
                in range(
                    0,
                    EXPECTED_EFFECT_COUNT,
                    3,
                )
            ),

        "integrity_effect_count":
            (
                int(
                    integrity.get(
                        "exact_effect_count"
                    )
                )
                ==
                EXPECTED_EFFECT_COUNT
            ),

        "integrity_probe_count":
            (
                int(
                    integrity.get(
                        "exact_probe_count"
                    )
                )
                ==
                EXPECTED_PROBE_COUNT
            ),

        "integrity_primary_probe_count":
            (
                int(
                    integrity.get(
                        "exact_primary_probe_count"
                    )
                )
                ==
                EXPECTED_PRIMARY_PROBE_COUNT
            ),

        "integrity_head_order":
            (
                integrity.get(
                    "exact_head_order"
                )
                ==
                EXPECTED_HEAD_ORDER
            ),

        "protocol_load_model_once":
            (
                protocol.get(
                    "load_model_once"
                )
                is True
            ),

        "protocol_checkpoint_write_blocked":
            (
                protocol.get(
                    "checkpoint_write_allowed"
                )
                is False
            ),

        "protocol_optimizer_step_blocked":
            (
                protocol.get(
                    "optimizer_step_allowed"
                )
                is False
            ),

        "protocol_knowledge_write_blocked":
            (
                protocol.get(
                    "knowledge_store_write_allowed"
                )
                is False
            ),

        "protocol_registry_write_blocked":
            (
                protocol.get(
                    "experiment_registry_write_allowed"
                )
                is False
            ),

        "protocol_result_write_allowed":
            (
                protocol.get(
                    "result_artifact_write_allowed"
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
            "EXP-0007 execution-spec "
            f"validation failed: {failed}"
        )

    return {
        "checks":
            checks,

        "spec":
            spec,

        "tasks":
            tasks,

        "protocol":
            protocol,

        "spec_sha256":
            spec_sha256,
    }


# ============================================================
# Frozen external bindings
# ============================================================

def validate_frozen_bindings(
    spec,
):

    protocol = spec[
        "execution_protocol"
    ]

    checkpoint_path = project_path(
        spec[
            "checkpoint_binding"
        ][
            "path"
        ]
    )

    runtime_path = project_path(
        protocol[
            "runtime_module"
        ]
    )

    intervention_path = project_path(
        protocol[
            "intervention_module"
        ]
    )

    for path in (
        checkpoint_path,
        runtime_path,
        intervention_path,
    ):
        require(
            path.exists(),
            (
                "Frozen execution binding "
                f"is missing: {path}"
            ),
        )

    actual_checkpoint_sha = file_sha256(
        checkpoint_path
    )

    actual_runtime_sha = file_sha256(
        runtime_path
    )

    actual_intervention_sha = file_sha256(
        intervention_path
    )

    checks = {
        "checkpoint_sha":
            (
                actual_checkpoint_sha
                ==
                EXPECTED_CHECKPOINT_SHA256
                ==
                spec[
                    "checkpoint_binding"
                ][
                    "sha256"
                ]
            ),

        "runtime_source_sha":
            (
                actual_runtime_sha
                ==
                protocol[
                    "runtime_sha256"
                ]
            ),

        "intervention_source_sha":
            (
                actual_intervention_sha
                ==
                protocol[
                    "intervention_module_sha256"
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
            "EXP-0007 frozen source/checkpoint "
            f"binding failed: {failed}"
        )

    return {
        "checks":
            checks,

        "checkpoint_path":
            checkpoint_path,

        "runtime_path":
            runtime_path,

        "intervention_path":
            intervention_path,

        "checkpoint_sha256":
            actual_checkpoint_sha,

        "runtime_sha256":
            actual_runtime_sha,

        "intervention_sha256":
            actual_intervention_sha,
    }


# ============================================================
# Approval gate
# ============================================================

def validate_approval(
    *,
    approval_path,
    spec_validation,
):

    approval_path = project_path(
        approval_path
    )

    if not approval_path.exists():
        raise PermissionError(
            "EXP-0007 execution blocked: "
            "P3 approval artifact does not exist."
        )

    approval = load_json(
        approval_path
    )

    bindings = approval.get(
        "bindings",
        {},
    )

    scope = approval.get(
        "authorized_scope",
        {},
    )

    checks = {
        "approval_version":
            (
                approval.get(
                    "approval_version"
                )
                ==
                EXPECTED_APPROVAL_VERSION
            ),

        "approval_id":
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

        "experiment_id":
            (
                approval.get(
                    "experiment_id"
                )
                ==
                EXPECTED_EXPERIMENT_ID
            ),

        "hypothesis_id":
            (
                approval.get(
                    "target_hypothesis_id"
                )
                ==
                EXPECTED_HYPOTHESIS_ID
            ),

        "permission":
            (
                approval.get(
                    "permission"
                )
                ==
                "P3"
            ),

        "approval_type":
            (
                approval.get(
                    "approval_type"
                )
                ==
                "explicit_human"
            ),

        "execution_authorized":
            (
                approval.get(
                    "execution_authorized"
                )
                is True
            ),

        "spec_id":
            (
                bindings.get(
                    "execution_spec_id"
                )
                ==
                EXPECTED_SPEC_ID
            ),

        "spec_sha":
            (
                bindings.get(
                    "execution_spec_sha256"
                )
                ==
                spec_validation[
                    "spec_sha256"
                ]
                ==
                EXPECTED_SPEC_SHA256
            ),

        "plan_id":
            (
                bindings.get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
            ),

        "plan_sha":
            (
                bindings.get(
                    "plan_sha256"
                )
                ==
                EXPECTED_PLAN_SHA256
            ),

        "checkpoint_sha":
            (
                bindings.get(
                    "checkpoint_sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        "scope_intervention":
            (
                scope.get(
                    "intervention"
                )
                ==
                "temporary_attention_head_ablation"
            ),

        "scope_scale":
            (
                float(
                    scope.get(
                        "head_scale"
                    )
                )
                ==
                0.0
            ),

        "scope_effect_count":
            (
                int(
                    scope.get(
                        "maximum_causal_effects"
                    )
                )
                ==
                EXPECTED_EFFECT_COUNT
            ),

        "scope_persistent_modification_blocked":
            (
                scope.get(
                    "persistent_weight_modification"
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
        raise PermissionError(
            "EXP-0007 P3 approval artifact "
            f"failed validation: {failed}"
        )

    return {
        "approval":
            approval,

        "checks":
            checks,

        "approval_path":
            approval_path,

        "approval_sha256":
            file_sha256(
                approval_path
            ),
    }


# ============================================================
# Frozen-task grouping
# ============================================================

def group_tasks_by_probe(
    tasks,
):

    groups = []

    for index in range(
        0,
        len(
            tasks
        ),
        3,
    ):

        group = tasks[
            index:
            index
            +
            3
        ]

        require(
            len(
                group
            )
            ==
            3,
            (
                "Execution task grouping "
                "is incomplete."
            ),
        )

        probe_ids = {
            item[
                "probe_id"
            ]
            for item
            in group
        }

        pair_ids = {
            item[
                "pair_id"
            ]
            for item
            in group
        }

        prompts = {
            item[
                "prompt"
            ]
            for item
            in group
        }

        require(
            len(
                probe_ids
            )
            ==
            1,
            (
                "Task group contains "
                "multiple probe IDs."
            ),
        )

        require(
            len(
                pair_ids
            )
            ==
            1,
            (
                "Task group contains "
                "multiple pair IDs."
            ),
        )

        require(
            len(
                prompts
            )
            ==
            1,
            (
                "Task group contains "
                "multiple prompts."
            ),
        )

        require(
            [
                item[
                    "head_id"
                ]
                for item
                in group
            ]
            ==
            EXPECTED_HEAD_ORDER,
            (
                "Frozen head order changed "
                "inside probe group."
            ),
        )

        groups.append(
            group
        )

    require(
        len(
            groups
        )
        ==
        EXPECTED_PROBE_COUNT,
        (
            "Expected exactly 20 "
            "probe groups."
        ),
    )

    return groups


# ============================================================
# Causal measurement
# ============================================================

def measure_margin(
    logits,
    lower_token_id,
    upper_token_id,
):
    next_logits = logits[
        0,
        -1,
        :
    ]

    lower_logit = float(
        next_logits[
            int(
                lower_token_id
            )
        ]
    )

    upper_logit = float(
        next_logits[
            int(
                upper_token_id
            )
        ]
    )

    return {
        "lower_logit":
            lower_logit,

        "upper_logit":
            upper_logit,

        "margin":
            (
                lower_logit
                -
                upper_logit
            ),
    }


def run_probe_group(
    *,
    runtime,
    group,
):

    exemplar = group[
        0
    ]

    context_ids = (
        [
            int(
                value
            )
            for value
            in exemplar[
                "prompt_token_ids"
            ]
        ]
        +
        [
            int(
                value
            )
            for value
            in exemplar[
                "shared_prefix_token_ids"
            ]
        ]
    )

    require(
        bool(
            context_ids
        ),
        (
            f"{exemplar['probe_id']}: "
            "empty scoring context."
        ),
    )

    require(
        len(
            context_ids
        )
        <=
        runtime.model.context_length,
        (
            f"{exemplar['probe_id']}: "
            "scoring context exceeds model "
            "context length."
        ),
    )

    input_ids = np.asarray(
        [
            context_ids
        ],
        dtype=np.int64,
    )

    baseline_logits = (
        runtime.model.forward(
            input_ids
        )
    )

    baseline = measure_margin(
        baseline_logits,
        exemplar[
            "lower_discriminative_token_id"
        ],
        exemplar[
            "upper_discriminative_token_id"
        ],
    )

    effect_records = []

    for task in group:

        (
            intervention_logits,
            intervention_record,
        ) = forward_with_head_scale(
            runtime=runtime,
            token_ids=input_ids,
            target_layer=int(
                task[
                    "layer"
                ]
            ),
            target_head=int(
                task[
                    "head"
                ]
            ),
            head_scale=float(
                task[
                    "head_scale"
                ]
            ),
        )

        intervention = measure_margin(
            intervention_logits,
            task[
                "lower_discriminative_token_id"
            ],
            task[
                "upper_discriminative_token_id"
            ],
        )

        ablation_delta = float(
            intervention[
                "margin"
            ]
            -
            baseline[
                "margin"
            ]
        )

        if (
            task[
                "correct_candidate"
            ]
            ==
            "lower"
        ):

            correctness_effect = (
                -ablation_delta
            )

        elif (
            task[
                "correct_candidate"
            ]
            ==
            "upper"
        ):

            correctness_effect = (
                ablation_delta
            )

        else:

            raise RuntimeError(
                f"{task['probe_id']}: "
                "unknown correct_candidate."
            )

        record = {
            "effect_index":
                int(
                    task[
                        "effect_index"
                    ]
                ),

            "pair_id":
                task[
                    "pair_id"
                ],

            "pair_role":
                task[
                    "pair_role"
                ],

            "probe_id":
                task[
                    "probe_id"
                ],

            "form":
                task[
                    "form"
                ],

            "correct_candidate":
                task[
                    "correct_candidate"
                ],

            "prompt":
                task[
                    "prompt"
                ],

            "scoring_context_token_ids":
                context_ids,

            "lower_label":
                task[
                    "lower_label"
                ],

            "upper_label":
                task[
                    "upper_label"
                ],

            "lower_discriminative_token_id":
                int(
                    task[
                        "lower_discriminative_token_id"
                    ]
                ),

            "upper_discriminative_token_id":
                int(
                    task[
                        "upper_discriminative_token_id"
                    ]
                ),

            "head_id":
                task[
                    "head_id"
                ],

            "layer":
                int(
                    task[
                        "layer"
                    ]
                ),

            "head":
                int(
                    task[
                        "head"
                    ]
                ),

            "head_scale":
                float(
                    task[
                        "head_scale"
                    ]
                ),

            "baseline_lower_logit":
                baseline[
                    "lower_logit"
                ],

            "baseline_upper_logit":
                baseline[
                    "upper_logit"
                ],

            "baseline_margin":
                baseline[
                    "margin"
                ],

            "ablated_lower_logit":
                intervention[
                    "lower_logit"
                ],

            "ablated_upper_logit":
                intervention[
                    "upper_logit"
                ],

            "ablated_margin":
                intervention[
                    "margin"
                ],

            "ablation_delta":
                ablation_delta,

            "raw_delta_sign":
                sign_label(
                    ablation_delta
                ),

            "correctness_effect":
                float(
                    correctness_effect
                ),

            "correctness_effect_positive":
                bool(
                    correctness_effect
                    >
                    0.0
                ),

            "intervention_record":
                intervention_record,
        }

        effect_records.append(
            record
        )

    return {
        "pair_id":
            exemplar[
                "pair_id"
            ],

        "pair_role":
            exemplar[
                "pair_role"
            ],

        "probe_id":
            exemplar[
                "probe_id"
            ],

        "prompt":
            exemplar[
                "prompt"
            ],

        "scoring_context_token_ids":
            context_ids,

        "baseline":
            baseline,

        "effects":
            effect_records,
    }


# ============================================================
# Analysis
# ============================================================

def primary_effects(
    effects,
    head_id=None,
):
    values = [
        item
        for item
        in effects
        if item[
            "pair_role"
        ]
        ==
        "primary_replication"
    ]

    if head_id is not None:
        values = [
            item
            for item
            in values
            if item[
                "head_id"
            ]
            ==
            head_id
        ]

    return values


def pair_head_effects(
    effects,
    *,
    pair_id,
    head_id,
):
    return [
        item
        for item
        in effects
        if item[
            "pair_id"
        ]
        ==
        pair_id
        and
        item[
            "head_id"
        ]
        ==
        head_id
    ]


def direction_stability_record(
    values,
):
    positive = sum(
        item[
            "ablation_delta"
        ]
        >
        0.0
        for item
        in values
    )

    negative = sum(
        item[
            "ablation_delta"
        ]
        <
        0.0
        for item
        in values
    )

    zero = len(
        values
    ) - positive - negative

    stable = (
        positive
        >=
        3
        or
        negative
        >=
        3
    )

    if positive >= 3:
        dominant_sign = "positive"

    elif negative >= 3:
        dominant_sign = "negative"

    else:
        dominant_sign = None

    return {
        "effect_count":
            len(
                values
            ),

        "positive_count":
            int(
                positive
            ),

        "negative_count":
            int(
                negative
            ),

        "zero_count":
            int(
                zero
            ),

        "stable":
            bool(
                stable
            ),

        "dominant_sign":
            dominant_sign,
    }


def analyze_results(
    *,
    effects,
    spec,
):

    primary_pair_ids = [
        "P12",
        "P67",
        "P78",
        "P89",
    ]

    primary_heads = [
        "L1H0",
        "L4H0",
    ]

    # --------------------------------------------------------
    # Per pair × head stability
    # --------------------------------------------------------

    pair_direction = {}

    for head_id in primary_heads:

        pair_direction[
            head_id
        ] = {}

        for pair_id in primary_pair_ids:

            values = pair_head_effects(
                effects,
                pair_id=pair_id,
                head_id=head_id,
            )

            require(
                len(
                    values
                )
                ==
                4,
                (
                    f"{head_id}/{pair_id}: "
                    "expected four effects."
                ),
            )

            pair_direction[
                head_id
            ][
                pair_id
            ] = (
                direction_stability_record(
                    values
                )
            )

    # --------------------------------------------------------
    # Cross-pair generalization
    # --------------------------------------------------------

    cross_pair = {}

    for head_id in primary_heads:

        stable_pairs = [
            pair_id
            for pair_id
            in primary_pair_ids
            if pair_direction[
                head_id
            ][
                pair_id
            ][
                "stable"
            ]
        ]

        cross_pair[
            head_id
        ] = {
            "stable_pair_ids":
                stable_pairs,

            "stable_pair_count":
                len(
                    stable_pairs
                ),

            "pair_count":
                4,

            "pass":
                (
                    len(
                        stable_pairs
                    )
                    >=
                    3
                ),
        }

    # --------------------------------------------------------
    # Correctness tracking
    # --------------------------------------------------------

    correctness = {}

    for head_id in primary_heads:

        values = primary_effects(
            effects,
            head_id=head_id,
        )

        positive_count = sum(
            item[
                "correctness_effect"
            ]
            >
            0.0
            for item
            in values
        )

        correctness[
            head_id
        ] = {
            "positive_count":
                int(
                    positive_count
                ),

            "probe_count":
                len(
                    values
                ),

            "strong_correctness_tracking":
                bool(
                    positive_count
                    >=
                    12
                ),
        }

    # --------------------------------------------------------
    # Positive control
    # --------------------------------------------------------

    p45_l1 = pair_head_effects(
        effects,
        pair_id="P45",
        head_id="L1H0",
    )

    p45_l4 = pair_head_effects(
        effects,
        pair_id="P45",
        head_id="L4H0",
    )

    require(
        len(
            p45_l1
        )
        ==
        4
        and
        len(
            p45_l4
        )
        ==
        4,
        (
            "P45 positive-control effect "
            "count is invalid."
        ),
    )

    l1_negative = sum(
        item[
            "ablation_delta"
        ]
        <
        0.0
        for item
        in p45_l1
    )

    l4_positive = sum(
        item[
            "ablation_delta"
        ]
        >
        0.0
        for item
        in p45_l4
    )

    positive_control = {
        "L1H0_negative_count":
            int(
                l1_negative
            ),

        "L1H0_pass":
            bool(
                l1_negative
                >=
                3
            ),

        "L4H0_positive_count":
            int(
                l4_positive
            ),

        "L4H0_pass":
            bool(
                l4_positive
                >=
                3
            ),
    }

    positive_control[
        "overall_pass"
    ] = bool(
        positive_control[
            "L1H0_pass"
        ]
        and
        positive_control[
            "L4H0_pass"
        ]
    )

    # --------------------------------------------------------
    # Weak control
    # --------------------------------------------------------

    mean_abs = {}

    for head_id in EXPECTED_HEAD_ORDER:

        values = primary_effects(
            effects,
            head_id=head_id,
        )

        require(
            len(
                values
            )
            ==
            16,
            (
                f"{head_id}: expected "
                "16 primary effects."
            ),
        )

        mean_abs[
            head_id
        ] = float(
            np.mean(
                [
                    abs(
                        item[
                            "ablation_delta"
                        ]
                    )
                    for item
                    in values
                ]
            )
        )

    weak_control = {
        "mean_absolute_delta":
            mean_abs,

        "pass":
            bool(
                mean_abs[
                    "L4H1"
                ]
                <
                mean_abs[
                    "L1H0"
                ]
                and
                mean_abs[
                    "L4H1"
                ]
                <
                mean_abs[
                    "L4H0"
                ]
            ),
    }

    # --------------------------------------------------------
    # Orientation diagnostic
    # --------------------------------------------------------

    orientation = {}

    for head_id in primary_heads:

        stable_signs = [
            pair_direction[
                head_id
            ][
                pair_id
            ][
                "dominant_sign"
            ]
            for pair_id
            in primary_pair_ids
            if pair_direction[
                head_id
            ][
                pair_id
            ][
                "stable"
            ]
        ]

        unique_signs = sorted(
            {
                value
                for value
                in stable_signs
                if value is not None
            }
        )

        if len(
            stable_signs
        ) < 2:

            interpretation = (
                "insufficient_stable_pairs"
            )

        elif len(
            unique_signs
        ) == 1:

            interpretation = (
                "global_order_bias_signal"
            )

        else:

            interpretation = (
                "pair_dependent_signal"
            )

        orientation[
            head_id
        ] = {
            "stable_pair_dominant_signs":
                stable_signs,

            "unique_dominant_signs":
                unique_signs,

            "interpretation":
                interpretation,
        }

    # --------------------------------------------------------
    # Outcome
    # --------------------------------------------------------

    if not positive_control[
        "overall_pass"
    ]:

        outcome = (
            "inconclusive_positive_"
            "control_failure"
        )

    elif not weak_control[
        "pass"
    ]:

        outcome = (
            "fails_cross_pair_"
            "replication"
        )

    else:

        generalization_pass_count = sum(
            cross_pair[
                head_id
            ][
                "pass"
            ]
            for head_id
            in primary_heads
        )

        correctness_pass_count = sum(
            correctness[
                head_id
            ][
                "strong_correctness_tracking"
            ]
            for head_id
            in primary_heads
        )

        if (
            generalization_pass_count
            ==
            2
            and
            correctness_pass_count
            ==
            0
        ):

            outcome = (
                "replicates_cross_pair_"
                "candidate_selection_bias"
            )

        elif (
            generalization_pass_count
            ==
            0
        ):

            outcome = (
                "fails_cross_pair_"
                "replication"
            )

        else:

            outcome = (
                "partial_replication"
            )

    return {
        "pair_direction_stability":
            pair_direction,

        "cross_pair_generalization":
            cross_pair,

        "strong_correctness_tracking":
            correctness,

        "positive_control":
            positive_control,

        "weak_control":
            weak_control,

        "orientation_diagnostic":
            orientation,

        "preregistered_outcome":
            outcome,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    spec_path = project_path(
        args.execution_spec
    )

    approval_path = project_path(
        args.approval
    )

    print()
    print("=" * 92)
    print(
        "STAGE L12D — EXP-0007 "
        "CONTROLLED RUNNER"
    )
    print("=" * 92)

    # ========================================================
    # 1. Validate execution spec
    # ========================================================

    spec_document = load_json(
        spec_path
    )

    spec_validation = validate_spec(
        spec_document=spec_document,
        spec_path=spec_path,
    )

    spec = spec_validation[
        "spec"
    ]

    print()
    print(
        "Execution spec:"
    )
    print(
        f"  {EXPECTED_SPEC_ID}"
    )

    print(
        "Spec SHA-256:"
    )
    print(
        "  "
        +
        spec_validation[
            "spec_sha256"
        ]
    )

    print()
    print(
        "Spec checks"
    )
    print(
        "-----------"
    )

    for name, passed in (
        spec_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    # ========================================================
    # 2. Validate frozen external bytes
    #
    # Still NO model load here.
    # ========================================================

    bindings = validate_frozen_bindings(
        spec
    )

    print()
    print(
        "Frozen binding checks"
    )
    print(
        "---------------------"
    )

    for name, passed in (
        bindings[
            "checks"
        ].items()
    ):

        print(
            f"{name:<46} "
            f"{passed}"
        )

    groups = group_tasks_by_probe(
        spec_validation[
            "tasks"
        ]
    )

    result_path = project_path(
        spec[
            "execution_protocol"
        ][
            "result_path"
        ]
    )

    approval_exists = (
        approval_path.exists()
    )

    result_exists = (
        result_path.exists()
    )

    print()
    print(
        "Gate state"
    )
    print(
        "----------"
    )

    print(
        f"Approval artifact exists: "
        f"{approval_exists}"
    )

    print(
        f"Result artifact exists: "
        f"{result_exists}"
    )

    print(
        f"Frozen probe groups: "
        f"{len(groups)}"
    )

    # ========================================================
    # Preflight-only boundary
    # ========================================================

    if args.preflight_only:

        print()
        print(
            "Preflight guarantees"
        )
        print(
            "--------------------"
        )

        print(
            "model_loaded: False"
        )

        print(
            "checkpoint_weights_loaded: False"
        )

        print(
            "intervention_executed: False"
        )

        print(
            "result_written: False"
        )

        print(
            "knowledge_store_mutated: False"
        )

        print(
            "experiment_registry_mutated: False"
        )

        print()
        print("=" * 92)
        print(
            "SUCCESS: EXP-0007 RUNNER "
            "PREFLIGHT PASSED."
        )
        print(
            "No model was loaded and no "
            "intervention was executed."
        )
        print("=" * 92)

        return

    # ========================================================
    # 3. HARD P3 APPROVAL GATE
    #
    # This occurs BEFORE LaboratoryRuntime(...)
    # and therefore before checkpoint weights are loaded.
    # ========================================================

    approval_validation = validate_approval(
        approval_path=approval_path,
        spec_validation=spec_validation,
    )

    if (
        result_path.exists()
        and
        not args.overwrite_result
    ):

        raise FileExistsError(
            "EXP-0007 result artifact already "
            f"exists: {result_path}\n"
            "Refusing to overwrite frozen causal "
            "evidence without --overwrite-result."
        )

    print()
    print(
        "P3 approval verified"
    )
    print(
        "--------------------"
    )

    print(
        "Approval ID:"
    )
    print(
        "  "
        +
        approval_validation[
            "approval"
        ][
            "approval_id"
        ]
    )

    print(
        "Approval SHA-256:"
    )
    print(
        "  "
        +
        approval_validation[
            "approval_sha256"
        ]
    )

    # ========================================================
    # 4. Capture knowledge state BEFORE model load
    # ========================================================

    knowledge_before = knowledge_hashes()

    # ========================================================
    # 5. Only NOW load the model
    # ========================================================

    print()
    print(
        "Loading mature MathGPT v0.3..."
    )

    runtime = LaboratoryRuntime(
        checkpoint_path=(
            bindings[
                "checkpoint_path"
            ]
        ),
        expected_checkpoint_sha256=(
            EXPECTED_CHECKPOINT_SHA256
        ),
    )

    require(
        runtime.checkpoint_sha256
        ==
        EXPECTED_CHECKPOINT_SHA256,
        (
            "Runtime checkpoint identity "
            "does not match frozen spec."
        ),
    )

    initial_model_sha = (
        model_state_sha256(
            runtime.model
        )
    )

    # ========================================================
    # 6. Run the 20 frozen probes × 3 heads
    # ========================================================

    probe_records = []
    effect_records = []

    print()
    print("=" * 92)
    print(
        "EXP-0007 CAUSAL EXECUTION"
    )
    print("=" * 92)

    for index, group in enumerate(
        groups,
        start=1,
    ):

        exemplar = group[
            0
        ]

        print()
        print(
            f"[{index:02d}/"
            f"{len(groups):02d}] "
            f"{exemplar['probe_id']} "
            f"| {exemplar['pair_id']} "
            f"| {exemplar['form']} "
            f"| correct="
            f"{exemplar['correct_candidate']}"
        )

        result = run_probe_group(
            runtime=runtime,
            group=group,
        )

        probe_records.append(
            result
        )

        for effect in result[
            "effects"
        ]:

            effect_records.append(
                effect
            )

            print(
                f"  {effect['head_id']}: "
                f"baseline={effect['baseline_margin']:+.6f} "
                f"ablated={effect['ablated_margin']:+.6f} "
                f"delta={effect['ablation_delta']:+.6f} "
                f"correctness="
                f"{effect['correctness_effect']:+.6f}"
            )

    # ========================================================
    # 7. Execution integrity
    # ========================================================

    require(
        len(
            effect_records
        )
        ==
        EXPECTED_EFFECT_COUNT,
        (
            "EXP-0007 did not produce "
            "exactly 60 effects."
        ),
    )

    effect_indexes = [
        int(
            item[
                "effect_index"
            ]
        )
        for item
        in effect_records
    ]

    frozen_effect_indexes = [
        int(
            item[
                "effect_index"
            ]
        )
        for item
        in spec_validation[
            "tasks"
        ]
    ]

    all_effects_finite = all(
        np.isfinite(
            item[
                "baseline_margin"
            ]
        )
        and
        np.isfinite(
            item[
                "ablated_margin"
            ]
        )
        and
        np.isfinite(
            item[
                "ablation_delta"
            ]
        )
        and
        np.isfinite(
            item[
                "correctness_effect"
            ]
        )
        for item
        in effect_records
    )

    selected_heads_zeroed = all(
        item[
            "intervention_record"
        ]
        is not None
        and
        float(
            item[
                "intervention_record"
            ][
                "scale"
            ]
        )
        ==
        0.0
        and
        float(
            item[
                "intervention_record"
            ][
                "intervened_last_token_l2"
            ]
        )
        ==
        0.0
        for item
        in effect_records
    )

    final_model_sha = (
        model_state_sha256(
            runtime.model
        )
    )

    knowledge_after = knowledge_hashes()

    execution_checks = {
        "spec_sha_matches":
            (
                spec_validation[
                    "spec_sha256"
                ]
                ==
                EXPECTED_SPEC_SHA256
            ),

        "approval_verified":
            all(
                approval_validation[
                    "checks"
                ].values()
            ),

        "checkpoint_sha_matches":
            (
                runtime.checkpoint_sha256
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        "probe_count_20":
            (
                len(
                    probe_records
                )
                ==
                EXPECTED_PROBE_COUNT
            ),

        "effect_count_60":
            (
                len(
                    effect_records
                )
                ==
                EXPECTED_EFFECT_COUNT
            ),

        "effect_order_preserved":
            (
                effect_indexes
                ==
                frozen_effect_indexes
            ),

        "all_effects_finite":
            bool(
                all_effects_finite
            ),

        "all_selected_heads_zeroed":
            bool(
                selected_heads_zeroed
            ),

        "model_parameters_unchanged":
            (
                initial_model_sha
                ==
                final_model_sha
            ),

        "knowledge_store_unchanged":
            (
                knowledge_before
                ==
                knowledge_after
            ),
    }

    failed_checks = [
        name
        for name, passed
        in execution_checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            "EXP-0007 execution integrity "
            f"failed: {failed_checks}"
        )

    # ========================================================
    # 8. Apply preregistered analysis
    # ========================================================

    analysis = analyze_results(
        effects=effect_records,
        spec=spec,
    )

    # ========================================================
    # 9. Result artifact
    # ========================================================

    summary = {
        "result_version":
            "EXP-0007.1",

        "runner_version":
            RUNNER_VERSION,

        "experiment_id":
            EXPECTED_EXPERIMENT_ID,

        "target_hypothesis_id":
            EXPECTED_HYPOTHESIS_ID,

        "status":
            "completed_execution",

        "evidence_level":
            "E3",

        "bindings": {
            "execution_spec_id":
                EXPECTED_SPEC_ID,

            "execution_spec_path":
                relative_project_path(
                    spec_path
                ),

            "execution_spec_sha256":
                spec_validation[
                    "spec_sha256"
                ],

            "plan_id":
                EXPECTED_PLAN_ID,

            "plan_sha256":
                EXPECTED_PLAN_SHA256,

            "checkpoint_sha256":
                EXPECTED_CHECKPOINT_SHA256,

            "approval_id":
                approval_validation[
                    "approval"
                ][
                    "approval_id"
                ],

            "approval_sha256":
                approval_validation[
                    "approval_sha256"
                ],
        },

        "model": (
            runtime.model_record()
        ),

        "execution": {
            "probe_count":
                len(
                    probe_records
                ),

            "primary_probe_count":
                EXPECTED_PRIMARY_PROBE_COUNT,

            "head_order":
                EXPECTED_HEAD_ORDER,

            "causal_effect_count":
                len(
                    effect_records
                ),

            "head_scale":
                0.0,

            "persistent_weight_modification":
                False,
        },

        "probe_results":
            probe_records,

        "effects":
            effect_records,

        "analysis":
            analysis,

        "integrity_checks":
            execution_checks,

        "model_state_sha256_before":
            initial_model_sha,

        "model_state_sha256_after":
            final_model_sha,

        "knowledge_hashes_before":
            knowledge_before,

        "knowledge_hashes_after":
            knowledge_after,
    }

    atomic_write_json(
        result_path,
        summary,
    )

    result_sha256 = file_sha256(
        result_path
    )

    # ========================================================
    # 10. Display summary
    # ========================================================

    print()
    print("=" * 92)
    print(
        "EXP-0007 PREREGISTERED RESULT"
    )
    print("=" * 92)

    print()
    print(
        "Positive control:"
    )
    print(
        f"  L1H0 pass: "
        f"{analysis['positive_control']['L1H0_pass']}"
    )
    print(
        f"  L4H0 pass: "
        f"{analysis['positive_control']['L4H0_pass']}"
    )
    print(
        f"  overall: "
        f"{analysis['positive_control']['overall_pass']}"
    )

    print()
    print(
        "Cross-pair generalization:"
    )

    for head_id in (
        "L1H0",
        "L4H0",
    ):

        record = (
            analysis[
                "cross_pair_generalization"
            ][
                head_id
            ]
        )

        print(
            f"  {head_id}: "
            f"{record['stable_pair_count']}/4 "
            f"pass={record['pass']}"
        )

    print()
    print(
        "Strong correctness tracking:"
    )

    for head_id in (
        "L1H0",
        "L4H0",
    ):

        record = (
            analysis[
                "strong_correctness_tracking"
            ][
                head_id
            ]
        )

        print(
            f"  {head_id}: "
            f"{record['positive_count']}/16 "
            f"pass="
            f"{record['strong_correctness_tracking']}"
        )

    print()
    print(
        "Weak control:"
    )
    print(
        f"  pass: "
        f"{analysis['weak_control']['pass']}"
    )

    print()
    print(
        "Orientation diagnostic:"
    )

    for head_id in (
        "L1H0",
        "L4H0",
    ):

        print(
            f"  {head_id}: "
            f"{analysis['orientation_diagnostic'][head_id]['interpretation']}"
        )

    print()
    print(
        "Preregistered outcome:"
    )
    print(
        "  "
        +
        analysis[
            "preregistered_outcome"
        ]
    )

    print()
    print(
        "Integrity checks"
    )
    print(
        "----------------"
    )

    for name, passed in (
        execution_checks.items()
    ):

        print(
            f"{name:<42} "
            f"{passed}"
        )

    print()
    print(
        "Result artifact:"
    )
    print(
        result_path
    )

    print()
    print(
        "Result SHA-256:"
    )
    print(
        result_sha256
    )

    print()
    print("=" * 92)
    print(
        "SUCCESS: EXP-0007 P3 EXECUTION "
        "COMPLETED WITH FROZEN PLAN."
    )
    print(
        "Knowledge state was not mutated."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
