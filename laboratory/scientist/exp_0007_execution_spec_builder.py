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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Identity
# ============================================================

ENGINE_VERSION = "L12C.1"

EXPECTED_PLAN_VERSION = "L12B.1"
EXPECTED_PLAN_ID = "PLAN-5825AD67CA0D2EC0"
EXPECTED_PLAN_SHA256 = (
    "3bb6eebe7da92546170e97decc9768388b483d08b06b07fb922ca3a3e30ef8cb"
)

EXPERIMENT_ID = "EXP-0007"
TARGET_HYPOTHESIS_ID = "HYP-V03-003"

EXPECTED_CHECKPOINT_SHA256 = (
    "4cd9b8f1c68e4079449519fd601b573f8ea48137bc74172c7b37e108c9707cf3"
)

EXPECTED_PRIMARY_PROBES = 16
EXPECTED_TOTAL_PROBES = 20

EXPECTED_HEADS = [
    "L1H0",
    "L4H0",
    "L4H1",
]

EXPECTED_EFFECTS = (
    EXPECTED_TOTAL_PROBES
    *
    len(
        EXPECTED_HEADS
    )
)


# ============================================================
# Paths
# ============================================================

DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0007.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec_exp_0007.json"
)

RUNTIME_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "core"
    / "runtime.py"
)

HEAD_ABLATION_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "tools"
    / "attention_head_ablation.py"
)

FUTURE_RUNNER_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "experiments"
    / "run_exp_0007.py"
)

FUTURE_APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0007.json"
)

FUTURE_RESULT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "artifacts"
    / "exp_0007_cross_pair_replication_summary.json"
)


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
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


def parse_head_id(
    head_id,
):
    if (
        not isinstance(
            head_id,
            str,
        )
        or
        not head_id.startswith(
            "L"
        )
        or
        "H"
        not in head_id
    ):
        raise ValueError(
            f"Invalid head ID: {head_id!r}"
        )

    layer_text, head_text = (
        head_id[
            1:
        ].split(
            "H",
            1,
        )
    )

    return (
        int(
            layer_text
        ),
        int(
            head_text
        ),
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L12C — bind the frozen "
            "EXP-0007 preregistered plan into "
            "an exact execution specification. "
            "No model is loaded and no "
            "intervention is executed."
        )
    )

    parser.add_argument(
        "--plan",
        default=str(
            DEFAULT_PLAN_PATH
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
# Plan validation
# ============================================================

def validate_plan(
    *,
    plan_document,
    plan_path,
):

    plan_sha256 = file_sha256(
        plan_path
    )

    plan = plan_document.get(
        "plan",
        {},
    )

    policy_checks = plan_document.get(
        "policy_checks",
        {},
    )

    checks = {
        "plan_version":
            (
                plan_document.get(
                    "plan_version"
                )
                ==
                EXPECTED_PLAN_VERSION
            ),

        "plan_id":
            (
                plan_document.get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
                and
                plan.get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
            ),

        "plan_sha256":
            (
                plan_sha256
                ==
                EXPECTED_PLAN_SHA256
            ),

        "experiment_id":
            (
                plan.get(
                    "experiment_id"
                )
                ==
                EXPERIMENT_ID
            ),

        "hypothesis_id":
            (
                plan.get(
                    "target_hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
            ),

        "checkpoint_sha256":
            (
                plan.get(
                    "checkpoint_binding",
                    {},
                ).get(
                    "sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        "primary_probe_count":
            (
                int(
                    plan.get(
                        "primary_probe_count"
                    )
                )
                ==
                EXPECTED_PRIMARY_PROBES
            ),

        "total_probe_count":
            (
                int(
                    plan.get(
                        "total_probe_count"
                    )
                )
                ==
                EXPECTED_TOTAL_PROBES
            ),

        "frozen_heads":
            (
                plan.get(
                    "frozen_heads",
                    {},
                ).get(
                    "primary"
                )
                ==
                [
                    "L1H0",
                    "L4H0",
                ]
                and
                plan.get(
                    "frozen_heads",
                    {},
                ).get(
                    "weak_control"
                )
                ==
                [
                    "L4H1",
                ]
            ),

        "intervention_head_order":
            (
                plan.get(
                    "intervention_policy",
                    {},
                ).get(
                    "all_heads"
                )
                ==
                EXPECTED_HEADS
            ),

        "expected_causal_effects":
            (
                int(
                    plan.get(
                        "intervention_policy",
                        {},
                    ).get(
                        "expected_total_causal_effects"
                    )
                )
                ==
                EXPECTED_EFFECTS
            ),

        "required_permission_p3":
            (
                plan.get(
                    "intervention_policy",
                    {},
                ).get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "human_approval_required":
            (
                plan.get(
                    "intervention_policy",
                    {},
                ).get(
                    "human_approval_required"
                )
                is True
            ),

        "execution_not_authorized":
            (
                plan.get(
                    "execution_authorized"
                )
                is False
                and
                plan.get(
                    "intervention_policy",
                    {},
                ).get(
                    "execution_authorized"
                )
                is False
            ),

        "p3_approval_absent":
            (
                plan.get(
                    "p3_approval_present"
                )
                is False
            ),

        "plan_policy_checks":
            (
                bool(
                    policy_checks
                )
                and
                all(
                    policy_checks.values()
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
            "L12C frozen-plan validation "
            f"failed: {failed}"
        )

    return {
        "checks":
            checks,

        "plan":
            plan,

        "plan_sha256":
            plan_sha256,
    }


# ============================================================
# Execution task construction
# ============================================================

def build_execution_tasks(
    plan,
):

    tasks = []

    effect_index = 0

    seen_probe_ids = set()

    frozen_pairs = plan[
        "frozen_pairs"
    ]

    for pair_index, pair in enumerate(
        frozen_pairs
    ):

        pair_id = pair[
            "pair_id"
        ]

        role = pair[
            "role"
        ]

        tokenization = pair[
            "tokenization"
        ]

        discriminative_index = int(
            tokenization[
                "first_discriminative_index"
            ]
        )

        lower_ids = [
            int(
                value
            )
            for value
            in tokenization[
                "lower_token_ids"
            ]
        ]

        upper_ids = [
            int(
                value
            )
            for value
            in tokenization[
                "upper_token_ids"
            ]
        ]

        require(
            lower_ids[
                :discriminative_index
            ]
            ==
            upper_ids[
                :discriminative_index
            ],
            (
                f"{pair_id}: candidate shared "
                "prefix does not match."
            ),
        )

        shared_prefix = (
            lower_ids[
                :discriminative_index
            ]
        )

        lower_token_id = int(
            tokenization[
                "lower_discriminative_token"
            ][
                "token_id"
            ]
        )

        upper_token_id = int(
            tokenization[
                "upper_discriminative_token"
            ][
                "token_id"
            ]
        )

        for probe_index, probe in enumerate(
            pair[
                "probes"
            ]
        ):

            probe_id = probe[
                "probe_id"
            ]

            require(
                probe_id
                not in
                seen_probe_ids,
                (
                    "Duplicate frozen probe ID: "
                    f"{probe_id}"
                ),
            )

            seen_probe_ids.add(
                probe_id
            )

            require(
                int(
                    probe[
                        "lower_discriminative_token_id"
                    ]
                )
                ==
                lower_token_id,
                (
                    f"{probe_id}: lower token "
                    "binding mismatch."
                ),
            )

            require(
                int(
                    probe[
                        "upper_discriminative_token_id"
                    ]
                )
                ==
                upper_token_id,
                (
                    f"{probe_id}: upper token "
                    "binding mismatch."
                ),
            )

            require(
                int(
                    probe[
                        "first_discriminative_index"
                    ]
                )
                ==
                discriminative_index,
                (
                    f"{probe_id}: discriminative "
                    "index binding mismatch."
                ),
            )

            for head_order, head_id in enumerate(
                EXPECTED_HEADS
            ):

                layer, head = parse_head_id(
                    head_id
                )

                effect_index += 1

                tasks.append(
                    {
                        "effect_index":
                            effect_index,

                        "pair_order":
                            pair_index,

                        "probe_order":
                            probe_index,

                        "head_order":
                            head_order,

                        "pair_id":
                            pair_id,

                        "pair_role":
                            role,

                        "probe_id":
                            probe_id,

                        "prompt":
                            probe[
                                "prompt"
                            ],

                        "prompt_token_ids":
                            [
                                int(
                                    value
                                )
                                for value
                                in probe[
                                    "prompt_token_ids"
                                ]
                            ],

                        "form":
                            probe[
                                "form"
                            ],

                        "correct_candidate":
                            probe[
                                "correct_candidate"
                            ],

                        "lower_label":
                            pair[
                                "lower_label"
                            ],

                        "upper_label":
                            pair[
                                "upper_label"
                            ],

                        "lower_continuation":
                            tokenization[
                                "lower_continuation"
                            ],

                        "upper_continuation":
                            tokenization[
                                "upper_continuation"
                            ],

                        "shared_prefix_token_ids":
                            shared_prefix,

                        "first_discriminative_index":
                            discriminative_index,

                        "lower_discriminative_token_id":
                            lower_token_id,

                        "upper_discriminative_token_id":
                            upper_token_id,

                        "head_id":
                            head_id,

                        "layer":
                            layer,

                        "head":
                            head,

                        "head_scale":
                            0.0,
                    }
                )

    require(
        len(
            seen_probe_ids
        )
        ==
        EXPECTED_TOTAL_PROBES,
        (
            "Execution task builder did not "
            "see exactly 20 unique probes."
        ),
    )

    require(
        len(
            tasks
        )
        ==
        EXPECTED_EFFECTS,
        (
            "Execution task count is not "
            f"{EXPECTED_EFFECTS}."
        ),
    )

    return tasks


# ============================================================
# Execution invariants
# ============================================================

def build_execution_invariants(
    plan,
    tasks,
):

    primary_tasks = [
        task
        for task
        in tasks
        if task[
            "pair_role"
        ]
        ==
        "primary_replication"
    ]

    control_tasks = [
        task
        for task
        in tasks
        if task[
            "pair_role"
        ]
        ==
        "positive_control"
    ]

    checks = {
        "task_count_60":
            (
                len(
                    tasks
                )
                ==
                60
            ),

        "primary_effect_count_48":
            (
                len(
                    primary_tasks
                )
                ==
                48
            ),

        "positive_control_effect_count_12":
            (
                len(
                    control_tasks
                )
                ==
                12
            ),

        "effect_indexes_sequential":
            (
                [
                    task[
                        "effect_index"
                    ]
                    for task
                    in tasks
                ]
                ==
                list(
                    range(
                        1,
                        61,
                    )
                )
            ),

        "head_order_frozen":
            all(
                tasks[
                    index:
                    index + 3
                ][
                    0
                ][
                    "head_id"
                ]
                ==
                "L1H0"
                and
                tasks[
                    index:
                    index + 3
                ][
                    1
                ][
                    "head_id"
                ]
                ==
                "L4H0"
                and
                tasks[
                    index:
                    index + 3
                ][
                    2
                ][
                    "head_id"
                ]
                ==
                "L4H1"
                for index
                in range(
                    0,
                    60,
                    3,
                )
            ),

        "all_scale_zero":
            all(
                task[
                    "head_scale"
                ]
                ==
                0.0
                for task
                in tasks
            ),

        "all_shared_prefixes_match_plan":
            all(
                isinstance(
                    task[
                        "shared_prefix_token_ids"
                    ],
                    list,
                )
                for task
                in tasks
            ),

        "primary_pairs_exact":
            (
                plan[
                    "primary_pair_ids"
                ]
                ==
                [
                    "P12",
                    "P67",
                    "P78",
                    "P89",
                ]
            ),

        "positive_control_exact":
            (
                plan[
                    "positive_control_pair_id"
                ]
                ==
                "P45"
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
            "L12C execution invariant "
            f"validation failed: {failed}"
        )

    return checks


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    plan_path = project_path(
        args.plan
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
            "EXP-0007 execution spec already "
            f"exists: {output_path}\n"
            "Use --overwrite only before P3 "
            "approval and only if intentionally "
            "rebuilding the same frozen plan."
        )

    print()
    print("=" * 88)
    print(
        "STAGE L12C — EXP-0007 "
        "EXECUTION SPEC BUILDER"
    )
    print("=" * 88)

    plan_document = load_json(
        plan_path
    )

    validation = validate_plan(
        plan_document=plan_document,
        plan_path=plan_path,
    )

    plan = validation[
        "plan"
    ]

    print()
    print(
        "Plan identity"
    )
    print(
        "-------------"
    )

    print(
        f"Plan ID: "
        f"{plan['plan_id']}"
    )

    print(
        f"Plan SHA-256: "
        f"{validation['plan_sha256']}"
    )

    print()
    print(
        "Plan checks"
    )
    print(
        "-----------"
    )

    for name, passed in (
        validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<44} "
            f"{passed}"
        )

    # ========================================================
    # Tool source binding
    # ========================================================

    for path in (
        RUNTIME_PATH,
        HEAD_ABLATION_PATH,
    ):

        if not path.exists():
            raise FileNotFoundError(
                path
            )

    runtime_sha256 = file_sha256(
        RUNTIME_PATH
    )

    head_ablation_sha256 = (
        file_sha256(
            HEAD_ABLATION_PATH
        )
    )

    # ========================================================
    # Exact execution tasks
    # ========================================================

    tasks = build_execution_tasks(
        plan
    )

    invariant_checks = (
        build_execution_invariants(
            plan,
            tasks,
        )
    )

    print()
    print(
        "Execution task summary"
    )
    print(
        "----------------------"
    )

    print(
        f"Total probes: "
        f"{EXPECTED_TOTAL_PROBES}"
    )

    print(
        f"Heads per probe: "
        f"{len(EXPECTED_HEADS)}"
    )

    print(
        f"Total causal effects: "
        f"{len(tasks)}"
    )

    print()
    print(
        "Frozen head order:"
    )

    for head_id in EXPECTED_HEADS:
        layer, head = parse_head_id(
            head_id
        )

        print(
            f"  {head_id} "
            f"(layer={layer}, head={head})"
        )

    print()
    print(
        "Execution invariant checks"
    )
    print(
        "--------------------------"
    )

    for name, passed in (
        invariant_checks.items()
    ):

        print(
            f"{name:<44} "
            f"{passed}"
        )

    # ========================================================
    # Frozen runner protocol
    # ========================================================

    execution_protocol = {
        "runner_path":
            relative_project_path(
                FUTURE_RUNNER_PATH
            ),

        "approval_path":
            relative_project_path(
                FUTURE_APPROVAL_PATH
            ),

        "result_path":
            relative_project_path(
                FUTURE_RESULT_PATH
            ),

        "load_model_once":
            True,

        "runtime_module":
            relative_project_path(
                RUNTIME_PATH
            ),

        "runtime_sha256":
            runtime_sha256,

        "intervention_module":
            relative_project_path(
                HEAD_ABLATION_PATH
            ),

        "intervention_module_sha256":
            head_ablation_sha256,

        "intervention_callable":
            (
                "forward_with_head_scale"
            ),

        "baseline_forward_callable":
            "model.forward",

        "scoring_context_protocol":
            (
                "For every probe, construct "
                "the scoring context as frozen "
                "prompt_token_ids followed by "
                "the frozen shared-prefix token "
                "IDs that occur before the first "
                "candidate-discriminative token."
            ),

        "baseline_margin":
            (
                "baseline_logit(lower_token_id) "
                "- baseline_logit(upper_token_id)"
            ),

        "ablated_margin":
            (
                "intervention_logit(lower_token_id) "
                "- intervention_logit(upper_token_id)"
            ),

        "ablation_delta":
            (
                "ablated_margin - baseline_margin"
            ),

        "correctness_effect": {
            "lower_correct":
                "-ablation_delta",

            "upper_correct":
                "+ablation_delta",
        },

        "head_scale":
            0.0,

        "persistent_parameter_changes":
            False,

        "checkpoint_write_allowed":
            False,

        "optimizer_step_allowed":
            False,

        "knowledge_store_write_allowed":
            False,

        "experiment_registry_write_allowed":
            False,

        "result_artifact_write_allowed":
            True,

        "approval_gate_position":
            (
                "The runner must verify the "
                "approval artifact and all frozen "
                "bindings before instantiating "
                "LaboratoryRuntime or loading "
                "checkpoint weights."
            ),
    }

    # ========================================================
    # Integrity requirements
    # ========================================================

    integrity_requirements = {
        "exact_plan_sha":
            EXPECTED_PLAN_SHA256,

        "exact_checkpoint_sha":
            EXPECTED_CHECKPOINT_SHA256,

        "exact_experiment_id":
            EXPERIMENT_ID,

        "exact_hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "exact_probe_count":
            EXPECTED_TOTAL_PROBES,

        "exact_primary_probe_count":
            EXPECTED_PRIMARY_PROBES,

        "exact_head_order":
            EXPECTED_HEADS,

        "exact_effect_count":
            EXPECTED_EFFECTS,

        "exact_head_scale":
            0.0,

        "all_effects_finite":
            True,

        "all_selected_heads_zeroed":
            True,

        "frozen_task_order_preserved":
            True,

        "candidate_orientation_preserved":
            True,

        "positive_control_evaluated":
            True,

        "weak_control_evaluated":
            True,

        "analysis_policy_applied_exactly":
            True,

        "no_persistent_weight_modification":
            True,

        "knowledge_store_not_mutated":
            True,

        "experiment_registry_not_mutated":
            True,
    }

    # ========================================================
    # Spec core + identity
    # ========================================================

    spec_core = {
        "experiment_id":
            EXPERIMENT_ID,

        "target_hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "plan_binding": {
            "plan_id":
                EXPECTED_PLAN_ID,

            "plan_path":
                relative_project_path(
                    plan_path
                ),

            "plan_sha256":
                validation[
                    "plan_sha256"
                ],
        },

        "checkpoint_binding":
            plan[
                "checkpoint_binding"
            ],

        "tokenizer_binding":
            plan[
                "tokenizer_binding"
            ],

        "analysis_policy":
            plan[
                "analysis_policy"
            ],

        "execution_protocol":
            execution_protocol,

        "execution_tasks":
            tasks,

        "integrity_requirements":
            integrity_requirements,

        "required_permission":
            "P3",

        "human_approval_required":
            True,

        "execution_authorized":
            False,

        "approval_present":
            False,
    }

    spec_id = (
        "SPEC-"
        +
        canonical_json_sha256(
            spec_core
        )[
            :16
        ].upper()
    )

    output = {
        "engine_version":
            ENGINE_VERSION,

        "spec_version":
            "L12C.1",

        "mode":
            (
                "frozen_execution_specification"
            ),

        "execution_spec_id":
            spec_id,

        "execution_spec":
            {
                "execution_spec_id":
                    spec_id,

                **spec_core,
            },

        "plan_checks":
            validation[
                "checks"
            ],

        "execution_invariant_checks":
            invariant_checks,

        "policy_checks": {
            "model_not_loaded":
                True,

            "checkpoint_weights_not_loaded":
                True,

            "intervention_not_executed":
                True,

            "result_artifact_not_created":
                (
                    not
                    FUTURE_RESULT_PATH.exists()
                ),

            "approval_artifact_not_created":
                (
                    not
                    FUTURE_APPROVAL_PATH.exists()
                ),

            "execution_not_authorized":
                True,

            "knowledge_store_not_mutated":
                True,

            "experiment_registry_not_mutated":
                True,
        },
    }

    failed_policy = [
        name
        for name, passed
        in output[
            "policy_checks"
        ].items()
        if not passed
    ]

    if failed_policy:
        raise RuntimeError(
            "L12C policy boundary failed: "
            f"{failed_policy}"
        )

    atomic_write_json(
        output_path,
        output,
    )

    saved = load_json(
        output_path
    )

    require(
        saved.get(
            "execution_spec_id"
        )
        ==
        spec_id,
        (
            "Saved execution spec identity "
            "mismatch."
        ),
    )

    spec_sha256 = file_sha256(
        output_path
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print("=" * 88)
    print(
        "L12C EXP-0007 EXECUTION SPEC SUMMARY"
    )
    print("=" * 88)

    print()
    print(
        "Execution Spec ID:"
    )
    print(
        f"  {spec_id}"
    )

    print()
    print(
        "Experiment:"
    )
    print(
        f"  {EXPERIMENT_ID}"
    )

    print(
        "Hypothesis:"
    )
    print(
        f"  {TARGET_HYPOTHESIS_ID}"
    )

    print()
    print(
        "Plan binding:"
    )
    print(
        f"  {EXPECTED_PLAN_ID}"
    )
    print(
        f"  {validation['plan_sha256']}"
    )

    print()
    print(
        "Checkpoint binding:"
    )
    print(
        "  "
        +
        plan[
            "checkpoint_binding"
        ][
            "sha256"
        ]
    )

    print()
    print(
        "Execution tasks:"
    )
    print(
        f"  {len(tasks)} causal effects"
    )

    print()
    print(
        "Future runner:"
    )
    print(
        "  "
        +
        execution_protocol[
            "runner_path"
        ]
    )

    print()
    print(
        "Future approval artifact:"
    )
    print(
        "  "
        +
        execution_protocol[
            "approval_path"
        ]
    )

    print()
    print(
        "Future result artifact:"
    )
    print(
        "  "
        +
        execution_protocol[
            "result_path"
        ]
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
        "Execution spec artifact:"
    )
    print(
        output_path
    )

    print()
    print(
        "Execution spec SHA-256:"
    )
    print(
        spec_sha256
    )

    print()
    print(
        "Policy checks"
    )
    print(
        "-------------"
    )

    for name, passed in (
        output[
            "policy_checks"
        ].items()
    ):

        print(
            f"{name:<44} "
            f"{passed}"
        )

    print()
    print("=" * 88)
    print(
        "SUCCESS: STAGE L12C EXP-0007 "
        "EXECUTION SPEC FROZEN."
    )
    print(
        "P3 approval is still absent and "
        "execution remains blocked."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
