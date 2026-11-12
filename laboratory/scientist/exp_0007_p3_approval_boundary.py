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

ENGINE_VERSION = "L12E.1"
APPROVAL_VERSION = "L12E.1"

EXPERIMENT_ID = "EXP-0007"
TARGET_HYPOTHESIS_ID = "HYP-V03-003"

REQUIRED_APPROVAL_PHRASE = (
    "I APPROVE EXP-0007 P3"
)

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

EXPECTED_CHECKPOINT_SHA256 = (
    "4cd9b8f1c68e4079449519fd601b573f8ea48137bc74172c7b37e108c9707cf3"
)

EXPECTED_CAUSAL_EFFECTS = 60


# ============================================================
# Paths
# ============================================================

DEFAULT_SPEC_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec_exp_0007.json"
)

DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0007.json"
)

DEFAULT_APPROVAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0007.json"
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


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L12E — create the explicit "
            "human P3 approval artifact for "
            "EXP-0007. This boundary does not "
            "load the model or execute the "
            "experiment."
        )
    )

    parser.add_argument(
        "--execution-spec",
        default=str(
            DEFAULT_SPEC_PATH
        ),
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
            DEFAULT_APPROVAL_PATH
        ),
    )

    parser.add_argument(
        "--approval-phrase",
        required=True,
        help=(
            "Must exactly equal: "
            f"{REQUIRED_APPROVAL_PHRASE}"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Normally disabled. Replacing an "
            "approval artifact requires an "
            "intentional explicit action."
        ),
    )

    return parser.parse_args()


# ============================================================
# Spec + plan validation
# ============================================================

def validate_bindings(
    *,
    spec_document,
    spec_path,
    plan_document,
    plan_path,
):

    spec_sha256 = file_sha256(
        spec_path
    )

    plan_sha256 = file_sha256(
        plan_path
    )

    spec = spec_document.get(
        "execution_spec",
        {},
    )

    plan = plan_document.get(
        "plan",
        {},
    )

    execution_protocol = spec.get(
        "execution_protocol",
        {},
    )

    integrity = spec.get(
        "integrity_requirements",
        {},
    )

    checks = {
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

        "spec_plan_binding":
            (
                spec.get(
                    "plan_binding",
                    {},
                ).get(
                    "plan_id"
                )
                ==
                EXPECTED_PLAN_ID
                and
                spec.get(
                    "plan_binding",
                    {},
                ).get(
                    "plan_sha256"
                )
                ==
                EXPECTED_PLAN_SHA256
            ),

        "experiment_id":
            (
                spec.get(
                    "experiment_id"
                )
                ==
                EXPERIMENT_ID
                and
                plan.get(
                    "experiment_id"
                )
                ==
                EXPERIMENT_ID
            ),

        "hypothesis_id":
            (
                spec.get(
                    "target_hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
                and
                plan.get(
                    "target_hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
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
                and
                plan.get(
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
                and
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
                spec.get(
                    "human_approval_required"
                )
                is True
                and
                plan.get(
                    "intervention_policy",
                    {},
                ).get(
                    "human_approval_required"
                )
                is True
            ),

        "spec_execution_not_authorized":
            (
                spec.get(
                    "execution_authorized"
                )
                is False
            ),

        "plan_execution_not_authorized":
            (
                plan.get(
                    "execution_authorized"
                )
                is False
            ),

        "effect_count_60":
            (
                int(
                    integrity.get(
                        "exact_effect_count"
                    )
                )
                ==
                EXPECTED_CAUSAL_EFFECTS
                and
                len(
                    spec.get(
                        "execution_tasks",
                        [],
                    )
                )
                ==
                EXPECTED_CAUSAL_EFFECTS
            ),

        "head_scale_zero":
            (
                float(
                    integrity.get(
                        "exact_head_scale"
                    )
                )
                ==
                0.0
                and
                all(
                    float(
                        task.get(
                            "head_scale"
                        )
                    )
                    ==
                    0.0

                    for task
                    in spec.get(
                        "execution_tasks",
                        [],
                    )
                )
            ),

        "optimizer_step_blocked":
            (
                execution_protocol.get(
                    "optimizer_step_allowed"
                )
                is False
            ),

        "checkpoint_write_blocked":
            (
                execution_protocol.get(
                    "checkpoint_write_allowed"
                )
                is False
            ),

        "knowledge_write_blocked":
            (
                execution_protocol.get(
                    "knowledge_store_write_allowed"
                )
                is False
            ),

        "registry_write_blocked":
            (
                execution_protocol.get(
                    "experiment_registry_write_allowed"
                )
                is False
            ),

        "persistent_weight_modification_blocked":
            (
                execution_protocol.get(
                    "persistent_parameter_changes"
                )
                is False
                and
                plan.get(
                    "intervention_policy",
                    {},
                ).get(
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
        raise RuntimeError(
            "L12E approval binding "
            f"validation failed: {failed}"
        )

    return {
        "checks":
            checks,

        "spec":
            spec,

        "plan":
            plan,

        "spec_sha256":
            spec_sha256,

        "plan_sha256":
            plan_sha256,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    spec_path = project_path(
        args.execution_spec
    )

    plan_path = project_path(
        args.plan
    )

    output_path = project_path(
        args.output
    )

    print()
    print("=" * 88)
    print(
        "STAGE L12E — EXP-0007 "
        "EXPLICIT P3 APPROVAL BOUNDARY"
    )
    print("=" * 88)

    # ========================================================
    # Exact human phrase
    # ========================================================

    phrase_matches = (
        args.approval_phrase
        ==
        REQUIRED_APPROVAL_PHRASE
    )

    print()
    print(
        "Human approval phrase"
    )
    print(
        "---------------------"
    )

    print(
        f"Required: "
        f"{REQUIRED_APPROVAL_PHRASE}"
    )

    print(
        f"Received: "
        f"{args.approval_phrase}"
    )

    print(
        f"Exact match: "
        f"{phrase_matches}"
    )

    if not phrase_matches:
        raise PermissionError(
            "EXP-0007 P3 approval denied: "
            "approval phrase did not match "
            "exactly."
        )

    if (
        output_path.exists()
        and
        not args.overwrite
    ):

        raise FileExistsError(
            "EXP-0007 P3 approval artifact "
            f"already exists: {output_path}\n"
            "Refusing to replace it without "
            "--overwrite."
        )

    # ========================================================
    # Load and verify immutable bindings
    #
    # No model load. No intervention.
    # ========================================================

    spec_document = load_json(
        spec_path
    )

    plan_document = load_json(
        plan_path
    )

    validation = validate_bindings(
        spec_document=spec_document,
        spec_path=spec_path,
        plan_document=plan_document,
        plan_path=plan_path,
    )

    print()
    print(
        "Frozen binding checks"
    )
    print(
        "---------------------"
    )

    for name, passed in (
        validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<50} "
            f"{passed}"
        )

    # ========================================================
    # Approval scope
    # ========================================================

    authorized_scope = {
        "intervention":
            (
                "temporary_attention_head_ablation"
            ),

        "head_scale":
            0.0,

        "maximum_causal_effects":
            EXPECTED_CAUSAL_EFFECTS,

        "allowed_heads": [
            "L1H0",
            "L4H0",
            "L4H1",
        ],

        "allowed_primary_pairs": [
            "P12",
            "P67",
            "P78",
            "P89",
        ],

        "allowed_positive_control_pair":
            "P45",

        "persistent_weight_modification":
            False,

        "optimizer_step_allowed":
            False,

        "checkpoint_write_allowed":
            False,

        "knowledge_store_write_allowed":
            False,

        "experiment_registry_write_allowed":
            False,

        "result_artifact_write_allowed":
            True,

        "authorization_limit":
            (
                "This approval applies only to "
                "the exact frozen EXP-0007 "
                "execution specification and "
                "does not authorize any other "
                "experiment, checkpoint, plan, "
                "head, pair, scale, or "
                "persistent model change."
            ),
    }

    approval_seed = {
        "approval_version":
            APPROVAL_VERSION,

        "human_statement":
            REQUIRED_APPROVAL_PHRASE,

        "experiment_id":
            EXPERIMENT_ID,

        "target_hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "permission":
            "P3",

        "execution_spec_id":
            EXPECTED_SPEC_ID,

        "execution_spec_sha256":
            validation[
                "spec_sha256"
            ],

        "plan_id":
            EXPECTED_PLAN_ID,

        "plan_sha256":
            validation[
                "plan_sha256"
            ],

        "checkpoint_sha256":
            EXPECTED_CHECKPOINT_SHA256,

        "authorized_scope":
            authorized_scope,
    }

    approval_id = (
        "APR-"
        +
        canonical_json_sha256(
            approval_seed
        )[
            :16
        ].upper()
    )

    approval = {
        "approval_version":
            APPROVAL_VERSION,

        "engine_version":
            ENGINE_VERSION,

        "approval_id":
            approval_id,

        "created_at_utc":
            utc_now(),

        "approval_type":
            "explicit_human",

        "human_statement":
            REQUIRED_APPROVAL_PHRASE,

        "experiment_id":
            EXPERIMENT_ID,

        "target_hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "permission":
            "P3",

        "execution_authorized":
            True,

        "bindings": {
            "execution_spec_id":
                EXPECTED_SPEC_ID,

            "execution_spec_path":
                relative_project_path(
                    spec_path
                ),

            "execution_spec_sha256":
                validation[
                    "spec_sha256"
                ],

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

            "checkpoint_sha256":
                EXPECTED_CHECKPOINT_SHA256,
        },

        "authorized_scope":
            authorized_scope,

        "policy": {
            "approval_creates_evidence":
                False,

            "approval_executes_experiment":
                False,

            "approval_loads_model":
                False,

            "approval_mutates_model":
                False,

            "approval_mutates_knowledge_store":
                False,

            "approval_mutates_experiment_registry":
                False,

            "runner_must_revalidate_all_bindings":
                True,
        },

        "binding_checks":
            validation[
                "checks"
            ],
    }

    atomic_write_json(
        output_path,
        approval,
    )

    saved = load_json(
        output_path
    )

    require(
        saved.get(
            "approval_id"
        )
        ==
        approval_id,
        (
            "Saved approval identity "
            "mismatch."
        ),
    )

    require(
        saved.get(
            "execution_authorized"
        )
        is True,
        (
            "Saved approval is not "
            "execution-authorizing."
        ),
    )

    approval_sha256 = file_sha256(
        output_path
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print("=" * 88)
    print(
        "L12E EXP-0007 P3 APPROVAL SUMMARY"
    )
    print("=" * 88)

    print()
    print(
        "Approval ID:"
    )
    print(
        f"  {approval_id}"
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
        "Permission:"
    )
    print(
        "  P3"
    )

    print(
        "Approval type:"
    )
    print(
        "  explicit_human"
    )

    print()
    print(
        "Execution spec:"
    )
    print(
        f"  {EXPECTED_SPEC_ID}"
    )
    print(
        "  "
        +
        validation[
            "spec_sha256"
        ]
    )

    print()
    print(
        "Plan:"
    )
    print(
        f"  {EXPECTED_PLAN_ID}"
    )
    print(
        "  "
        +
        validation[
            "plan_sha256"
        ]
    )

    print()
    print(
        "Checkpoint SHA-256:"
    )
    print(
        "  "
        +
        EXPECTED_CHECKPOINT_SHA256
    )

    print()
    print(
        "Authorized scope:"
    )
    print(
        "  temporary attention-head ablation"
    )
    print(
        "  head_scale = 0.0"
    )
    print(
        "  maximum causal effects = 60"
    )
    print(
        "  persistent weight modification = False"
    )

    print()
    print(
        "Approval artifact:"
    )
    print(
        output_path
    )

    print()
    print(
        "Approval SHA-256:"
    )
    print(
        approval_sha256
    )

    print()
    print(
        "Boundary guarantees"
    )
    print(
        "-------------------"
    )

    print(
        "model_loaded: False"
    )
    print(
        "experiment_executed: False"
    )
    print(
        "evidence_created: False"
    )
    print(
        "model_mutated: False"
    )
    print(
        "knowledge_store_mutated: False"
    )
    print(
        "experiment_registry_mutated: False"
    )

    print()
    print("=" * 88)
    print(
        "SUCCESS: STAGE L12E EXP-0007 "
        "P3 APPROVAL ARTIFACT CREATED."
    )
    print(
        "Approval alone did not execute "
        "EXP-0007."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
