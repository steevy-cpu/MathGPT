import argparse
import hashlib
import json
import sys

from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BOUNDARY_VERSION = "L9D.1"
APPROVAL_VERSION = "P3A.1"
EXPECTED_SPEC_VERSION = "L9B.1"
EXPECTED_EXPERIMENT_ID = "EXP-0005"
EXPECTED_PERMISSION = "P3"
CONFIRMATION_PHRASE = "I APPROVE EXP-0005 P3"

DEFAULT_SPEC_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_execution_spec.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "p3_approval_exp_0005.json"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path):
    path = project_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(path, value):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)
        file.write("\n")
    temporary.replace(path)


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
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
    return hashlib.sha256(encoded).hexdigest()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Stage L9D — explicit human P3 approval boundary for EXP-0005. "
            "Preview-only by default."
        )
    )

    parser.add_argument(
        "--execution-spec",
        default=str(DEFAULT_SPEC_PATH),
        help="Path to the frozen L9B execution specification.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for the P3 approval artifact.",
    )

    parser.add_argument(
        "--approve",
        action="store_true",
        help=(
            "Explicitly request creation of the P3 approval artifact. "
            "This still requires the exact confirmation phrase."
        ),
    )

    parser.add_argument(
        "--confirmation",
        default=None,
        help="Exact human confirmation phrase. Required with --approve.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing approval artifact only after "
            "re-confirming the exact approval phrase."
        ),
    )

    return parser.parse_args()


def validate_execution_spec(spec_artifact):
    spec = spec_artifact.get("execution_spec", {})
    payload = spec.get("experiment_payload", {})
    approval_contract = spec.get("approval_contract", {})
    source_plan = spec.get("source_plan", {})
    runner_contract = spec.get("runner_contract", {})
    intervention = payload.get("intervention", {})

    checks = {
        "spec_version": (
            spec_artifact.get("engine_version") == EXPECTED_SPEC_VERSION
        ),
        "spec_checks_present": bool(
            spec_artifact.get("overall_checks")
        ),
        "spec_checks_passed": (
            bool(spec_artifact.get("overall_checks"))
            and all(spec_artifact["overall_checks"].values())
        ),
        "execution_spec_id_present": (
            isinstance(spec.get("execution_spec_id"), str)
            and spec["execution_spec_id"].startswith("SPEC-")
        ),
        "experiment_is_exp_0005": (
            payload.get("experiment_id") == EXPECTED_EXPERIMENT_ID
        ),
        "plan_id_present": (
            isinstance(payload.get("plan_id"), str)
            and payload["plan_id"].startswith("PLAN-")
        ),
        "permission_is_p3": (
            approval_contract.get("required_permission")
            == EXPECTED_PERMISSION
        ),
        "human_approval_required": (
            approval_contract.get("human_approval_required") is True
        ),
        "spec_does_not_self_authorize": (
            approval_contract.get("execution_authorized") is False
            and approval_contract.get("may_execute_now") is False
        ),
        "checkpoint_sha_valid": (
            isinstance(
                payload.get("checkpoint", {}).get("sha256"),
                str,
            )
            and len(payload["checkpoint"]["sha256"]) == 64
        ),
        "source_plan_sha_valid": (
            isinstance(source_plan.get("sha256"), str)
            and len(source_plan["sha256"]) == 64
        ),
        "temporary_intervention": (
            intervention.get("type") == "attention_head_ablation"
            and float(intervention.get("baseline_scale")) == 1.0
            and float(intervention.get("ablation_scale")) == 0.0
            and intervention.get(
                "model_weights_persistently_modified"
            ) is False
        ),
        "runner_requires_approval": (
            "--approval"
            in runner_contract.get(
                "runner_required_arguments",
                [],
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
            "L9D execution-spec validation failed: "
            f"{failed}"
        )

    return checks


def resolve_binding(*, spec_artifact, spec_path):
    spec = spec_artifact["execution_spec"]
    payload = spec["experiment_payload"]
    source_plan = spec["source_plan"]

    spec_sha256 = file_sha256(spec_path)

    plan_path = project_path(source_plan["path"])
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    actual_plan_sha256 = file_sha256(plan_path)
    expected_plan_sha256 = source_plan["sha256"]

    if actual_plan_sha256 != expected_plan_sha256:
        raise RuntimeError(
            "L9D source-plan SHA mismatch.\n"
            f"Expected: {expected_plan_sha256}\n"
            f"Actual:   {actual_plan_sha256}"
        )

    checkpoint_path = project_path(
        payload["checkpoint"]["path"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    actual_checkpoint_sha256 = file_sha256(
        checkpoint_path
    )

    expected_checkpoint_sha256 = (
        payload["checkpoint"]["sha256"]
    )

    if (
        actual_checkpoint_sha256
        != expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "L9D checkpoint SHA mismatch.\n"
            f"Expected: {expected_checkpoint_sha256}\n"
            f"Actual:   {actual_checkpoint_sha256}"
        )

    runner_path = project_path(
        spec["runner_contract"]["runner_path"]
    )

    if not runner_path.exists():
        raise FileNotFoundError(runner_path)

    return {
        "experiment_id": payload["experiment_id"],
        "plan_id": payload["plan_id"],
        "execution_spec_id": spec["execution_spec_id"],
        "execution_spec_sha256": spec_sha256,
        "source_plan_sha256": actual_plan_sha256,
        "checkpoint_sha256": actual_checkpoint_sha256,
    }


def build_approval_artifact(*, binding, spec_artifact):
    approval_id = (
        "APR-"
        + canonical_json_sha256(binding)[:16].upper()
    )

    spec = spec_artifact["execution_spec"]
    payload = spec["experiment_payload"]

    return {
        "approval_version": APPROVAL_VERSION,
        "boundary_version": BOUNDARY_VERSION,
        "approval_id": approval_id,
        "created_at_utc": utc_now(),
        "decision": "approved",
        "permission": "P3",
        "approved_by": "human",
        "scope": "temporary_causal_intervention_only",
        "binding": binding,
        "approved_action": {
            "experiment_id": payload["experiment_id"],
            "hypothesis_id": payload["hypothesis_id"],
            "intervention_type": (
                payload["intervention"]["type"]
            ),
            "baseline_scale": (
                payload["intervention"]["baseline_scale"]
            ),
            "ablation_scale": (
                payload["intervention"]["ablation_scale"]
            ),
            "target_heads": [
                head["head_id"]
                for head in payload["target_heads"]
            ],
            "probe_count": len(payload["probes"]),
            "persistent_weight_modification": False,
        },
        "checks": {
            "explicit_human_approval": True,
            "confirmation_phrase_verified": True,
            "binding_verified_before_approval": True,
            "checkpoint_verified_before_approval": True,
            "source_plan_verified_before_approval": True,
            "execution_spec_verified_before_approval": True,
            "approval_does_not_execute_experiment": True,
            "approval_does_not_mutate_knowledge_store": True,
            "approval_does_not_change_model_weights": True,
        },
        "note": (
            "This artifact authorizes only the exact "
            "temporary P3 causal intervention bound above. "
            "It does not itself execute the experiment."
        ),
    }


def validate_generated_approval(*, approval, binding):
    checks = {
        "approval_version": (
            approval.get("approval_version")
            == APPROVAL_VERSION
        ),
        "approval_id_present": (
            isinstance(approval.get("approval_id"), str)
            and approval["approval_id"].startswith("APR-")
        ),
        "decision_approved": (
            approval.get("decision") == "approved"
        ),
        "permission_p3": (
            approval.get("permission") == "P3"
        ),
        "approved_by_human": (
            approval.get("approved_by") == "human"
        ),
        "temporary_scope": (
            approval.get("scope")
            == "temporary_causal_intervention_only"
        ),
        "exact_binding": (
            approval.get("binding") == binding
        ),
        "explicit_human_approval": (
            approval.get("checks", {}).get(
                "explicit_human_approval"
            )
            is True
        ),
        "approval_does_not_execute": (
            approval.get("checks", {}).get(
                "approval_does_not_execute_experiment"
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
            "Generated P3 approval failed validation: "
            f"{failed}"
        )

    return checks


def print_boundary_summary(
    *,
    spec_artifact,
    binding,
    output_path,
):
    spec = spec_artifact["execution_spec"]
    payload = spec["experiment_payload"]
    intervention = payload["intervention"]

    print()
    print("=" * 76)
    print("STAGE L9D — P3 APPROVAL BOUNDARY")
    print("=" * 76)

    print()
    print("Experiment:")
    print(f"  {binding['experiment_id']}")

    print()
    print("Hypothesis:")
    print(f"  {payload['hypothesis_id']}")

    print()
    print("Plan:")
    print(f"  {binding['plan_id']}")

    print()
    print("Execution Spec:")
    print(f"  {binding['execution_spec_id']}")

    print()
    print("Permission:")
    print("  P3 — temporary causal intervention")

    print()
    print("Intervention:")
    print(f"  {intervention['type']}")
    print(
        f"  baseline scale = "
        f"{intervention['baseline_scale']}"
    )
    print(
        f"  ablation scale = "
        f"{intervention['ablation_scale']}"
    )

    print()
    print("Target heads:")

    for head in payload["target_heads"]:
        print(
            f"  {head['head_id']} — "
            f"{head['role']}"
        )

    print()
    print("Probes:")
    print(f"  {len(payload['probes'])}")

    print()
    print("Persistent weight modification:")
    print("  False")

    print()
    print("Exact binding")
    print("-------------")

    for name, value in binding.items():
        print(f"{name}:")
        print(f"  {value}")

    print()
    print("Approval artifact:")
    print(f"  {output_path}")


def main():
    args = parse_arguments()

    spec_path = project_path(
        args.execution_spec
    )

    output_path = project_path(
        args.output
    )

    spec_artifact = load_json(spec_path)

    spec_checks = validate_execution_spec(
        spec_artifact
    )

    binding = resolve_binding(
        spec_artifact=spec_artifact,
        spec_path=spec_path,
    )

    print_boundary_summary(
        spec_artifact=spec_artifact,
        binding=binding,
        output_path=output_path,
    )

    print()
    print("Boundary checks")
    print("---------------")

    for name, passed in spec_checks.items():
        print(f"{name:<42} {passed}")

    # ========================================================
    # PREVIEW-ONLY DEFAULT
    # ========================================================

    if not args.approve:
        print()
        print("Approval status:")

        if output_path.exists():
            print(
                "  approval artifact already exists"
            )
            print(
                "  NOTE: this preview did not "
                "create or modify it."
            )
        else:
            print("  NOT APPROVED")

        print()
        print("Execution authorized by this run:")
        print("  False")

        print()
        print("Model loaded:")
        print("  False")

        print()
        print("Causal intervention executed:")
        print("  False")

        print()
        print("Knowledge store mutated:")
        print("  False")

        print()
        print(
            "To grant approval later, an explicit "
            "human approval command must include "
            "the exact confirmation phrase."
        )

        print()
        print("=" * 76)
        print(
            "SUCCESS: STAGE L9D APPROVAL "
            "BOUNDARY PREVIEW PASSED."
        )
        print("No approval was granted.")
        print("=" * 76)
        return

    # ========================================================
    # EXPLICIT APPROVAL PATH
    # ========================================================

    if args.confirmation != CONFIRMATION_PHRASE:
        raise PermissionError(
            "P3 approval denied: exact human "
            "confirmation phrase was not provided.\n"
            "Required phrase:\n"
            f"{CONFIRMATION_PHRASE}"
        )

    if (
        output_path.exists()
        and
        not args.overwrite
    ):
        raise FileExistsError(
            f"P3 approval artifact already exists: "
            f"{output_path}\n"
            "Use --overwrite only if you explicitly "
            "intend to re-authorize this exact "
            "frozen experiment."
        )

    approval = build_approval_artifact(
        binding=binding,
        spec_artifact=spec_artifact,
    )

    validate_generated_approval(
        approval=approval,
        binding=binding,
    )

    atomic_write_json(
        output_path,
        approval,
    )

    saved = load_json(
        output_path
    )

    saved_checks = validate_generated_approval(
        approval=saved,
        binding=binding,
    )

    print()
    print("Generated approval checks")
    print("-------------------------")

    for name, passed in saved_checks.items():
        print(f"{name:<42} {passed}")

    print()
    print("Approval ID:")
    print(f"  {saved['approval_id']}")

    print()
    print("Decision:")
    print("  approved")

    print()
    print("Permission:")
    print("  P3")

    print()
    print("Approval artifact created:")
    print(f"  {output_path}")

    print()
    print("Experiment executed by L9D:")
    print("  False")

    print()
    print("=" * 76)
    print(
        "SUCCESS: P3 APPROVAL ARTIFACT CREATED."
    )
    print(
        "The experiment is authorized for the "
        "exact frozen binding, but has not been "
        "executed."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
