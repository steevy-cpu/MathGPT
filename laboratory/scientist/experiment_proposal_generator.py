import argparse
import copy
import hashlib
import json
import sys

from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from laboratory.knowledge.store import KnowledgeStore
from laboratory.knowledge.query import KnowledgeQuery
from laboratory.knowledge.snapshot import (
    SNAPSHOT_VERSION,
    build_snapshot,
    validate_snapshot,
    sha256_json,
)


ENGINE_VERSION = "L10A.1"
EXPECTED_HANDOFF_VERSION = "L8D.1"

EXPECTED_HYPOTHESIS_ID = "HYP-V03-001"
EXPECTED_ACTION = "design_discriminating_experiment"
NEW_EXPERIMENT_ID = "EXP-0006"

DEFAULT_HANDOFF_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "research_action_handoff.json"
)

DEFAULT_PROPOSAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_proposal.json"
)

DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)

DEFAULT_SNAPSHOT_REPORT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "knowledge_snapshot_report.json"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def relative_project_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


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
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def index_by(records, field):
    return {record[field]: record for record in records}


def capture_backups(paths):
    backups = {}
    for path in paths:
        path = Path(path)
        backups[path] = path.read_bytes() if path.exists() else None
    return backups


def restore_backups(backups):
    for path, content in backups.items():
        path = Path(path)
        if content is None:
            if path.exists():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Stage L10A — convert an L8D experiment-design handoff into a "
            "registered discriminating experiment proposal without planning "
            "or executing the experiment."
        )
    )
    parser.add_argument("--handoff", default=str(DEFAULT_HANDOFF_PATH))
    parser.add_argument("--proposal", default=str(DEFAULT_PROPOSAL_PATH))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument(
        "--snapshot-report",
        default=str(DEFAULT_SNAPSHOT_REPORT_PATH),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the EXP-0006 proposal without modifying L7.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permit regeneration of the proposal artifact. This never "
            "bypasses identity or knowledge-state checks."
        ),
    )
    return parser.parse_args()


def validate_handoff(*, handoff, live_knowledge_sha256):
    h = handoff.get("handoff", {})
    hypothesis = h.get("hypothesis", {})
    action = h.get("research_action", {})
    experiment = h.get("experiment", {})
    permission = h.get("permission_gate", {})
    execution = h.get("execution_gate", {})
    source = h.get("source_decision", {})

    checks = {
        "handoff_version": (
            handoff.get("engine_version") == EXPECTED_HANDOFF_VERSION
        ),
        "handoff_checks_present": bool(handoff.get("overall_checks")),
        "handoff_checks_passed": (
            bool(handoff.get("overall_checks"))
            and all(handoff["overall_checks"].values())
        ),
        "selected_hypothesis": (
            hypothesis.get("hypothesis_id") == EXPECTED_HYPOTHESIS_ID
        ),
        "selected_hypothesis_weakened": (
            hypothesis.get("current_status") == "weakened"
        ),
        "mixed_evidence": (
            hypothesis.get("evidence_classification") == "mixed"
        ),
        "design_action_selected": (
            action.get("action") == EXPECTED_ACTION
        ),
        "priority_high": (
            action.get("priority") == "high"
        ),
        "no_registered_experiment": (
            experiment.get("experiment_id") is None
            and experiment.get("registered") is False
        ),
        "no_permission_resolved_yet": (
            permission.get("required_permission") is None
        ),
        "human_approval_not_applicable_yet": (
            permission.get("human_approval_required") is False
        ),
        "experiment_design_required": (
            execution.get("status") == "experiment_design_required"
        ),
        "execution_blocked": (
            execution.get("may_execute_now") is False
        ),
        "execution_not_attempted": (
            execution.get("execution_attempted") is False
        ),
        "knowledge_not_mutated_by_handoff": (
            execution.get("knowledge_store_mutated") is False
        ),
        "handoff_matches_live_knowledge": (
            source.get("knowledge_state_sha256") == live_knowledge_sha256
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "L10A handoff validation failed: "
            f"{failed}"
        )

    return checks


def validate_scientific_context(*, data):
    hypotheses = index_by(
        data["hypotheses"]["records"],
        "hypothesis_id",
    )
    findings = index_by(
        data["findings"]["records"],
        "finding_id",
    )
    experiments = index_by(
        data["experiments"]["records"],
        "experiment_id",
    )

    require(
        EXPECTED_HYPOTHESIS_ID in hypotheses,
        f"Missing target hypothesis {EXPECTED_HYPOTHESIS_ID}.",
    )

    target = hypotheses[EXPECTED_HYPOTHESIS_ID]

    checks = {
        "target_hypothesis_weakened": (
            target.get("status") == "weakened"
        ),
        "support_finding_present": (
            "FND-0003" in findings
            and "FND-0003" in target.get("evidence_for", [])
        ),
        "weakening_finding_present": (
            "FND-0004" in findings
            and "FND-0004" in target.get("evidence_against", [])
        ),
        "support_is_e3": (
            findings.get("FND-0003", {}).get("evidence_level") == "E3"
        ),
        "weakening_is_e3": (
            findings.get("FND-0004", {}).get("evidence_level") == "E3"
        ),
        "exp_0005_completed": (
            experiments.get("EXP-0005", {}).get("status") == "completed"
        ),
        "four_five_replication_present": (
            "FND-0007" in findings
            and findings["FND-0007"].get("evidence_level") == "E3"
        ),
        "target_model_scope_present": isinstance(
            target.get("model_scope"),
            dict,
        ),
        "target_checkpoint_sha_present": (
            isinstance(
                target.get("model_scope", {}).get("checkpoint_sha256"),
                str,
            )
            and len(target["model_scope"]["checkpoint_sha256"]) == 64
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "L10A scientific-context validation failed: "
            f"{failed}"
        )

    return {
        "checks": checks,
        "target_hypothesis": target,
        "support_finding": findings["FND-0003"],
        "weakening_finding": findings["FND-0004"],
        "four_five_replication": findings["FND-0007"],
    }


def build_proposal(
    *,
    handoff,
    handoff_path,
    handoff_sha256,
    live_knowledge_sha256,
    scientific_context,
):
    target = scientific_context["target_hypothesis"]
    model_scope = copy.deepcopy(target["model_scope"])

    proposal_core = {
        "experiment_id": NEW_EXPERIMENT_ID,
        "title": (
            "Cross-pair numerical candidate identity versus arithmetic correctness"
        ),
        "question": (
            "Across multiple fixed numerical candidate pairs, do the causal "
            "effects of L1H0, L3H3, and L4H0 track the semantically correct "
            "arithmetic answer, or do they track particular candidate-token "
            "identities?"
        ),
        "target_hypothesis_id": EXPECTED_HYPOTHESIS_ID,
        "target_hypothesis_statement": target.get("statement"),
        "model_scope": model_scope,
        "scientific_problem": {
            "supporting_anchor": {
                "finding_id": "FND-0003",
                "summary": (
                    "On the mature v0.3 2+2 anchor, L1H0 supported four while "
                    "L3H3 and L4H0 opposed the preferred margin."
                ),
            },
            "weakening_generalization": {
                "finding_id": "FND-0004",
                "summary": (
                    "Across six addition contexts, those head roles did not "
                    "generalize consistently with arithmetic correctness."
                ),
            },
            "established_positive_control": {
                "finding_id": "FND-0007",
                "summary": (
                    "EXP-0005 established that L1H0 consistently favored four "
                    "and L4H0 consistently favored five across 12 fixed "
                    "four-versus-five contexts."
                ),
            },
        },
        "competing_explanations": [
            {
                "explanation_id": "A_ARITHMETIC_CORRECTNESS",
                "statement": (
                    "The selected heads form a general arithmetic correctness "
                    "circuit. Their causal effects should systematically follow "
                    "which candidate is the correct numerical answer, even "
                    "when candidate-token identity changes."
                ),
            },
            {
                "explanation_id": "B_PAIR_SPECIFIC_TOKEN_BIAS",
                "statement": (
                    "The apparent arithmetic roles are largely driven by "
                    "candidate-token identity. The strong L1H0-four and "
                    "L4H0-five directions should be strongest for the "
                    "four-versus-five pair and should not generally follow "
                    "correctness for other number pairs."
                ),
            },
            {
                "explanation_id": "C_BROADER_NUMBER_TOKEN_BIAS",
                "statement": (
                    "The network contains a broader number-token selection "
                    "mechanism, but individual causal directions are "
                    "associated with token identities or candidate pairs "
                    "rather than semantic correctness."
                ),
            },
        ],
        "design_constraints_for_next_stage": {
            "must_include_positive_control_pair": "four_vs_five",
            "must_include_non_four_five_pairs": True,
            "candidate_pairs_should_be_fixed_within_probe_family": True,
            "semantic_correctness_must_be_balanced": True,
            "same_target_heads_as_prior_work": [
                "L1H0",
                "L3H3",
                "L4H0",
                "L4H1",
            ],
            "L4H1_role": "weak_control",
            "measurement_type": "temporary_attention_head_ablation",
            "required_permission_when_executed": "P3",
            "exact_probes_frozen_here": False,
            "decision_thresholds_frozen_here": False,
            "note": (
                "L10A proposes and registers the scientific experiment. The "
                "next planning stage must freeze exact candidate pairs, "
                "prompts, orientation, measurements, and decision thresholds "
                "before execution."
            ),
        },
        "expected_discrimination": {
            "supports_general_arithmetic_circuit_if": (
                "Across multiple number pairs, head effects systematically "
                "track the semantically correct candidate rather than a fixed "
                "token identity."
            ),
            "further_weakens_general_arithmetic_circuit_if": (
                "Effects remain tied to specific candidate identities or pairs "
                "and fail to follow semantic correctness across new number pairs."
            ),
            "supports_broader_token_bias_followup_if": (
                "Stable pair- or token-specific directions emerge for several "
                "number pairs, suggesting a broader numerical token-selection "
                "mechanism."
            ),
        },
        "required_permission": "P3",
        "proposal_only": True,
        "registered_status": "planned",
        "execution_authorized": False,
        "planning_complete": False,
    }

    proposal_seed = {
        "request_id": handoff["handoff"]["request_id"],
        "knowledge_state_sha256": live_knowledge_sha256,
        "proposal_core": proposal_core,
    }

    proposal_hash = hashlib.sha256(
        json.dumps(
            proposal_seed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    proposal_id = "PROP-" + proposal_hash[:16].upper()

    return {
        "engine_version": ENGINE_VERSION,
        "created_at_utc": utc_now(),
        "proposal_id": proposal_id,
        "mode": "deterministic_experiment_proposal",
        "source_handoff": {
            "path": relative_project_path(handoff_path),
            "sha256": handoff_sha256,
            "request_id": handoff["handoff"]["request_id"],
            "knowledge_state_sha256": live_knowledge_sha256,
        },
        "proposal": proposal_core,
        "policy": {
            "registers_planned_experiment": True,
            "executes_experiment": False,
            "loads_model": False,
            "performs_intervention": False,
            "creates_evidence": False,
            "changes_hypothesis_status": False,
            "grants_permission": False,
            "freezes_exact_experiment_plan": False,
        },
    }


def prepare_registration(
    *,
    data,
    proposal,
    overwrite,
):
    experiments = copy.deepcopy(data["experiments"])
    hypotheses = copy.deepcopy(data["hypotheses"])
    research_state = copy.deepcopy(data["research_state"])

    experiment_index = index_by(
        experiments["records"],
        "experiment_id",
    )
    hypothesis_index = index_by(
        hypotheses["records"],
        "hypothesis_id",
    )

    existing = experiment_index.get(NEW_EXPERIMENT_ID)

    if existing is not None:
        same_target = (
            existing.get("hypothesis_ids")
            == [EXPECTED_HYPOTHESIS_ID]
        )
        same_status = existing.get("status") == "planned"
        same_question = (
            existing.get("question")
            == proposal["proposal"]["question"]
        )

        if same_target and same_status and same_question:
            return {
                "already_registered": True,
                "experiments": experiments,
                "hypotheses": hypotheses,
                "research_state": research_state,
            }

        if not overwrite:
            raise RuntimeError(
                "EXP-0006 already exists with different content. Review the "
                "existing record before using --overwrite."
            )

    target_hypothesis = hypothesis_index[
        EXPECTED_HYPOTHESIS_ID
    ]

    created_at = utc_now()

    experiment_record = {
        "experiment_id": NEW_EXPERIMENT_ID,
        "title": proposal["proposal"]["title"],
        "question": proposal["proposal"]["question"],
        "status": "planned",
        "evidence_level": None,
        "model_scope": copy.deepcopy(
            proposal["proposal"]["model_scope"]
        ),
        "summary_artifact": None,
        "hypothesis_ids": [EXPECTED_HYPOTHESIS_ID],
        "finding_ids": [],
        "required_permission": "P3",
        "proposal_id": proposal["proposal_id"],
        "created_at_utc": created_at,
    }

    if existing is None:
        experiments["records"].append(
            experiment_record
        )
    else:
        position = experiments["records"].index(existing)
        experiments["records"][position] = experiment_record

    tested_by = list(
        target_hypothesis.get(
            "tested_by",
            [],
        )
    )

    if NEW_EXPERIMENT_ID not in tested_by:
        tested_by.append(NEW_EXPERIMENT_ID)

    target_hypothesis["tested_by"] = tested_by
    target_hypothesis["updated_at_utc"] = created_at

    state = research_state["state"]
    state["next_recommended_experiment"] = NEW_EXPERIMENT_ID
    state["last_experiment_proposal"] = {
        "proposal_id": proposal["proposal_id"],
        "experiment_id": NEW_EXPERIMENT_ID,
        "hypothesis_id": EXPECTED_HYPOTHESIS_ID,
        "created_at_utc": created_at,
    }

    experiments["updated_at_utc"] = created_at
    hypotheses["updated_at_utc"] = created_at
    research_state["updated_at_utc"] = created_at

    return {
        "already_registered": False,
        "experiments": experiments,
        "hypotheses": hypotheses,
        "research_state": research_state,
    }


def validate_registered_state(*, store, proposal_id):
    validation = store.validate()

    require(
        validation["passed"],
        "L10A registered knowledge store failed validation.",
    )

    data = store.load_all()

    experiments = index_by(
        data["experiments"]["records"],
        "experiment_id",
    )
    hypotheses = index_by(
        data["hypotheses"]["records"],
        "hypothesis_id",
    )
    state = data["research_state"]["state"]

    experiment = experiments[NEW_EXPERIMENT_ID]
    hypothesis = hypotheses[EXPECTED_HYPOTHESIS_ID]

    checks = {
        "knowledge_store_valid": bool(validation["passed"]),
        "exp_0006_registered": NEW_EXPERIMENT_ID in experiments,
        "exp_0006_planned": experiment.get("status") == "planned",
        "exp_0006_has_no_evidence": (
            experiment.get("evidence_level") is None
            and experiment.get("summary_artifact") is None
            and experiment.get("finding_ids") == []
        ),
        "exp_0006_targets_hyp_001": (
            experiment.get("hypothesis_ids")
            == [EXPECTED_HYPOTHESIS_ID]
        ),
        "exp_0006_proposal_bound": (
            experiment.get("proposal_id") == proposal_id
        ),
        "hypothesis_status_unchanged": (
            hypothesis.get("status") == "weakened"
        ),
        "hypothesis_tested_by_exp_0006": (
            NEW_EXPERIMENT_ID
            in hypothesis.get("tested_by", [])
        ),
        "research_state_recommends_exp_0006": (
            state.get("next_recommended_experiment")
            == NEW_EXPERIMENT_ID
        ),
        "no_exp_0006_evidence_record": all(
            record.get("experiment_id") != NEW_EXPERIMENT_ID
            for record in data["evidence_ledger"]["records"]
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "L10A registered-state validation failed: "
            f"{failed}"
        )

    return {
        "validation": validation,
        "checks": checks,
    }


def regenerate_snapshot(
    *,
    store,
    snapshot_path,
    snapshot_report_path,
):
    knowledge = KnowledgeQuery(store=store)
    snapshot = build_snapshot(store, knowledge)
    validation = validate_snapshot(snapshot)

    require(
        validation["passed"],
        "L10A regenerated scientist snapshot failed validation.",
    )

    atomic_write_json(
        snapshot_path,
        snapshot,
    )

    saved = load_json(snapshot_path)
    saved_validation = validate_snapshot(saved)

    require(
        saved_validation["passed"],
        "L10A saved scientist snapshot failed validation.",
    )

    report = {
        "stage": "L7D",
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at_utc": utc_now(),
        "snapshot_path": relative_project_path(
            snapshot_path
        ),
        "knowledge_state_sha256": snapshot[
            "knowledge_state_sha256"
        ],
        "knowledge_counts": snapshot[
            "knowledge_counts"
        ],
        "active_hypothesis_count": len(
            snapshot["hypotheses"]["active"]
        ),
        "supported_hypothesis_count": len(
            snapshot["hypotheses"]["supported"]
        ),
        "weakened_hypothesis_count": len(
            snapshot["hypotheses"]["weakened"]
        ),
        "planned_experiment_count": len(
            snapshot["experiments"]["planned"]
        ),
        "next_recommended_experiment": snapshot[
            "research_focus"
        ][
            "next_recommended_experiment"
        ],
        "snapshot_validation": saved_validation,
        "passed": True,
    }

    atomic_write_json(
        snapshot_report_path,
        report,
    )

    return {
        "snapshot": snapshot,
        "validation": saved_validation,
    }


def main():
    args = parse_arguments()

    handoff_path = project_path(
        args.handoff
    )
    proposal_path = project_path(
        args.proposal
    )
    snapshot_path = project_path(
        args.snapshot
    )
    snapshot_report_path = project_path(
        args.snapshot_report
    )

    print()
    print("=" * 80)
    print(
        "STAGE L10A — EXPERIMENT PROPOSAL GENERATOR"
    )
    print("=" * 80)

    store = KnowledgeStore()
    store_validation = store.validate()

    require(
        store_validation["passed"],
        "Current L7 knowledge store is invalid.",
    )

    data = store.load_all()
    live_knowledge_sha256 = sha256_json(data)

    handoff = load_json(handoff_path)
    handoff_sha256 = file_sha256(
        handoff_path
    )

    print()
    print("Current knowledge fingerprint:")
    print(live_knowledge_sha256)
    print()
    print("Research-action handoff SHA:")
    print(handoff_sha256)

    handoff_checks = validate_handoff(
        handoff=handoff,
        live_knowledge_sha256=live_knowledge_sha256,
    )

    scientific_context = validate_scientific_context(
        data=data
    )

    print()
    print("Handoff checks")
    print("--------------")

    for name, passed in handoff_checks.items():
        print(f"{name:<46} {passed}")

    print()
    print("Scientific-context checks")
    print("-------------------------")

    for name, passed in scientific_context["checks"].items():
        print(f"{name:<46} {passed}")

    proposal = build_proposal(
        handoff=handoff,
        handoff_path=handoff_path,
        handoff_sha256=handoff_sha256,
        live_knowledge_sha256=live_knowledge_sha256,
        scientific_context=scientific_context,
    )

    print()
    print("=" * 80)
    print("L10A PROPOSED EXPERIMENT")
    print("=" * 80)
    print()
    print("Proposal ID:")
    print(f"  {proposal['proposal_id']}")
    print()
    print("Experiment:")
    print(f"  {NEW_EXPERIMENT_ID}")
    print()
    print("Title:")
    print("  " + proposal["proposal"]["title"])
    print()
    print("Question:")
    print("  " + proposal["proposal"]["question"])
    print()
    print("Target hypothesis:")
    print(f"  {EXPECTED_HYPOTHESIS_ID}")
    print()
    print("Required permission when executed:")
    print("  P3")
    print()
    print("Exact probes frozen:")
    print("  False")
    print()
    print("Execution authorized:")
    print("  False")

    prepared = prepare_registration(
        data=data,
        proposal=proposal,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        print()
        print("Proposed L7 changes")
        print("-------------------")
        print("  Register EXP-0006 as planned")
        print(
            "  Add EXP-0006 to "
            "HYP-V03-001.tested_by"
        )
        print(
            "  Set next recommended experiment "
            "to EXP-0006"
        )
        print(
            "  Regenerate scientist_context.json"
        )
        print()
        print("Knowledge files modified: False")
        print()
        print("=" * 80)
        print("SUCCESS: L10A DRY RUN PASSED.")
        print("=" * 80)
        return

    paths = [
        store.collection_path("experiments"),
        store.collection_path("hypotheses"),
        store.collection_path("research_state"),
        snapshot_path,
        snapshot_report_path,
        proposal_path,
    ]

    backups = capture_backups(paths)

    try:
        store.save(
            "experiments",
            prepared["experiments"],
        )
        store.save(
            "hypotheses",
            prepared["hypotheses"],
        )
        store.save(
            "research_state",
            prepared["research_state"],
        )

        registered_validation = validate_registered_state(
            store=store,
            proposal_id=proposal["proposal_id"],
        )

        snapshot_result = regenerate_snapshot(
            store=store,
            snapshot_path=snapshot_path,
            snapshot_report_path=snapshot_report_path,
        )

        proposal["registration"] = {
            "registered": True,
            "experiment_status": "planned",
            "evidence_created": False,
            "hypothesis_status_changed": False,
            "old_knowledge_state_sha256": live_knowledge_sha256,
            "new_knowledge_state_sha256": snapshot_result[
                "snapshot"
            ][
                "knowledge_state_sha256"
            ],
            "checks": registered_validation["checks"],
        }

        proposal["overall_checks"] = {
            "handoff_valid": all(
                handoff_checks.values()
            ),
            "scientific_context_valid": all(
                scientific_context["checks"].values()
            ),
            "registration_valid": all(
                registered_validation["checks"].values()
            ),
            "snapshot_valid": bool(
                snapshot_result["validation"]["passed"]
            ),
            "experiment_not_executed": True,
            "model_not_loaded": True,
            "intervention_not_performed": True,
            "evidence_not_created": True,
            "hypothesis_status_not_changed": True,
            "exact_plan_not_frozen": True,
        }

        atomic_write_json(
            proposal_path,
            proposal,
        )

    except Exception:
        restore_backups(backups)
        raise

    print()
    print("=" * 80)
    print("L10A REGISTRATION SUMMARY")
    print("=" * 80)
    print()
    print("Registered experiment:")
    print("  EXP-0006")
    print()
    print("Status:")
    print("  planned")
    print()
    print("Evidence created:")
    print("  False")
    print()
    print("Hypothesis status changed:")
    print("  False")
    print()
    print("Next recommended experiment:")
    print("  EXP-0006")
    print()
    print("New knowledge fingerprint:")
    print(
        "  "
        + snapshot_result[
            "snapshot"
        ][
            "knowledge_state_sha256"
        ]
    )
    print()
    print("Proposal artifact:")
    print(f"  {proposal_path}")
    print()
    print("Overall checks")
    print("--------------")

    for name, passed in proposal[
        "overall_checks"
    ].items():
        print(f"{name:<40} {passed}")

    print()
    print("=" * 80)
    print(
        "SUCCESS: STAGE L10A EXPERIMENT "
        "PROPOSAL + REGISTRATION COMPLETE."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
