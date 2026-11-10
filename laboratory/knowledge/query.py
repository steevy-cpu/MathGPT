import argparse
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
    KnowledgeStore,
)


# ============================================================
# Query engine
# ============================================================

class KnowledgeQuery:
    """
    Read-only query interface over the Stage-L7
    scientific knowledge store.

    This class does not:
        - modify hypotheses
        - create findings
        - update confidence
        - plan experiments

    Those capabilities belong to later stages.

    L7C only retrieves and connects existing
    scientific knowledge.
    """

    def __init__(
        self,
        store=None,
    ):

        if store is None:

            store = KnowledgeStore()

        self.store = store

        # Validate before answering scientific queries.
        self.validation = (
            self.store.validate()
        )

        self.data = (
            self.store.load_all()
        )

        self.hypotheses = (
            self.data[
                "hypotheses"
            ][
                "records"
            ]
        )

        self.findings = (
            self.data[
                "findings"
            ][
                "records"
            ]
        )

        self.experiments = (
            self.data[
                "experiments"
            ][
                "records"
            ]
        )

        self.evidence_records = (
            self.data[
                "evidence_ledger"
            ][
                "records"
            ]
        )

        self.research_state = (
            self.data[
                "research_state"
            ][
                "state"
            ]
        )

        self.hypothesis_index = {
            item[
                "hypothesis_id"
            ]:
                item

            for item
            in self.hypotheses
        }

        self.finding_index = {
            item[
                "finding_id"
            ]:
                item

            for item
            in self.findings
        }

        self.experiment_index = {
            item[
                "experiment_id"
            ]:
                item

            for item
            in self.experiments
        }

        self.evidence_index = {
            item[
                "evidence_record_id"
            ]:
                item

            for item
            in self.evidence_records
        }

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _copy(
        value,
    ):

        # JSON round-trip gives callers detached
        # read-only-style records.
        return json.loads(
            json.dumps(
                value
            )
        )

    @staticmethod
    def _sort_by_id(
        values,
        field,
    ):

        return sorted(
            values,
            key=lambda item:
                item.get(
                    field,
                    "",
                ),
        )

    # ========================================================
    # Overall summary
    # ========================================================

    def summary(
        self,
    ):

        statuses = {}

        for hypothesis in self.hypotheses:

            status = hypothesis[
                "status"
            ]

            statuses[
                status
            ] = (
                statuses.get(
                    status,
                    0,
                )
                +
                1
            )

        experiment_statuses = {}

        for experiment in self.experiments:

            status = experiment[
                "status"
            ]

            experiment_statuses[
                status
            ] = (
                experiment_statuses.get(
                    status,
                    0,
                )
                +
                1
            )

        return {
            "schema_version":
                self.validation[
                    "schema_version"
                ],

            "hypothesis_count":
                len(
                    self.hypotheses
                ),

            "finding_count":
                len(
                    self.findings
                ),

            "experiment_count":
                len(
                    self.experiments
                ),

            "evidence_record_count":
                len(
                    self.evidence_records
                ),

            "hypothesis_statuses":
                statuses,

            "experiment_statuses":
                experiment_statuses,

            "current_research_question":
                self.research_state.get(
                    "current_research_question"
                ),

            "last_completed_experiment":
                self.research_state.get(
                    "last_completed_experiment"
                ),

            "next_recommended_experiment":
                self.research_state.get(
                    "next_recommended_experiment"
                ),

            "scientist_status":
                self.research_state.get(
                    "scientist_status"
                ),

            "knowledge_validation":
                bool(
                    self.validation[
                        "passed"
                    ]
                ),
        }

    # ========================================================
    # Hypotheses
    # ========================================================

    def list_hypotheses(
        self,
        status=None,
    ):

        values = self.hypotheses

        if status is not None:

            values = [
                item

                for item
                in values

                if item.get(
                    "status"
                )
                ==
                status
            ]

        return self._copy(
            self._sort_by_id(
                values,
                "hypothesis_id",
            )
        )

    def active_hypotheses(
        self,
    ):

        active_ids = set(
            self.research_state.get(
                "active_hypotheses",
                [],
            )
        )

        values = [
            item

            for item
            in self.hypotheses

            if item[
                "hypothesis_id"
            ]
            in active_ids
        ]

        return self._copy(
            self._sort_by_id(
                values,
                "hypothesis_id",
            )
        )

    def get_hypothesis(
        self,
        hypothesis_id,
    ):

        if (
            hypothesis_id
            not in
            self.hypothesis_index
        ):

            raise KeyError(
                f"Unknown hypothesis: "
                f"{hypothesis_id}"
            )

        hypothesis = self.hypothesis_index[
            hypothesis_id
        ]

        supporting_findings = [
            self.finding_index[
                finding_id
            ]

            for finding_id
            in hypothesis.get(
                "evidence_for",
                [],
            )
        ]

        weakening_findings = [
            self.finding_index[
                finding_id
            ]

            for finding_id
            in hypothesis.get(
                "evidence_against",
                [],
            )
        ]

        experiments = [
            self.experiment_index[
                experiment_id
            ]

            for experiment_id
            in hypothesis.get(
                "tested_by",
                [],
            )
        ]

        return self._copy(
            {
                "hypothesis":
                    hypothesis,

                "supporting_findings":
                    supporting_findings,

                "weakening_findings":
                    weakening_findings,

                "tested_by_experiments":
                    experiments,
            }
        )

    # ========================================================
    # Findings
    # ========================================================

    def list_findings(
        self,
        *,
        experiment_id=None,
        hypothesis_id=None,
        tag=None,
    ):

        values = list(
            self.findings
        )

        if experiment_id is not None:

            values = [
                finding

                for finding
                in values

                if finding.get(
                    "source_experiment_id"
                )
                ==
                experiment_id
            ]

        if hypothesis_id is not None:

            values = [
                finding

                for finding
                in values

                if (
                    hypothesis_id
                    in
                    finding.get(
                        "supports_hypotheses",
                        [],
                    )

                    or

                    hypothesis_id
                    in
                    finding.get(
                        "weakens_hypotheses",
                        [],
                    )
                )
            ]

        if tag is not None:

            tag_lower = (
                str(
                    tag
                )
                .lower()
            )

            values = [
                finding

                for finding
                in values

                if any(
                    tag_lower
                    ==
                    str(
                        candidate
                    ).lower()

                    for candidate
                    in finding.get(
                        "tags",
                        [],
                    )
                )
            ]

        return self._copy(
            self._sort_by_id(
                values,
                "finding_id",
            )
        )

    def get_finding(
        self,
        finding_id,
    ):

        if (
            finding_id
            not in
            self.finding_index
        ):

            raise KeyError(
                f"Unknown finding: "
                f"{finding_id}"
            )

        finding = self.finding_index[
            finding_id
        ]

        experiment = (
            self.experiment_index[
                finding[
                    "source_experiment_id"
                ]
            ]
        )

        supported = [
            self.hypothesis_index[
                hypothesis_id
            ]

            for hypothesis_id
            in finding.get(
                "supports_hypotheses",
                [],
            )
        ]

        weakened = [
            self.hypothesis_index[
                hypothesis_id
            ]

            for hypothesis_id
            in finding.get(
                "weakens_hypotheses",
                [],
            )
        ]

        return self._copy(
            {
                "finding":
                    finding,

                "source_experiment":
                    experiment,

                "supports":
                    supported,

                "weakens":
                    weakened,
            }
        )

    # ========================================================
    # Experiments
    # ========================================================

    def list_experiments(
        self,
        status=None,
    ):

        values = self.experiments

        if status is not None:

            values = [
                experiment

                for experiment
                in values

                if experiment.get(
                    "status"
                )
                ==
                status
            ]

        return self._copy(
            self._sort_by_id(
                values,
                "experiment_id",
            )
        )

    def get_experiment(
        self,
        experiment_id,
    ):

        if (
            experiment_id
            not in
            self.experiment_index
        ):

            raise KeyError(
                f"Unknown experiment: "
                f"{experiment_id}"
            )

        experiment = self.experiment_index[
            experiment_id
        ]

        findings = [
            self.finding_index[
                finding_id
            ]

            for finding_id
            in experiment.get(
                "finding_ids",
                [],
            )
        ]

        hypotheses = [
            self.hypothesis_index[
                hypothesis_id
            ]

            for hypothesis_id
            in experiment.get(
                "hypothesis_ids",
                [],
            )
        ]

        evidence = (
            self.evidence_for_experiment(
                experiment_id
            )
        )

        return self._copy(
            {
                "experiment":
                    experiment,

                "findings":
                    findings,

                "hypotheses":
                    hypotheses,

                "evidence_record":
                    evidence,
            }
        )

    # ========================================================
    # Evidence ledger
    # ========================================================

    def evidence_for_experiment(
        self,
        experiment_id,
    ):

        if (
            experiment_id
            not in
            self.experiment_index
        ):

            raise KeyError(
                f"Unknown experiment: "
                f"{experiment_id}"
            )

        matches = [
            item

            for item
            in self.evidence_records

            if item.get(
                "experiment_id"
            )
            ==
            experiment_id
        ]

        if not matches:

            return None

        if len(
            matches
        ) != 1:

            raise RuntimeError(
                f"Experiment {experiment_id} "
                "has multiple evidence ledger "
                "records."
            )

        return self._copy(
            matches[
                0
            ]
        )

    def get_evidence_record(
        self,
        evidence_record_id,
    ):

        if (
            evidence_record_id
            not in
            self.evidence_index
        ):

            raise KeyError(
                f"Unknown evidence record: "
                f"{evidence_record_id}"
            )

        return self._copy(
            self.evidence_index[
                evidence_record_id
            ]
        )

    # ========================================================
    # Checkpoint queries
    # ========================================================

    def experiments_for_checkpoint(
        self,
        checkpoint_sha256,
    ):

        values = []

        for experiment in self.experiments:

            model_scope = experiment.get(
                "model_scope",
                {},
            )

            if (
                model_scope.get(
                    "checkpoint_sha256"
                )
                ==
                checkpoint_sha256
            ):

                values.append(
                    experiment
                )

        return self._copy(
            self._sort_by_id(
                values,
                "experiment_id",
            )
        )

    def findings_for_checkpoint(
        self,
        checkpoint_sha256,
    ):

        values = []

        for finding in self.findings:

            model_scope = finding.get(
                "model_scope",
                {},
            )

            if (
                model_scope.get(
                    "checkpoint_sha256"
                )
                ==
                checkpoint_sha256
            ):

                values.append(
                    finding
                )

        return self._copy(
            self._sort_by_id(
                values,
                "finding_id",
            )
        )

    # ========================================================
    # Open research questions
    # ========================================================

    def open_questions(
        self,
    ):

        records = []

        seen = set()

        # ----------------------------------------------------
        # Program-level questions
        # ----------------------------------------------------

        for question in (
            self.research_state.get(
                "open_questions",
                [],
            )
        ):

            key = (
                "research_state",
                question,
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            records.append(
                {
                    "source_type":
                        "research_state",

                    "source_id":
                        None,

                    "question":
                        question,
                }
            )

        # ----------------------------------------------------
        # Hypothesis-level questions
        # ----------------------------------------------------

        for hypothesis in self.hypotheses:

            if (
                hypothesis.get(
                    "status"
                )
                in
                {
                    "retired",
                    "falsified",
                }
            ):

                continue

            for question in (
                hypothesis.get(
                    "open_questions",
                    [],
                )
            ):

                key = (
                    hypothesis[
                        "hypothesis_id"
                    ],
                    question,
                )

                if key in seen:

                    continue

                seen.add(
                    key
                )

                records.append(
                    {
                        "source_type":
                            "hypothesis",

                        "source_id":
                            hypothesis[
                                "hypothesis_id"
                            ],

                        "question":
                            question,
                    }
                )

        return self._copy(
            records
        )

    # ========================================================
    # Next recommended experiment
    # ========================================================

    def next_recommended_experiment(
        self,
    ):

        experiment_id = (
            self.research_state.get(
                "next_recommended_experiment"
            )
        )

        if experiment_id is None:

            return None

        return self.get_experiment(
            experiment_id
        )

    # ========================================================
    # Text search
    # ========================================================

    def search(
        self,
        text,
    ):

        query = (
            str(
                text
            )
            .strip()
            .lower()
        )

        if not query:

            raise ValueError(
                "Search text cannot be empty."
            )

        results = []

        # ----------------------------------------------------
        # Hypotheses
        # ----------------------------------------------------

        for item in self.hypotheses:

            haystack = " ".join(
                [
                    item.get(
                        "hypothesis_id",
                        "",
                    ),

                    item.get(
                        "title",
                        "",
                    ),

                    item.get(
                        "statement",
                        "",
                    ),

                    " ".join(
                        item.get(
                            "tags",
                            [],
                        )
                    ),
                ]
            ).lower()

            if query in haystack:

                results.append(
                    {
                        "type":
                            "hypothesis",

                        "id":
                            item[
                                "hypothesis_id"
                            ],

                        "title":
                            item.get(
                                "title"
                            ),

                        "status":
                            item.get(
                                "status"
                            ),
                    }
                )

        # ----------------------------------------------------
        # Findings
        # ----------------------------------------------------

        for item in self.findings:

            haystack = " ".join(
                [
                    item.get(
                        "finding_id",
                        "",
                    ),

                    item.get(
                        "title",
                        "",
                    ),

                    item.get(
                        "statement",
                        "",
                    ),

                    " ".join(
                        item.get(
                            "tags",
                            [],
                        )
                    ),
                ]
            ).lower()

            if query in haystack:

                results.append(
                    {
                        "type":
                            "finding",

                        "id":
                            item[
                                "finding_id"
                            ],

                        "title":
                            item.get(
                                "title"
                            ),

                        "experiment":
                            item.get(
                                "source_experiment_id"
                            ),
                    }
                )

        # ----------------------------------------------------
        # Experiments
        # ----------------------------------------------------

        for item in self.experiments:

            haystack = " ".join(
                [
                    item.get(
                        "experiment_id",
                        "",
                    ),

                    item.get(
                        "title",
                        "",
                    ),

                    item.get(
                        "question",
                        "",
                    ),
                ]
            ).lower()

            if query in haystack:

                results.append(
                    {
                        "type":
                            "experiment",

                        "id":
                            item[
                                "experiment_id"
                            ],

                        "title":
                            item.get(
                                "title"
                            ),

                        "status":
                            item.get(
                                "status"
                            ),
                    }
                )

        return self._copy(
            results
        )


# ============================================================
# CLI helpers
# ============================================================

def print_json(
    value,
):

    print(
        json.dumps(
            value,
            indent=4,
            ensure_ascii=False,
        )
    )


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L7C — query the MathGPT "
            "scientific knowledge store."
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    subparsers.add_parser(
        "summary",
        help=(
            "Show overall scientific "
            "knowledge status."
        ),
    )

    # --------------------------------------------------------
    # active
    # --------------------------------------------------------

    subparsers.add_parser(
        "active",
        help=(
            "Show active hypotheses."
        ),
    )

    # --------------------------------------------------------
    # hypotheses
    # --------------------------------------------------------

    hypotheses_parser = (
        subparsers.add_parser(
            "hypotheses",
            help=(
                "List hypotheses."
            ),
        )
    )

    hypotheses_parser.add_argument(
        "--status",
        default=None,
    )

    # --------------------------------------------------------
    # hypothesis
    # --------------------------------------------------------

    hypothesis_parser = (
        subparsers.add_parser(
            "hypothesis",
            help=(
                "Show one hypothesis and "
                "its evidence."
            ),
        )
    )

    hypothesis_parser.add_argument(
        "hypothesis_id",
    )

    # --------------------------------------------------------
    # experiment
    # --------------------------------------------------------

    experiment_parser = (
        subparsers.add_parser(
            "experiment",
            help=(
                "Show one experiment."
            ),
        )
    )

    experiment_parser.add_argument(
        "experiment_id",
    )

    # --------------------------------------------------------
    # finding
    # --------------------------------------------------------

    finding_parser = (
        subparsers.add_parser(
            "finding",
            help=(
                "Show one finding."
            ),
        )
    )

    finding_parser.add_argument(
        "finding_id",
    )

    # --------------------------------------------------------
    # findings
    # --------------------------------------------------------

    findings_parser = (
        subparsers.add_parser(
            "findings",
            help=(
                "List/filter findings."
            ),
        )
    )

    findings_parser.add_argument(
        "--experiment-id",
        default=None,
    )

    findings_parser.add_argument(
        "--hypothesis-id",
        default=None,
    )

    findings_parser.add_argument(
        "--tag",
        default=None,
    )

    # --------------------------------------------------------
    # evidence
    # --------------------------------------------------------

    evidence_parser = (
        subparsers.add_parser(
            "evidence",
            help=(
                "Show the verified raw "
                "evidence record for an "
                "experiment."
            ),
        )
    )

    evidence_parser.add_argument(
        "experiment_id",
    )

    # --------------------------------------------------------
    # checkpoint
    # --------------------------------------------------------

    checkpoint_parser = (
        subparsers.add_parser(
            "checkpoint",
            help=(
                "Show experiments and findings "
                "for a checkpoint SHA-256."
            ),
        )
    )

    checkpoint_parser.add_argument(
        "sha256",
    )

    # --------------------------------------------------------
    # questions
    # --------------------------------------------------------

    subparsers.add_parser(
        "questions",
        help=(
            "Show unresolved scientific "
            "questions."
        ),
    )

    # --------------------------------------------------------
    # next
    # --------------------------------------------------------

    subparsers.add_parser(
        "next",
        help=(
            "Show the currently recommended "
            "next experiment."
        ),
    )

    # --------------------------------------------------------
    # search
    # --------------------------------------------------------

    search_parser = (
        subparsers.add_parser(
            "search",
            help=(
                "Search hypotheses, findings, "
                "and experiments."
            ),
        )
    )

    search_parser.add_argument(
        "text",
    )

    return parser


# ============================================================
# CLI
# ============================================================

def main():

    parser = build_parser()

    args = parser.parse_args()

    knowledge = KnowledgeQuery()

    if args.command == "summary":

        print_json(
            knowledge.summary()
        )

    elif args.command == "active":

        print_json(
            knowledge.active_hypotheses()
        )

    elif args.command == "hypotheses":

        print_json(
            knowledge.list_hypotheses(
                status=args.status,
            )
        )

    elif args.command == "hypothesis":

        print_json(
            knowledge.get_hypothesis(
                args.hypothesis_id
            )
        )

    elif args.command == "experiment":

        print_json(
            knowledge.get_experiment(
                args.experiment_id
            )
        )

    elif args.command == "finding":

        print_json(
            knowledge.get_finding(
                args.finding_id
            )
        )

    elif args.command == "findings":

        print_json(
            knowledge.list_findings(
                experiment_id=(
                    args.experiment_id
                ),
                hypothesis_id=(
                    args.hypothesis_id
                ),
                tag=args.tag,
            )
        )

    elif args.command == "evidence":

        print_json(
            knowledge.evidence_for_experiment(
                args.experiment_id
            )
        )

    elif args.command == "checkpoint":

        print_json(
            {
                "checkpoint_sha256":
                    args.sha256,

                "experiments":
                    (
                        knowledge
                        .experiments_for_checkpoint(
                            args.sha256
                        )
                    ),

                "findings":
                    (
                        knowledge
                        .findings_for_checkpoint(
                            args.sha256
                        )
                    ),
            }
        )

    elif args.command == "questions":

        print_json(
            knowledge.open_questions()
        )

    elif args.command == "next":

        print_json(
            knowledge.next_recommended_experiment()
        )

    elif args.command == "search":

        print_json(
            knowledge.search(
                args.text
            )
        )

    else:

        raise RuntimeError(
            f"Unhandled command: "
            f"{args.command}"
        )


if __name__ == "__main__":

    main()