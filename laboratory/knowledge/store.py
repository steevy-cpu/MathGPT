import json

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path


# ============================================================
# Project / knowledge paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


KNOWLEDGE_DIR = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
)


SCHEMA_VERSION = (
    "L7.1"
)


COLLECTION_FILES = {
    "hypotheses":
        "hypotheses.json",

    "findings":
        "findings.json",

    "experiments":
        "experiment_registry.json",

    "research_state":
        "research_state.json",

    "evidence_ledger":
        "evidence_ledger.json",
}


# ============================================================
# Allowed states
# ============================================================

HYPOTHESIS_STATUSES = {
    "proposed",
    "active",
    "supported",
    "weakened",
    "falsified",
    "unresolved",
    "retired",
}


EXPERIMENT_STATUSES = {
    "planned",
    "running",
    "completed",
    "failed",
    "cancelled",
}


EVIDENCE_LEVELS = {
    None,
    "E0",
    "E1",
    "E2",
    "E3",
}


# ============================================================
# Exceptions
# ============================================================

class KnowledgeValidationError(
    RuntimeError
):
    pass


# ============================================================
# Helpers
# ============================================================

def utc_now():

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def atomic_write_json(
    path,
    value,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_name(
            path.name
            +
            ".tmp"
        )
    )

    with open(
        temporary_path,
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

    temporary_path.replace(
        path
    )


def require(
    condition,
    message,
):

    if not condition:

        raise KnowledgeValidationError(
            message
        )


def empty_evidence_ledger():

    return {
        "schema_version":
            SCHEMA_VERSION,

        "collection":
            "evidence_ledger",

        "updated_at_utc":
            utc_now(),

        "records":
            [],
    }


# ============================================================
# Knowledge Store
# ============================================================

class KnowledgeStore:

    def __init__(
        self,
        root=None,
    ):

        if root is None:

            root = (
                KNOWLEDGE_DIR
            )

        self.root = Path(
            root
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Paths
    # ========================================================

    def collection_path(
        self,
        collection,
    ):

        if (
            collection
            not in
            COLLECTION_FILES
        ):

            raise KeyError(
                f"Unknown knowledge collection: "
                f"{collection}"
            )

        return (
            self.root
            /
            COLLECTION_FILES[
                collection
            ]
        )

    # ========================================================
    # Read
    # ========================================================

    def load(
        self,
        collection,
    ):

        path = self.collection_path(
            collection
        )

        # L7B introduces the evidence ledger after the
        # original L7A bootstrap. Returning an empty ledger
        # here keeps existing L7A stores backward-compatible.
        if (
            collection
            ==
            "evidence_ledger"
            and
            not path.exists()
        ):

            return (
                empty_evidence_ledger()
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

    def load_all(
        self,
    ):

        return {
            collection:
                self.load(
                    collection
                )

            for collection
            in COLLECTION_FILES
        }

    # ========================================================
    # Write
    # ========================================================

    def save(
        self,
        collection,
        value,
    ):

        atomic_write_json(
            self.collection_path(
                collection
            ),
            value,
        )

    # ========================================================
    # Indexes
    # ========================================================

    @staticmethod
    def _index(
        records,
        id_field,
    ):

        return {
            record[
                id_field
            ]:
                record

            for record
            in records
        }

    def experiment_index(
        self,
        data,
    ):

        return self._index(
            data[
                "experiments"
            ][
                "records"
            ],
            "experiment_id",
        )

    def finding_index(
        self,
        data,
    ):

        return self._index(
            data[
                "findings"
            ][
                "records"
            ],
            "finding_id",
        )

    def hypothesis_index(
        self,
        data,
    ):

        return self._index(
            data[
                "hypotheses"
            ][
                "records"
            ],
            "hypothesis_id",
        )

    def evidence_index(
        self,
        data,
    ):

        return self._index(
            data[
                "evidence_ledger"
            ][
                "records"
            ],
            "evidence_record_id",
        )

    # ========================================================
    # Validation helpers
    # ========================================================

    def _validate_collection_header(
        self,
        *,
        value,
        expected_collection,
    ):

        require(
            isinstance(
                value,
                dict,
            ),
            (
                f"{expected_collection} "
                "must be a JSON object."
            ),
        )

        require(
            value.get(
                "schema_version"
            )
            ==
            SCHEMA_VERSION,
            (
                f"{expected_collection} "
                "schema_version must be "
                f"{SCHEMA_VERSION}."
            ),
        )

        require(
            value.get(
                "collection"
            )
            ==
            expected_collection,
            (
                f"Expected collection "
                f"{expected_collection}."
            ),
        )

    def _validate_unique_ids(
        self,
        *,
        records,
        id_field,
        collection,
    ):

        ids = []

        for record in records:

            require(
                isinstance(
                    record,
                    dict,
                ),
                (
                    f"{collection} contains "
                    "a non-object record."
                ),
            )

            identifier = record.get(
                id_field
            )

            require(
                isinstance(
                    identifier,
                    str,
                )
                and
                bool(
                    identifier.strip()
                ),
                (
                    f"{collection} record "
                    f"missing {id_field}."
                ),
            )

            ids.append(
                identifier
            )

        require(
            len(
                ids
            )
            ==
            len(
                set(
                    ids
                )
            ),
            (
                f"Duplicate IDs found in "
                f"{collection}."
            ),
        )

    # ========================================================
    # Full validation
    # ========================================================

    def validate(
        self,
    ):

        data = self.load_all()

        hypotheses = (
            data[
                "hypotheses"
            ]
        )

        findings = (
            data[
                "findings"
            ]
        )

        experiments = (
            data[
                "experiments"
            ]
        )

        research_state = (
            data[
                "research_state"
            ]
        )

        evidence_ledger = (
            data[
                "evidence_ledger"
            ]
        )

        # ----------------------------------------------------
        # Headers
        # ----------------------------------------------------

        for name, value in (
            (
                "hypotheses",
                hypotheses,
            ),
            (
                "findings",
                findings,
            ),
            (
                "experiments",
                experiments,
            ),
            (
                "research_state",
                research_state,
            ),
            (
                "evidence_ledger",
                evidence_ledger,
            ),
        ):

            self._validate_collection_header(
                value=value,
                expected_collection=name,
            )

        # ----------------------------------------------------
        # Record arrays
        # ----------------------------------------------------

        for collection in (
            hypotheses,
            findings,
            experiments,
            evidence_ledger,
        ):

            require(
                isinstance(
                    collection.get(
                        "records"
                    ),
                    list,
                ),
                (
                    f"{collection['collection']} "
                    "records must be a list."
                ),
            )

        # ----------------------------------------------------
        # Unique IDs
        # ----------------------------------------------------

        self._validate_unique_ids(
            records=(
                hypotheses[
                    "records"
                ]
            ),
            id_field="hypothesis_id",
            collection="hypotheses",
        )

        self._validate_unique_ids(
            records=(
                findings[
                    "records"
                ]
            ),
            id_field="finding_id",
            collection="findings",
        )

        self._validate_unique_ids(
            records=(
                experiments[
                    "records"
                ]
            ),
            id_field="experiment_id",
            collection="experiments",
        )

        self._validate_unique_ids(
            records=(
                evidence_ledger[
                    "records"
                ]
            ),
            id_field="evidence_record_id",
            collection="evidence_ledger",
        )

        hypothesis_index = self._index(
            hypotheses[
                "records"
            ],
            "hypothesis_id",
        )

        finding_index = self._index(
            findings[
                "records"
            ],
            "finding_id",
        )

        experiment_index = self._index(
            experiments[
                "records"
            ],
            "experiment_id",
        )

        evidence_index = self._index(
            evidence_ledger[
                "records"
            ],
            "evidence_record_id",
        )

        # ----------------------------------------------------
        # Hypotheses
        # ----------------------------------------------------

        for hypothesis in (
            hypotheses[
                "records"
            ]
        ):

            hypothesis_id = (
                hypothesis[
                    "hypothesis_id"
                ]
            )

            require(
                hypothesis.get(
                    "status"
                )
                in
                HYPOTHESIS_STATUSES,
                (
                    f"{hypothesis_id}: "
                    "invalid hypothesis status."
                ),
            )

            require(
                isinstance(
                    hypothesis.get(
                        "statement"
                    ),
                    str,
                ),
                (
                    f"{hypothesis_id}: "
                    "statement missing."
                ),
            )

            confidence = hypothesis.get(
                "confidence"
            )

            require(
                isinstance(
                    confidence,
                    dict,
                ),
                (
                    f"{hypothesis_id}: "
                    "confidence must be an object."
                ),
            )

            score = confidence.get(
                "score"
            )

            if score is not None:

                require(
                    isinstance(
                        score,
                        (
                            int,
                            float,
                        ),
                    )
                    and
                    0.0
                    <=
                    float(
                        score
                    )
                    <=
                    1.0,
                    (
                        f"{hypothesis_id}: "
                        "confidence score must "
                        "be between 0 and 1."
                    ),
                )

            for finding_id in (
                hypothesis.get(
                    "evidence_for",
                    []
                )
            ):

                require(
                    finding_id
                    in
                    finding_index,
                    (
                        f"{hypothesis_id}: "
                        f"unknown evidence_for "
                        f"{finding_id}."
                    ),
                )

            for finding_id in (
                hypothesis.get(
                    "evidence_against",
                    []
                )
            ):

                require(
                    finding_id
                    in
                    finding_index,
                    (
                        f"{hypothesis_id}: "
                        f"unknown evidence_against "
                        f"{finding_id}."
                    ),
                )

            for experiment_id in (
                hypothesis.get(
                    "tested_by",
                    []
                )
            ):

                require(
                    experiment_id
                    in
                    experiment_index,
                    (
                        f"{hypothesis_id}: "
                        f"unknown experiment "
                        f"{experiment_id}."
                    ),
                )

        # ----------------------------------------------------
        # Findings
        # ----------------------------------------------------

        for finding in (
            findings[
                "records"
            ]
        ):

            finding_id = (
                finding[
                    "finding_id"
                ]
            )

            experiment_id = (
                finding.get(
                    "source_experiment_id"
                )
            )

            require(
                experiment_id
                in
                experiment_index,
                (
                    f"{finding_id}: "
                    f"unknown source experiment "
                    f"{experiment_id}."
                ),
            )

            require(
                finding.get(
                    "evidence_level"
                )
                in
                EVIDENCE_LEVELS,
                (
                    f"{finding_id}: "
                    "invalid evidence level."
                ),
            )

            require(
                isinstance(
                    finding.get(
                        "statement"
                    ),
                    str,
                ),
                (
                    f"{finding_id}: "
                    "statement missing."
                ),
            )

            for hypothesis_id in (
                finding.get(
                    "supports_hypotheses",
                    []
                )
            ):

                require(
                    hypothesis_id
                    in
                    hypothesis_index,
                    (
                        f"{finding_id}: "
                        f"unknown supported "
                        f"hypothesis "
                        f"{hypothesis_id}."
                    ),
                )

            for hypothesis_id in (
                finding.get(
                    "weakens_hypotheses",
                    []
                )
            ):

                require(
                    hypothesis_id
                    in
                    hypothesis_index,
                    (
                        f"{finding_id}: "
                        f"unknown weakened "
                        f"hypothesis "
                        f"{hypothesis_id}."
                    ),
                )

        # ----------------------------------------------------
        # Evidence ledger
        # ----------------------------------------------------

        for evidence in (
            evidence_ledger[
                "records"
            ]
        ):

            evidence_id = (
                evidence[
                    "evidence_record_id"
                ]
            )

            experiment_id = (
                evidence.get(
                    "experiment_id"
                )
            )

            require(
                experiment_id
                in
                experiment_index,
                (
                    f"{evidence_id}: "
                    f"unknown experiment "
                    f"{experiment_id}."
                ),
            )

            require(
                evidence.get(
                    "evidence_level"
                )
                in
                EVIDENCE_LEVELS,
                (
                    f"{evidence_id}: "
                    "invalid evidence level."
                ),
            )

            artifact_path = evidence.get(
                "artifact_path"
            )

            require(
                isinstance(
                    artifact_path,
                    str,
                )
                and
                bool(
                    artifact_path
                ),
                (
                    f"{evidence_id}: "
                    "artifact_path missing."
                ),
            )

            artifact_sha256 = evidence.get(
                "artifact_sha256"
            )

            require(
                isinstance(
                    artifact_sha256,
                    str,
                )
                and
                len(
                    artifact_sha256
                )
                ==
                64,
                (
                    f"{evidence_id}: "
                    "invalid SHA-256."
                ),
            )

            require(
                isinstance(
                    evidence.get(
                        "facts"
                    ),
                    dict,
                ),
                (
                    f"{evidence_id}: "
                    "facts must be an object."
                ),
            )

        # ----------------------------------------------------
        # Experiments
        # ----------------------------------------------------

        for experiment in (
            experiments[
                "records"
            ]
        ):

            experiment_id = (
                experiment[
                    "experiment_id"
                ]
            )

            require(
                experiment.get(
                    "status"
                )
                in
                EXPERIMENT_STATUSES,
                (
                    f"{experiment_id}: "
                    "invalid experiment status."
                ),
            )

            require(
                experiment.get(
                    "evidence_level"
                )
                in
                EVIDENCE_LEVELS,
                (
                    f"{experiment_id}: "
                    "invalid evidence level."
                ),
            )

            for finding_id in (
                experiment.get(
                    "finding_ids",
                    []
                )
            ):

                require(
                    finding_id
                    in
                    finding_index,
                    (
                        f"{experiment_id}: "
                        f"unknown finding "
                        f"{finding_id}."
                    ),
                )

            for hypothesis_id in (
                experiment.get(
                    "hypothesis_ids",
                    []
                )
            ):

                require(
                    hypothesis_id
                    in
                    hypothesis_index,
                    (
                        f"{experiment_id}: "
                        f"unknown hypothesis "
                        f"{hypothesis_id}."
                    ),
                )

            ingestion = experiment.get(
                "artifact_ingestion"
            )

            if ingestion is not None:

                require(
                    ingestion.get(
                        "status"
                    )
                    ==
                    "verified",
                    (
                        f"{experiment_id}: "
                        "artifact ingestion "
                        "is not verified."
                    ),
                )

                evidence_record_id = (
                    ingestion.get(
                        "evidence_record_id"
                    )
                )

                require(
                    evidence_record_id
                    in
                    evidence_index,
                    (
                        f"{experiment_id}: "
                        "unknown evidence record "
                        f"{evidence_record_id}."
                    ),
                )

                require(
                    evidence_index[
                        evidence_record_id
                    ][
                        "experiment_id"
                    ]
                    ==
                    experiment_id,
                    (
                        f"{experiment_id}: "
                        "evidence record belongs "
                        "to a different experiment."
                    ),
                )

        # ----------------------------------------------------
        # Reciprocal hypothesis/finding references
        # ----------------------------------------------------

        for hypothesis in (
            hypotheses[
                "records"
            ]
        ):

            hypothesis_id = (
                hypothesis[
                    "hypothesis_id"
                ]
            )

            for finding_id in (
                hypothesis.get(
                    "evidence_for",
                    []
                )
            ):

                require(
                    hypothesis_id
                    in
                    finding_index[
                        finding_id
                    ].get(
                        "supports_hypotheses",
                        []
                    ),
                    (
                        f"Reference mismatch: "
                        f"{hypothesis_id} / "
                        f"{finding_id}."
                    ),
                )

            for finding_id in (
                hypothesis.get(
                    "evidence_against",
                    []
                )
            ):

                require(
                    hypothesis_id
                    in
                    finding_index[
                        finding_id
                    ].get(
                        "weakens_hypotheses",
                        []
                    ),
                    (
                        f"Reference mismatch: "
                        f"{hypothesis_id} / "
                        f"{finding_id}."
                    ),
                )

        # ----------------------------------------------------
        # Research state
        # ----------------------------------------------------

        state = research_state.get(
            "state"
        )

        require(
            isinstance(
                state,
                dict,
            ),
            (
                "research_state.state "
                "must be an object."
            ),
        )

        for field in (
            "active_hypotheses",
            "weakened_hypotheses",
        ):

            for hypothesis_id in (
                state.get(
                    field,
                    []
                )
            ):

                require(
                    hypothesis_id
                    in
                    hypothesis_index,
                    (
                        f"Research state {field} "
                        "references unknown "
                        f"{hypothesis_id}."
                    ),
                )

        last_experiment = state.get(
            "last_completed_experiment"
        )

        if last_experiment is not None:

            require(
                last_experiment
                in
                experiment_index,
                (
                    "Unknown last completed "
                    f"experiment "
                    f"{last_experiment}."
                ),
            )

        next_experiment = state.get(
            "next_recommended_experiment"
        )

        if next_experiment is not None:

            require(
                next_experiment
                in
                experiment_index,
                (
                    "Unknown next recommended "
                    f"experiment "
                    f"{next_experiment}."
                ),
            )

        for experiment_id in (
            state.get(
                "ingested_experiments",
                []
            )
        ):

            require(
                experiment_id
                in
                experiment_index,
                (
                    "Unknown ingested experiment "
                    f"{experiment_id}."
                ),
            )

        return {
            "schema_version":
                SCHEMA_VERSION,

            "hypotheses":
                len(
                    hypothesis_index
                ),

            "findings":
                len(
                    finding_index
                ),

            "experiments":
                len(
                    experiment_index
                ),

            "evidence_records":
                len(
                    evidence_index
                ),

            "active_hypotheses":
                len(
                    state.get(
                        "active_hypotheses",
                        []
                    )
                ),

            "last_completed_experiment":
                last_experiment,

            "next_recommended_experiment":
                next_experiment,

            "passed":
                True,
        }

    # ========================================================
    # Human-readable summary
    # ========================================================

    def print_summary(
        self,
    ):

        validation = self.validate()

        print()
        print(
            "=" * 72
        )

        print(
            "MATHGPT SCIENTIFIC KNOWLEDGE STORE"
        )

        print(
            "=" * 72
        )

        print()

        print(
            f"Schema: "
            f"{validation['schema_version']}"
        )

        print(
            f"Hypotheses: "
            f"{validation['hypotheses']}"
        )

        print(
            f"Findings: "
            f"{validation['findings']}"
        )

        print(
            f"Experiments: "
            f"{validation['experiments']}"
        )

        print(
            f"Evidence records: "
            f"{validation['evidence_records']}"
        )

        print(
            f"Active hypotheses: "
            f"{validation['active_hypotheses']}"
        )

        print()

        print(
            f"Last completed experiment: "
            f"{validation['last_completed_experiment']}"
        )

        print(
            f"Next recommended experiment: "
            f"{validation['next_recommended_experiment']}"
        )

        print()

        print(
            "SUCCESS: KNOWLEDGE STORE VALID."
        )

        return validation