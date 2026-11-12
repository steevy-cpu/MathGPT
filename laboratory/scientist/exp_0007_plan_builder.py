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
# Imports
# ============================================================

from tokenizer.bpe_tokenizer import (
    BPETokenizer,
)

from laboratory.knowledge.store import (
    KnowledgeStore,
)


# ============================================================
# Identity
# ============================================================

ENGINE_VERSION = "L12B.1"

EXPECTED_PROPOSAL_VERSION = "L12A.1"

EXPERIMENT_ID = "EXP-0007"
TARGET_HYPOTHESIS_ID = "HYP-V03-003"

EXPECTED_PROPOSAL_ID = (
    "PROP-8E7FB86CBDFF9DE6"
)

EXPECTED_MODEL = (
    "MathGPT-General-v0.3"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "4cd9b8f1c68e4079449519fd601b573f8ea48137bc74172c7b37e108c9707cf3"
)

EXPECTED_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "general_pretraining_v0.3"
    / "general_pretraining_v0_3_300m_full"
    / "mathgpt_step_292969.npz"
)


# ============================================================
# Paths
# ============================================================

DEFAULT_PROPOSAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_proposal_exp_0007.json"
)

DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0007.json"
)

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_50m_v0.2"
)

TOKENIZER_CONFIG_PATH = (
    TOKENIZER_DIR
    / "tokenizer_config.json"
)

TOKENIZER_MERGES_PATH = (
    TOKENIZER_DIR
    / "merges.json"
)

TOKENIZER_VOCAB_PATH = (
    TOKENIZER_DIR
    / "vocab.json"
)


# ============================================================
# Frozen experimental design
# ============================================================

PRIMARY_HEADS = [
    "L1H0",
    "L4H0",
]

WEAK_CONTROL_HEADS = [
    "L4H1",
]

ALL_HEADS = (
    PRIMARY_HEADS
    +
    WEAK_CONTROL_HEADS
)

PRIMARY_PAIR_IDS = [
    "P12",
    "P67",
    "P78",
    "P89",
]

POSITIVE_CONTROL_PAIR_ID = "P45"

EXP_0006_PRIMARY_PAIR_IDS = {
    "P23",
    "P34",
    "P45",
    "P56",
}


PAIR_DEFINITIONS = [
    {
        "pair_id":
            "P12",

        "role":
            "primary_replication",

        "lower_label":
            "one",

        "upper_label":
            "two",

        "lower_continuation":
            " one.",

        "upper_continuation":
            " two.",

        "probes": [
            {
                "probe_id":
                    "P12-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Zero plus one equals",
            },
            {
                "probe_id":
                    "P12-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "One plus one equals",
            },
            {
                "probe_id":
                    "P12-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "0 + 1 =",
            },
            {
                "probe_id":
                    "P12-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "1 + 1 =",
            },
        ],
    },
    {
        "pair_id":
            "P67",

        "role":
            "primary_replication",

        "lower_label":
            "six",

        "upper_label":
            "seven",

        "lower_continuation":
            " six.",

        "upper_continuation":
            " seven.",

        "probes": [
            {
                "probe_id":
                    "P67-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Three plus three equals",
            },
            {
                "probe_id":
                    "P67-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Three plus four equals",
            },
            {
                "probe_id":
                    "P67-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "3 + 3 =",
            },
            {
                "probe_id":
                    "P67-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "3 + 4 =",
            },
        ],
    },
    {
        "pair_id":
            "P78",

        "role":
            "primary_replication",

        "lower_label":
            "seven",

        "upper_label":
            "eight",

        "lower_continuation":
            " seven.",

        "upper_continuation":
            " eight.",

        "probes": [
            {
                "probe_id":
                    "P78-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Three plus four equals",
            },
            {
                "probe_id":
                    "P78-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Four plus four equals",
            },
            {
                "probe_id":
                    "P78-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "3 + 4 =",
            },
            {
                "probe_id":
                    "P78-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "4 + 4 =",
            },
        ],
    },
    {
        "pair_id":
            "P89",

        "role":
            "primary_replication",

        "lower_label":
            "eight",

        "upper_label":
            "nine",

        "lower_continuation":
            " eight.",

        "upper_continuation":
            " nine.",

        "probes": [
            {
                "probe_id":
                    "P89-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Four plus four equals",
            },
            {
                "probe_id":
                    "P89-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Four plus five equals",
            },
            {
                "probe_id":
                    "P89-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "4 + 4 =",
            },
            {
                "probe_id":
                    "P89-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "4 + 5 =",
            },
        ],
    },
    {
        "pair_id":
            "P45",

        "role":
            "positive_control",

        "lower_label":
            "four",

        "upper_label":
            "five",

        "lower_continuation":
            " four.",

        "upper_continuation":
            " five.",

        "probes": [
            {
                "probe_id":
                    "P45-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Two plus two equals",
            },
            {
                "probe_id":
                    "P45-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Two plus three equals",
            },
            {
                "probe_id":
                    "P45-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "2 + 2 =",
            },
            {
                "probe_id":
                    "P45-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "2 + 3 =",
            },
        ],
    },
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
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


def index_by(
    records,
    field,
):
    return {
        record[field]:
            record
        for record
        in records
    }


def knowledge_file_hashes(
    store,
):
    result = {}

    for collection in (
        "hypotheses",
        "findings",
        "experiments",
        "research_state",
        "evidence_ledger",
    ):

        path = store.collection_path(
            collection
        )

        result[
            collection
        ] = file_sha256(
            path
        )

    return result


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L12B — freeze the exact "
            "preregistered EXP-0007 replication "
            "plan and verify tokenizer feasibility. "
            "This stage does not load the model or "
            "execute an intervention."
        )
    )

    parser.add_argument(
        "--proposal",
        default=str(
            DEFAULT_PROPOSAL_PATH
        ),
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
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
# Tokenizer
# ============================================================

def load_tokenizer():

    for path in (
        TOKENIZER_CONFIG_PATH,
        TOKENIZER_MERGES_PATH,
        TOKENIZER_VOCAB_PATH,
    ):

        if not path.exists():
            raise FileNotFoundError(path)

    return BPETokenizer.from_files(
        config_path=(
            TOKENIZER_CONFIG_PATH
        ),
        merges_path=(
            TOKENIZER_MERGES_PATH
        ),
        vocab_path=(
            TOKENIZER_VOCAB_PATH
        ),
    )


def token_display(
    tokenizer,
    token_id,
):

    raw_bytes = tokenizer.token_bytes[
        int(
            token_id
        )
    ]

    try:
        text = raw_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError:
        text = None

    return {
        "token_id":
            int(
                token_id
            ),

        "utf8_text":
            text,

        "bytes_hex":
            raw_bytes.hex(),
    }


def first_discriminative_position(
    lower_ids,
    upper_ids,
):

    common_length = min(
        len(
            lower_ids
        ),
        len(
            upper_ids
        ),
    )

    for index in range(
        common_length
    ):

        if (
            lower_ids[
                index
            ]
            !=
            upper_ids[
                index
            ]
        ):

            return index

    raise RuntimeError(
        "Candidate continuations have no "
        "token-level discriminative position "
        "before one continuation ends."
    )


def freeze_pair_tokenization(
    *,
    tokenizer,
    pair,
):

    lower_text = pair[
        "lower_continuation"
    ]

    upper_text = pair[
        "upper_continuation"
    ]

    lower_ids = tokenizer.encode(
        lower_text,
        add_bos=False,
        add_eos=False,
    )

    upper_ids = tokenizer.encode(
        upper_text,
        add_bos=False,
        add_eos=False,
    )

    require(
        bool(
            lower_ids
        ),
        (
            f"{pair['pair_id']}: lower "
            "continuation tokenized empty."
        ),
    )

    require(
        bool(
            upper_ids
        ),
        (
            f"{pair['pair_id']}: upper "
            "continuation tokenized empty."
        ),
    )

    require(
        tokenizer.decode(
            lower_ids
        )
        ==
        lower_text,
        (
            f"{pair['pair_id']}: lower "
            "continuation round-trip failed."
        ),
    )

    require(
        tokenizer.decode(
            upper_ids
        )
        ==
        upper_text,
        (
            f"{pair['pair_id']}: upper "
            "continuation round-trip failed."
        ),
    )

    discriminative_index = (
        first_discriminative_position(
            lower_ids,
            upper_ids,
        )
    )

    lower_token_id = int(
        lower_ids[
            discriminative_index
        ]
    )

    upper_token_id = int(
        upper_ids[
            discriminative_index
        ]
    )

    require(
        lower_token_id
        !=
        upper_token_id,
        (
            f"{pair['pair_id']}: "
            "discriminative token IDs "
            "are unexpectedly equal."
        ),
    )

    return {
        "lower_continuation":
            lower_text,

        "upper_continuation":
            upper_text,

        "lower_token_ids":
            [
                int(
                    token_id
                )
                for token_id
                in lower_ids
            ],

        "upper_token_ids":
            [
                int(
                    token_id
                )
                for token_id
                in upper_ids
            ],

        "first_discriminative_index":
            int(
                discriminative_index
            ),

        "lower_discriminative_token":
            token_display(
                tokenizer,
                lower_token_id,
            ),

        "upper_discriminative_token":
            token_display(
                tokenizer,
                upper_token_id,
            ),

        "scoring_protocol":
            (
                "Prompt and continuation are "
                "encoded separately. The causal "
                "margin is evaluated at the first "
                "continuation-token position where "
                "lower and upper tokenizations "
                "differ."
            ),
    }


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    proposal,
    proposal_path,
    snapshot,
    snapshot_path,
    store,
):

    proposal_record = proposal.get(
        "proposal",
        {},
    )

    # --------------------------------------------------------
    # Compact scientist snapshot view
    #
    # Important:
    # compact_experiment() intentionally omits some
    # registry-only metadata such as required_permission.
    # Snapshot checks therefore validate only fields that
    # the snapshot contract actually exposes.
    # --------------------------------------------------------

    planned_index = index_by(
        snapshot.get(
            "experiments",
            {},
        ).get(
            "planned",
            [],
        ),
        "experiment_id",
    )

    planned_snapshot = planned_index.get(
        EXPERIMENT_ID
    )

    # --------------------------------------------------------
    # Authoritative L7 experiment registry view
    #
    # Permission, proposal metadata, and full experiment
    # identity must be validated against experiment_registry,
    # not against the compact scientist snapshot.
    # --------------------------------------------------------

    experiment_collection = store.load(
        "experiments"
    )

    registry_index = index_by(
        experiment_collection.get(
            "records",
            [],
        ),
        "experiment_id",
    )

    planned_registry = registry_index.get(
        EXPERIMENT_ID
    )

    checks = {
        "proposal_version":
            (
                proposal.get(
                    "engine_version"
                )
                ==
                EXPECTED_PROPOSAL_VERSION
            ),

        "proposal_checks_passed":
            (
                bool(
                    proposal.get(
                        "input_checks"
                    )
                )
                and
                all(
                    proposal[
                        "input_checks"
                    ].values()
                )
            ),

        "proposal_id":
            (
                proposal_record.get(
                    "proposal_id"
                )
                ==
                EXPECTED_PROPOSAL_ID
            ),

        "proposal_experiment":
            (
                proposal_record.get(
                    "experiment_id"
                )
                ==
                EXPERIMENT_ID
            ),

        "proposal_hypothesis":
            (
                proposal_record.get(
                    "target_hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
            ),

        "proposal_execution_blocked":
            (
                proposal_record.get(
                    "execution_authorized"
                )
                is False
            ),

        "proposal_exact_probes_unfrozen":
            (
                proposal_record.get(
                    "design",
                    {},
                ).get(
                    "exact_probes_frozen"
                )
                is False
            ),

        # L12A created the proposal from the pre-registration
        # state and then registered EXP-0007 transactionally.
        # The current knowledge fingerprint should therefore
        # be newer/different from the one embedded in the
        # proposal input record.
        "knowledge_state_advanced_after_registration":
            (
                proposal.get(
                    "inputs",
                    {},
                ).get(
                    "knowledge_state_sha256"
                )
                !=
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        # ----------------------------------------------------
        # Snapshot-visible planned experiment checks
        # ----------------------------------------------------

        "snapshot_exp_0007_planned":
            (
                planned_snapshot
                is not None
                and
                planned_snapshot.get(
                    "status"
                )
                ==
                "planned"
            ),

        "snapshot_hypothesis_link":
            (
                planned_snapshot
                is not None
                and
                TARGET_HYPOTHESIS_ID
                in
                planned_snapshot.get(
                    "hypothesis_ids",
                    []
                )
            ),

        "snapshot_evidence_none":
            (
                planned_snapshot
                is not None
                and
                planned_snapshot.get(
                    "evidence_level"
                )
                is None
            ),

        "snapshot_no_findings":
            (
                planned_snapshot
                is not None
                and
                planned_snapshot.get(
                    "finding_ids"
                )
                ==
                []
            ),

        "snapshot_model_scope":
            (
                planned_snapshot
                is not None
                and
                planned_snapshot.get(
                    "model_scope",
                    {},
                ).get(
                    "model"
                )
                ==
                EXPECTED_MODEL
            ),

        "snapshot_checkpoint_sha":
            (
                planned_snapshot
                is not None
                and
                planned_snapshot.get(
                    "model_scope",
                    {},
                ).get(
                    "checkpoint_sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        # ----------------------------------------------------
        # Authoritative experiment-registry checks
        # ----------------------------------------------------

        "registry_exp_0007_present":
            (
                planned_registry
                is not None
            ),

        "registry_status_planned":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "status"
                )
                ==
                "planned"
            ),

        "registry_hypothesis_link":
            (
                planned_registry
                is not None
                and
                TARGET_HYPOTHESIS_ID
                in
                planned_registry.get(
                    "hypothesis_ids",
                    []
                )
            ),

        "registry_evidence_none":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "evidence_level"
                )
                is None
            ),

        "registry_permission_p3":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "registry_no_findings":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "finding_ids"
                )
                ==
                []
            ),

        "registry_proposal_id_matches":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "proposal",
                    {},
                ).get(
                    "proposal_id"
                )
                ==
                EXPECTED_PROPOSAL_ID
            ),

        "registry_exact_probes_unfrozen":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "proposal",
                    {},
                ).get(
                    "exact_probes_frozen"
                )
                is False
            ),

        "registry_execution_blocked":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "proposal",
                    {},
                ).get(
                    "execution_authorized"
                )
                is False
            ),

        "registry_model_scope":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "model_scope",
                    {},
                ).get(
                    "model"
                )
                ==
                EXPECTED_MODEL
            ),

        "registry_checkpoint_sha":
            (
                planned_registry
                is not None
                and
                planned_registry.get(
                    "model_scope",
                    {},
                ).get(
                    "checkpoint_sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
            ),

        "snapshot_registry_status_agree":
            (
                planned_snapshot
                is not None
                and
                planned_registry
                is not None
                and
                planned_snapshot.get(
                    "status"
                )
                ==
                planned_registry.get(
                    "status"
                )
            ),

        "snapshot_registry_model_scope_agree":
            (
                planned_snapshot
                is not None
                and
                planned_registry
                is not None
                and
                planned_snapshot.get(
                    "model_scope"
                )
                ==
                planned_registry.get(
                    "model_scope"
                )
            ),

        "next_recommended_exp_0007":
            (
                snapshot.get(
                    "research_focus",
                    {},
                ).get(
                    "next_recommended_experiment"
                )
                ==
                EXPERIMENT_ID
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
            "L12B input validation failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        # Use the full registry record downstream whenever
        # metadata not represented in compact snapshots may
        # matter.
        "planned_experiment":
            planned_registry,

        "planned_experiment_snapshot":
            planned_snapshot,

        "proposal_sha256":
            file_sha256(
                proposal_path
            ),

        "snapshot_sha256":
            file_sha256(
                snapshot_path
            ),
    }


# ============================================================
# Frozen probe construction
# ============================================================

def build_frozen_pairs(
    tokenizer,
):

    frozen_pairs = []

    seen_probe_ids = set()

    for pair in PAIR_DEFINITIONS:

        tokenization = (
            freeze_pair_tokenization(
                tokenizer=tokenizer,
                pair=pair,
            )
        )

        frozen_probes = []

        for probe in pair[
            "probes"
        ]:

            probe_id = probe[
                "probe_id"
            ]

            require(
                probe_id
                not in
                seen_probe_ids,
                (
                    "Duplicate probe ID: "
                    f"{probe_id}"
                ),
            )

            seen_probe_ids.add(
                probe_id
            )

            prompt_ids = tokenizer.encode(
                probe[
                    "prompt"
                ],
                add_bos=False,
                add_eos=False,
            )

            require(
                bool(
                    prompt_ids
                ),
                (
                    f"{probe_id}: prompt "
                    "tokenized empty."
                ),
            )

            require(
                tokenizer.decode(
                    prompt_ids
                )
                ==
                probe[
                    "prompt"
                ],
                (
                    f"{probe_id}: prompt "
                    "round-trip failed."
                ),
            )

            frozen_probes.append(
                {
                    **probe,

                    "prompt_token_ids":
                        [
                            int(
                                token_id
                            )
                            for token_id
                            in prompt_ids
                        ],

                    "lower_continuation":
                        pair[
                            "lower_continuation"
                        ],

                    "upper_continuation":
                        pair[
                            "upper_continuation"
                        ],

                    "lower_discriminative_token_id":
                        tokenization[
                            "lower_discriminative_token"
                        ][
                            "token_id"
                        ],

                    "upper_discriminative_token_id":
                        tokenization[
                            "upper_discriminative_token"
                        ][
                            "token_id"
                        ],

                    "first_discriminative_index":
                        tokenization[
                            "first_discriminative_index"
                        ],
                }
            )

        frozen_pairs.append(
            {
                "pair_id":
                    pair[
                        "pair_id"
                    ],

                "role":
                    pair[
                        "role"
                    ],

                "lower_label":
                    pair[
                        "lower_label"
                    ],

                "upper_label":
                    pair[
                        "upper_label"
                    ],

                "tokenization":
                    tokenization,

                "probe_count":
                    len(
                        frozen_probes
                    ),

                "probes":
                    frozen_probes,
            }
        )

    return frozen_pairs


# ============================================================
# Design validation
# ============================================================

def validate_frozen_design(
    frozen_pairs,
):

    pair_index = index_by(
        frozen_pairs,
        "pair_id",
    )

    primary_pairs = [
        pair
        for pair
        in frozen_pairs
        if pair[
            "role"
        ]
        ==
        "primary_replication"
    ]

    positive_controls = [
        pair
        for pair
        in frozen_pairs
        if pair[
            "role"
        ]
        ==
        "positive_control"
    ]

    primary_probes = [
        probe
        for pair
        in primary_pairs
        for probe
        in pair[
            "probes"
        ]
    ]

    all_probes = [
        probe
        for pair
        in frozen_pairs
        for probe
        in pair[
            "probes"
        ]
    ]

    checks = {
        "five_total_pairs":
            (
                len(
                    frozen_pairs
                )
                ==
                5
            ),

        "four_primary_pairs":
            (
                len(
                    primary_pairs
                )
                ==
                4
            ),

        "primary_pair_ids_exact":
            (
                [
                    pair[
                        "pair_id"
                    ]
                    for pair
                    in primary_pairs
                ]
                ==
                PRIMARY_PAIR_IDS
            ),

        "primary_pairs_new":
            all(
                pair[
                    "pair_id"
                ]
                not in
                EXP_0006_PRIMARY_PAIR_IDS
                for pair
                in primary_pairs
            ),

        "one_positive_control":
            (
                len(
                    positive_controls
                )
                ==
                1
                and
                positive_controls[
                    0
                ][
                    "pair_id"
                ]
                ==
                POSITIVE_CONTROL_PAIR_ID
            ),

        "four_probes_per_pair":
            all(
                pair[
                    "probe_count"
                ]
                ==
                4
                for pair
                in frozen_pairs
            ),

        "sixteen_primary_probes":
            (
                len(
                    primary_probes
                )
                ==
                16
            ),

        "twenty_total_probes":
            (
                len(
                    all_probes
                )
                ==
                20
            ),

        "primary_correctness_balanced":
            all(
                sum(
                    probe[
                        "correct_candidate"
                    ]
                    ==
                    "lower"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                and
                sum(
                    probe[
                        "correct_candidate"
                    ]
                    ==
                    "upper"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                for pair
                in primary_pairs
            ),

        "primary_form_balanced":
            all(
                sum(
                    probe[
                        "form"
                    ]
                    ==
                    "word"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                and
                sum(
                    probe[
                        "form"
                    ]
                    ==
                    "symbol"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                for pair
                in primary_pairs
            ),

        "all_discriminative_tokens_distinct":
            all(
                pair[
                    "tokenization"
                ][
                    "lower_discriminative_token"
                ][
                    "token_id"
                ]
                !=
                pair[
                    "tokenization"
                ][
                    "upper_discriminative_token"
                ][
                    "token_id"
                ]
                for pair
                in frozen_pairs
            ),

        "all_candidate_roundtrips_verified":
            all(
                bool(
                    pair[
                        "tokenization"
                    ][
                        "lower_token_ids"
                    ]
                )
                and
                bool(
                    pair[
                        "tokenization"
                    ][
                        "upper_token_ids"
                    ]
                )
                for pair
                in frozen_pairs
            ),

        "primary_heads_exact":
            (
                PRIMARY_HEADS
                ==
                [
                    "L1H0",
                    "L4H0",
                ]
            ),

        "weak_control_exact":
            (
                WEAK_CONTROL_HEADS
                ==
                [
                    "L4H1",
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
            "L12B frozen design failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "pair_index":
            pair_index,

        "primary_probe_count":
            len(
                primary_probes
            ),

        "total_probe_count":
            len(
                all_probes
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    proposal_path = project_path(
        args.proposal
    )

    snapshot_path = project_path(
        args.snapshot
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
            "EXP-0007 frozen plan already "
            f"exists: {output_path}\n"
            "Use --overwrite only before an "
            "execution spec or approval is "
            "created and only if the design "
            "must intentionally change."
        )

    print()
    print("=" * 86)
    print(
        "STAGE L12B — EXP-0007 "
        "PREREGISTERED PLAN BUILDER"
    )
    print("=" * 86)

    proposal = load_json(
        proposal_path
    )

    snapshot = load_json(
        snapshot_path
    )

    store = KnowledgeStore()

    knowledge_before = (
        knowledge_file_hashes(
            store
        )
    )

    input_validation = (
        validate_inputs(
            proposal=proposal,
            proposal_path=(
                proposal_path
            ),
            snapshot=snapshot,
            snapshot_path=(
                snapshot_path
            ),
            store=store,
        )
    )

    print()
    print(
        "Input checks"
    )
    print(
        "------------"
    )

    for name, passed in (
        input_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    print()
    print(
        "Loading tokenizer only..."
    )

    tokenizer = load_tokenizer()

    tokenizer_hashes = {
        "config_sha256":
            file_sha256(
                TOKENIZER_CONFIG_PATH
            ),

        "merges_sha256":
            file_sha256(
                TOKENIZER_MERGES_PATH
            ),

        "vocab_sha256":
            file_sha256(
                TOKENIZER_VOCAB_PATH
            ),
    }

    print(
        "Tokenizer loaded:"
    )
    print(
        "  tokenizer_50m_v0.2"
    )

    # --------------------------------------------------------
    # Bind exact checkpoint bytes without loading model.
    # --------------------------------------------------------

    require(
        EXPECTED_CHECKPOINT_PATH.exists(),
        (
            "Mature v0.3 checkpoint is "
            "missing."
        ),
    )

    print()
    print(
        "Verifying mature-v0.3 checkpoint "
        "SHA-256..."
    )

    actual_checkpoint_sha = (
        file_sha256(
            EXPECTED_CHECKPOINT_PATH
        )
    )

    require(
        actual_checkpoint_sha
        ==
        EXPECTED_CHECKPOINT_SHA256,
        (
            "Mature v0.3 checkpoint SHA "
            "does not match frozen identity."
        ),
    )

    frozen_pairs = build_frozen_pairs(
        tokenizer
    )

    design_validation = (
        validate_frozen_design(
            frozen_pairs
        )
    )

    print()
    print(
        "Frozen candidate pairs"
    )
    print(
        "----------------------"
    )

    for pair in frozen_pairs:

        tokenization = pair[
            "tokenization"
        ]

        print()
        print(
            f"{pair['pair_id']} "
            f"[{pair['role']}]"
        )

        print(
            f"  {pair['lower_label']} vs "
            f"{pair['upper_label']}"
        )

        print(
            "  lower continuation IDs: "
            f"{tokenization['lower_token_ids']}"
        )

        print(
            "  upper continuation IDs: "
            f"{tokenization['upper_token_ids']}"
        )

        print(
            "  discriminative index: "
            f"{tokenization['first_discriminative_index']}"
        )

        print(
            "  discriminative token IDs: "
            f"{tokenization['lower_discriminative_token']['token_id']} "
            "vs "
            f"{tokenization['upper_discriminative_token']['token_id']}"
        )

    print()
    print(
        "Frozen design checks"
    )
    print(
        "--------------------"
    )

    for name, passed in (
        design_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    # ========================================================
    # Frozen preregistered analysis policy
    # ========================================================

    analysis_policy = {
        "canonical_margin":
            (
                "logit(lower discriminative token) "
                "- logit(upper discriminative token)"
            ),

        "ablation_delta":
            (
                "ablated_margin - baseline_margin"
            ),

        "raw_direction": {
            "negative_delta":
                (
                    "intact head supports LOWER "
                    "relative to UPPER"
                ),

            "positive_delta":
                (
                    "intact head supports UPPER "
                    "relative to LOWER"
                ),
        },

        "correctness_effect": {
            "lower_correct":
                "-ablation_delta",

            "upper_correct":
                "+ablation_delta",

            "positive_effect":
                (
                    "intact head supports the "
                    "correct candidate"
                ),

            "negative_effect":
                (
                    "intact head suppresses the "
                    "correct candidate"
                ),
        },

        "primary_replication_tests": {
            "within_pair_direction_stability":
                {
                    "unit":
                        "head × primary pair",

                    "pass_rule":
                        (
                            "At least 3 of 4 raw "
                            "ablation deltas have "
                            "the same non-zero sign."
                        ),

                    "threshold":
                        "3/4",
                },

            "cross_pair_generalization":
                {
                    "unit":
                        "primary head",

                    "pass_rule":
                        (
                            "Within-pair direction "
                            "stability passes on at "
                            "least 3 of the 4 new "
                            "primary pairs."
                        ),

                    "threshold":
                        "3/4 pairs",
                },

            "strong_correctness_tracking":
                {
                    "unit":
                        "primary head",

                    "pass_rule":
                        (
                            "At least 12 of 16 "
                            "primary correctness "
                            "effects are positive."
                        ),

                    "threshold":
                        "12/16",

                    "interpretation":
                        (
                            "Passing this test is "
                            "evidence for a strong "
                            "arithmetic-correctness "
                            "account and therefore "
                            "works against the "
                            "candidate-selection "
                            "replication claim."
                        ),
                },
        },

        "positive_control_tests": {
            "L1H0":
                {
                    "expected_raw_delta_sign":
                        "negative",

                    "threshold":
                        ">= 3/4 P45 probes",
                },

            "L4H0":
                {
                    "expected_raw_delta_sign":
                        "positive",

                    "threshold":
                        ">= 3/4 P45 probes",
                },

            "overall_pass":
                (
                    "Both L1H0 and L4H0 "
                    "positive-control tests pass."
                ),
        },

        "weak_control_test": {
            "head":
                "L4H1",

            "metric":
                (
                    "mean absolute raw ablation "
                    "delta across the 16 primary "
                    "replication probes"
                ),

            "pass_rule":
                (
                    "L4H1 mean absolute delta is "
                    "strictly lower than the mean "
                    "absolute delta of each primary "
                    "head on the same probes."
                ),
        },

        "orientation_diagnostic": {
            "purpose":
                (
                    "Distinguish pair/token-specific "
                    "stability from a trivial global "
                    "LOWER-versus-UPPER ordering bias."
                ),

            "metric":
                (
                    "For each primary head, compare "
                    "the dominant raw-delta sign "
                    "across the stable primary pairs."
                ),

            "global_order_bias_signal":
                (
                    "The same dominant raw-delta "
                    "sign occurs in every stable "
                    "primary pair."
                ),

            "pair_dependent_signal":
                (
                    "At least two stable primary "
                    "pairs have opposite dominant "
                    "raw-delta signs."
                ),

            "note":
                (
                    "This diagnostic does not alter "
                    "the preregistered primary "
                    "replication pass/fail result; "
                    "it refines mechanism "
                    "interpretation."
                ),
        },

        "outcome_policy": [
            {
                "outcome":
                    (
                        "inconclusive_positive_"
                        "control_failure"
                    ),

                "condition":
                    (
                        "P45 overall positive "
                        "control fails."
                    ),
            },
            {
                "outcome":
                    (
                        "replicates_cross_pair_"
                        "candidate_selection_bias"
                    ),

                "condition":
                    (
                        "Positive control passes; "
                        "both L1H0 and L4H0 pass "
                        "cross-pair generalization; "
                        "neither primary head passes "
                        "strong correctness tracking; "
                        "and the L4H1 weak-control "
                        "criterion passes."
                    ),
            },
            {
                "outcome":
                    "partial_replication",

                "condition":
                    (
                        "Positive control passes and "
                        "exactly one of L1H0/L4H0 "
                        "passes cross-pair "
                        "generalization, or both pass "
                        "but one also passes strong "
                        "correctness tracking, with "
                        "weak-control behavior still "
                        "acceptable."
                    ),
            },
            {
                "outcome":
                    (
                        "fails_cross_pair_"
                        "replication"
                    ),

                "condition":
                    (
                        "Positive control passes but "
                        "neither primary head passes "
                        "cross-pair generalization, "
                        "or the weak-control "
                        "criterion fails."
                    ),
            },
        ],
    }

    intervention_policy = {
        "instrument":
            "attention_head_ablation",

        "intervention":
            (
                "Temporarily zero the selected "
                "attention head output for one "
                "forward pass."
            ),

        "ablation_scale":
            0.0,

        "persistent_weight_modification":
            False,

        "primary_heads":
            PRIMARY_HEADS,

        "weak_control_heads":
            WEAK_CONTROL_HEADS,

        "all_heads":
            ALL_HEADS,

        "effects_per_probe":
            len(
                ALL_HEADS
            ),

        "expected_total_probe_count":
            20,

        "expected_total_causal_effects":
            (
                20
                *
                len(
                    ALL_HEADS
                )
            ),

        "required_permission":
            "P3",

        "human_approval_required":
            True,

        "execution_authorized":
            False,
    }

    plan_core = {
        "experiment_id":
            EXPERIMENT_ID,

        "target_hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "title":
            (
                "Independent cross-pair numerical "
                "candidate-selection replication"
            ),

        "question":
            (
                "Across four previously untested "
                "numerical candidate pairs, do "
                "L1H0 and L4H0 preserve "
                "context-stable causal pair/token "
                "directions more reliably than "
                "they track arithmetic correctness?"
            ),

        "proposal": {
            "proposal_id":
                EXPECTED_PROPOSAL_ID,

            "proposal_path":
                relative_project_path(
                    proposal_path
                ),

            "proposal_sha256":
                input_validation[
                    "proposal_sha256"
                ],
        },

        "knowledge_state": {
            "knowledge_state_sha256":
                snapshot[
                    "knowledge_state_sha256"
                ],

            "scientist_context_path":
                relative_project_path(
                    snapshot_path
                ),

            "scientist_context_sha256":
                input_validation[
                    "snapshot_sha256"
                ],
        },

        "model_scope":
            input_validation[
                "planned_experiment"
            ][
                "model_scope"
            ],

        "checkpoint_binding": {
            "path":
                relative_project_path(
                    EXPECTED_CHECKPOINT_PATH
                ),

            "sha256":
                actual_checkpoint_sha,
        },

        "tokenizer_binding": {
            "artifact":
                "tokenizer_50m_v0.2",

            "directory":
                relative_project_path(
                    TOKENIZER_DIR
                ),

            **tokenizer_hashes,

            "encoding_protocol":
                (
                    "Prompt and candidate "
                    "continuation are encoded "
                    "separately without BOS/EOS."
                ),
        },

        "frozen_heads":
            {
                "primary":
                    PRIMARY_HEADS,

                "weak_control":
                    WEAK_CONTROL_HEADS,
            },

        "frozen_pairs":
            frozen_pairs,

        "primary_pair_ids":
            PRIMARY_PAIR_IDS,

        "positive_control_pair_id":
            POSITIVE_CONTROL_PAIR_ID,

        "primary_probe_count":
            design_validation[
                "primary_probe_count"
            ],

        "total_probe_count":
            design_validation[
                "total_probe_count"
            ],

        "analysis_policy":
            analysis_policy,

        "intervention_policy":
            intervention_policy,

        "design_constraints": [
            (
                "P12/P67/P78/P89 are the only "
                "primary replication pairs."
            ),
            (
                "P45 is a positive control only "
                "and cannot contribute to primary "
                "replication success."
            ),
            (
                "Each primary pair has exactly "
                "two lower-correct and two "
                "upper-correct probes."
            ),
            (
                "Each primary pair has exactly "
                "two word-form and two symbol-form "
                "probes."
            ),
            (
                "L1H0 and L4H0 are the only "
                "primary heads."
            ),
            (
                "L4H1 is the only weak-control "
                "head."
            ),
            (
                "L3H3 is excluded because it is "
                "reserved for HYP-V03-004."
            ),
            (
                "No prompt, candidate pair, head, "
                "token ID, threshold, or outcome "
                "rule may change after this plan "
                "is frozen without creating a new "
                "plan identity."
            ),
        ],

        "execution_authorized":
            False,

        "p3_approval_present":
            False,
    }

    plan_id = (
        "PLAN-"
        +
        canonical_json_sha256(
            plan_core
        )[
            :16
        ].upper()
    )

    output = {
        "engine_version":
            ENGINE_VERSION,

        "plan_version":
            "L12B.1",

        "mode":
            (
                "frozen_preregistered_"
                "replication_plan"
            ),

        "plan_id":
            plan_id,

        "plan":
            {
                "plan_id":
                    plan_id,

                **plan_core,
            },

        "input_checks":
            input_validation[
                "checks"
            ],

        "design_checks":
            design_validation[
                "checks"
            ],

        "policy_checks": {
            "model_not_loaded":
                True,

            "experiment_not_executed":
                True,

            "knowledge_store_not_mutated":
                True,

            "evidence_not_created":
                True,

            "findings_not_created":
                True,

            "hypothesis_status_not_mutated":
                True,

            "execution_not_authorized":
                True,

            "p3_approval_not_created":
                True,
        },
    }

    atomic_write_json(
        output_path,
        output,
    )

    knowledge_after = (
        knowledge_file_hashes(
            store
        )
    )

    require(
        knowledge_before
        ==
        knowledge_after,
        (
            "L12B unexpectedly mutated the "
            "knowledge store."
        ),
    )

    saved = load_json(
        output_path
    )

    require(
        saved.get(
            "plan_id"
        )
        ==
        plan_id,
        (
            "Saved EXP-0007 plan identity "
            "mismatch."
        ),
    )

    plan_sha256 = file_sha256(
        output_path
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print("=" * 86)
    print(
        "L12B EXP-0007 FROZEN PLAN SUMMARY"
    )
    print("=" * 86)

    print()
    print(
        "Plan ID:"
    )
    print(
        f"  {plan_id}"
    )

    print()
    print(
        "Experiment:"
    )
    print(
        "  EXP-0007"
    )

    print(
        "Hypothesis:"
    )
    print(
        "  HYP-V03-003"
    )

    print()
    print(
        "Primary pairs:"
    )

    for pair_id in (
        PRIMARY_PAIR_IDS
    ):

        print(
            f"  {pair_id}"
        )

    print()
    print(
        "Positive control:"
    )
    print(
        "  P45"
    )

    print()
    print(
        "Primary heads:"
    )
    print(
        "  L1H0"
    )
    print(
        "  L4H0"
    )

    print()
    print(
        "Weak control:"
    )
    print(
        "  L4H1"
    )

    print()
    print(
        "Primary probes:"
    )
    print(
        "  16"
    )

    print(
        "Total probes:"
    )
    print(
        "  20"
    )

    print(
        "Causal effects:"
    )
    print(
        f"  {intervention_policy['expected_total_causal_effects']}"
    )

    print()
    print(
        "Checkpoint SHA-256:"
    )
    print(
        "  "
        +
        actual_checkpoint_sha
    )

    print()
    print(
        "Plan artifact:"
    )
    print(
        output_path
    )

    print()
    print(
        "Plan SHA-256:"
    )
    print(
        plan_sha256
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
            f"{name:<42} "
            f"{passed}"
        )

    print()
    print("=" * 86)
    print(
        "SUCCESS: STAGE L12B EXP-0007 "
        "PREREGISTERED PLAN FROZEN."
    )
    print(
        "No model was loaded, no evidence was "
        "created, and P3 execution remains blocked."
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
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
# Imports
# ============================================================

from tokenizer.bpe_tokenizer import (
    BPETokenizer,
)

from laboratory.knowledge.store import (
    KnowledgeStore,
)


# ============================================================
# Identity
# ============================================================

ENGINE_VERSION = "L12B.1"

EXPECTED_PROPOSAL_VERSION = "L12A.1"

EXPERIMENT_ID = "EXP-0007"
TARGET_HYPOTHESIS_ID = "HYP-V03-003"

EXPECTED_PROPOSAL_ID = (
    "PROP-8E7FB86CBDFF9DE6"
)

EXPECTED_MODEL = (
    "MathGPT-General-v0.3"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "4cd9b8f1c68e4079449519fd601b573f8ea48137bc74172c7b37e108c9707cf3"
)

EXPECTED_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "general_pretraining_v0.3"
    / "general_pretraining_v0_3_300m_full"
    / "mathgpt_step_292969.npz"
)


# ============================================================
# Paths
# ============================================================

DEFAULT_PROPOSAL_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_proposal_exp_0007.json"
)

DEFAULT_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "knowledge"
    / "scientist_context.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "laboratory"
    / "scientist"
    / "state"
    / "experiment_plan_exp_0007.json"
)

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_50m_v0.2"
)

TOKENIZER_CONFIG_PATH = (
    TOKENIZER_DIR
    / "tokenizer_config.json"
)

TOKENIZER_MERGES_PATH = (
    TOKENIZER_DIR
    / "merges.json"
)

TOKENIZER_VOCAB_PATH = (
    TOKENIZER_DIR
    / "vocab.json"
)


# ============================================================
# Frozen experimental design
# ============================================================

PRIMARY_HEADS = [
    "L1H0",
    "L4H0",
]

WEAK_CONTROL_HEADS = [
    "L4H1",
]

ALL_HEADS = (
    PRIMARY_HEADS
    +
    WEAK_CONTROL_HEADS
)

PRIMARY_PAIR_IDS = [
    "P12",
    "P67",
    "P78",
    "P89",
]

POSITIVE_CONTROL_PAIR_ID = "P45"

EXP_0006_PRIMARY_PAIR_IDS = {
    "P23",
    "P34",
    "P45",
    "P56",
}


PAIR_DEFINITIONS = [
    {
        "pair_id":
            "P12",

        "role":
            "primary_replication",

        "lower_label":
            "one",

        "upper_label":
            "two",

        "lower_continuation":
            " one.",

        "upper_continuation":
            " two.",

        "probes": [
            {
                "probe_id":
                    "P12-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Zero plus one equals",
            },
            {
                "probe_id":
                    "P12-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "One plus one equals",
            },
            {
                "probe_id":
                    "P12-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "0 + 1 =",
            },
            {
                "probe_id":
                    "P12-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "1 + 1 =",
            },
        ],
    },
    {
        "pair_id":
            "P67",

        "role":
            "primary_replication",

        "lower_label":
            "six",

        "upper_label":
            "seven",

        "lower_continuation":
            " six.",

        "upper_continuation":
            " seven.",

        "probes": [
            {
                "probe_id":
                    "P67-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Three plus three equals",
            },
            {
                "probe_id":
                    "P67-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Three plus four equals",
            },
            {
                "probe_id":
                    "P67-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "3 + 3 =",
            },
            {
                "probe_id":
                    "P67-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "3 + 4 =",
            },
        ],
    },
    {
        "pair_id":
            "P78",

        "role":
            "primary_replication",

        "lower_label":
            "seven",

        "upper_label":
            "eight",

        "lower_continuation":
            " seven.",

        "upper_continuation":
            " eight.",

        "probes": [
            {
                "probe_id":
                    "P78-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Three plus four equals",
            },
            {
                "probe_id":
                    "P78-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Four plus four equals",
            },
            {
                "probe_id":
                    "P78-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "3 + 4 =",
            },
            {
                "probe_id":
                    "P78-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "4 + 4 =",
            },
        ],
    },
    {
        "pair_id":
            "P89",

        "role":
            "primary_replication",

        "lower_label":
            "eight",

        "upper_label":
            "nine",

        "lower_continuation":
            " eight.",

        "upper_continuation":
            " nine.",

        "probes": [
            {
                "probe_id":
                    "P89-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Four plus four equals",
            },
            {
                "probe_id":
                    "P89-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Four plus five equals",
            },
            {
                "probe_id":
                    "P89-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "4 + 4 =",
            },
            {
                "probe_id":
                    "P89-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "4 + 5 =",
            },
        ],
    },
    {
        "pair_id":
            "P45",

        "role":
            "positive_control",

        "lower_label":
            "four",

        "upper_label":
            "five",

        "lower_continuation":
            " four.",

        "upper_continuation":
            " five.",

        "probes": [
            {
                "probe_id":
                    "P45-W-L",

                "form":
                    "word",

                "correct_candidate":
                    "lower",

                "prompt":
                    "Two plus two equals",
            },
            {
                "probe_id":
                    "P45-W-U",

                "form":
                    "word",

                "correct_candidate":
                    "upper",

                "prompt":
                    "Two plus three equals",
            },
            {
                "probe_id":
                    "P45-S-L",

                "form":
                    "symbol",

                "correct_candidate":
                    "lower",

                "prompt":
                    "2 + 2 =",
            },
            {
                "probe_id":
                    "P45-S-U",

                "form":
                    "symbol",

                "correct_candidate":
                    "upper",

                "prompt":
                    "2 + 3 =",
            },
        ],
    },
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
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


def index_by(
    records,
    field,
):
    return {
        record[field]:
            record
        for record
        in records
    }


def knowledge_file_hashes(
    store,
):
    result = {}

    for collection in (
        "hypotheses",
        "findings",
        "experiments",
        "research_state",
        "evidence_ledger",
    ):

        path = store.collection_path(
            collection
        )

        result[
            collection
        ] = file_sha256(
            path
        )

    return result


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Stage L12B — freeze the exact "
            "preregistered EXP-0007 replication "
            "plan and verify tokenizer feasibility. "
            "This stage does not load the model or "
            "execute an intervention."
        )
    )

    parser.add_argument(
        "--proposal",
        default=str(
            DEFAULT_PROPOSAL_PATH
        ),
    )

    parser.add_argument(
        "--snapshot",
        default=str(
            DEFAULT_SNAPSHOT_PATH
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
# Tokenizer
# ============================================================

def load_tokenizer():

    for path in (
        TOKENIZER_CONFIG_PATH,
        TOKENIZER_MERGES_PATH,
        TOKENIZER_VOCAB_PATH,
    ):

        if not path.exists():
            raise FileNotFoundError(path)

    return BPETokenizer.from_files(
        config_path=(
            TOKENIZER_CONFIG_PATH
        ),
        merges_path=(
            TOKENIZER_MERGES_PATH
        ),
        vocab_path=(
            TOKENIZER_VOCAB_PATH
        ),
    )


def token_display(
    tokenizer,
    token_id,
):

    raw_bytes = tokenizer.token_bytes[
        int(
            token_id
        )
    ]

    try:
        text = raw_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError:
        text = None

    return {
        "token_id":
            int(
                token_id
            ),

        "utf8_text":
            text,

        "bytes_hex":
            raw_bytes.hex(),
    }


def first_discriminative_position(
    lower_ids,
    upper_ids,
):

    common_length = min(
        len(
            lower_ids
        ),
        len(
            upper_ids
        ),
    )

    for index in range(
        common_length
    ):

        if (
            lower_ids[
                index
            ]
            !=
            upper_ids[
                index
            ]
        ):

            return index

    raise RuntimeError(
        "Candidate continuations have no "
        "token-level discriminative position "
        "before one continuation ends."
    )


def freeze_pair_tokenization(
    *,
    tokenizer,
    pair,
):

    lower_text = pair[
        "lower_continuation"
    ]

    upper_text = pair[
        "upper_continuation"
    ]

    lower_ids = tokenizer.encode(
        lower_text,
        add_bos=False,
        add_eos=False,
    )

    upper_ids = tokenizer.encode(
        upper_text,
        add_bos=False,
        add_eos=False,
    )

    require(
        bool(
            lower_ids
        ),
        (
            f"{pair['pair_id']}: lower "
            "continuation tokenized empty."
        ),
    )

    require(
        bool(
            upper_ids
        ),
        (
            f"{pair['pair_id']}: upper "
            "continuation tokenized empty."
        ),
    )

    require(
        tokenizer.decode(
            lower_ids
        )
        ==
        lower_text,
        (
            f"{pair['pair_id']}: lower "
            "continuation round-trip failed."
        ),
    )

    require(
        tokenizer.decode(
            upper_ids
        )
        ==
        upper_text,
        (
            f"{pair['pair_id']}: upper "
            "continuation round-trip failed."
        ),
    )

    discriminative_index = (
        first_discriminative_position(
            lower_ids,
            upper_ids,
        )
    )

    lower_token_id = int(
        lower_ids[
            discriminative_index
        ]
    )

    upper_token_id = int(
        upper_ids[
            discriminative_index
        ]
    )

    require(
        lower_token_id
        !=
        upper_token_id,
        (
            f"{pair['pair_id']}: "
            "discriminative token IDs "
            "are unexpectedly equal."
        ),
    )

    return {
        "lower_continuation":
            lower_text,

        "upper_continuation":
            upper_text,

        "lower_token_ids":
            [
                int(
                    token_id
                )
                for token_id
                in lower_ids
            ],

        "upper_token_ids":
            [
                int(
                    token_id
                )
                for token_id
                in upper_ids
            ],

        "first_discriminative_index":
            int(
                discriminative_index
            ),

        "lower_discriminative_token":
            token_display(
                tokenizer,
                lower_token_id,
            ),

        "upper_discriminative_token":
            token_display(
                tokenizer,
                upper_token_id,
            ),

        "scoring_protocol":
            (
                "Prompt and continuation are "
                "encoded separately. The causal "
                "margin is evaluated at the first "
                "continuation-token position where "
                "lower and upper tokenizations "
                "differ."
            ),
    }


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    *,
    proposal,
    proposal_path,
    snapshot,
    snapshot_path,
):

    proposal_record = proposal.get(
        "proposal",
        {},
    )

    planned_index = index_by(
        snapshot.get(
            "experiments",
            {},
        ).get(
            "planned",
            [],
        ),
        "experiment_id",
    )

    planned = planned_index.get(
        EXPERIMENT_ID
    )

    checks = {
        "proposal_version":
            (
                proposal.get(
                    "engine_version"
                )
                ==
                EXPECTED_PROPOSAL_VERSION
            ),

        "proposal_checks_passed":
            (
                bool(
                    proposal.get(
                        "input_checks"
                    )
                )
                and
                all(
                    proposal[
                        "input_checks"
                    ].values()
                )
            ),

        "proposal_id":
            (
                proposal_record.get(
                    "proposal_id"
                )
                ==
                EXPECTED_PROPOSAL_ID
            ),

        "proposal_experiment":
            (
                proposal_record.get(
                    "experiment_id"
                )
                ==
                EXPERIMENT_ID
            ),

        "proposal_hypothesis":
            (
                proposal_record.get(
                    "target_hypothesis_id"
                )
                ==
                TARGET_HYPOTHESIS_ID
            ),

        "proposal_execution_blocked":
            (
                proposal_record.get(
                    "execution_authorized"
                )
                is False
            ),

        "proposal_exact_probes_unfrozen":
            (
                proposal_record.get(
                    "design",
                    {},
                ).get(
                    "exact_probes_frozen"
                )
                is False
            ),

        "knowledge_state_matches_proposal":
            (
                proposal.get(
                    "inputs",
                    {},
                ).get(
                    "knowledge_state_sha256"
                )
                !=
                snapshot.get(
                    "knowledge_state_sha256"
                )
            ),

        # The knowledge fingerprint MUST differ:
        # L12A wrote the proposal from the old
        # state and then registered EXP-0007.
        "exp_0007_planned":
            (
                planned
                is not None
                and
                planned.get(
                    "status"
                )
                ==
                "planned"
            ),

        "planned_hypothesis_link":
            (
                planned
                is not None
                and
                TARGET_HYPOTHESIS_ID
                in
                planned.get(
                    "hypothesis_ids",
                    []
                )
            ),

        "planned_evidence_none":
            (
                planned
                is not None
                and
                planned.get(
                    "evidence_level"
                )
                is None
            ),

        "planned_permission_p3":
            (
                planned
                is not None
                and
                planned.get(
                    "required_permission"
                )
                ==
                "P3"
            ),

        "planned_no_findings":
            (
                planned
                is not None
                and
                planned.get(
                    "finding_ids"
                )
                ==
                []
            ),

        "next_recommended_exp_0007":
            (
                snapshot.get(
                    "research_focus",
                    {},
                ).get(
                    "next_recommended_experiment"
                )
                ==
                EXPERIMENT_ID
            ),

        "model_scope":
            (
                planned
                is not None
                and
                planned.get(
                    "model_scope",
                    {},
                ).get(
                    "model"
                )
                ==
                EXPECTED_MODEL
            ),

        "checkpoint_sha":
            (
                planned
                is not None
                and
                planned.get(
                    "model_scope",
                    {},
                ).get(
                    "checkpoint_sha256"
                )
                ==
                EXPECTED_CHECKPOINT_SHA256
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
            "L12B input validation failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "planned_experiment":
            planned,

        "proposal_sha256":
            file_sha256(
                proposal_path
            ),

        "snapshot_sha256":
            file_sha256(
                snapshot_path
            ),
    }


# ============================================================
# Frozen probe construction
# ============================================================

def build_frozen_pairs(
    tokenizer,
):

    frozen_pairs = []

    seen_probe_ids = set()

    for pair in PAIR_DEFINITIONS:

        tokenization = (
            freeze_pair_tokenization(
                tokenizer=tokenizer,
                pair=pair,
            )
        )

        frozen_probes = []

        for probe in pair[
            "probes"
        ]:

            probe_id = probe[
                "probe_id"
            ]

            require(
                probe_id
                not in
                seen_probe_ids,
                (
                    "Duplicate probe ID: "
                    f"{probe_id}"
                ),
            )

            seen_probe_ids.add(
                probe_id
            )

            prompt_ids = tokenizer.encode(
                probe[
                    "prompt"
                ],
                add_bos=False,
                add_eos=False,
            )

            require(
                bool(
                    prompt_ids
                ),
                (
                    f"{probe_id}: prompt "
                    "tokenized empty."
                ),
            )

            require(
                tokenizer.decode(
                    prompt_ids
                )
                ==
                probe[
                    "prompt"
                ],
                (
                    f"{probe_id}: prompt "
                    "round-trip failed."
                ),
            )

            frozen_probes.append(
                {
                    **probe,

                    "prompt_token_ids":
                        [
                            int(
                                token_id
                            )
                            for token_id
                            in prompt_ids
                        ],

                    "lower_continuation":
                        pair[
                            "lower_continuation"
                        ],

                    "upper_continuation":
                        pair[
                            "upper_continuation"
                        ],

                    "lower_discriminative_token_id":
                        tokenization[
                            "lower_discriminative_token"
                        ][
                            "token_id"
                        ],

                    "upper_discriminative_token_id":
                        tokenization[
                            "upper_discriminative_token"
                        ][
                            "token_id"
                        ],

                    "first_discriminative_index":
                        tokenization[
                            "first_discriminative_index"
                        ],
                }
            )

        frozen_pairs.append(
            {
                "pair_id":
                    pair[
                        "pair_id"
                    ],

                "role":
                    pair[
                        "role"
                    ],

                "lower_label":
                    pair[
                        "lower_label"
                    ],

                "upper_label":
                    pair[
                        "upper_label"
                    ],

                "tokenization":
                    tokenization,

                "probe_count":
                    len(
                        frozen_probes
                    ),

                "probes":
                    frozen_probes,
            }
        )

    return frozen_pairs


# ============================================================
# Design validation
# ============================================================

def validate_frozen_design(
    frozen_pairs,
):

    pair_index = index_by(
        frozen_pairs,
        "pair_id",
    )

    primary_pairs = [
        pair
        for pair
        in frozen_pairs
        if pair[
            "role"
        ]
        ==
        "primary_replication"
    ]

    positive_controls = [
        pair
        for pair
        in frozen_pairs
        if pair[
            "role"
        ]
        ==
        "positive_control"
    ]

    primary_probes = [
        probe
        for pair
        in primary_pairs
        for probe
        in pair[
            "probes"
        ]
    ]

    all_probes = [
        probe
        for pair
        in frozen_pairs
        for probe
        in pair[
            "probes"
        ]
    ]

    checks = {
        "five_total_pairs":
            (
                len(
                    frozen_pairs
                )
                ==
                5
            ),

        "four_primary_pairs":
            (
                len(
                    primary_pairs
                )
                ==
                4
            ),

        "primary_pair_ids_exact":
            (
                [
                    pair[
                        "pair_id"
                    ]
                    for pair
                    in primary_pairs
                ]
                ==
                PRIMARY_PAIR_IDS
            ),

        "primary_pairs_new":
            all(
                pair[
                    "pair_id"
                ]
                not in
                EXP_0006_PRIMARY_PAIR_IDS
                for pair
                in primary_pairs
            ),

        "one_positive_control":
            (
                len(
                    positive_controls
                )
                ==
                1
                and
                positive_controls[
                    0
                ][
                    "pair_id"
                ]
                ==
                POSITIVE_CONTROL_PAIR_ID
            ),

        "four_probes_per_pair":
            all(
                pair[
                    "probe_count"
                ]
                ==
                4
                for pair
                in frozen_pairs
            ),

        "sixteen_primary_probes":
            (
                len(
                    primary_probes
                )
                ==
                16
            ),

        "twenty_total_probes":
            (
                len(
                    all_probes
                )
                ==
                20
            ),

        "primary_correctness_balanced":
            all(
                sum(
                    probe[
                        "correct_candidate"
                    ]
                    ==
                    "lower"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                and
                sum(
                    probe[
                        "correct_candidate"
                    ]
                    ==
                    "upper"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                for pair
                in primary_pairs
            ),

        "primary_form_balanced":
            all(
                sum(
                    probe[
                        "form"
                    ]
                    ==
                    "word"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                and
                sum(
                    probe[
                        "form"
                    ]
                    ==
                    "symbol"
                    for probe
                    in pair[
                        "probes"
                    ]
                )
                ==
                2
                for pair
                in primary_pairs
            ),

        "all_discriminative_tokens_distinct":
            all(
                pair[
                    "tokenization"
                ][
                    "lower_discriminative_token"
                ][
                    "token_id"
                ]
                !=
                pair[
                    "tokenization"
                ][
                    "upper_discriminative_token"
                ][
                    "token_id"
                ]
                for pair
                in frozen_pairs
            ),

        "all_candidate_roundtrips_verified":
            all(
                bool(
                    pair[
                        "tokenization"
                    ][
                        "lower_token_ids"
                    ]
                )
                and
                bool(
                    pair[
                        "tokenization"
                    ][
                        "upper_token_ids"
                    ]
                )
                for pair
                in frozen_pairs
            ),

        "primary_heads_exact":
            (
                PRIMARY_HEADS
                ==
                [
                    "L1H0",
                    "L4H0",
                ]
            ),

        "weak_control_exact":
            (
                WEAK_CONTROL_HEADS
                ==
                [
                    "L4H1",
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
            "L12B frozen design failed: "
            f"{failed}"
        )

    return {
        "checks":
            checks,

        "pair_index":
            pair_index,

        "primary_probe_count":
            len(
                primary_probes
            ),

        "total_probe_count":
            len(
                all_probes
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    proposal_path = project_path(
        args.proposal
    )

    snapshot_path = project_path(
        args.snapshot
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
            "EXP-0007 frozen plan already "
            f"exists: {output_path}\n"
            "Use --overwrite only before an "
            "execution spec or approval is "
            "created and only if the design "
            "must intentionally change."
        )

    print()
    print("=" * 86)
    print(
        "STAGE L12B — EXP-0007 "
        "PREREGISTERED PLAN BUILDER"
    )
    print("=" * 86)

    proposal = load_json(
        proposal_path
    )

    snapshot = load_json(
        snapshot_path
    )

    store = KnowledgeStore()

    knowledge_before = (
        knowledge_file_hashes(
            store
        )
    )

    input_validation = (
        validate_inputs(
            proposal=proposal,
            proposal_path=(
                proposal_path
            ),
            snapshot=snapshot,
            snapshot_path=(
                snapshot_path
            ),
        )
    )

    print()
    print(
        "Input checks"
    )
    print(
        "------------"
    )

    for name, passed in (
        input_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    print()
    print(
        "Loading tokenizer only..."
    )

    tokenizer = load_tokenizer()

    tokenizer_hashes = {
        "config_sha256":
            file_sha256(
                TOKENIZER_CONFIG_PATH
            ),

        "merges_sha256":
            file_sha256(
                TOKENIZER_MERGES_PATH
            ),

        "vocab_sha256":
            file_sha256(
                TOKENIZER_VOCAB_PATH
            ),
    }

    print(
        "Tokenizer loaded:"
    )
    print(
        "  tokenizer_50m_v0.2"
    )

    # --------------------------------------------------------
    # Bind exact checkpoint bytes without loading model.
    # --------------------------------------------------------

    require(
        EXPECTED_CHECKPOINT_PATH.exists(),
        (
            "Mature v0.3 checkpoint is "
            "missing."
        ),
    )

    print()
    print(
        "Verifying mature-v0.3 checkpoint "
        "SHA-256..."
    )

    actual_checkpoint_sha = (
        file_sha256(
            EXPECTED_CHECKPOINT_PATH
        )
    )

    require(
        actual_checkpoint_sha
        ==
        EXPECTED_CHECKPOINT_SHA256,
        (
            "Mature v0.3 checkpoint SHA "
            "does not match frozen identity."
        ),
    )

    frozen_pairs = build_frozen_pairs(
        tokenizer
    )

    design_validation = (
        validate_frozen_design(
            frozen_pairs
        )
    )

    print()
    print(
        "Frozen candidate pairs"
    )
    print(
        "----------------------"
    )

    for pair in frozen_pairs:

        tokenization = pair[
            "tokenization"
        ]

        print()
        print(
            f"{pair['pair_id']} "
            f"[{pair['role']}]"
        )

        print(
            f"  {pair['lower_label']} vs "
            f"{pair['upper_label']}"
        )

        print(
            "  lower continuation IDs: "
            f"{tokenization['lower_token_ids']}"
        )

        print(
            "  upper continuation IDs: "
            f"{tokenization['upper_token_ids']}"
        )

        print(
            "  discriminative index: "
            f"{tokenization['first_discriminative_index']}"
        )

        print(
            "  discriminative token IDs: "
            f"{tokenization['lower_discriminative_token']['token_id']} "
            "vs "
            f"{tokenization['upper_discriminative_token']['token_id']}"
        )

    print()
    print(
        "Frozen design checks"
    )
    print(
        "--------------------"
    )

    for name, passed in (
        design_validation[
            "checks"
        ].items()
    ):

        print(
            f"{name:<48} "
            f"{passed}"
        )

    # ========================================================
    # Frozen preregistered analysis policy
    # ========================================================

    analysis_policy = {
        "canonical_margin":
            (
                "logit(lower discriminative token) "
                "- logit(upper discriminative token)"
            ),

        "ablation_delta":
            (
                "ablated_margin - baseline_margin"
            ),

        "raw_direction": {
            "negative_delta":
                (
                    "intact head supports LOWER "
                    "relative to UPPER"
                ),

            "positive_delta":
                (
                    "intact head supports UPPER "
                    "relative to LOWER"
                ),
        },

        "correctness_effect": {
            "lower_correct":
                "-ablation_delta",

            "upper_correct":
                "+ablation_delta",

            "positive_effect":
                (
                    "intact head supports the "
                    "correct candidate"
                ),

            "negative_effect":
                (
                    "intact head suppresses the "
                    "correct candidate"
                ),
        },

        "primary_replication_tests": {
            "within_pair_direction_stability":
                {
                    "unit":
                        "head × primary pair",

                    "pass_rule":
                        (
                            "At least 3 of 4 raw "
                            "ablation deltas have "
                            "the same non-zero sign."
                        ),

                    "threshold":
                        "3/4",
                },

            "cross_pair_generalization":
                {
                    "unit":
                        "primary head",

                    "pass_rule":
                        (
                            "Within-pair direction "
                            "stability passes on at "
                            "least 3 of the 4 new "
                            "primary pairs."
                        ),

                    "threshold":
                        "3/4 pairs",
                },

            "strong_correctness_tracking":
                {
                    "unit":
                        "primary head",

                    "pass_rule":
                        (
                            "At least 12 of 16 "
                            "primary correctness "
                            "effects are positive."
                        ),

                    "threshold":
                        "12/16",

                    "interpretation":
                        (
                            "Passing this test is "
                            "evidence for a strong "
                            "arithmetic-correctness "
                            "account and therefore "
                            "works against the "
                            "candidate-selection "
                            "replication claim."
                        ),
                },
        },

        "positive_control_tests": {
            "L1H0":
                {
                    "expected_raw_delta_sign":
                        "negative",

                    "threshold":
                        ">= 3/4 P45 probes",
                },

            "L4H0":
                {
                    "expected_raw_delta_sign":
                        "positive",

                    "threshold":
                        ">= 3/4 P45 probes",
                },

            "overall_pass":
                (
                    "Both L1H0 and L4H0 "
                    "positive-control tests pass."
                ),
        },

        "weak_control_test": {
            "head":
                "L4H1",

            "metric":
                (
                    "mean absolute raw ablation "
                    "delta across the 16 primary "
                    "replication probes"
                ),

            "pass_rule":
                (
                    "L4H1 mean absolute delta is "
                    "strictly lower than the mean "
                    "absolute delta of each primary "
                    "head on the same probes."
                ),
        },

        "orientation_diagnostic": {
            "purpose":
                (
                    "Distinguish pair/token-specific "
                    "stability from a trivial global "
                    "LOWER-versus-UPPER ordering bias."
                ),

            "metric":
                (
                    "For each primary head, compare "
                    "the dominant raw-delta sign "
                    "across the stable primary pairs."
                ),

            "global_order_bias_signal":
                (
                    "The same dominant raw-delta "
                    "sign occurs in every stable "
                    "primary pair."
                ),

            "pair_dependent_signal":
                (
                    "At least two stable primary "
                    "pairs have opposite dominant "
                    "raw-delta signs."
                ),

            "note":
                (
                    "This diagnostic does not alter "
                    "the preregistered primary "
                    "replication pass/fail result; "
                    "it refines mechanism "
                    "interpretation."
                ),
        },

        "outcome_policy": [
            {
                "outcome":
                    (
                        "inconclusive_positive_"
                        "control_failure"
                    ),

                "condition":
                    (
                        "P45 overall positive "
                        "control fails."
                    ),
            },
            {
                "outcome":
                    (
                        "replicates_cross_pair_"
                        "candidate_selection_bias"
                    ),

                "condition":
                    (
                        "Positive control passes; "
                        "both L1H0 and L4H0 pass "
                        "cross-pair generalization; "
                        "neither primary head passes "
                        "strong correctness tracking; "
                        "and the L4H1 weak-control "
                        "criterion passes."
                    ),
            },
            {
                "outcome":
                    "partial_replication",

                "condition":
                    (
                        "Positive control passes and "
                        "exactly one of L1H0/L4H0 "
                        "passes cross-pair "
                        "generalization, or both pass "
                        "but one also passes strong "
                        "correctness tracking, with "
                        "weak-control behavior still "
                        "acceptable."
                    ),
            },
            {
                "outcome":
                    (
                        "fails_cross_pair_"
                        "replication"
                    ),

                "condition":
                    (
                        "Positive control passes but "
                        "neither primary head passes "
                        "cross-pair generalization, "
                        "or the weak-control "
                        "criterion fails."
                    ),
            },
        ],
    }

    intervention_policy = {
        "instrument":
            "attention_head_ablation",

        "intervention":
            (
                "Temporarily zero the selected "
                "attention head output for one "
                "forward pass."
            ),

        "ablation_scale":
            0.0,

        "persistent_weight_modification":
            False,

        "primary_heads":
            PRIMARY_HEADS,

        "weak_control_heads":
            WEAK_CONTROL_HEADS,

        "all_heads":
            ALL_HEADS,

        "effects_per_probe":
            len(
                ALL_HEADS
            ),

        "expected_total_probe_count":
            20,

        "expected_total_causal_effects":
            (
                20
                *
                len(
                    ALL_HEADS
                )
            ),

        "required_permission":
            "P3",

        "human_approval_required":
            True,

        "execution_authorized":
            False,
    }

    plan_core = {
        "experiment_id":
            EXPERIMENT_ID,

        "target_hypothesis_id":
            TARGET_HYPOTHESIS_ID,

        "title":
            (
                "Independent cross-pair numerical "
                "candidate-selection replication"
            ),

        "question":
            (
                "Across four previously untested "
                "numerical candidate pairs, do "
                "L1H0 and L4H0 preserve "
                "context-stable causal pair/token "
                "directions more reliably than "
                "they track arithmetic correctness?"
            ),

        "proposal": {
            "proposal_id":
                EXPECTED_PROPOSAL_ID,

            "proposal_path":
                relative_project_path(
                    proposal_path
                ),

            "proposal_sha256":
                input_validation[
                    "proposal_sha256"
                ],
        },

        "knowledge_state": {
            "knowledge_state_sha256":
                snapshot[
                    "knowledge_state_sha256"
                ],

            "scientist_context_path":
                relative_project_path(
                    snapshot_path
                ),

            "scientist_context_sha256":
                input_validation[
                    "snapshot_sha256"
                ],
        },

        "model_scope":
            input_validation[
                "planned_experiment"
            ][
                "model_scope"
            ],

        "checkpoint_binding": {
            "path":
                relative_project_path(
                    EXPECTED_CHECKPOINT_PATH
                ),

            "sha256":
                actual_checkpoint_sha,
        },

        "tokenizer_binding": {
            "artifact":
                "tokenizer_50m_v0.2",

            "directory":
                relative_project_path(
                    TOKENIZER_DIR
                ),

            **tokenizer_hashes,

            "encoding_protocol":
                (
                    "Prompt and candidate "
                    "continuation are encoded "
                    "separately without BOS/EOS."
                ),
        },

        "frozen_heads":
            {
                "primary":
                    PRIMARY_HEADS,

                "weak_control":
                    WEAK_CONTROL_HEADS,
            },

        "frozen_pairs":
            frozen_pairs,

        "primary_pair_ids":
            PRIMARY_PAIR_IDS,

        "positive_control_pair_id":
            POSITIVE_CONTROL_PAIR_ID,

        "primary_probe_count":
            design_validation[
                "primary_probe_count"
            ],

        "total_probe_count":
            design_validation[
                "total_probe_count"
            ],

        "analysis_policy":
            analysis_policy,

        "intervention_policy":
            intervention_policy,

        "design_constraints": [
            (
                "P12/P67/P78/P89 are the only "
                "primary replication pairs."
            ),
            (
                "P45 is a positive control only "
                "and cannot contribute to primary "
                "replication success."
            ),
            (
                "Each primary pair has exactly "
                "two lower-correct and two "
                "upper-correct probes."
            ),
            (
                "Each primary pair has exactly "
                "two word-form and two symbol-form "
                "probes."
            ),
            (
                "L1H0 and L4H0 are the only "
                "primary heads."
            ),
            (
                "L4H1 is the only weak-control "
                "head."
            ),
            (
                "L3H3 is excluded because it is "
                "reserved for HYP-V03-004."
            ),
            (
                "No prompt, candidate pair, head, "
                "token ID, threshold, or outcome "
                "rule may change after this plan "
                "is frozen without creating a new "
                "plan identity."
            ),
        ],

        "execution_authorized":
            False,

        "p3_approval_present":
            False,
    }

    plan_id = (
        "PLAN-"
        +
        canonical_json_sha256(
            plan_core
        )[
            :16
        ].upper()
    )

    output = {
        "engine_version":
            ENGINE_VERSION,

        "plan_version":
            "L12B.1",

        "mode":
            (
                "frozen_preregistered_"
                "replication_plan"
            ),

        "plan_id":
            plan_id,

        "plan":
            {
                "plan_id":
                    plan_id,

                **plan_core,
            },

        "input_checks":
            input_validation[
                "checks"
            ],

        "design_checks":
            design_validation[
                "checks"
            ],

        "policy_checks": {
            "model_not_loaded":
                True,

            "experiment_not_executed":
                True,

            "knowledge_store_not_mutated":
                True,

            "evidence_not_created":
                True,

            "findings_not_created":
                True,

            "hypothesis_status_not_mutated":
                True,

            "execution_not_authorized":
                True,

            "p3_approval_not_created":
                True,
        },
    }

    atomic_write_json(
        output_path,
        output,
    )

    knowledge_after = (
        knowledge_file_hashes(
            store
        )
    )

    require(
        knowledge_before
        ==
        knowledge_after,
        (
            "L12B unexpectedly mutated the "
            "knowledge store."
        ),
    )

    saved = load_json(
        output_path
    )

    require(
        saved.get(
            "plan_id"
        )
        ==
        plan_id,
        (
            "Saved EXP-0007 plan identity "
            "mismatch."
        ),
    )

    plan_sha256 = file_sha256(
        output_path
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print("=" * 86)
    print(
        "L12B EXP-0007 FROZEN PLAN SUMMARY"
    )
    print("=" * 86)

    print()
    print(
        "Plan ID:"
    )
    print(
        f"  {plan_id}"
    )

    print()
    print(
        "Experiment:"
    )
    print(
        "  EXP-0007"
    )

    print(
        "Hypothesis:"
    )
    print(
        "  HYP-V03-003"
    )

    print()
    print(
        "Primary pairs:"
    )

    for pair_id in (
        PRIMARY_PAIR_IDS
    ):

        print(
            f"  {pair_id}"
        )

    print()
    print(
        "Positive control:"
    )
    print(
        "  P45"
    )

    print()
    print(
        "Primary heads:"
    )
    print(
        "  L1H0"
    )
    print(
        "  L4H0"
    )

    print()
    print(
        "Weak control:"
    )
    print(
        "  L4H1"
    )

    print()
    print(
        "Primary probes:"
    )
    print(
        "  16"
    )

    print(
        "Total probes:"
    )
    print(
        "  20"
    )

    print(
        "Causal effects:"
    )
    print(
        f"  {intervention_policy['expected_total_causal_effects']}"
    )

    print()
    print(
        "Checkpoint SHA-256:"
    )
    print(
        "  "
        +
        actual_checkpoint_sha
    )

    print()
    print(
        "Plan artifact:"
    )
    print(
        output_path
    )

    print()
    print(
        "Plan SHA-256:"
    )
    print(
        plan_sha256
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
            f"{name:<42} "
            f"{passed}"
        )

    print()
    print("=" * 86)
    print(
        "SUCCESS: STAGE L12B EXP-0007 "
        "PREREGISTERED PLAN FROZEN."
    )
    print(
        "No model was loaded, no evidence was "
        "created, and P3 execution remains blocked."
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
