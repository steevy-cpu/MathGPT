import argparse
import json
import shlex
import subprocess
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

PROFILER_SCRIPT = (
    PROJECT_ROOT
    /
    "laboratory/tools/"
    "causal_vocabulary_profiler.py"
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "EXP-0003 — compare L0H1 "
            "vocabulary-wide causal signatures "
            "across multiple contexts."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "laboratory/experiments/"
            "exp_0003_contexts.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "laboratory/artifacts/"
            "exp_0003_profiles"
        ),
    )

    parser.add_argument(
        "--summary",
        type=str,
        default=(
            "laboratory/artifacts/"
            "exp_0003_signature_summary.json"
        ),
    )

    parser.add_argument(
        "--matrix-npz",
        type=str,
        default=(
            "laboratory/artifacts/"
            "exp_0003_signature_matrix.npz"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
    )

    return parser.parse_args()


# ============================================================
# Paths / JSON
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


def load_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Commands
# ============================================================

def run_command(
    command,
):

    command = [
        str(
            value
        )

        for value
        in command
    ]

    print()
    print(
        "$ "
        +
        shlex.join(
            command
        )
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


# ============================================================
# Similarity
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b,
):

    vector_a = np.asarray(
        vector_a,
        dtype=np.float64,
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float64,
    )

    denominator = (
        np.linalg.norm(
            vector_a
        )
        *
        np.linalg.norm(
            vector_b
        )
    )

    if denominator == 0.0:

        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
        /
        denominator
    )


def centered_cosine_similarity(
    vector_a,
    vector_b,
):

    vector_a = np.asarray(
        vector_a,
        dtype=np.float64,
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float64,
    )

    centered_a = (
        vector_a
        -
        np.mean(
            vector_a
        )
    )

    centered_b = (
        vector_b
        -
        np.mean(
            vector_b
        )
    )

    return cosine_similarity(
        centered_a,
        centered_b,
    )


# ============================================================
# Matrix builder
# ============================================================

def similarity_matrix(
    vectors,
    similarity_function,
):

    count = len(
        vectors
    )

    matrix = np.zeros(
        (
            count,
            count,
        ),
        dtype=np.float64,
    )

    for i in range(
        count
    ):

        for j in range(
            count
        ):

            matrix[
                i,
                j
            ] = similarity_function(
                vectors[
                    i
                ],
                vectors[
                    j
                ],
            )

    return matrix


# ============================================================
# Group pair averages
# ============================================================

def pair_values(
    *,
    matrix,
    records,
    condition,
):

    values = []

    count = len(
        records
    )

    for i in range(
        count
    ):

        for j in range(
            i + 1,
            count,
        ):

            if condition(
                records[
                    i
                ],
                records[
                    j
                ],
            ):

                values.append(
                    float(
                        matrix[
                            i,
                            j
                        ]
                    )
                )

    return values


def summarize_values(
    values,
):

    if not values:

        return {
            "count":
                0,

            "mean":
                None,

            "min":
                None,

            "max":
                None,
        }

    return {
        "count":
            int(
                len(
                    values
                )
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


# ============================================================
# Tracked token helper
# ============================================================

def tracked_effect(
    profile,
    text,
):

    for record in profile[
        "tracked_texts"
    ]:

        if (
            record[
                "text"
            ]
            ==
            text
        ):

            first = (
                record[
                    "first_token"
                ]
            )

            if first is None:

                return None

            return float(
                first[
                    "causal_logit_effect"
                ]
            )

    return None


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    config_path = project_path(
        args.config
    )

    output_dir = project_path(
        args.output_dir
    )

    summary_path = project_path(
        args.summary
    )

    matrix_npz_path = project_path(
        args.matrix_npz
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_npz_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_json(
        config_path
    )

    layer = int(
        config[
            "target_layer"
        ]
    )

    head = int(
        config[
            "target_head"
        ]
    )

    head_id = (
        f"L{layer}H{head}"
    )

    print()
    print(
        "=" * 90
    )

    print(
        "EXP-0003 — CAUSAL VOCABULARY "
        "SIGNATURE SIMILARITY"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Target head: "
        f"{head_id}"
    )

    print(
        f"Context count: "
        f"{len(config['contexts'])}"
    )

    records = []

    vectors = []

    # ========================================================
    # Profile every context
    # ========================================================

    for (
        index,
        context,
    ) in enumerate(
        config[
            "contexts"
        ],
        start=1,
    ):

        context_id = (
            context[
                "id"
            ]
        )

        scorer_path = project_path(
            context[
                "scorer_artifact"
            ]
        )

        scorer = load_json(
            scorer_path
        )

        prompt = (
            scorer[
                "prompt"
            ]
        )

        shared_prefix_ids = [
            int(
                token_id
            )

            for token_id
            in scorer[
                "discriminative_analysis"
            ][
                "shared_prefix_token_ids"
            ]
        ]

        print()
        print(
            "=" * 90
        )

        print(
            f"[{index}/"
            f"{len(config['contexts'])}] "
            f"{context_id}"
        )

        print(
            "=" * 90
        )

        print(
            f"Category: "
            f"{context['category']}"
        )

        print(
            f"Prompt: "
            f"{prompt}"
        )

        print(
            f"Shared prefix IDs: "
            f"{shared_prefix_ids}"
        )

        output_prefix = (
            output_dir
            /
            (
                f"{context_id.lower()}"
                f"_vocab_profile"
            )
        )

        command = [
            sys.executable,
            PROFILER_SCRIPT,

            "--checkpoint",
            args.checkpoint,

            "--prompt",
            prompt,

            "--layer",
            str(
                layer
            ),

            "--head",
            str(
                head
            ),

            "--top-k",
            str(
                args.top_k
            ),

            "--output-prefix",
            output_prefix,
        ]

        if (
            args.expected_sha256
            is not None
        ):

            command.extend(
                [
                    "--expected-sha256",
                    args.expected_sha256,
                ]
            )

        for token_id in shared_prefix_ids:

            command.extend(
                [
                    "--append-token-id",
                    str(
                        token_id
                    ),
                ]
            )

        run_command(
            command
        )

        profile_json_path = (
            output_prefix.with_suffix(
                ".json"
            )
        )

        profile_npz_path = (
            output_prefix.with_suffix(
                ".npz"
            )
        )

        profile = load_json(
            profile_json_path
        )

        with np.load(
            profile_npz_path
        ) as artifact:

            vector = np.array(
                artifact[
                    "causal_logit_effect"
                ],
                dtype=np.float64,
                copy=True,
            )

        if (
            vector.shape
            !=
            (
                16000,
            )
        ):

            raise RuntimeError(
                f"{context_id} causal vector "
                f"has unexpected shape "
                f"{vector.shape}."
            )

        vectors.append(
            vector
        )

        records.append(
            {
                "id":
                    context_id,

                "category":
                    context[
                        "category"
                    ],

                "prompt":
                    prompt,

                "shared_prefix_token_ids":
                    shared_prefix_ids,

                "effect_mean":
                    float(
                        profile[
                            "effect_statistics"
                        ][
                            "mean"
                        ]
                    ),

                "effect_std":
                    float(
                        profile[
                            "effect_statistics"
                        ][
                            "std"
                        ]
                    ),

                "effect_l2":
                    float(
                        profile[
                            "effect_statistics"
                        ][
                            "l2_norm"
                        ]
                    ),

                "effect_four":
                    tracked_effect(
                        profile,
                        "four",
                    ),

                "effect_five_first_token":
                    tracked_effect(
                        profile,
                        "five",
                    ),

                "effect_two":
                    tracked_effect(
                        profile,
                        "two",
                    ),

                "effect_three_first_token":
                    tracked_effect(
                        profile,
                        "three",
                    ),

                "profile_json":
                    str(
                        profile_json_path
                    ),

                "profile_npz":
                    str(
                        profile_npz_path
                    ),
            }
        )

    # ========================================================
    # Similarity matrices
    # ========================================================

    raw_cosine = (
        similarity_matrix(
            vectors,
            cosine_similarity,
        )
    )

    centered_cosine = (
        similarity_matrix(
            vectors,
            centered_cosine_similarity,
        )
    )

    # ========================================================
    # Group summaries
    # ========================================================

    addition_addition = pair_values(
        matrix=centered_cosine,

        records=records,

        condition=lambda a, b: (
            a[
                "category"
            ]
            ==
            "addition"
            and
            b[
                "category"
            ]
            ==
            "addition"
        ),
    )

    control_control = pair_values(
        matrix=centered_cosine,

        records=records,

        condition=lambda a, b: (
            a[
                "category"
            ]
            !=
            "addition"
            and
            b[
                "category"
            ]
            !=
            "addition"
        ),
    )

    addition_control = pair_values(
        matrix=centered_cosine,

        records=records,

        condition=lambda a, b: (
            (
                a[
                    "category"
                ]
                ==
                "addition"
            )
            !=
            (
                b[
                    "category"
                ]
                ==
                "addition"
            )
        ),
    )

    group_summary = {
        "centered_cosine": {
            "addition_vs_addition":
                summarize_values(
                    addition_addition
                ),

            "control_vs_control":
                summarize_values(
                    control_control
                ),

            "addition_vs_control":
                summarize_values(
                    addition_control
                ),
        }
    }

    # ========================================================
    # Find nearest neighbor for each context
    # ========================================================

    nearest_neighbors = []

    for i, record in enumerate(
        records
    ):

        candidates = []

        for j, other in enumerate(
            records
        ):

            if i == j:

                continue

            candidates.append(
                {
                    "id":
                        other[
                            "id"
                        ],

                    "category":
                        other[
                            "category"
                        ],

                    "raw_cosine":
                        float(
                            raw_cosine[
                                i,
                                j
                            ]
                        ),

                    "centered_cosine":
                        float(
                            centered_cosine[
                                i,
                                j
                            ]
                        ),
                }
            )

        candidates.sort(
            key=lambda item: (
                item[
                    "centered_cosine"
                ]
            ),
            reverse=True,
        )

        nearest_neighbors.append(
            {
                "id":
                    record[
                        "id"
                    ],

                "nearest":
                    candidates[
                        0
                    ],
            }
        )

    # ========================================================
    # Save vector matrix
    # ========================================================

    stacked_vectors = np.stack(
        vectors,
        axis=0,
    )

    context_ids = np.asarray(
        [
            record[
                "id"
            ]

            for record
            in records
        ],
        dtype="<U16",
    )

    categories = np.asarray(
        [
            record[
                "category"
            ]

            for record
            in records
        ],
        dtype="<U64",
    )

    np.savez(
        matrix_npz_path,

        context_ids=(
            context_ids
        ),

        categories=(
            categories
        ),

        causal_effect_vectors=(
            stacked_vectors.astype(
                np.float32
            )
        ),

        raw_cosine_similarity=(
            raw_cosine.astype(
                np.float32
            )
        ),

        centered_cosine_similarity=(
            centered_cosine.astype(
                np.float32
            )
        ),
    )

    # ========================================================
    # JSON report
    # ========================================================

    report = {
        "experiment_id":
            config[
                "experiment_id"
            ],

        "title":
            config[
                "title"
            ],

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "checkpoint":
            args.checkpoint,

        "expected_checkpoint_sha256":
            args.expected_sha256,

        "target_component": {
            "layer":
                layer,

            "head":
                head,

            "head_id":
                head_id,
        },

        "context_count":
            int(
                len(
                    records
                )
            ),

        "contexts":
            records,

        "similarity": {
            "definition": {
                "raw_cosine":
                    (
                        "Cosine similarity between "
                        "the full 16,000-dimensional "
                        "causal logit effect vectors."
                    ),

                "centered_cosine":
                    (
                        "Cosine similarity after "
                        "subtracting each effect "
                        "vector's vocabulary-wide "
                        "mean."
                    ),
            },

            "context_order":
                [
                    record[
                        "id"
                    ]

                    for record
                    in records
                ],

            "raw_cosine_matrix":
                raw_cosine.tolist(),

            "centered_cosine_matrix":
                centered_cosine.tolist(),

            "nearest_neighbors":
                nearest_neighbors,

            "group_summary":
                group_summary,
        },

        "artifacts": {
            "summary_json":
                str(
                    summary_path
                ),

            "matrix_npz":
                str(
                    matrix_npz_path
                ),
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # Display context effects
    # ========================================================

    print()
    print(
        "=" * 104
    )

    print(
        "CONTEXT EFFECT SUMMARY"
    )

    print(
        "=" * 104
    )

    print()

    print(
        (
            f"{'ID':<6}"
            f"{'Category':<24}"
            f"{'L2':>12}"
            f"{'four':>12}"
            f"{'fiv':>12}"
            f"{'two':>12}"
            f"{'thre':>12}"
        )
    )

    print(
        "-" * 90
    )

    for record in records:

        print(
            (
                f"{record['id']:<6}"
                f"{record['category']:<24}"
                f"{record['effect_l2']:>12.4f}"
                f"{record['effect_four']:>+12.5f}"
                f"{record['effect_five_first_token']:>+12.5f}"
                f"{record['effect_two']:>+12.5f}"
                f"{record['effect_three_first_token']:>+12.5f}"
            )
        )

    # ========================================================
    # Raw cosine matrix
    # ========================================================

    ids = [
        record[
            "id"
        ]

        for record
        in records
    ]

    print()
    print(
        "=" * 104
    )

    print(
        "RAW COSINE SIMILARITY"
    )

    print(
        "=" * 104
    )

    print()

    print(
        f"{'':<6}"
        +
        "".join(
            f"{context_id:>8}"

            for context_id
            in ids
        )
    )

    for i, context_id in enumerate(
        ids
    ):

        row = (
            f"{context_id:<6}"
        )

        for j in range(
            len(
                ids
            )
        ):

            row += (
                f"{raw_cosine[i, j]:>8.3f}"
            )

        print(
            row
        )

    # ========================================================
    # Centered matrix
    # ========================================================

    print()
    print(
        "=" * 104
    )

    print(
        "CENTERED COSINE SIMILARITY"
    )

    print(
        "=" * 104
    )

    print()

    print(
        f"{'':<6}"
        +
        "".join(
            f"{context_id:>8}"

            for context_id
            in ids
        )
    )

    for i, context_id in enumerate(
        ids
    ):

        row = (
            f"{context_id:<6}"
        )

        for j in range(
            len(
                ids
            )
        ):

            row += (
                f"{centered_cosine[i, j]:>8.3f}"
            )

        print(
            row
        )

    # ========================================================
    # Nearest neighbors
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "NEAREST CAUSAL-SIGNATURE NEIGHBORS"
    )

    print(
        "=" * 80
    )

    print()

    for item in nearest_neighbors:

        nearest = (
            item[
                "nearest"
            ]
        )

        print(
            (
                f"{item['id']:<6} "
                f"→ {nearest['id']:<6} "
                f"centered_cos="
                f"{nearest['centered_cosine']:+.6f}"
            )
        )

    # ========================================================
    # Group comparison
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "GROUP COMPARISON — CENTERED COSINE"
    )

    print(
        "=" * 80
    )

    print()

    for (
        name,
        summary,
    ) in (
        group_summary[
            "centered_cosine"
        ].items()
    ):

        print(
            f"{name}:"
        )

        print(
            f"  pairs = "
            f"{summary['count']}"
        )

        print(
            f"  mean  = "
            f"{summary['mean']}"
        )

        print(
            f"  min   = "
            f"{summary['min']}"
        )

        print(
            f"  max   = "
            f"{summary['max']}"
        )

        print()

    print(
        "Artifacts"
    )

    print(
        "---------"
    )

    print(
        f"Summary JSON:"
    )

    print(
        summary_path
    )

    print()

    print(
        f"Matrix NPZ:"
    )

    print(
        matrix_npz_path
    )

    print()
    print(
        "=" * 90
    )

    print(
        "SUCCESS: EXP-0003 CAUSAL "
        "SIGNATURE COMPARISON COMPLETE."
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":

    main()