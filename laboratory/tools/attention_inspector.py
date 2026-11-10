import argparse
import json
import math
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

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Laboratory imports
# ============================================================

from laboratory.core.runtime import (
    LAB_TOOL_VERSION,
    MARS_PROTOCOL_VERSION,
    LaboratoryRuntime,
    sha256_file,
    token_text,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.5 attention_inspector — "
            "inspect causal self-attention patterns "
            "inside MathGPT."
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
        "--prompt",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--append-token-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Append exact token IDs after the "
            "encoded prompt."
        ),
    )

    parser.add_argument(
        "--query-position",
        type=int,
        default=-1,
        help=(
            "Token position whose attention row "
            "will be inspected. -1 means last token."
        ),
    )

    parser.add_argument(
        "--focus-position",
        action="append",
        type=int,
        default=[],
        help=(
            "Token position of special interest. "
            "Repeat for multiple positions."
        ),
    )

    parser.add_argument(
        "--top-k-positions",
        type=int,
        default=6,
        help=(
            "How many attended token positions "
            "to report per head."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=str,
        help=(
            "Produces <prefix>.json and "
            "<prefix>.npz."
        ),
    )

    return parser.parse_args()


# ============================================================
# Entropy helpers
# ============================================================

def attention_entropy(
    probabilities,
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    safe = np.maximum(
        probabilities,
        1e-30,
    )

    return float(
        -np.sum(
            probabilities
            *
            np.log(
                safe
            )
        )
    )


def normalized_entropy(
    probabilities,
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    valid_count = int(
        np.sum(
            probabilities
            >
            0.0
        )
    )

    if valid_count <= 1:

        return 0.0

    entropy = attention_entropy(
        probabilities
    )

    maximum_entropy = math.log(
        valid_count
    )

    return float(
        entropy
        /
        maximum_entropy
    )


# ============================================================
# Head analysis
# ============================================================

def analyze_head(
    *,
    tokenizer,
    attention_row,
    context_token_ids,
    query_position,
    layer_index,
    head_index,
    focus_positions,
    top_k,
):

    attention_row = np.asarray(
        attention_row,
        dtype=np.float64,
    )

    row_sum = float(
        np.sum(
            attention_row
        )
    )

    entropy = attention_entropy(
        attention_row
    )

    entropy_normalized = (
        normalized_entropy(
            attention_row
        )
    )

    top_k = min(
        int(
            top_k
        ),
        len(
            attention_row
        ),
    )

    order = np.argsort(
        -attention_row,
        kind="stable",
    )[
        :top_k
    ]

    top_positions = []

    for rank, position in enumerate(
        order,
        start=1,
    ):

        position = int(
            position
        )

        token_id = int(
            context_token_ids[
                position
            ]
        )

        top_positions.append(
            {
                "rank":
                    int(
                        rank
                    ),

                "position":
                    position,

                "token_id":
                    token_id,

                "token_text":
                    token_text(
                        tokenizer,
                        token_id,
                    ),

                "attention_weight":
                    float(
                        attention_row[
                            position
                        ]
                    ),
            }
        )

    focus_records = []

    focus_mass = 0.0

    for position in focus_positions:

        weight = float(
            attention_row[
                position
            ]
        )

        focus_mass += weight

        token_id = int(
            context_token_ids[
                position
            ]
        )

        focus_records.append(
            {
                "position":
                    int(
                        position
                    ),

                "token_id":
                    token_id,

                "token_text":
                    token_text(
                        tokenizer,
                        token_id,
                    ),

                "attention_weight":
                    weight,
            }
        )

    query_token_id = int(
        context_token_ids[
            query_position
        ]
    )

    self_attention_weight = float(
        attention_row[
            query_position
        ]
    )

    strongest_position = int(
        np.argmax(
            attention_row
        )
    )

    strongest_token_id = int(
        context_token_ids[
            strongest_position
        ]
    )

    return {
        "layer":
            int(
                layer_index
            ),

        "head":
            int(
                head_index
            ),

        "head_id":
            (
                f"L{layer_index}H{head_index}"
            ),

        "query": {
            "position":
                int(
                    query_position
                ),

            "token_id":
                query_token_id,

            "token_text":
                token_text(
                    tokenizer,
                    query_token_id,
                ),
        },

        "row_sum":
            row_sum,

        "entropy":
            entropy,

        "normalized_entropy":
            entropy_normalized,

        "self_attention_weight":
            self_attention_weight,

        "strongest_position":
            strongest_position,

        "strongest_token_id":
            strongest_token_id,

        "strongest_token_text":
            token_text(
                tokenizer,
                strongest_token_id,
            ),

        "strongest_attention_weight":
            float(
                attention_row[
                    strongest_position
                ]
            ),

        "focus_mass":
            float(
                focus_mass
            ),

        "focus_positions":
            focus_records,

        "top_positions":
            top_positions,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print()
    print(
        "=" * 76
    )

    print(
        "L6.5 — ATTENTION INSPECTOR"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Permission level: P2"
    )

    print(
        "Evidence level: E2 "
        "(internal correlation)"
    )

    print()

    # ========================================================
    # Runtime
    # ========================================================

    runtime = LaboratoryRuntime(
        checkpoint_path=(
            args.checkpoint
        ),

        expected_checkpoint_sha256=(
            args.expected_sha256
        ),
    )

    # ========================================================
    # Exact context
    # ========================================================

    prompt_token_ids = (
        runtime.encode_prompt(
            args.prompt
        )
    )

    appended_token_ids = [
        int(
            token_id
        )

        for token_id
        in args.append_token_id
    ]

    context_token_ids = (
        list(
            prompt_token_ids
        )
        +
        appended_token_ids
    )

    sequence_length = len(
        context_token_ids
    )

    if (
        sequence_length
        >
        runtime.model.context_length
    ):

        raise ValueError(
            "Input exceeds context length."
        )

    if sequence_length == 0:

        raise ValueError(
            "Context cannot be empty."
        )

    # ========================================================
    # Query position
    # ========================================================

    if args.query_position < 0:

        query_position = (
            sequence_length
            +
            args.query_position
        )

    else:

        query_position = (
            args.query_position
        )

    if not (
        0
        <=
        query_position
        <
        sequence_length
    ):

        raise ValueError(
            "Query position is outside "
            "the context."
        )

    # ========================================================
    # Focus positions
    # ========================================================

    focus_positions = [
        int(
            position
        )

        for position
        in args.focus_position
    ]

    for position in focus_positions:

        if not (
            0
            <=
            position
            <=
            query_position
        ):

            raise ValueError(
                f"Focus position {position} "
                "is outside the causal context "
                "available to the query."
            )

    # ========================================================
    # Forward trace
    # ========================================================

    input_ids = np.asarray(
        [
            context_token_ids
        ],
        dtype=np.int64,
    )

    traced_logits, trace = (
        runtime.forward_with_trace(
            input_ids
        )
    )

    normal_logits = (
        runtime.model.forward(
            input_ids
        )
    )

    forward_exact = bool(
        np.array_equal(
            traced_logits,
            normal_logits,
        )
    )

    forward_max_difference = float(
        np.max(
            np.abs(
                traced_logits.astype(
                    np.float64
                )
                -
                normal_logits.astype(
                    np.float64
                )
            )
        )
    )

    # ========================================================
    # Inspect attention heads
    # ========================================================

    head_results = []

    raw_arrays = {
        "context_token_ids":
            np.asarray(
                context_token_ids,
                dtype=np.int64,
            )
    }

    for layer_index in range(
        runtime.model.num_layers
    ):

        trace_key = (
            f"block_{layer_index}."
            f"attention_weights"
        )

        attention = np.asarray(
            trace[
                trace_key
            ],
            dtype=np.float32,
        )

        expected_shape = (
            1,
            runtime.model.num_heads,
            sequence_length,
            sequence_length,
        )

        if (
            attention.shape
            !=
            expected_shape
        ):

            raise RuntimeError(
                f"{trace_key} has shape "
                f"{attention.shape}, expected "
                f"{expected_shape}."
            )

        raw_arrays[
            f"block_{layer_index}__attention_weights"
        ] = np.array(
            attention,
            copy=True,
        )

        for head_index in range(
            runtime.model.num_heads
        ):

            row = (
                attention[
                    0,
                    head_index,
                    query_position,
                    :
                ]
            )

            result = analyze_head(
                tokenizer=(
                    runtime.tokenizer
                ),

                attention_row=(
                    row
                ),

                context_token_ids=(
                    context_token_ids
                ),

                query_position=(
                    query_position
                ),

                layer_index=(
                    layer_index
                ),

                head_index=(
                    head_index
                ),

                focus_positions=(
                    focus_positions
                ),

                top_k=(
                    args.top_k_positions
                ),
            )

            head_results.append(
                result
            )

    # ========================================================
    # Layer summaries
    # ========================================================

    layer_summaries = []

    for layer_index in range(
        runtime.model.num_layers
    ):

        layer_heads = [
            head

            for head
            in head_results

            if head[
                "layer"
            ]
            ==
            layer_index
        ]

        focus_values = [
            head[
                "focus_mass"
            ]

            for head
            in layer_heads
        ]

        entropy_values = [
            head[
                "normalized_entropy"
            ]

            for head
            in layer_heads
        ]

        self_values = [
            head[
                "self_attention_weight"
            ]

            for head
            in layer_heads
        ]

        strongest_focus_head = max(
            layer_heads,
            key=lambda head: (
                head[
                    "focus_mass"
                ]
            ),
        )

        layer_summaries.append(
            {
                "layer":
                    int(
                        layer_index
                    ),

                "mean_focus_mass":
                    float(
                        np.mean(
                            focus_values
                        )
                    ),

                "max_focus_mass":
                    float(
                        np.max(
                            focus_values
                        )
                    ),

                "max_focus_head":
                    strongest_focus_head[
                        "head_id"
                    ],

                "mean_normalized_entropy":
                    float(
                        np.mean(
                            entropy_values
                        )
                    ),

                "mean_self_attention":
                    float(
                        np.mean(
                            self_values
                        )
                    ),
            }
        )

    # ========================================================
    # Rank heads by focus mass
    # ========================================================

    focus_ranking = sorted(
        head_results,
        key=lambda head: (
            head[
                "focus_mass"
            ]
        ),
        reverse=True,
    )

    # ========================================================
    # Integrity
    # ========================================================

    row_sum_errors = [
        abs(
            head[
                "row_sum"
            ]
            -
            1.0
        )

        for head
        in head_results
    ]

    maximum_row_sum_error = float(
        max(
            row_sum_errors
        )
    )

    all_values_finite = all(
        np.isfinite(
            head[
                "row_sum"
            ]
        )
        and
        np.isfinite(
            head[
                "entropy"
            ]
        )
        and
        np.isfinite(
            head[
                "focus_mass"
            ]
        )

        for head
        in head_results
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "trace_matches_gpt_forward":
            forward_exact,

        "expected_head_count":
            (
                len(
                    head_results
                )
                ==
                (
                    runtime.model.num_layers
                    *
                    runtime.model.num_heads
                )
            ),

        "attention_values_finite":
            bool(
                all_values_finite
            ),

        "attention_rows_sum_to_one":
            (
                maximum_row_sum_error
                <=
                1e-5
            ),

        "focus_positions_valid":
            all(
                0
                <=
                position
                <=
                query_position

                for position
                in focus_positions
            ),
    }

    all_checks_passed = all(
        checks.values()
    )

    # ========================================================
    # Output paths
    # ========================================================

    output_prefix = Path(
        args.output_prefix
    )

    if not output_prefix.is_absolute():

        output_prefix = (
            PROJECT_ROOT
            /
            output_prefix
        )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_prefix.with_suffix(
            ".json"
        )
    )

    npz_path = (
        output_prefix.with_suffix(
            ".npz"
        )
    )

    # ========================================================
    # Save raw attention
    # ========================================================

    np.savez(
        npz_path,
        **raw_arrays,
    )

    npz_sha256 = (
        sha256_file(
            npz_path
        )
    )

    # ========================================================
    # Context records
    # ========================================================

    context_records = [
        {
            "position":
                int(
                    position
                ),

            "token_id":
                int(
                    token_id
                ),

            "token_text":
                token_text(
                    runtime.tokenizer,
                    token_id,
                ),

            "is_query":
                bool(
                    position
                    ==
                    query_position
                ),

            "is_focus":
                bool(
                    position
                    in
                    focus_positions
                ),
        }

        for (
            position,
            token_id,
        ) in enumerate(
            context_token_ids
        )
    ]

    # ========================================================
    # Report
    # ========================================================

    report = {
        "mars_protocol":
            MARS_PROTOCOL_VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "tool": {
            "name":
                "attention_inspector",

            "version":
                LAB_TOOL_VERSION,

            "permission_level":
                "P2",

            "evidence_level":
                "E2",
        },

        "status":
            (
                "success"
                if all_checks_passed
                else "failed"
            ),

        "model":
            runtime.model_record(),

        "input": {
            "prompt":
                args.prompt,

            "prompt_token_ids":
                prompt_token_ids,

            "appended_token_ids":
                appended_token_ids,

            "context":
                context_records,

            "query_position":
                int(
                    query_position
                ),

            "focus_positions":
                focus_positions,
        },

        "attention": {
            "head_count":
                int(
                    len(
                        head_results
                    )
                ),

            "heads":
                head_results,

            "layer_summaries":
                layer_summaries,

            "focus_ranking":
                [
                    {
                        "rank":
                            int(
                                rank
                            ),

                        "head_id":
                            head[
                                "head_id"
                            ],

                        "layer":
                            head[
                                "layer"
                            ],

                        "head":
                            head[
                                "head"
                            ],

                        "focus_mass":
                            head[
                                "focus_mass"
                            ],

                        "strongest_position":
                            head[
                                "strongest_position"
                            ],

                        "strongest_token_text":
                            head[
                                "strongest_token_text"
                            ],

                        "normalized_entropy":
                            head[
                                "normalized_entropy"
                            ],
                    }

                    for (
                        rank,
                        head,
                    ) in enumerate(
                        focus_ranking,
                        start=1,
                    )
                ],
        },

        "forward_verification": {
            "exact":
                forward_exact,

            "max_absolute_difference":
                forward_max_difference,
        },

        "integrity": {
            "maximum_attention_row_sum_error":
                maximum_row_sum_error,
        },

        "artifacts": {
            "json":
                str(
                    json_path
                ),

            "raw_npz":
                str(
                    npz_path
                ),

            "raw_npz_sha256":
                npz_sha256,
        },

        "checks":
            checks,
    }

    # ========================================================
    # Save JSON
    # ========================================================

    with open(
        json_path,
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
    # Display
    # ========================================================

    print(
        "Input context"
    )

    print(
        "-------------"
    )

    for record in context_records:

        markers = []

        if record[
            "is_focus"
        ]:

            markers.append(
                "FOCUS"
            )

        if record[
            "is_query"
        ]:

            markers.append(
                "QUERY"
            )

        marker_text = (
            ""
            if not markers
            else
            "  [" + ", ".join(
                markers
            ) + "]"
        )

        print(
            (
                f"[{record['position']}] "
                f"id={record['token_id']:<5} "
                f"text={repr(record['token_text'])}"
                f"{marker_text}"
            )
        )

    print()
    print(
        "=" * 76
    )

    print(
        "ATTENTION BY LAYER / HEAD"
    )

    print(
        "=" * 76
    )

    for layer_index in range(
        runtime.model.num_layers
    ):

        print()
        print(
            f"Layer {layer_index}"
        )

        print(
            "-" * 40
        )

        for head in head_results:

            if (
                head[
                    "layer"
                ]
                !=
                layer_index
            ):

                continue

            print()
            print(
                (
                    f"{head['head_id']} | "
                    f"focus_mass="
                    f"{head['focus_mass']:.4f} | "
                    f"self="
                    f"{head['self_attention_weight']:.4f} | "
                    f"entropy="
                    f"{head['normalized_entropy']:.4f}"
                )
            )

            print(
                "  Top attended:"
            )

            for item in (
                head[
                    "top_positions"
                ]
            ):

                print(
                    (
                        f"    #{item['rank']} "
                        f"pos={item['position']} "
                        f"text={repr(item['token_text']):<12} "
                        f"weight="
                        f"{item['attention_weight']:.6f}"
                    )
                )

    # ========================================================
    # Focus ranking
    # ========================================================

    if focus_positions:

        print()
        print(
            "=" * 76
        )

        print(
            "HEADS RANKED BY FOCUS MASS"
        )

        print(
            "=" * 76
        )

        print()

        for rank, head in enumerate(
            focus_ranking,
            start=1,
        ):

            print(
                (
                    f"#{rank:<2} "
                    f"{head['head_id']:<5} "
                    f"focus_mass="
                    f"{head['focus_mass']:.6f} "
                    f"strongest="
                    f"{repr(head['strongest_token_text'])}"
                )
            )

    # ========================================================
    # Layer summary
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "LAYER SUMMARY"
    )

    print(
        "=" * 76
    )

    print()

    print(
        (
            f"{'Layer':<8}"
            f"{'Mean Focus':>14}"
            f"{'Max Focus':>14}"
            f"{'Best Head':>14}"
            f"{'Entropy':>14}"
        )
    )

    print(
        "-" * 64
    )

    for layer in layer_summaries:

        print(
            (
                f"{layer['layer']:<8}"
                f"{layer['mean_focus_mass']:>14.6f}"
                f"{layer['max_focus_mass']:>14.6f}"
                f"{layer['max_focus_head']:>14}"
                f"{layer['mean_normalized_entropy']:>14.6f}"
            )
        )

    # ========================================================
    # Verification
    # ========================================================

    print()
    print(
        "Instrument verification"
    )

    print(
        "-----------------------"
    )

    print(
        f"Trace logits exact: "
        f"{forward_exact}"
    )

    print(
        f"Max logit difference: "
        f"{forward_max_difference}"
    )

    print(
        f"Maximum attention row-sum error: "
        f"{maximum_row_sum_error}"
    )

    print()

    print(
        "Checks"
    )

    print(
        "------"
    )

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    print()
    print(
        "Artifacts"
    )

    print(
        "---------"
    )

    print(
        f"JSON: "
        f"{json_path}"
    )

    print(
        f"NPZ:  "
        f"{npz_path}"
    )

    print(
        f"NPZ SHA-256:"
    )

    print(
        npz_sha256
    )

    if not all_checks_passed:

        raise RuntimeError(
            "L6.5 attention inspector failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.5 ATTENTION "
        "INSPECTOR PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()