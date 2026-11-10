import argparse
import json
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
# Capture presets
# ============================================================

def residual_capture_points(
    runtime,
):

    points = [
        "embedding_output"
    ]

    for block_index in range(
        runtime.model.num_layers
    ):

        points.append(
            f"block_{block_index}.output"
        )

    points.append(
        "final_layernorm"
    )

    return points


def component_capture_points(
    runtime,
):

    points = [
        "embedding_output"
    ]

    for block_index in range(
        runtime.model.num_layers
    ):

        prefix = (
            f"block_{block_index}"
        )

        points.extend(
            [
                f"{prefix}.attention_output",
                f"{prefix}.attention_residual",
                f"{prefix}.mlp_output",
                f"{prefix}.output",
            ]
        )

    points.append(
        "final_layernorm"
    )

    return points


def attention_capture_points(
    runtime,
):

    return [
        (
            f"block_{block_index}."
            f"attention_weights"
        )

        for block_index
        in range(
            runtime.model.num_layers
        )
    ]


def all_capture_points(
    trace,
):

    return list(
        trace.keys()
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.3 activation_capture — "
            "read-only capture of MathGPT "
            "internal forward activations."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="MathGPT checkpoint.",
    )

    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help=(
            "Optional expected checkpoint "
            "SHA-256."
        ),
    )

    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
        help="Prompt to inspect.",
    )

    parser.add_argument(
        "--append-token-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Append an exact token ID after "
            "the encoded prompt. Repeat this "
            "flag to append multiple tokens."
        ),
    )

    parser.add_argument(
        "--preset",
        choices=[
            "residuals",
            "components",
            "attention",
            "all",
        ],
        default="residuals",
        help="Which internal states to capture.",
    )

    parser.add_argument(
        "--capture",
        action="append",
        default=[],
        help=(
            "Optional exact trace key. "
            "Repeat to add specific capture "
            "points beyond the selected preset."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=str,
        help=(
            "Output prefix. Produces "
            "<prefix>.json and <prefix>.npz."
        ),
    )

    return parser.parse_args()


# ============================================================
# Activation statistics
# ============================================================

def array_statistics(
    array,
):

    array = np.asarray(
        array
    )

    finite = bool(
        np.all(
            np.isfinite(
                array
            )
        )
    )

    result = {
        "shape":
            [
                int(
                    dimension
                )

                for dimension
                in array.shape
            ],

        "dtype":
            str(
                array.dtype
            ),

        "element_count":
            int(
                array.size
            ),

        "finite":
            finite,
    }

    if array.size == 0:

        result.update(
            {
                "mean":
                    None,

                "std":
                    None,

                "min":
                    None,

                "max":
                    None,

                "l2_norm":
                    None,
            }
        )

        return result

    array64 = array.astype(
        np.float64,
        copy=False,
    )

    result.update(
        {
            "mean":
                float(
                    np.mean(
                        array64
                    )
                ),

            "std":
                float(
                    np.std(
                        array64
                    )
                ),

            "min":
                float(
                    np.min(
                        array64
                    )
                ),

            "max":
                float(
                    np.max(
                        array64
                    )
                ),

            "l2_norm":
                float(
                    np.linalg.norm(
                        array64
                    )
                ),
        }
    )

    # ========================================================
    # Last-token statistics
    #
    # Standard activation tensors:
    #     (B, T, D)
    #
    # Attention:
    #     (B, H, T, T)
    # ========================================================

    if array.ndim == 3:

        last_token = (
            array64[
                0,
                -1,
                :
            ]
        )

        result[
            "last_token"
        ] = {
            "mean":
                float(
                    np.mean(
                        last_token
                    )
                ),

            "std":
                float(
                    np.std(
                        last_token
                    )
                ),

            "min":
                float(
                    np.min(
                        last_token
                    )
                ),

            "max":
                float(
                    np.max(
                        last_token
                    )
                ),

            "l2_norm":
                float(
                    np.linalg.norm(
                        last_token
                    )
                ),
        }

    elif array.ndim == 4:

        # Attention weights:
        # last query position for each head.
        last_attention = (
            array64[
                0,
                :,
                -1,
                :
            ]
        )

        row_sums = np.sum(
            last_attention,
            axis=-1,
        )

        result[
            "last_query_attention"
        ] = {
            "head_count":
                int(
                    last_attention.shape[
                        0
                    ]
                ),

            "row_sum_min":
                float(
                    np.min(
                        row_sums
                    )
                ),

            "row_sum_max":
                float(
                    np.max(
                        row_sums
                    )
                ),

            "mean_entropy":
                float(
                    np.mean(
                        -np.sum(
                            last_attention
                            *
                            np.log(
                                np.maximum(
                                    last_attention,
                                    1e-30,
                                )
                            ),
                            axis=-1,
                        )
                    )
                ),
        }

    return result


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
        "L6.3 — ACTIVATION CAPTURE"
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
    # Construct exact token context
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

    for token_id in appended_token_ids:

        if not (
            0
            <=
            token_id
            <
            runtime.model.vocab_size
        ):

            raise ValueError(
                f"Appended token ID "
                f"{token_id} is outside "
                "the vocabulary."
            )

    context_token_ids = (
        list(
            prompt_token_ids
        )
        +
        appended_token_ids
    )

    if (
        len(
            context_token_ids
        )
        >
        runtime.model.context_length
    ):

        raise ValueError(
            "Prompt + appended tokens "
            "exceed model context length."
        )

    input_ids = np.asarray(
        [
            context_token_ids
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Traced forward pass
    # ========================================================

    traced_logits, trace = (
        runtime.forward_with_trace(
            input_ids
        )
    )

    # ========================================================
    # Exact equivalence check
    # ========================================================

    normal_logits = (
        runtime.model.forward(
            input_ids
        )
    )

    logits_exact = bool(
        np.array_equal(
            traced_logits,
            normal_logits,
        )
    )

    max_logit_difference = float(
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
    # Select capture points
    # ========================================================

    if (
        args.preset
        ==
        "residuals"
    ):

        selected_points = (
            residual_capture_points(
                runtime
            )
        )

    elif (
        args.preset
        ==
        "components"
    ):

        selected_points = (
            component_capture_points(
                runtime
            )
        )

    elif (
        args.preset
        ==
        "attention"
    ):

        selected_points = (
            attention_capture_points(
                runtime
            )
        )

    else:

        selected_points = (
            all_capture_points(
                trace
            )
        )

    for capture_name in args.capture:

        if (
            capture_name
            not in selected_points
        ):

            selected_points.append(
                capture_name
            )

    # ========================================================
    # Validate capture names
    # ========================================================

    missing_points = [
        capture_name

        for capture_name
        in selected_points

        if capture_name
        not in trace
    ]

    if missing_points:

        print()
        print(
            "Available trace keys:"
        )

        for key in sorted(
            trace.keys()
        ):

            print(
                f"  {key}"
            )

        raise KeyError(
            "Unknown capture points: "
            f"{missing_points}"
        )

    # ========================================================
    # Build raw artifact payload
    # ========================================================

    raw_arrays = {}

    summaries = {}

    for capture_name in selected_points:

        array = np.asarray(
            trace[
                capture_name
            ]
        )

        artifact_key = (
            capture_name
            .replace(
                ".",
                "__",
            )
        )

        raw_arrays[
            artifact_key
        ] = np.array(
            array,
            copy=True,
        )

        summaries[
            capture_name
        ] = {
            "artifact_key":
                artifact_key,

            **array_statistics(
                array
            ),
        }

    # Save the final next-token logits as a useful
    # reference without storing the entire B×T×V
    # tensor.
    next_token_logits = (
        traced_logits[
            0,
            -1,
            :
        ]
    )

    raw_arrays[
        "next_token_logits"
    ] = np.array(
        next_token_logits,
        copy=True,
    )

    # Also preserve the exact input IDs.
    raw_arrays[
        "context_token_ids"
    ] = np.asarray(
        context_token_ids,
        dtype=np.int64,
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
    # Save raw arrays
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
    # Integrity checks
    # ========================================================

    all_captures_finite = all(
        summary[
            "finite"
        ]

        for summary
        in summaries.values()
    )

    next_logits_finite = bool(
        np.all(
            np.isfinite(
                next_token_logits
            )
        )
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "trace_matches_gpt_forward":
            logits_exact,

        "all_capture_points_present":
            (
                len(
                    missing_points
                )
                ==
                0
            ),

        "all_captures_finite":
            all_captures_finite,

        "next_token_logits_finite":
            next_logits_finite,

        "context_within_limit":
            (
                len(
                    context_token_ids
                )
                <=
                runtime.model.context_length
            ),
    }

    all_checks_passed = all(
        checks.values()
    )

    # ========================================================
    # Token records
    # ========================================================

    prompt_token_records = [
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
        }

        for (
            position,
            token_id,
        ) in enumerate(
            prompt_token_ids
        )
    ]

    appended_token_records = [
        {
            "append_index":
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
        }

        for (
            position,
            token_id,
        ) in enumerate(
            appended_token_ids
        )
    ]

    context_token_records = [
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
        }

        for (
            position,
            token_id,
        ) in enumerate(
            context_token_ids
        )
    ]

    # ========================================================
    # JSON report
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
                "activation_capture",

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
                [
                    int(
                        value
                    )

                    for value
                    in prompt_token_ids
                ],

            "prompt_tokens":
                prompt_token_records,

            "appended_token_ids":
                appended_token_ids,

            "appended_tokens":
                appended_token_records,

            "context_token_count":
                int(
                    len(
                        context_token_ids
                    )
                ),

            "context_tokens":
                context_token_records,
        },

        "capture": {
            "preset":
                args.preset,

            "capture_point_count":
                int(
                    len(
                        selected_points
                    )
                ),

            "capture_points":
                selected_points,

            "summaries":
                summaries,
        },

        "forward_verification": {
            "exact":
                logits_exact,

            "max_absolute_difference":
                max_logit_difference,
        },

        "artifacts": {
            "raw_npz":
                str(
                    npz_path
                ),

            "raw_npz_sha256":
                npz_sha256,

            "json":
                str(
                    json_path
                ),
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
        "Checkpoint"
    )

    print(
        "----------"
    )

    print(
        runtime.checkpoint_path
    )

    print()

    print(
        f"SHA-256:"
    )

    print(
        runtime.checkpoint_sha256
    )

    print()

    print(
        "Input context"
    )

    print(
        "-------------"
    )

    print(
        f"Prompt: "
        f"{repr(args.prompt)}"
    )

    print()

    print(
        "Prompt tokens:"
    )

    for record in prompt_token_records:

        print(
            (
                f"  [{record['position']}] "
                f"id={record['token_id']:<5} "
                f"text={repr(record['token_text'])}"
            )
        )

    if appended_token_records:

        print()
        print(
            "Explicit appended tokens:"
        )

        for record in appended_token_records:

            print(
                (
                    f"  id={record['token_id']:<5} "
                    f"text={repr(record['token_text'])}"
                )
            )

    print()

    print(
        f"Final context length: "
        f"{len(context_token_ids)}"
    )

    print()

    print(
        "Forward verification"
    )

    print(
        "--------------------"
    )

    print(
        f"Exact logits: "
        f"{logits_exact}"
    )

    print(
        f"Max absolute difference: "
        f"{max_logit_difference}"
    )

    print()

    print(
        "Captured activations"
    )

    print(
        "--------------------"
    )

    for capture_name in selected_points:

        summary = (
            summaries[
                capture_name
            ]
        )

        print()

        print(
            capture_name
        )

        print(
            (
                f"  shape="
                f"{summary['shape']} "
                f"dtype="
                f"{summary['dtype']}"
            )
        )

        print(
            (
                f"  mean="
                f"{summary['mean']:+.6f} "
                f"std="
                f"{summary['std']:.6f} "
                f"L2="
                f"{summary['l2_norm']:.6f}"
            )
        )

        if (
            "last_token"
            in summary
        ):

            last = (
                summary[
                    "last_token"
                ]
            )

            print(
                (
                    f"  last-token "
                    f"mean={last['mean']:+.6f} "
                    f"std={last['std']:.6f} "
                    f"L2={last['l2_norm']:.6f}"
                )
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

    print()

    print(
        "Checks"
    )

    print(
        "------"
    )

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    # ========================================================
    # Acceptance
    # ========================================================

    if not all_checks_passed:

        raise RuntimeError(
            "L6.3 activation capture failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.3 ACTIVATION "
        "CAPTURE PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()