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
    token_statistics,
    token_text,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.6 gradient_attribution — "
            "measure local sensitivity of a "
            "target logit margin to internal "
            "MathGPT activations."
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
    )

    parser.add_argument(
        "--scorer-artifact",
        required=True,
        type=str,
        help=(
            "Validated L6.2 continuation-scorer "
            "artifact containing discriminative "
            "target tokens."
        ),
    )

    parser.add_argument(
        "--preferred-target-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--comparison-target-index",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# JSON helper
# ============================================================

def load_json(
    path,
):

    path = Path(
        path
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
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

        return (
            path,
            json.load(
                file
            ),
        )


# ============================================================
# Extract discriminative targets
# ============================================================

def extract_discriminative_targets(
    scorer_artifact,
):

    candidates = (
        scorer_artifact[
            "discriminative_analysis"
        ][
            "candidates"
        ]
    )

    records = []

    for candidate in candidates:

        if not candidate.get(
            "has_discriminative_token",
            False,
        ):

            continue

        records.append(
            {
                "candidate_index":
                    int(
                        candidate[
                            "candidate_index"
                        ]
                    ),

                "token_id":
                    int(
                        candidate[
                            "token_id"
                        ]
                    ),

                "token_text":
                    candidate[
                        "token_text"
                    ],
            }
        )

    return records


# ============================================================
# Attribution statistics
# ============================================================

def attribution_statistics(
    *,
    activation,
    gradient,
):
    """
    Summarize one activation/gradient pair.

    Both are expected to have shape:
        (B, T, D)

    We focus especially on the final token,
    because that is the representation being
    used to choose the next token.
    """

    activation = np.asarray(
        activation,
        dtype=np.float64,
    )

    gradient = np.asarray(
        gradient,
        dtype=np.float64,
    )

    if (
        activation.shape
        !=
        gradient.shape
    ):

        raise ValueError(
            "Activation and gradient shapes "
            "must match."
        )

    if activation.ndim != 3:

        raise ValueError(
            "Expected activation tensors with "
            "shape (batch, sequence, d_model)."
        )

    activation_last = (
        activation[
            0,
            -1,
            :
        ]
    )

    gradient_last = (
        gradient[
            0,
            -1,
            :
        ]
    )

    activation_norm = float(
        np.linalg.norm(
            activation_last
        )
    )

    gradient_norm = float(
        np.linalg.norm(
            gradient_last
        )
    )

    dot_product = float(
        np.dot(
            activation_last,
            gradient_last,
        )
    )

    denominator = (
        activation_norm
        *
        gradient_norm
    )

    if denominator > 0.0:

        cosine = float(
            dot_product
            /
            denominator
        )

    else:

        cosine = 0.0

    elementwise_product = (
        activation_last
        *
        gradient_last
    )

    return {
        "shape":
            [
                int(
                    dimension
                )

                for dimension
                in activation.shape
            ],

        "activation_last_token_l2":
            activation_norm,

        "gradient_last_token_l2":
            gradient_norm,

        "gradient_dot_activation":
            dot_product,

        "absolute_gradient_dot_activation":
            float(
                abs(
                    dot_product
                )
            ),

        "gradient_activation_cosine":
            cosine,

        "mean_abs_gradient_x_activation":
            float(
                np.mean(
                    np.abs(
                        elementwise_product
                    )
                )
            ),

        "max_abs_gradient_x_activation":
            float(
                np.max(
                    np.abs(
                        elementwise_product
                    )
                )
            ),

        "gradient_finite":
            bool(
                np.all(
                    np.isfinite(
                        gradient
                    )
                )
            ),

        "activation_finite":
            bool(
                np.all(
                    np.isfinite(
                        activation
                    )
                )
            ),
    }


# ============================================================
# One Transformer-block backward pass with captures
# ============================================================

def backward_block_with_trace(
    *,
    block,
    grad_output,
):
    """
    Reproduce TransformerBlock.backward()
    while exposing selected internal gradients.

    No parameter values are changed.

    Parameter gradient buffers ARE temporarily
    populated, exactly as during ordinary
    backpropagation. The laboratory zeros them
    before and after the experiment.
    """

    # ========================================================
    # output =
    # attention_residual + mlp_output
    # ========================================================

    grad_block_output = np.array(
        grad_output,
        copy=True,
    )

    grad_mlp_output = np.array(
        grad_output,
        copy=True,
    )

    # ========================================================
    # MLP backward
    # ========================================================

    grad_normalized_attention = (
        block.mlp.backward(
            grad_mlp_output
        )
    )

    grad_attention_residual_norm = (
        block.norm2.backward(
            grad_normalized_attention
        )
    )

    # ========================================================
    # Merge second residual
    # ========================================================

    grad_attention_residual = (
        grad_output
        +
        grad_attention_residual_norm
    )

    grad_attention_output = np.array(
        grad_attention_residual,
        copy=True,
    )

    # ========================================================
    # Attention backward
    # ========================================================

    grad_normalized_x = (
        block.attention.backward(
            grad_attention_output
        )
    )

    grad_x_norm = (
        block.norm1.backward(
            grad_normalized_x
        )
    )

    # ========================================================
    # Merge first residual
    # ========================================================

    grad_input = (
        grad_attention_residual
        +
        grad_x_norm
    )

    return (
        grad_input,
        {
            "block_output":
                grad_block_output,

            "mlp_output":
                grad_mlp_output,

            "attention_output":
                grad_attention_output,

            "block_input":
                np.array(
                    grad_input,
                    copy=True,
                ),
        },
    )


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
        "L6.6 — GRADIENT ATTRIBUTION"
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
        "(local sensitivity / correlation)"
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
    # Load L6.2 artifact
    # ========================================================

    (
        scorer_path,
        scorer_artifact,
    ) = load_json(
        args.scorer_artifact
    )

    scorer_checkpoint_hash = (
        scorer_artifact[
            "model"
        ][
            "checkpoint_sha256"
        ]
    )

    if (
        scorer_checkpoint_hash
        !=
        runtime.checkpoint_sha256
    ):

        raise RuntimeError(
            "L6.2 artifact checkpoint does "
            "not match current checkpoint."
        )

    if (
        scorer_artifact[
            "prompt"
        ]
        !=
        args.prompt
    ):

        raise RuntimeError(
            "L6.2 artifact prompt does not "
            "match current prompt."
        )

    shared_prefix_ids = [
        int(
            token_id
        )

        for token_id
        in scorer_artifact[
            "discriminative_analysis"
        ][
            "shared_prefix_token_ids"
        ]
    ]

    appended_token_ids = [
        int(
            token_id
        )

        for token_id
        in args.append_token_id
    ]

    if (
        shared_prefix_ids
        !=
        appended_token_ids
    ):

        raise RuntimeError(
            "Appended token IDs must match "
            "the L6.2 shared discriminative "
            "prefix exactly."
        )

    target_records = (
        extract_discriminative_targets(
            scorer_artifact
        )
    )

    if len(
        target_records
    ) < 2:

        raise RuntimeError(
            "Need at least two discriminative "
            "targets."
        )

    if not (
        0
        <=
        args.preferred_target_index
        <
        len(
            target_records
        )
    ):

        raise ValueError(
            "Preferred target index invalid."
        )

    if not (
        0
        <=
        args.comparison_target_index
        <
        len(
            target_records
        )
    ):

        raise ValueError(
            "Comparison target index invalid."
        )

    if (
        args.preferred_target_index
        ==
        args.comparison_target_index
    ):

        raise ValueError(
            "Preferred and comparison targets "
            "must be different."
        )

    preferred = (
        target_records[
            args.preferred_target_index
        ]
    )

    comparison = (
        target_records[
            args.comparison_target_index
        ]
    )

    preferred_token_id = int(
        preferred[
            "token_id"
        ]
    )

    comparison_token_id = int(
        comparison[
            "token_id"
        ]
    )

    # ========================================================
    # Exact context
    # ========================================================

    prompt_token_ids = (
        runtime.encode_prompt(
            args.prompt
        )
    )

    context_token_ids = (
        list(
            prompt_token_ids
        )
        +
        appended_token_ids
    )

    input_ids = np.asarray(
        [
            context_token_ids
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Clear any old gradient state
    # ========================================================

    runtime.model.zero_grad()

    # ========================================================
    # Forward with trace
    #
    # IMPORTANT:
    # Do not call another forward pass before backward.
    # The component caches created here are the caches used
    # by the backward pass below.
    # ========================================================

    logits, trace = (
        runtime.forward_with_trace(
            input_ids
        )
    )

    next_token_logits = (
        logits[
            0,
            -1,
            :
        ]
    )

    preferred_stats = (
        token_statistics(
            next_token_logits,
            preferred_token_id,
        )
    )

    comparison_stats = (
        token_statistics(
            next_token_logits,
            comparison_token_id,
        )
    )

    actual_logit_margin = float(
        preferred_stats[
            "logit"
        ]
        -
        comparison_stats[
            "logit"
        ]
    )

    # ========================================================
    # Objective
    #
    # margin =
    # logit(preferred)
    # -
    # logit(comparison)
    #
    # Since:
    # logits = hidden @ E^T
    #
    # dMargin/dHidden =
    # E[preferred] - E[comparison]
    # ========================================================

    embedding_weight = (
        runtime
        .model
        .input_embedding
        .token_embedding
        .weight
    )

    grad_final_hidden = np.zeros_like(
        trace[
            "final_layernorm"
        ],
        dtype=np.float32,
    )

    grad_final_hidden[
        0,
        -1,
        :
    ] = (
        embedding_weight[
            preferred_token_id
        ]
        -
        embedding_weight[
            comparison_token_id
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    # ========================================================
    # Attribution records
    # ========================================================

    activation_gradients = {
        "final_layernorm":
            np.array(
                grad_final_hidden,
                copy=True,
            )
    }

    # ========================================================
    # Final LayerNorm backward
    # ========================================================

    grad_x = (
        runtime
        .model
        .final_norm
        .backward(
            grad_final_hidden
        )
    )

    # grad_x is now gradient wrt block_4.output.

    # ========================================================
    # Walk transformer blocks backward
    # ========================================================

    component_gradient_records = {}

    for block_index in reversed(
        range(
            runtime.model.num_layers
        )
    ):

        block = (
            runtime.model.blocks[
                block_index
            ]
        )

        prefix = (
            f"block_{block_index}"
        )

        # Current grad_x is gradient with respect
        # to this block's output.
        activation_gradients[
            f"{prefix}.output"
        ] = np.array(
            grad_x,
            copy=True,
        )

        (
            grad_input,
            internal_grads,
        ) = backward_block_with_trace(
            block=block,
            grad_output=grad_x,
        )

        component_gradient_records[
            prefix
        ] = {
            "attention_output":
                np.array(
                    internal_grads[
                        "attention_output"
                    ],
                    copy=True,
                ),

            "mlp_output":
                np.array(
                    internal_grads[
                        "mlp_output"
                    ],
                    copy=True,
                ),
        }

        grad_x = (
            grad_input
        )

    # After the first block, grad_x is gradient
    # wrt embedding_output.
    activation_gradients[
        "embedding_output"
    ] = np.array(
        grad_x,
        copy=True,
    )

    # ========================================================
    # Build attribution summaries
    # ========================================================

    stage_order = [
        "embedding_output"
    ]

    for block_index in range(
        runtime.model.num_layers
    ):

        stage_order.append(
            f"block_{block_index}.output"
        )

    stage_order.append(
        "final_layernorm"
    )

    residual_attributions = []

    for stage_name in stage_order:

        statistics = (
            attribution_statistics(
                activation=(
                    trace[
                        stage_name
                    ]
                ),

                gradient=(
                    activation_gradients[
                        stage_name
                    ]
                ),
            )
        )

        residual_attributions.append(
            {
                "stage":
                    stage_name,

                **statistics,
            }
        )

    # ========================================================
    # Attention + MLP subcomponent attribution
    # ========================================================

    subcomponent_attributions = []

    for block_index in range(
        runtime.model.num_layers
    ):

        prefix = (
            f"block_{block_index}"
        )

        for component_name in (
            "attention_output",
            "mlp_output",
        ):

            trace_key = (
                f"{prefix}."
                f"{component_name}"
            )

            gradient = (
                component_gradient_records[
                    prefix
                ][
                    component_name
                ]
            )

            statistics = (
                attribution_statistics(
                    activation=(
                        trace[
                            trace_key
                        ]
                    ),

                    gradient=(
                        gradient
                    ),
                )
            )

            subcomponent_attributions.append(
                {
                    "layer":
                        int(
                            block_index
                        ),

                    "component":
                        component_name,

                    "component_id":
                        (
                            f"{prefix}."
                            f"{component_name}"
                        ),

                    **statistics,
                }
            )

    # ========================================================
    # Rankings
    # ========================================================

    residual_by_gradient_norm = sorted(
        residual_attributions,
        key=lambda record: (
            record[
                "gradient_last_token_l2"
            ]
        ),
        reverse=True,
    )

    components_by_gradient_norm = sorted(
        subcomponent_attributions,
        key=lambda record: (
            record[
                "gradient_last_token_l2"
            ]
        ),
        reverse=True,
    )

    components_by_abs_dot = sorted(
        subcomponent_attributions,
        key=lambda record: (
            record[
                "absolute_gradient_dot_activation"
            ]
        ),
        reverse=True,
    )

    # ========================================================
    # Save raw gradient artifact
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

    raw_arrays = {
        "context_token_ids":
            np.asarray(
                context_token_ids,
                dtype=np.int64,
            )
    }

    for (
        stage_name,
        gradient,
    ) in activation_gradients.items():

        key = (
            "grad__"
            +
            stage_name.replace(
                ".",
                "__",
            )
        )

        raw_arrays[
            key
        ] = np.array(
            gradient,
            copy=True,
        )

    for (
        prefix,
        component_records,
    ) in component_gradient_records.items():

        for (
            component_name,
            gradient,
        ) in component_records.items():

            key = (
                "grad__"
                +
                prefix
                +
                "__"
                +
                component_name
            )

            raw_arrays[
                key
            ] = np.array(
                gradient,
                copy=True,
            )

    np.savez(
        npz_path,
        **raw_arrays,
    )

    npz_hash = (
        sha256_file(
            npz_path
        )
    )

    # ========================================================
    # Integrity
    # ========================================================

    all_residual_finite = all(
        record[
            "gradient_finite"
        ]
        and
        record[
            "activation_finite"
        ]

        for record
        in residual_attributions
    )

    all_components_finite = all(
        record[
            "gradient_finite"
        ]
        and
        record[
            "activation_finite"
        ]

        for record
        in subcomponent_attributions
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "scorer_checkpoint_matches":
            (
                scorer_checkpoint_hash
                ==
                runtime.checkpoint_sha256
            ),

        "scorer_prompt_matches":
            (
                scorer_artifact[
                    "prompt"
                ]
                ==
                args.prompt
            ),

        "shared_prefix_matches":
            (
                shared_prefix_ids
                ==
                appended_token_ids
            ),

        "preferred_and_comparison_distinct":
            (
                preferred_token_id
                !=
                comparison_token_id
            ),

        "residual_gradients_finite":
            bool(
                all_residual_finite
            ),

        "component_gradients_finite":
            bool(
                all_components_finite
            ),

        "expected_residual_stage_count":
            (
                len(
                    residual_attributions
                )
                ==
                (
                    runtime.model.num_layers
                    +
                    2
                )
            ),

        "expected_subcomponent_count":
            (
                len(
                    subcomponent_attributions
                )
                ==
                (
                    runtime.model.num_layers
                    *
                    2
                )
            ),
    }

    all_checks_passed = all(
        checks.values()
    )

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
                "gradient_attribution",

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

            "context_token_ids":
                context_token_ids,
        },

        "objective": {
            "type":
                "next_token_logit_margin",

            "preferred": {
                "token_id":
                    preferred_token_id,

                "token_text":
                    token_text(
                        runtime.tokenizer,
                        preferred_token_id,
                    ),

                "rank":
                    preferred_stats[
                        "rank"
                    ],

                "probability":
                    preferred_stats[
                        "probability"
                    ],

                "logit":
                    preferred_stats[
                        "logit"
                    ],
            },

            "comparison": {
                "token_id":
                    comparison_token_id,

                "token_text":
                    token_text(
                        runtime.tokenizer,
                        comparison_token_id,
                    ),

                "rank":
                    comparison_stats[
                        "rank"
                    ],

                "probability":
                    comparison_stats[
                        "probability"
                    ],

                "logit":
                    comparison_stats[
                        "logit"
                    ],
            },

            "logit_margin":
                actual_logit_margin,
        },

        "attribution": {
            "interpretation":
                (
                    "Gradient quantities represent "
                    "local sensitivity of the preferred "
                    "minus comparison logit margin. "
                    "They are correlational/local and "
                    "do not establish causality."
                ),

            "residual_stream":
                residual_attributions,

            "subcomponents":
                subcomponent_attributions,

            "residual_ranking_by_gradient_norm":
                [
                    {
                        "rank":
                            int(
                                rank
                            ),

                        "stage":
                            item[
                                "stage"
                            ],

                        "gradient_last_token_l2":
                            item[
                                "gradient_last_token_l2"
                            ],

                        "gradient_dot_activation":
                            item[
                                "gradient_dot_activation"
                            ],
                    }

                    for (
                        rank,
                        item,
                    ) in enumerate(
                        residual_by_gradient_norm,
                        start=1,
                    )
                ],

            "component_ranking_by_gradient_norm":
                [
                    {
                        "rank":
                            int(
                                rank
                            ),

                        "component_id":
                            item[
                                "component_id"
                            ],

                        "gradient_last_token_l2":
                            item[
                                "gradient_last_token_l2"
                            ],

                        "gradient_dot_activation":
                            item[
                                "gradient_dot_activation"
                            ],
                    }

                    for (
                        rank,
                        item,
                    ) in enumerate(
                        components_by_gradient_norm,
                        start=1,
                    )
                ],

            "component_ranking_by_abs_gradient_dot_activation":
                [
                    {
                        "rank":
                            int(
                                rank
                            ),

                        "component_id":
                            item[
                                "component_id"
                            ],

                        "absolute_gradient_dot_activation":
                            item[
                                "absolute_gradient_dot_activation"
                            ],

                        "gradient_dot_activation":
                            item[
                                "gradient_dot_activation"
                            ],
                    }

                    for (
                        rank,
                        item,
                    ) in enumerate(
                        components_by_abs_dot,
                        start=1,
                    )
                ],
        },

        "artifacts": {
            "scorer_artifact":
                str(
                    scorer_path
                ),

            "json":
                str(
                    json_path
                ),

            "raw_gradient_npz":
                str(
                    npz_path
                ),

            "raw_gradient_npz_sha256":
                npz_hash,
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
    # Clear gradient buffers after measurement
    # ========================================================

    runtime.model.zero_grad()

    # ========================================================
    # Display
    # ========================================================

    print(
        "Objective"
    )

    print(
        "---------"
    )

    print(
        (
            f"preferred: "
            f"{repr(token_text(runtime.tokenizer, preferred_token_id))} "
            f"(id={preferred_token_id})"
        )
    )

    print(
        (
            f"comparison: "
            f"{repr(token_text(runtime.tokenizer, comparison_token_id))} "
            f"(id={comparison_token_id})"
        )
    )

    print()

    print(
        f"Preferred rank: "
        f"{preferred_stats['rank']:,}"
    )

    print(
        f"Comparison rank: "
        f"{comparison_stats['rank']:,}"
    )

    print(
        f"Logit margin: "
        f"{actual_logit_margin:+.8f}"
    )

    # ========================================================
    # Residual table
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "RESIDUAL-STREAM GRADIENT ATTRIBUTION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        (
            f"{'Stage':<24}"
            f"{'Grad L2':>14}"
            f"{'Act L2':>14}"
            f"{'Grad·Act':>14}"
            f"{'Cosine':>12}"
        )
    )

    print(
        "-" * 78
    )

    for item in residual_attributions:

        print(
            (
                f"{item['stage']:<24}"
                f"{item['gradient_last_token_l2']:>14.6f}"
                f"{item['activation_last_token_l2']:>14.6f}"
                f"{item['gradient_dot_activation']:>+14.6f}"
                f"{item['gradient_activation_cosine']:>12.6f}"
            )
        )

    # ========================================================
    # Component table
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "ATTENTION / MLP LOCAL SENSITIVITY"
    )

    print(
        "=" * 76
    )

    print()

    print(
        (
            f"{'Component':<32}"
            f"{'Grad L2':>14}"
            f"{'Grad·Act':>14}"
            f"{'|Grad·Act|':>14}"
        )
    )

    print(
        "-" * 76
    )

    for item in subcomponent_attributions:

        print(
            (
                f"{item['component_id']:<32}"
                f"{item['gradient_last_token_l2']:>14.6f}"
                f"{item['gradient_dot_activation']:>+14.6f}"
                f"{item['absolute_gradient_dot_activation']:>14.6f}"
            )
        )

    # ========================================================
    # Ranked candidates
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "TOP COMPONENTS BY GRADIENT NORM"
    )

    print(
        "=" * 76
    )

    print()

    for rank, item in enumerate(
        components_by_gradient_norm,
        start=1,
    ):

        print(
            (
                f"#{rank:<2} "
                f"{item['component_id']:<32} "
                f"grad_l2="
                f"{item['gradient_last_token_l2']:.6f}"
            )
        )

    print()
    print(
        "=" * 76
    )

    print(
        "TOP COMPONENTS BY |GRADIENT · ACTIVATION|"
    )

    print(
        "=" * 76
    )

    print()

    for rank, item in enumerate(
        components_by_abs_dot,
        start=1,
    ):

        print(
            (
                f"#{rank:<2} "
                f"{item['component_id']:<32} "
                f"dot="
                f"{item['gradient_dot_activation']:+.6f}"
            )
        )

    # ========================================================
    # Checks
    # ========================================================

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
        npz_hash
    )

    if not all_checks_passed:

        raise RuntimeError(
            "L6.6 gradient attribution failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.6 GRADIENT "
        "ATTRIBUTION PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()