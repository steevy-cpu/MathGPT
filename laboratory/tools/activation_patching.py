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

from laboratory.tools.attention_head_ablation import (
    load_json,
    extract_targets,
    measure_targets,
    distribution_metrics,
)

from model.causal_attention import (
    causal_attention,
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.10 activation_patching — "
            "patch one attention head's "
            "last-token activation from a "
            "donor context into a recipient "
            "context."
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
        "--donor-prompt",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--recipient-prompt",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--donor-append-token-id",
        action="append",
        type=int,
        default=[],
    )

    parser.add_argument(
        "--recipient-append-token-id",
        action="append",
        type=int,
        default=[],
    )

    parser.add_argument(
        "--recipient-scorer-artifact",
        required=True,
        type=str,
        help=(
            "Validated L6.2 scorer artifact "
            "for the recipient context."
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
        "--layer",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--head",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# Vector measurements
# ============================================================

def vector_comparison(
    donor,
    recipient,
):

    donor = np.asarray(
        donor,
        dtype=np.float64,
    )

    recipient = np.asarray(
        recipient,
        dtype=np.float64,
    )

    if donor.shape != recipient.shape:

        raise ValueError(
            "Donor and recipient activation "
            "vectors must have identical shape."
        )

    donor_norm = float(
        np.linalg.norm(
            donor
        )
    )

    recipient_norm = float(
        np.linalg.norm(
            recipient
        )
    )

    difference = (
        donor
        -
        recipient
    )

    difference_norm = float(
        np.linalg.norm(
            difference
        )
    )

    dot_product = float(
        np.dot(
            donor,
            recipient,
        )
    )

    denominator = (
        donor_norm
        *
        recipient_norm
    )

    if denominator > 0.0:

        cosine_similarity = float(
            dot_product
            /
            denominator
        )

    else:

        cosine_similarity = 0.0

    return {
        "dimension":
            int(
                donor.size
            ),

        "donor_l2":
            donor_norm,

        "recipient_l2":
            recipient_norm,

        "l2_distance":
            difference_norm,

        "dot_product":
            dot_product,

        "cosine_similarity":
            cosine_similarity,

        "exactly_equal":
            bool(
                np.array_equal(
                    donor,
                    recipient,
                )
            ),
    }


# ============================================================
# Raw attention forward exposing heads
# ============================================================

def attention_forward_expose_heads(
    *,
    attention,
    x,
):

    batch_size = int(
        x.shape[
            0
        ]
    )

    sequence_length = int(
        x.shape[
            1
        ]
    )

    # ========================================================
    # Q / K / V
    # ========================================================

    query = (
        attention
        .query_projection
        .forward(
            x
        )
    )

    key = (
        attention
        .key_projection
        .forward(
            x
        )
    )

    value = (
        attention
        .value_projection
        .forward(
            x
        )
    )

    # ========================================================
    # Split heads
    # ========================================================

    query_heads = (
        attention.split_heads(
            query
        )
    )

    key_heads = (
        attention.split_heads(
            key
        )
    )

    value_heads = (
        attention.split_heads(
            value
        )
    )

    # ========================================================
    # Flatten B × H
    # ========================================================

    query_flat = (
        query_heads.reshape(
            batch_size
            *
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    key_flat = (
        key_heads.reshape(
            batch_size
            *
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    value_flat = (
        value_heads.reshape(
            batch_size
            *
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    # ========================================================
    # Causal attention
    # ========================================================

    (
        attention_output_flat,
        attention_weights_flat,
    ) = causal_attention(
        query_flat,
        key_flat,
        value_flat,
    )

    # ========================================================
    # Restore head dimension
    # ========================================================

    attention_output_heads = (
        attention_output_flat.reshape(
            batch_size,
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
    )

    attention_weights = (
        attention_weights_flat.reshape(
            batch_size,
            attention.num_heads,
            sequence_length,
            sequence_length,
        )
    )

    # ========================================================
    # Combine + Wo
    # ========================================================

    combined = (
        attention.combine_heads(
            attention_output_heads
        )
    )

    output = (
        attention
        .output_projection
        .forward(
            combined
        )
    )

    return (
        output.astype(
            np.float32,
            copy=False,
        ),

        attention_weights,

        attention_output_heads,
    )


# ============================================================
# Capture one head's final-position activation
# ============================================================

def capture_head_last_token(
    *,
    runtime,
    token_ids,
    target_layer,
    target_head,
):

    model = (
        runtime.model
    )

    token_ids = np.asarray(
        token_ids,
        dtype=np.int64,
    )

    if token_ids.ndim == 1:

        token_ids = token_ids[
            None,
            :
        ]

    x = (
        model
        .input_embedding
        .forward(
            token_ids
        )
    )

    # ========================================================
    # Walk up to target layer
    # ========================================================

    for (
        block_index,
        block,
    ) in enumerate(
        model.blocks
    ):

        normalized_x = (
            block.norm1.forward(
                x
            )
        )

        if (
            block_index
            ==
            target_layer
        ):

            (
                _,
                attention_weights,
                attention_output_heads,
            ) = (
                attention_forward_expose_heads(
                    attention=(
                        block.attention
                    ),

                    x=(
                        normalized_x
                    ),
                )
            )

            head_vector = np.array(
                attention_output_heads[
                    0,
                    target_head,
                    -1,
                    :
                ],
                copy=True,
            )

            attention_row = np.array(
                attention_weights[
                    0,
                    target_head,
                    -1,
                    :
                ],
                copy=True,
            )

            return (
                head_vector,
                attention_row,
            )

        # ====================================================
        # Ordinary block propagation before target layer
        # ====================================================

        (
            attention_output,
            _,
        ) = (
            block
            .attention
            .forward(
                normalized_x
            )
        )

        attention_residual = (
            x
            +
            attention_output
        )

        normalized_attention = (
            block.norm2.forward(
                attention_residual
            )
        )

        mlp_output = (
            block.mlp.forward(
                normalized_attention
            )
        )

        x = (
            attention_residual
            +
            mlp_output
        )

    raise RuntimeError(
        "Target layer was not reached."
    )


# ============================================================
# Forward with optional head activation patch
# ============================================================

def forward_with_head_patch(
    *,
    runtime,
    token_ids,
    target_layer,
    target_head,
    patch_vector=None,
):

    model = (
        runtime.model
    )

    token_ids = np.asarray(
        token_ids,
        dtype=np.int64,
    )

    if token_ids.ndim == 1:

        token_ids = token_ids[
            None,
            :
        ]

    if patch_vector is not None:

        patch_vector = np.asarray(
            patch_vector,
            dtype=np.float32,
        )

        expected_shape = (
            model
            .blocks[
                target_layer
            ]
            .attention
            .head_dim,
        )

        if (
            patch_vector.shape
            !=
            expected_shape
        ):

            raise ValueError(
                "Patch vector has incorrect "
                f"shape {patch_vector.shape}; "
                f"expected {expected_shape}."
            )

    x = (
        model
        .input_embedding
        .forward(
            token_ids
        )
    )

    patch_record = None

    # ========================================================
    # Transformer blocks
    # ========================================================

    for (
        block_index,
        block,
    ) in enumerate(
        model.blocks
    ):

        normalized_x = (
            block.norm1.forward(
                x
            )
        )

        # ====================================================
        # Target attention layer
        # ====================================================

        if (
            block_index
            ==
            target_layer
        ):

            (
                _,
                attention_weights,
                attention_output_heads,
            ) = (
                attention_forward_expose_heads(
                    attention=(
                        block.attention
                    ),

                    x=(
                        normalized_x
                    ),
                )
            )

            original_vector = np.array(
                attention_output_heads[
                    0,
                    target_head,
                    -1,
                    :
                ],
                copy=True,
            )

            patched_vector = np.array(
                original_vector,
                copy=True,
            )

            if patch_vector is not None:

                attention_output_heads[
                    0,
                    target_head,
                    -1,
                    :
                ] = (
                    patch_vector
                )

                patched_vector = np.array(
                    patch_vector,
                    copy=True,
                )

            combined = (
                block
                .attention
                .combine_heads(
                    attention_output_heads
                )
            )

            attention_output = (
                block
                .attention
                .output_projection
                .forward(
                    combined
                )
            )

            patch_record = {
                "layer":
                    int(
                        target_layer
                    ),

                "head":
                    int(
                        target_head
                    ),

                "head_id":
                    (
                        f"L{target_layer}"
                        f"H{target_head}"
                    ),

                "position":
                    int(
                        token_ids.shape[
                            1
                        ]
                        -
                        1
                    ),

                "applied":
                    bool(
                        patch_vector
                        is not None
                    ),

                "original_last_token_l2":
                    float(
                        np.linalg.norm(
                            original_vector.astype(
                                np.float64
                            )
                        )
                    ),

                "patched_last_token_l2":
                    float(
                        np.linalg.norm(
                            patched_vector.astype(
                                np.float64
                            )
                        )
                    ),

                "patch_l2_distance":
                    float(
                        np.linalg.norm(
                            (
                                patched_vector
                                -
                                original_vector
                            ).astype(
                                np.float64
                            )
                        )
                    ),

                "attention_row_sum":
                    float(
                        np.sum(
                            attention_weights[
                                0,
                                target_head,
                                -1,
                                :
                            ],
                            dtype=np.float64,
                        )
                    ),
            }

        # ====================================================
        # Ordinary layer
        # ====================================================

        else:

            (
                attention_output,
                _,
            ) = (
                block
                .attention
                .forward(
                    normalized_x
                )
            )

        # ====================================================
        # Residual + MLP
        # ====================================================

        attention_residual = (
            x
            +
            attention_output
        )

        normalized_attention = (
            block.norm2.forward(
                attention_residual
            )
        )

        mlp_output = (
            block.mlp.forward(
                normalized_attention
            )
        )

        x = (
            attention_residual
            +
            mlp_output
        )

    # ========================================================
    # Final LayerNorm
    # ========================================================

    hidden = (
        model.final_norm.forward(
            x
        )
    )

    # ========================================================
    # Tied output projection
    # ========================================================

    embedding_weight = (
        model
        .input_embedding
        .token_embedding
        .weight
    )

    logits = (
        hidden
        @
        embedding_weight.T
    )

    return (
        logits.astype(
            np.float32,
            copy=False,
        ),

        patch_record,
    )


# ============================================================
# L6.2 baseline cross-check
# ============================================================

def scorer_crosscheck(
    *,
    scorer_artifact,
    measured_targets,
):

    scorer_candidates = (
        scorer_artifact[
            "discriminative_analysis"
        ][
            "candidates"
        ]
    )

    scorer_by_token = {}

    for candidate in scorer_candidates:

        if not candidate.get(
            "has_discriminative_token",
            False,
        ):

            continue

        scorer_by_token[
            int(
                candidate[
                    "token_id"
                ]
            )
        ] = candidate

    details = []

    for name in (
        "preferred",
        "comparison",
    ):

        measured = (
            measured_targets[
                name
            ]
        )

        token_id = int(
            measured[
                "token_id"
            ]
        )

        scorer = (
            scorer_by_token[
                token_id
            ]
        )

        rank_match = (
            int(
                measured[
                    "rank"
                ]
            )
            ==
            int(
                scorer[
                    "rank"
                ]
            )
        )

        probability_difference = abs(
            float(
                measured[
                    "probability"
                ]
            )
            -
            float(
                scorer[
                    "probability"
                ]
            )
        )

        details.append(
            {
                "token_id":
                    token_id,

                "token_text":
                    measured[
                        "token_text"
                    ],

                "rank_match":
                    bool(
                        rank_match
                    ),

                "probability_difference":
                    float(
                        probability_difference
                    ),

                "probability_match":
                    bool(
                        probability_difference
                        <=
                        1e-12
                    ),
            }
        )

    passed = all(
        (
            detail[
                "rank_match"
            ]
            and
            detail[
                "probability_match"
            ]
        )

        for detail
        in details
    )

    return {
        "passed":
            bool(
                passed
            ),

        "details":
            details,
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
        "L6.10 — ACTIVATION PATCHING"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Permission level: P3"
    )

    print(
        "Evidence level: E3 "
        "(causal activation intervention)"
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
    # Validate layer/head
    # ========================================================

    if not (
        0
        <=
        args.layer
        <
        runtime.model.num_layers
    ):

        raise ValueError(
            f"Layer must be between 0 and "
            f"{runtime.model.num_layers - 1}."
        )

    if not (
        0
        <=
        args.head
        <
        runtime.model.num_heads
    ):

        raise ValueError(
            f"Head must be between 0 and "
            f"{runtime.model.num_heads - 1}."
        )

    head_id = (
        f"L{args.layer}H{args.head}"
    )

    # ========================================================
    # Recipient scorer artifact
    # ========================================================

    (
        scorer_path,
        scorer_artifact,
    ) = load_json(
        args.recipient_scorer_artifact
    )

    if (
        scorer_artifact[
            "model"
        ][
            "checkpoint_sha256"
        ]
        !=
        runtime.checkpoint_sha256
    ):

        raise RuntimeError(
            "Recipient L6.2 checkpoint "
            "does not match current checkpoint."
        )

    if (
        scorer_artifact[
            "prompt"
        ]
        !=
        args.recipient_prompt
    ):

        raise RuntimeError(
            "Recipient L6.2 prompt mismatch."
        )

    expected_recipient_prefix = [
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

    recipient_append_ids = [
        int(
            token_id
        )

        for token_id
        in args.recipient_append_token_id
    ]

    if (
        expected_recipient_prefix
        !=
        recipient_append_ids
    ):

        raise RuntimeError(
            "Recipient appended token IDs "
            "do not match the L6.2 "
            "discriminative context."
        )

    target_records = (
        extract_targets(
            scorer_artifact
        )
    )

    if len(
        target_records
    ) < 2:

        raise RuntimeError(
            "At least two recipient "
            "targets are required."
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
            "Preferred and comparison "
            "targets must differ."
        )

    preferred_token_id = int(
        target_records[
            args.preferred_target_index
        ][
            "token_id"
        ]
    )

    comparison_token_id = int(
        target_records[
            args.comparison_target_index
        ][
            "token_id"
        ]
    )

    # ========================================================
    # Build exact donor context
    # ========================================================

    donor_prompt_ids = (
        runtime.encode_prompt(
            args.donor_prompt
        )
    )

    donor_append_ids = [
        int(
            token_id
        )

        for token_id
        in args.donor_append_token_id
    ]

    donor_context_ids = (
        list(
            donor_prompt_ids
        )
        +
        donor_append_ids
    )

    # ========================================================
    # Build exact recipient context
    # ========================================================

    recipient_prompt_ids = (
        runtime.encode_prompt(
            args.recipient_prompt
        )
    )

    recipient_context_ids = (
        list(
            recipient_prompt_ids
        )
        +
        recipient_append_ids
    )

    # ========================================================
    # Context validation
    # ========================================================

    if len(
        donor_context_ids
    ) == 0:

        raise ValueError(
            "Donor context is empty."
        )

    if len(
        recipient_context_ids
    ) == 0:

        raise ValueError(
            "Recipient context is empty."
        )

    if (
        len(
            donor_context_ids
        )
        >
        runtime.model.context_length
    ):

        raise ValueError(
            "Donor context exceeds "
            "model context length."
        )

    if (
        len(
            recipient_context_ids
        )
        >
        runtime.model.context_length
    ):

        raise ValueError(
            "Recipient context exceeds "
            "model context length."
        )

    donor_input_ids = np.asarray(
        [
            donor_context_ids
        ],
        dtype=np.int64,
    )

    recipient_input_ids = np.asarray(
        [
            recipient_context_ids
        ],
        dtype=np.int64,
    )

    # ========================================================
    # True recipient baseline
    # ========================================================

    baseline_logits = (
        runtime.model.forward(
            recipient_input_ids
        )
    )

    baseline_targets = (
        measure_targets(
            runtime=runtime,

            logits=baseline_logits,

            preferred_token_id=(
                preferred_token_id
            ),

            comparison_token_id=(
                comparison_token_id
            ),
        )
    )

    # ========================================================
    # L6.2 recipient cross-check
    # ========================================================

    scorer_check = (
        scorer_crosscheck(
            scorer_artifact=(
                scorer_artifact
            ),

            measured_targets=(
                baseline_targets
            ),
        )
    )

    # ========================================================
    # Capture donor activation
    # ========================================================

    (
        donor_vector,
        donor_attention_row,
    ) = capture_head_last_token(
        runtime=runtime,

        token_ids=donor_input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            args.head
        ),
    )

    # ========================================================
    # Capture recipient activation
    # ========================================================

    (
        recipient_vector,
        recipient_attention_row,
    ) = capture_head_last_token(
        runtime=runtime,

        token_ids=recipient_input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            args.head
        ),
    )

    vector_metrics = (
        vector_comparison(
            donor_vector,
            recipient_vector,
        )
    )

    # ========================================================
    # Control 1:
    # Manual forward without patch.
    # ========================================================

    (
        manual_logits,
        manual_record,
    ) = forward_with_head_patch(
        runtime=runtime,

        token_ids=recipient_input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            args.head
        ),

        patch_vector=None,
    )

    manual_exact = bool(
        np.array_equal(
            baseline_logits,
            manual_logits,
        )
    )

    manual_max_difference = float(
        np.max(
            np.abs(
                baseline_logits.astype(
                    np.float64
                )
                -
                manual_logits.astype(
                    np.float64
                )
            )
        )
    )

    # ========================================================
    # Control 2:
    # Self patch.
    #
    # Replace recipient activation with its own
    # captured activation.
    #
    # This MUST preserve logits exactly.
    # ========================================================

    (
        self_patch_logits,
        self_patch_record,
    ) = forward_with_head_patch(
        runtime=runtime,

        token_ids=recipient_input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            args.head
        ),

        patch_vector=(
            recipient_vector
        ),
    )

    self_patch_exact = bool(
        np.array_equal(
            baseline_logits,
            self_patch_logits,
        )
    )

    self_patch_max_difference = float(
        np.max(
            np.abs(
                baseline_logits.astype(
                    np.float64
                )
                -
                self_patch_logits.astype(
                    np.float64
                )
            )
        )
    )

    # ========================================================
    # Real donor → recipient patch
    # ========================================================

    (
        patched_logits,
        patch_record,
    ) = forward_with_head_patch(
        runtime=runtime,

        token_ids=recipient_input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            args.head
        ),

        patch_vector=(
            donor_vector
        ),
    )

    patched_targets = (
        measure_targets(
            runtime=runtime,

            logits=patched_logits,

            preferred_token_id=(
                preferred_token_id
            ),

            comparison_token_id=(
                comparison_token_id
            ),
        )
    )

    # ========================================================
    # Patch effect
    # ========================================================

    margin_delta = float(
        patched_targets[
            "logit_margin"
        ]
        -
        baseline_targets[
            "logit_margin"
        ]
    )

    preferred_rank_delta = int(
        patched_targets[
            "preferred"
        ][
            "rank"
        ]
        -
        baseline_targets[
            "preferred"
        ][
            "rank"
        ]
    )

    comparison_rank_delta = int(
        patched_targets[
            "comparison"
        ][
            "rank"
        ]
        -
        baseline_targets[
            "comparison"
        ][
            "rank"
        ]
    )

    distribution_change = (
        distribution_metrics(
            baseline_logits[
                0,
                -1,
                :
            ],

            patched_logits[
                0,
                -1,
                :
            ],
        )
    )

    # ========================================================
    # Self-patch / donor-recipient identity
    # ========================================================

    contexts_identical = bool(
        donor_context_ids
        ==
        recipient_context_ids
    )

    actual_patch_exact = bool(
        np.array_equal(
            baseline_logits,
            patched_logits,
        )
    )

    actual_patch_max_difference = float(
        np.max(
            np.abs(
                baseline_logits.astype(
                    np.float64
                )
                -
                patched_logits.astype(
                    np.float64
                )
            )
        )
    )

    # ========================================================
    # Raw artifact
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

    np.savez(
        npz_path,

        donor_context_token_ids=np.asarray(
            donor_context_ids,
            dtype=np.int64,
        ),

        recipient_context_token_ids=np.asarray(
            recipient_context_ids,
            dtype=np.int64,
        ),

        donor_head_last_token=np.asarray(
            donor_vector,
            dtype=np.float32,
        ),

        recipient_head_last_token=np.asarray(
            recipient_vector,
            dtype=np.float32,
        ),

        patch_difference=np.asarray(
            donor_vector
            -
            recipient_vector,
            dtype=np.float32,
        ),

        donor_attention_row=np.asarray(
            donor_attention_row,
            dtype=np.float32,
        ),

        recipient_attention_row=np.asarray(
            recipient_attention_row,
            dtype=np.float32,
        ),

        baseline_next_token_logits=np.asarray(
            baseline_logits[
                0,
                -1,
                :
            ],
            dtype=np.float32,
        ),

        patched_next_token_logits=np.asarray(
            patched_logits[
                0,
                -1,
                :
            ],
            dtype=np.float32,
        ),
    )

    npz_hash = (
        sha256_file(
            npz_path
        )
    )

    # ========================================================
    # Integrity
    # ========================================================

    measurements_finite = all(
        np.isfinite(
            value
        )

        for value in [
            baseline_targets[
                "logit_margin"
            ],

            patched_targets[
                "logit_margin"
            ],

            margin_delta,

            distribution_change[
                "kl_baseline_to_intervention"
            ],

            distribution_change[
                "probability_l1_distance"
            ],

            distribution_change[
                "logit_l2_difference"
            ],

            vector_metrics[
                "donor_l2"
            ],

            vector_metrics[
                "recipient_l2"
            ],

            vector_metrics[
                "l2_distance"
            ],
        ]
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "recipient_l6_2_crosscheck":
            bool(
                scorer_check[
                    "passed"
                ]
            ),

        "manual_unpatched_forward_matches_gpt":
            manual_exact,

        "self_patch_matches_gpt":
            self_patch_exact,

        "patch_record_created":
            (
                patch_record
                is not None
            ),

        "attention_rows_valid":
            (
                abs(
                    float(
                        np.sum(
                            donor_attention_row,
                            dtype=np.float64,
                        )
                    )
                    -
                    1.0
                )
                <=
                1e-5
                and
                abs(
                    float(
                        np.sum(
                            recipient_attention_row,
                            dtype=np.float64,
                        )
                    )
                    -
                    1.0
                )
                <=
                1e-5
            ),

        "measurements_finite":
            bool(
                measurements_finite
            ),
    }

    # ========================================================
    # Additional mandatory self-patch condition
    #
    # If donor and recipient contexts are identical,
    # the REAL patch must also be exact.
    # ========================================================

    if contexts_identical:

        checks[
            "identical_context_patch_is_exact"
        ] = (
            actual_patch_exact
        )

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
                "activation_patching",

            "version":
                LAB_TOOL_VERSION,

            "permission_level":
                "P3",

            "evidence_level":
                "E3",
        },

        "status":
            (
                "success"
                if all_checks_passed
                else "failed"
            ),

        "model":
            runtime.model_record(),

        "patch_site": {
            "layer":
                int(
                    args.layer
                ),

            "head":
                int(
                    args.head
                ),

            "head_id":
                head_id,

            "position":
                "last_token",

            "activation_space":
                (
                    "per-head causal-attention "
                    "output before combine_heads "
                    "and before Wo"
                ),

            "dimension":
                int(
                    donor_vector.size
                ),
        },

        "donor": {
            "prompt":
                args.donor_prompt,

            "prompt_token_ids":
                donor_prompt_ids,

            "appended_token_ids":
                donor_append_ids,

            "context_token_ids":
                donor_context_ids,
        },

        "recipient": {
            "prompt":
                args.recipient_prompt,

            "prompt_token_ids":
                recipient_prompt_ids,

            "appended_token_ids":
                recipient_append_ids,

            "context_token_ids":
                recipient_context_ids,
        },

        "targets": {
            "preferred_token_id":
                preferred_token_id,

            "preferred_token_text":
                token_text(
                    runtime.tokenizer,
                    preferred_token_id,
                ),

            "comparison_token_id":
                comparison_token_id,

            "comparison_token_text":
                token_text(
                    runtime.tokenizer,
                    comparison_token_id,
                ),
        },

        "activation_comparison":
            vector_metrics,

        "controls": {
            "contexts_identical":
                contexts_identical,

            "manual_unpatched": {
                "exact":
                    manual_exact,

                "max_absolute_difference":
                    manual_max_difference,

                "record":
                    manual_record,
            },

            "self_patch": {
                "exact":
                    self_patch_exact,

                "max_absolute_difference":
                    self_patch_max_difference,

                "record":
                    self_patch_record,
            },

            "actual_patch_exact":
                actual_patch_exact,

            "actual_patch_max_absolute_difference":
                actual_patch_max_difference,

            "recipient_l6_2_crosscheck":
                scorer_check,
        },

        "baseline":
            baseline_targets,

        "patched":
            patched_targets,

        "effect": {
            "logit_margin_delta":
                margin_delta,

            "preferred_rank_delta":
                preferred_rank_delta,

            "comparison_rank_delta":
                comparison_rank_delta,

            "preferred_still_wins":
                bool(
                    patched_targets[
                        "preferred_wins"
                    ]
                ),

            "distribution_change":
                distribution_change,
        },

        "patch_record":
            patch_record,

        "interpretation": {
            "scope":
                (
                    "The intervention tests whether "
                    "replacing this recipient head "
                    "activation with the donor head "
                    "activation changes the recipient "
                    "next-token behavior."
                ),

            "warning":
                (
                    "A patch effect demonstrates "
                    "causal dependence on the "
                    "transplanted activation for "
                    "these contexts. It does not "
                    "by itself establish a general "
                    "semantic function for the head."
                ),
        },

        "artifacts": {
            "recipient_scorer_artifact":
                str(
                    scorer_path
                ),

            "json":
                str(
                    json_path
                ),

            "raw_npz":
                str(
                    npz_path
                ),

            "raw_npz_sha256":
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
    # Display
    # ========================================================

    print(
        "Patch site"
    )

    print(
        "----------"
    )

    print(
        f"Head: {head_id}"
    )

    print(
        "Position: last token"
    )

    print(
        f"Activation dimension: "
        f"{donor_vector.size}"
    )

    print()

    # ========================================================
    # Contexts
    # ========================================================

    print(
        "Donor"
    )

    print(
        "-----"
    )

    print(
        repr(
            args.donor_prompt
        )
    )

    print(
        f"Context IDs: "
        f"{donor_context_ids}"
    )

    print()

    print(
        "Recipient"
    )

    print(
        "---------"
    )

    print(
        repr(
            args.recipient_prompt
        )
    )

    print(
        f"Context IDs: "
        f"{recipient_context_ids}"
    )

    print()

    # ========================================================
    # Activation comparison
    # ========================================================

    print(
        "=" * 76
    )

    print(
        "DONOR / RECIPIENT ACTIVATION COMPARISON"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Donor L2: "
        f"{vector_metrics['donor_l2']:.8f}"
    )

    print(
        f"Recipient L2: "
        f"{vector_metrics['recipient_l2']:.8f}"
    )

    print(
        f"L2 distance: "
        f"{vector_metrics['l2_distance']:.8f}"
    )

    print(
        f"Cosine similarity: "
        f"{vector_metrics['cosine_similarity']:.8f}"
    )

    print(
        f"Exactly equal: "
        f"{vector_metrics['exactly_equal']}"
    )

    # ========================================================
    # Controls
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PATCHING CONTROLS"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Manual unpatched exact: "
        f"{manual_exact}"
    )

    print(
        f"Manual max difference: "
        f"{manual_max_difference}"
    )

    print()

    print(
        f"Self-patch exact: "
        f"{self_patch_exact}"
    )

    print(
        f"Self-patch max difference: "
        f"{self_patch_max_difference}"
    )

    print()

    print(
        f"Donor/recipient contexts identical: "
        f"{contexts_identical}"
    )

    if contexts_identical:

        print(
            f"Actual donor patch exact: "
            f"{actual_patch_exact}"
        )

        print(
            f"Actual patch max difference: "
            f"{actual_patch_max_difference}"
        )

    # ========================================================
    # Behavioral result
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PATCH EFFECT"
    )

    print(
        "=" * 76
    )

    print()

    print(
        (
            f"{'':<18}"
            f"{'Baseline':>16}"
            f"{'Patched':>16}"
            f"{'Delta':>16}"
        )
    )

    print(
        "-" * 66
    )

    print(
        (
            f"{'Logit margin':<18}"
            f"{baseline_targets['logit_margin']:>+16.8f}"
            f"{patched_targets['logit_margin']:>+16.8f}"
            f"{margin_delta:>+16.8f}"
        )
    )

    print(
        (
            f"{'Preferred rank':<18}"
            f"{baseline_targets['preferred']['rank']:>16,}"
            f"{patched_targets['preferred']['rank']:>16,}"
            f"{preferred_rank_delta:>+16d}"
        )
    )

    print(
        (
            f"{'Comparison rank':<18}"
            f"{baseline_targets['comparison']['rank']:>16,}"
            f"{patched_targets['comparison']['rank']:>16,}"
            f"{comparison_rank_delta:>+16d}"
        )
    )

    print()

    print(
        "Distribution change:"
    )

    print(
        (
            f"  KL(base || patch): "
            f"{distribution_change['kl_baseline_to_intervention']:.8f}"
        )
    )

    print(
        (
            f"  probability L1: "
            f"{distribution_change['probability_l1_distance']:.8f}"
        )
    )

    print(
        (
            f"  logit L2: "
            f"{distribution_change['logit_l2_difference']:.8f}"
        )
    )

    print(
        (
            f"  top-1 changed: "
            f"{distribution_change['top1_changed']}"
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

    # ========================================================
    # Acceptance
    # ========================================================

    if not all_checks_passed:

        raise RuntimeError(
            "L6.10 activation patching failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.10 ACTIVATION "
        "PATCHING PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()