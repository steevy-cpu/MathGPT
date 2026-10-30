import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from model.gpt import GPT

from training.loss import (
    cross_entropy_loss,
    cross_entropy_loss_and_backward,
)


# ============================================================
# Tiny GPT for numerical gradient checking
# ============================================================

VOCAB_SIZE = 7
CONTEXT_LENGTH = 3

D_MODEL = 4
NUM_LAYERS = 2
NUM_HEADS = 2
D_FF = 6

EPSILON = 1e-3


def numerical_gradient(
    tensor,
    index,
    loss_function,
):
    """
    Central finite difference.
    """

    original_value = float(
        tensor[index]
    )

    tensor[index] = (
        original_value
        + EPSILON
    )

    loss_plus = float(
        loss_function()
    )

    tensor[index] = (
        original_value
        - EPSILON
    )

    loss_minus = float(
        loss_function()
    )

    tensor[index] = (
        original_value
    )

    return (
        loss_plus
        - loss_minus
    ) / (
        2.0
        * EPSILON
    )


def model_loss(
    model,
    token_ids,
    targets,
):
    """
    Real GPT next-token loss.
    """

    logits = model.forward(
        token_ids
    )

    return cross_entropy_loss(
        logits,
        targets,
    )


def collect_parameter_gradients(
    model,
):
    """
    Collect every unique parameter gradient.

    Weight tying does NOT add a second
    parameter matrix. The token embedding
    gradient already contains both:

        input lookup contribution
        +
        output projection contribution
    """

    gradients = []

    # ========================================================
    # Embeddings
    # ========================================================

    gradients.append(
        (
            "token_embedding.weight",
            model
            .input_embedding
            .token_embedding
            .dweight,
        )
    )

    gradients.append(
        (
            "position_embedding.weight",
            model
            .input_embedding
            .position_embedding
            .dweight,
        )
    )

    # ========================================================
    # Transformer blocks
    # ========================================================

    for (
        block_index,
        block,
    ) in enumerate(
        model.blocks
    ):

        prefix = (
            f"block_{block_index}"
        )

        # LayerNorm 1
        gradients.append(
            (
                f"{prefix}.norm1.gamma",
                block.norm1.dgamma,
            )
        )

        gradients.append(
            (
                f"{prefix}.norm1.beta",
                block.norm1.dbeta,
            )
        )

        # Attention projections
        projections = [
            (
                "query",
                block.attention
                .query_projection,
            ),
            (
                "key",
                block.attention
                .key_projection,
            ),
            (
                "value",
                block.attention
                .value_projection,
            ),
            (
                "output",
                block.attention
                .output_projection,
            ),
        ]

        for (
            projection_name,
            projection,
        ) in projections:

            gradients.append(
                (
                    (
                        f"{prefix}.attention."
                        f"{projection_name}.weight"
                    ),
                    projection.dweight,
                )
            )

            if (
                projection.dbias
                is not None
            ):

                gradients.append(
                    (
                        (
                            f"{prefix}.attention."
                            f"{projection_name}.bias"
                        ),
                        projection.dbias,
                    )
                )

        # LayerNorm 2
        gradients.append(
            (
                f"{prefix}.norm2.gamma",
                block.norm2.dgamma,
            )
        )

        gradients.append(
            (
                f"{prefix}.norm2.beta",
                block.norm2.dbeta,
            )
        )

        # MLP
        gradients.append(
            (
                f"{prefix}.mlp.fc_in.weight",
                block.mlp.fc_in.dweight,
            )
        )

        gradients.append(
            (
                f"{prefix}.mlp.fc_in.bias",
                block.mlp.fc_in.dbias,
            )
        )

        gradients.append(
            (
                f"{prefix}.mlp.fc_out.weight",
                block.mlp.fc_out.dweight,
            )
        )

        gradients.append(
            (
                f"{prefix}.mlp.fc_out.bias",
                block.mlp.fc_out.dbias,
            )
        )

    # ========================================================
    # Final LayerNorm
    # ========================================================

    gradients.append(
        (
            "final_norm.gamma",
            model.final_norm.dgamma,
        )
    )

    gradients.append(
        (
            "final_norm.beta",
            model.final_norm.dbeta,
        )
    )

    return gradients


def main():

    print()
    print(
        "GPT End-to-End Backward Test"
    )

    print(
        "============================"
    )

    model = GPT(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        use_bias=True,
        epsilon=1e-5,
        tie_embeddings=True,
        seed=42,
    )

    # ========================================================
    # Real next-token structure
    #
    # X:
    #     [2, 4, 2]
    #
    # Y:
    #     [4, 2, 5]
    #
    # First two positions are actual shifts.
    # Last target represents the next token
    # after the context.
    # ========================================================

    token_ids = np.array(
        [
            [
                2,
                4,
                2,
            ]
        ],
        dtype=np.int64,
    )

    targets = np.array(
        [
            [
                4,
                2,
                5,
            ]
        ],
        dtype=np.int64,
    )

    # ========================================================
    # REAL forward + loss + backward
    # ========================================================

    logits = model.forward(
        token_ids
    )

    (
        loss,
        grad_logits,
    ) = cross_entropy_loss_and_backward(
        logits,
        targets,
    )

    model.zero_grad()

    model.backward(
        grad_logits
    )

    # Save analytical gradients before
    # numerical forward passes overwrite caches.
    embedding_gradient = (
        model
        .input_embedding
        .token_embedding
        .dweight
        .copy()
    )

    position_gradient = (
        model
        .input_embedding
        .position_embedding
        .dweight
        .copy()
    )

    attention_gradient = (
        model
        .blocks[0]
        .attention
        .output_projection
        .dweight
        .copy()
    )

    mlp_gradient = (
        model
        .blocks[1]
        .mlp
        .fc_out
        .dweight
        .copy()
    )

    final_norm_gradient = (
        model
        .final_norm
        .dgamma
        .copy()
    )

    print()
    print(
        f"Token IDs: "
        f"{token_ids}"
    )

    print(
        f"Targets:   "
        f"{targets}"
    )

    print()

    print(
        f"Logits shape: "
        f"{logits.shape}"
    )

    print(
        f"Loss: "
        f"{loss:.6f}"
    )

    print(
        f"dLogits shape: "
        f"{grad_logits.shape}"
    )

    # ========================================================
    # Loss / dLogits structural checks
    # ========================================================

    loss_finite_passed = (
        np.isfinite(
            loss
        )
    )

    grad_logits_finite_passed = (
        np.all(
            np.isfinite(
                grad_logits
            )
        )
    )

    grad_logits_shape_passed = (
        grad_logits.shape
        == logits.shape
    )

    grad_logits_zero_sum_passed = (
        np.allclose(
            np.sum(
                grad_logits,
                axis=-1,
            ),
            0.0,
            atol=1e-6,
        )
    )

    # ========================================================
    # Numerical gradient 1:
    # tied token embedding
    #
    # This is especially important because the
    # same matrix participates in BOTH ends
    # of the model.
    # ========================================================

    embedding_index = (
        2,
        1,
    )

    numerical_embedding = (
        numerical_gradient(
            model
            .input_embedding
            .token_embedding
            .weight,
            embedding_index,
            lambda: model_loss(
                model,
                token_ids,
                targets,
            ),
        )
    )

    analytical_embedding = float(
        embedding_gradient[
            embedding_index
        ]
    )

    # ========================================================
    # Numerical gradient 2:
    # positional embedding
    # ========================================================

    position_index = (
        1,
        2,
    )

    numerical_position = (
        numerical_gradient(
            model
            .input_embedding
            .position_embedding
            .weight,
            position_index,
            lambda: model_loss(
                model,
                token_ids,
                targets,
            ),
        )
    )

    analytical_position = float(
        position_gradient[
            position_index
        ]
    )

    # ========================================================
    # Numerical gradient 3:
    # Block 0 attention output projection
    # ========================================================

    attention_index = (
        1,
        2,
    )

    numerical_attention = (
        numerical_gradient(
            model
            .blocks[0]
            .attention
            .output_projection
            .weight,
            attention_index,
            lambda: model_loss(
                model,
                token_ids,
                targets,
            ),
        )
    )

    analytical_attention = float(
        attention_gradient[
            attention_index
        ]
    )

    # ========================================================
    # Numerical gradient 4:
    # Block 1 MLP output projection
    # ========================================================

    mlp_index = (
        4,
        1,
    )

    numerical_mlp = (
        numerical_gradient(
            model
            .blocks[1]
            .mlp
            .fc_out
            .weight,
            mlp_index,
            lambda: model_loss(
                model,
                token_ids,
                targets,
            ),
        )
    )

    analytical_mlp = float(
        mlp_gradient[
            mlp_index
        ]
    )

    # ========================================================
    # Numerical gradient 5:
    # Final LayerNorm gamma
    # ========================================================

    norm_index = 3

    numerical_norm = (
        numerical_gradient(
            model
            .final_norm
            .gamma,
            norm_index,
            lambda: model_loss(
                model,
                token_ids,
                targets,
            ),
        )
    )

    analytical_norm = float(
        final_norm_gradient[
            norm_index
        ]
    )

    # ========================================================
    # Compare numerical vs analytical
    # ========================================================

    embedding_passed = np.isclose(
        numerical_embedding,
        analytical_embedding,
        atol=8e-3,
        rtol=8e-3,
    )

    position_passed = np.isclose(
        numerical_position,
        analytical_position,
        atol=8e-3,
        rtol=8e-3,
    )

    attention_passed = np.isclose(
        numerical_attention,
        analytical_attention,
        atol=8e-3,
        rtol=8e-3,
    )

    mlp_passed = np.isclose(
        numerical_mlp,
        analytical_mlp,
        atol=8e-3,
        rtol=8e-3,
    )

    norm_passed = np.isclose(
        numerical_norm,
        analytical_norm,
        atol=8e-3,
        rtol=8e-3,
    )

    # ========================================================
    # Audit EVERY stored gradient
    # ========================================================

    all_gradients = (
        collect_parameter_gradients(
            model
        )
    )

    total_gradient_scalars = sum(
        gradient.size
        for _, gradient
        in all_gradients
    )

    parameter_count = (
        model.parameter_count()
    )

    gradient_count_passed = (
        total_gradient_scalars
        == parameter_count
    )

    all_stored_gradients_finite = all(
        np.all(
            np.isfinite(
                gradient
            )
        )
        for _, gradient
        in all_gradients
    )

    total_gradient_squared = sum(
        float(
            np.sum(
                gradient.astype(
                    np.float64
                ) ** 2
            )
        )
        for _, gradient
        in all_gradients
    )

    global_gradient_norm = (
        np.sqrt(
            total_gradient_squared
        )
    )

    nonzero_gradient_passed = (
        global_gradient_norm
        > 0.0
    )

    # ========================================================
    # Display numerical comparisons
    # ========================================================

    print()
    print(
        "Numerical Gradient Checks"
    )

    print(
        "-------------------------"
    )

    comparisons = [
        (
            "Tied embedding",
            numerical_embedding,
            analytical_embedding,
            embedding_passed,
        ),

        (
            "Position embedding",
            numerical_position,
            analytical_position,
            position_passed,
        ),

        (
            "Block 0 Attention Wo",
            numerical_attention,
            analytical_attention,
            attention_passed,
        ),

        (
            "Block 1 MLP W2",
            numerical_mlp,
            analytical_mlp,
            mlp_passed,
        ),

        (
            "Final LayerNorm gamma",
            numerical_norm,
            analytical_norm,
            norm_passed,
        ),
    ]

    for (
        name,
        numerical,
        analytical,
        passed,
    ) in comparisons:

        print()

        print(
            name
        )

        print(
            f"Numerical:  "
            f"{numerical:.6f}"
        )

        print(
            f"Analytical: "
            f"{analytical:.6f}"
        )

        print(
            f"Passed:     "
            f"{passed}"
        )

    all_numerical_passed = (
        embedding_passed
        and
        position_passed
        and
        attention_passed
        and
        mlp_passed
        and
        norm_passed
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print(
        "Full Gradient Audit"
    )

    print(
        "-------------------"
    )

    print()

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Gradient scalars stored: "
        f"{total_gradient_scalars:,}"
    )

    print(
        f"Global gradient norm: "
        f"{global_gradient_norm:.6f}"
    )

    print()

    print(
        f"Loss finite: "
        f"{loss_finite_passed}"
    )

    print(
        f"dLogits shape passed: "
        f"{grad_logits_shape_passed}"
    )

    print(
        f"dLogits finite: "
        f"{grad_logits_finite_passed}"
    )

    print(
        f"dLogits rows sum to zero: "
        f"{grad_logits_zero_sum_passed}"
    )

    print(
        f"All numerical gradients passed: "
        f"{all_numerical_passed}"
    )

    print(
        f"Gradient count = parameter count: "
        f"{gradient_count_passed}"
    )

    print(
        f"All stored gradients finite: "
        f"{all_stored_gradients_finite}"
    )

    print(
        f"Global gradient non-zero: "
        f"{nonzero_gradient_passed}"
    )

    all_passed = (
        loss_finite_passed
        and
        grad_logits_shape_passed
        and
        grad_logits_finite_passed
        and
        grad_logits_zero_sum_passed
        and
        all_numerical_passed
        and
        gradient_count_passed
        and
        all_stored_gradients_finite
        and
        nonzero_gradient_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: END-TO-END GPT "
            "BACKPROPAGATION VALIDATED."
        )

    else:

        print(
            "FAILURE: One or more end-to-end "
            "gradient checks failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()