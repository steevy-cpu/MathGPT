import numpy as np

from gpt import GPT


VOCAB_SIZE = 7
CONTEXT_LENGTH = 3

D_MODEL = 4
NUM_LAYERS = 2
NUM_HEADS = 2
D_FF = 6

BATCH_SIZE = 1
SEQUENCE_LENGTH = 3

EPSILON = 1e-3


def scalar_loss(
    model,
    token_ids,
    grad_logits,
):
    """
    Artificial scalar loss:

        L =
        sum(
            GPT(token_ids)
            * grad_logits
        )
    """

    logits = model.forward(
        token_ids
    )

    return float(
        np.sum(
            logits
            * grad_logits
        )
    )


def numerical_gradient(
    tensor,
    index,
    loss_function,
):
    original_value = float(
        tensor[index]
    )

    tensor[index] = (
        original_value
        + EPSILON
    )

    loss_plus = (
        loss_function()
    )

    tensor[index] = (
        original_value
        - EPSILON
    )

    loss_minus = (
        loss_function()
    )

    tensor[index] = original_value

    return (
        loss_plus
        - loss_minus
    ) / (
        2.0
        * EPSILON
    )


def main():

    print()
    print(
        "Full GPT Backward Test"
    )

    print(
        "======================"
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

    rng = np.random.default_rng(
        123
    )

    grad_logits = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            VOCAB_SIZE,
        ),
    ).astype(
        np.float32
    )

    # ========================================================
    # Analytical forward + backward
    # ========================================================

    logits = model.forward(
        token_ids
    )

    model.zero_grad()

    model.backward(
        grad_logits
    )

    # Save gradients BEFORE numerical
    # forward passes overwrite caches.
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

    block0_q_gradient = (
        model
        .blocks[0]
        .attention
        .query_projection
        .dweight
        .copy()
    )

    block1_mlp_gradient = (
        model
        .blocks[1]
        .mlp
        .fc_in
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
        f"Token IDs shape: "
        f"{token_ids.shape}"
    )

    print(
        f"Logits shape: "
        f"{logits.shape}"
    )

    print(
        f"Embedding gradient shape: "
        f"{embedding_gradient.shape}"
    )

    # ========================================================
    # 1. Tied embedding gradient
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
            lambda: scalar_loss(
                model,
                token_ids,
                grad_logits,
            ),
        )
    )

    analytical_embedding = float(
        embedding_gradient[
            embedding_index
        ]
    )

    # ========================================================
    # 2. Position embedding
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
            lambda: scalar_loss(
                model,
                token_ids,
                grad_logits,
            ),
        )
    )

    analytical_position = float(
        position_gradient[
            position_index
        ]
    )

    # ========================================================
    # 3. Block 0 Attention Wq
    # ========================================================

    q_index = (
        1,
        2,
    )

    numerical_q = (
        numerical_gradient(
            model
            .blocks[0]
            .attention
            .query_projection
            .weight,
            q_index,
            lambda: scalar_loss(
                model,
                token_ids,
                grad_logits,
            ),
        )
    )

    analytical_q = float(
        block0_q_gradient[
            q_index
        ]
    )

    # ========================================================
    # 4. Block 1 MLP W1
    # ========================================================

    mlp_index = (
        2,
        4,
    )

    numerical_mlp = (
        numerical_gradient(
            model
            .blocks[1]
            .mlp
            .fc_in
            .weight,
            mlp_index,
            lambda: scalar_loss(
                model,
                token_ids,
                grad_logits,
            ),
        )
    )

    analytical_mlp = float(
        block1_mlp_gradient[
            mlp_index
        ]
    )

    # ========================================================
    # 5. Final LayerNorm gamma
    # ========================================================

    norm_index = 3

    numerical_norm = (
        numerical_gradient(
            model
            .final_norm
            .gamma,
            norm_index,
            lambda: scalar_loss(
                model,
                token_ids,
                grad_logits,
            ),
        )
    )

    analytical_norm = float(
        final_norm_gradient[
            norm_index
        ]
    )

    # ========================================================
    # Compare
    # ========================================================

    embedding_passed = np.isclose(
        numerical_embedding,
        analytical_embedding,
        atol=6e-3,
        rtol=6e-3,
    )

    position_passed = np.isclose(
        numerical_position,
        analytical_position,
        atol=6e-3,
        rtol=6e-3,
    )

    q_passed = np.isclose(
        numerical_q,
        analytical_q,
        atol=6e-3,
        rtol=6e-3,
    )

    mlp_passed = np.isclose(
        numerical_mlp,
        analytical_mlp,
        atol=6e-3,
        rtol=6e-3,
    )

    norm_passed = np.isclose(
        numerical_norm,
        analytical_norm,
        atol=6e-3,
        rtol=6e-3,
    )

    finite_passed = (
        np.all(
            np.isfinite(
                embedding_gradient
            )
        )
        and
        np.all(
            np.isfinite(
                position_gradient
            )
        )
        and
        np.all(
            np.isfinite(
                block0_q_gradient
            )
        )
        and
        np.all(
            np.isfinite(
                block1_mlp_gradient
            )
        )
        and
        np.all(
            np.isfinite(
                final_norm_gradient
            )
        )
    )

    # ========================================================
    # Display
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
            "Block 0 Attention Wq",
            numerical_q,
            analytical_q,
            q_passed,
        ),

        (
            "Block 1 MLP W1",
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
            f"{name}"
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

    all_gradients_passed = (
        embedding_passed
        and
        position_passed
        and
        q_passed
        and
        mlp_passed
        and
        norm_passed
    )

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"All representative gradients passed: "
        f"{all_gradients_passed}"
    )

    print(
        f"Finite gradients passed: "
        f"{finite_passed}"
    )

    if (
        all_gradients_passed
        and
        finite_passed
    ):

        print()

        print(
            "SUCCESS: Full GPT backward "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Full GPT backward test failed."
        )


if __name__ == "__main__":
    main()