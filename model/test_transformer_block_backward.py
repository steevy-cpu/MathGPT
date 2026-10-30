import numpy as np

from transformer_block import (
    TransformerBlock
)


BATCH_SIZE = 1
SEQUENCE_LENGTH = 3

D_MODEL = 4
NUM_HEADS = 2
D_FF = 6

EPSILON = 1e-3


def scalar_loss(
    block,
    x,
    grad_output,
):
    """
    Artificial scalar loss:

        L =
        sum(
            TransformerBlock(x)
            * grad_output
        )
    """

    (
        output,
        _,
    ) = block.forward(
        x
    )

    return float(
        np.sum(
            output
            * grad_output
        )
    )


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


def main():

    print()
    print(
        "Transformer Block Backward Test"
    )

    print(
        "==============================="
    )

    rng = np.random.default_rng(
        123
    )

    block = TransformerBlock(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        use_bias=True,
        epsilon=1e-5,
        seed=42,
    )

    # Non-trivial LayerNorm parameters
    block.norm1.gamma[:] = np.array(
        [
            0.9,
            1.1,
            0.8,
            1.2,
        ],
        dtype=np.float32,
    )

    block.norm2.gamma[:] = np.array(
        [
            1.2,
            0.85,
            1.05,
            0.95,
        ],
        dtype=np.float32,
    )

    x = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            D_MODEL,
        ),
    ).astype(
        np.float32
    )

    grad_output = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            D_MODEL,
        ),
    ).astype(
        np.float32
    )

    # ========================================================
    # Analytical forward + backward
    # ========================================================

    (
        output,
        attention_weights,
    ) = block.forward(
        x
    )

    grad_input = block.backward(
        grad_output
    )

    analytical_dx = (
        grad_input.copy()
    )

    analytical_norm1_gamma = (
        block.norm1.dgamma.copy()
    )

    analytical_norm2_beta = (
        block.norm2.dbeta.copy()
    )

    analytical_attention_wq = (
        block
        .attention
        .query_projection
        .dweight
        .copy()
    )

    analytical_mlp_w1 = (
        block
        .mlp
        .fc_in
        .dweight
        .copy()
    )

    print()
    print(
        f"Input shape: "
        f"{x.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"Attention shape: "
        f"{attention_weights.shape}"
    )

    print(
        f"dInput shape: "
        f"{grad_input.shape}"
    )

    # ========================================================
    # Numerical dX
    # ========================================================

    x_index = (
        0,
        1,
        2,
    )

    numerical_dx = numerical_gradient(
        x,
        x_index,
        lambda: scalar_loss(
            block,
            x,
            grad_output,
        ),
    )

    expected_dx = float(
        analytical_dx[
            x_index
        ]
    )

    # ========================================================
    # Numerical norm1 gamma
    # ========================================================

    norm1_gamma_index = 2

    numerical_norm1_gamma = (
        numerical_gradient(
            block.norm1.gamma,
            norm1_gamma_index,
            lambda: scalar_loss(
                block,
                x,
                grad_output,
            ),
        )
    )

    expected_norm1_gamma = float(
        analytical_norm1_gamma[
            norm1_gamma_index
        ]
    )

    # ========================================================
    # Numerical norm2 beta
    # ========================================================

    norm2_beta_index = 1

    numerical_norm2_beta = (
        numerical_gradient(
            block.norm2.beta,
            norm2_beta_index,
            lambda: scalar_loss(
                block,
                x,
                grad_output,
            ),
        )
    )

    expected_norm2_beta = float(
        analytical_norm2_beta[
            norm2_beta_index
        ]
    )

    # ========================================================
    # Numerical attention Wq
    # ========================================================

    attention_wq_index = (
        1,
        2,
    )

    numerical_attention_wq = (
        numerical_gradient(
            block
            .attention
            .query_projection
            .weight,
            attention_wq_index,
            lambda: scalar_loss(
                block,
                x,
                grad_output,
            ),
        )
    )

    expected_attention_wq = float(
        analytical_attention_wq[
            attention_wq_index
        ]
    )

    # ========================================================
    # Numerical MLP W1
    # ========================================================

    mlp_w1_index = (
        2,
        4,
    )

    numerical_mlp_w1 = (
        numerical_gradient(
            block
            .mlp
            .fc_in
            .weight,
            mlp_w1_index,
            lambda: scalar_loss(
                block,
                x,
                grad_output,
            ),
        )
    )

    expected_mlp_w1 = float(
        analytical_mlp_w1[
            mlp_w1_index
        ]
    )

    # ========================================================
    # Comparisons
    # ========================================================

    dx_passed = np.isclose(
        numerical_dx,
        expected_dx,
        atol=4e-3,
        rtol=4e-3,
    )

    norm1_gamma_passed = (
        np.isclose(
            numerical_norm1_gamma,
            expected_norm1_gamma,
            atol=4e-3,
            rtol=4e-3,
        )
    )

    norm2_beta_passed = (
        np.isclose(
            numerical_norm2_beta,
            expected_norm2_beta,
            atol=4e-3,
            rtol=4e-3,
        )
    )

    attention_wq_passed = (
        np.isclose(
            numerical_attention_wq,
            expected_attention_wq,
            atol=4e-3,
            rtol=4e-3,
        )
    )

    mlp_w1_passed = (
        np.isclose(
            numerical_mlp_w1,
            expected_mlp_w1,
            atol=4e-3,
            rtol=4e-3,
        )
    )

    shape_passed = (
        grad_input.shape
        == x.shape
    )

    dtype_passed = (
        grad_input.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                grad_input
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_norm1_gamma
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_norm2_beta
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_attention_wq
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_mlp_w1
            )
        )
    )

    # ========================================================
    # Strong causal-gradient test
    #
    # Loss depends only on output position 0.
    #
    # Future input positions should receive
    # zero gradient through causal attention.
    #
    # HOWEVER:
    # the residual at position 0 only connects
    # to position 0 as well, so future positions
    # still receive zero total gradient.
    # ========================================================

    causal_grad_output = np.zeros_like(
        grad_output
    )

    causal_grad_output[
        0,
        0,
        :
    ] = np.array(
        [
            1.0,
            -0.5,
            0.25,
            -0.75,
        ],
        dtype=np.float32,
    )

    # Need a fresh forward because backward
    # uses the latest caches.
    block.forward(
        x
    )

    causal_grad_input = (
        block.backward(
            causal_grad_output
        )
    )

    future_input_zero = (
        np.allclose(
            causal_grad_input[
                0,
                1:,
                :
            ],
            0.0,
            atol=1e-6,
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
            "dX",
            numerical_dx,
            expected_dx,
            dx_passed,
        ),

        (
            "Norm1 dGamma",
            numerical_norm1_gamma,
            expected_norm1_gamma,
            norm1_gamma_passed,
        ),

        (
            "Norm2 dBeta",
            numerical_norm2_beta,
            expected_norm2_beta,
            norm2_beta_passed,
        ),

        (
            "Attention dWq",
            numerical_attention_wq,
            expected_attention_wq,
            attention_wq_passed,
        ),

        (
            "MLP dW1",
            numerical_mlp_w1,
            expected_mlp_w1,
            mlp_w1_passed,
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
            f"{name} numerical:  "
            f"{numerical:.6f}"
        )

        print(
            f"{name} analytical: "
            f"{analytical:.6f}"
        )

        print(
            f"{name} passed:     "
            f"{passed}"
        )

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Input shape passed: "
        f"{shape_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"Finite gradients passed: "
        f"{finite_passed}"
    )

    print(
        f"Future input gradients blocked: "
        f"{future_input_zero}"
    )

    all_gradients_passed = (
        dx_passed
        and
        norm1_gamma_passed
        and
        norm2_beta_passed
        and
        attention_wq_passed
        and
        mlp_w1_passed
    )

    print(
        f"All numerical gradients passed: "
        f"{all_gradients_passed}"
    )

    if (
        all_gradients_passed
        and shape_passed
        and dtype_passed
        and finite_passed
        and future_input_zero
    ):

        print()

        print(
            "SUCCESS: Transformer block "
            "backward tests passed."
        )

    else:

        raise AssertionError(
            "Transformer block backward "
            "test failed."
        )


if __name__ == "__main__":
    main()