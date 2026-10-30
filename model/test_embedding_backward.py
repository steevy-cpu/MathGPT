import numpy as np

from input_embedding import (
    InputEmbedding
)


VOCAB_SIZE = 6
CONTEXT_LENGTH = 4
D_MODEL = 3

BATCH_SIZE = 2
SEQUENCE_LENGTH = 3

EPSILON = 1e-3


def scalar_loss(
    embedding,
    token_ids,
    grad_output,
):
    """
    L =
        sum(
            InputEmbedding(token_ids)
            * grad_output
        )
    """

    output = embedding.forward(
        token_ids
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
    original_value = float(
        tensor[index]
    )

    tensor[index] = (
        original_value
        + EPSILON
    )

    loss_plus = loss_function()

    tensor[index] = (
        original_value
        - EPSILON
    )

    loss_minus = loss_function()

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
        "Embedding Backward Test"
    )

    print(
        "======================="
    )

    embedding = InputEmbedding(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        seed=42,
    )

    # Token 2 appears THREE times.
    # Token 1 appears TWO times.
    token_ids = np.array(
        [
            [2, 1, 2],
            [3, 2, 1],
        ],
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        123
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
    # Analytical backward
    # ========================================================

    output = embedding.forward(
        token_ids
    )

    embedding.backward(
        grad_output
    )

    token_gradient = (
        embedding
        .token_embedding
        .dweight
        .copy()
    )

    position_gradient = (
        embedding
        .position_embedding
        .dweight
        .copy()
    )

    print()
    print(
        f"Token IDs shape: "
        f"{token_ids.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"Token gradient shape: "
        f"{token_gradient.shape}"
    )

    print(
        f"Position gradient shape: "
        f"{position_gradient.shape}"
    )

    # ========================================================
    # Test repeated-token accumulation
    # ========================================================

    token_2_expected = np.sum(
        grad_output[
            token_ids == 2
        ],
        axis=0,
        dtype=np.float32,
    )

    token_2_actual = (
        token_gradient[
            2
        ]
    )

    repeated_token_passed = (
        np.allclose(
            token_2_actual,
            token_2_expected,
            atol=1e-6,
        )
    )

    # ========================================================
    # Position-gradient accumulation
    #
    # Position 1 appears once in every batch item.
    # ========================================================

    position_1_expected = np.sum(
        grad_output[
            :,
            1,
            :
        ],
        axis=0,
        dtype=np.float32,
    )

    position_1_actual = (
        position_gradient[
            1
        ]
    )

    position_sum_passed = (
        np.allclose(
            position_1_actual,
            position_1_expected,
            atol=1e-6,
        )
    )

    # ========================================================
    # Numerical token embedding gradient
    # ========================================================

    token_weight_index = (
        2,
        1,
    )

    numerical_token_gradient = (
        numerical_gradient(
            embedding
            .token_embedding
            .weight,
            token_weight_index,
            lambda: scalar_loss(
                embedding,
                token_ids,
                grad_output,
            ),
        )
    )

    analytical_token_gradient = float(
        token_gradient[
            token_weight_index
        ]
    )

    token_numerical_passed = (
        np.isclose(
            numerical_token_gradient,
            analytical_token_gradient,
            atol=2e-3,
            rtol=2e-3,
        )
    )

    # ========================================================
    # Numerical position embedding gradient
    # ========================================================

    position_weight_index = (
        1,
        2,
    )

    numerical_position_gradient = (
        numerical_gradient(
            embedding
            .position_embedding
            .weight,
            position_weight_index,
            lambda: scalar_loss(
                embedding,
                token_ids,
                grad_output,
            ),
        )
    )

    analytical_position_gradient = float(
        position_gradient[
            position_weight_index
        ]
    )

    position_numerical_passed = (
        np.isclose(
            numerical_position_gradient,
            analytical_position_gradient,
            atol=2e-3,
            rtol=2e-3,
        )
    )

    # ========================================================
    # Unused rows should remain zero
    # ========================================================

    unused_token_passed = (
        np.allclose(
            token_gradient[
                5
            ],
            0.0,
            atol=1e-7,
        )
    )

    unused_position_passed = (
        np.allclose(
            position_gradient[
                3
            ],
            0.0,
            atol=1e-7,
        )
    )

    # ========================================================
    # Other checks
    # ========================================================

    dtype_passed = (
        token_gradient.dtype
        == np.float32
        and
        position_gradient.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                token_gradient
            )
        )
        and
        np.all(
            np.isfinite(
                position_gradient
            )
        )
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Repeated Token Test"
    )

    print(
        "-------------------"
    )

    print(
        "Token 2 expected gradient:"
    )

    print(
        token_2_expected
    )

    print(
        "Token 2 actual gradient:"
    )

    print(
        token_2_actual
    )

    print()

    print(
        "Numerical Gradient Checks"
    )

    print(
        "-------------------------"
    )

    print()

    print(
        f"Token numerical:  "
        f"{numerical_token_gradient:.6f}"
    )

    print(
        f"Token analytical: "
        f"{analytical_token_gradient:.6f}"
    )

    print()

    print(
        f"Position numerical:  "
        f"{numerical_position_gradient:.6f}"
    )

    print(
        f"Position analytical: "
        f"{analytical_position_gradient:.6f}"
    )

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Repeated token accumulation passed: "
        f"{repeated_token_passed}"
    )

    print(
        f"Position accumulation passed: "
        f"{position_sum_passed}"
    )

    print(
        f"Token numerical gradient passed: "
        f"{token_numerical_passed}"
    )

    print(
        f"Position numerical gradient passed: "
        f"{position_numerical_passed}"
    )

    print(
        f"Unused token row zero: "
        f"{unused_token_passed}"
    )

    print(
        f"Unused position row zero: "
        f"{unused_position_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"Finite gradients passed: "
        f"{finite_passed}"
    )

    all_passed = (
        repeated_token_passed
        and
        position_sum_passed
        and
        token_numerical_passed
        and
        position_numerical_passed
        and
        unused_token_passed
        and
        unused_position_passed
        and
        dtype_passed
        and
        finite_passed
    )

    if all_passed:

        print()

        print(
            "SUCCESS: Embedding backward "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Embedding backward test failed."
        )


if __name__ == "__main__":
    main()