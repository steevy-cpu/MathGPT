import numpy as np

from causal_attention import (
    causal_attention,
    causal_attention_backward,
)


BATCH_SIZE = 1
SEQUENCE_LENGTH = 3
HEAD_DIM = 2

EPSILON = 1e-3


def scalar_loss(
    query,
    key,
    value,
    grad_output,
):
    """
    Artificial scalar loss:

        L =
        sum(
            attention(Q,K,V)
            * grad_output
        )
    """

    (
        output,
        _,
    ) = causal_attention(
        query,
        key,
        value,
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
    Central finite difference for one
    selected tensor element.
    """

    original_value = float(
        tensor[
            index
        ]
    )

    tensor[
        index
    ] = (
        original_value
        + EPSILON
    )

    loss_plus = (
        loss_function()
    )

    tensor[
        index
    ] = (
        original_value
        - EPSILON
    )

    loss_minus = (
        loss_function()
    )

    tensor[
        index
    ] = (
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
        "Causal Attention Backward Test"
    )

    print(
        "=============================="
    )

    rng = np.random.default_rng(
        123
    )

    query = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        ),
    ).astype(
        np.float32
    )

    key = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        ),
    ).astype(
        np.float32
    )

    value = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        ),
    ).astype(
        np.float32
    )

    grad_output = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        ),
    ).astype(
        np.float32
    )

    # ========================================================
    # Forward
    # ========================================================

    (
        output,
        attention_weights,
    ) = causal_attention(
        query,
        key,
        value,
    )

    # ========================================================
    # Analytical backward
    # ========================================================

    (
        grad_query,
        grad_key,
        grad_value,
    ) = causal_attention_backward(
        query,
        key,
        value,
        attention_weights,
        grad_output,
    )

    print()
    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"dQ shape: "
        f"{grad_query.shape}"
    )

    print(
        f"dK shape: "
        f"{grad_key.shape}"
    )

    print(
        f"dV shape: "
        f"{grad_value.shape}"
    )

    # ========================================================
    # Numerical dQ
    # ========================================================

    query_index = (
        0,
        2,
        1,
    )

    numerical_dq = (
        numerical_gradient(
            query,
            query_index,
            lambda: scalar_loss(
                query,
                key,
                value,
                grad_output,
            ),
        )
    )

    analytical_dq = float(
        grad_query[
            query_index
        ]
    )

    # ========================================================
    # Numerical dK
    # ========================================================

    key_index = (
        0,
        1,
        0,
    )

    numerical_dk = (
        numerical_gradient(
            key,
            key_index,
            lambda: scalar_loss(
                query,
                key,
                value,
                grad_output,
            ),
        )
    )

    analytical_dk = float(
        grad_key[
            key_index
        ]
    )

    # ========================================================
    # Numerical dV
    # ========================================================

    value_index = (
        0,
        1,
        1,
    )

    numerical_dv = (
        numerical_gradient(
            value,
            value_index,
            lambda: scalar_loss(
                query,
                key,
                value,
                grad_output,
            ),
        )
    )

    analytical_dv = float(
        grad_value[
            value_index
        ]
    )

    # ========================================================
    # Gradient comparisons
    # ========================================================

    dq_passed = np.isclose(
        numerical_dq,
        analytical_dq,
        atol=2e-3,
        rtol=2e-3,
    )

    dk_passed = np.isclose(
        numerical_dk,
        analytical_dk,
        atol=2e-3,
        rtol=2e-3,
    )

    dv_passed = np.isclose(
        numerical_dv,
        analytical_dv,
        atol=2e-3,
        rtol=2e-3,
    )

    # ========================================================
    # Shape / dtype / finite checks
    # ========================================================

    shapes_passed = (
        grad_query.shape
        == query.shape
        and
        grad_key.shape
        == key.shape
        and
        grad_value.shape
        == value.shape
    )

    dtype_passed = (
        grad_query.dtype
        == np.float32
        and
        grad_key.dtype
        == np.float32
        and
        grad_value.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                grad_query
            )
        )
        and
        np.all(
            np.isfinite(
                grad_key
            )
        )
        and
        np.all(
            np.isfinite(
                grad_value
            )
        )
    )

    # ========================================================
    # Strong causal-gradient test
    #
    # Create a loss ONLY from output position 0.
    #
    # Position 0 cannot see positions 1 or 2.
    #
    # Therefore future K/V positions should
    # receive zero gradient.
    # ========================================================

    causal_grad_output = (
        np.zeros_like(
            grad_output
        )
    )

    causal_grad_output[
        0,
        0,
        :
    ] = np.array(
        [
            1.0,
            -0.5,
        ],
        dtype=np.float32,
    )

    (
        causal_dq,
        causal_dk,
        causal_dv,
    ) = causal_attention_backward(
        query,
        key,
        value,
        attention_weights,
        causal_grad_output,
    )

    future_key_zero = (
        np.allclose(
            causal_dk[
                0,
                1:,
                :
            ],
            0.0,
            atol=1e-7,
        )
    )

    future_value_zero = (
        np.allclose(
            causal_dv[
                0,
                1:,
                :
            ],
            0.0,
            atol=1e-7,
        )
    )

    causal_gradient_passed = (
        future_key_zero
        and
        future_value_zero
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

    print()

    print(
        f"dQ numerical:  "
        f"{numerical_dq:.6f}"
    )

    print(
        f"dQ analytical: "
        f"{analytical_dq:.6f}"
    )

    print()

    print(
        f"dK numerical:  "
        f"{numerical_dk:.6f}"
    )

    print(
        f"dK analytical: "
        f"{analytical_dk:.6f}"
    )

    print()

    print(
        f"dV numerical:  "
        f"{numerical_dv:.6f}"
    )

    print(
        f"dV analytical: "
        f"{analytical_dv:.6f}"
    )

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"dQ gradient passed: "
        f"{dq_passed}"
    )

    print(
        f"dK gradient passed: "
        f"{dk_passed}"
    )

    print(
        f"dV gradient passed: "
        f"{dv_passed}"
    )

    print(
        f"Shapes passed: "
        f"{shapes_passed}"
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
        f"Future K gradients blocked: "
        f"{future_key_zero}"
    )

    print(
        f"Future V gradients blocked: "
        f"{future_value_zero}"
    )

    print(
        f"Causal gradient test passed: "
        f"{causal_gradient_passed}"
    )

    if (
        dq_passed
        and dk_passed
        and dv_passed
        and shapes_passed
        and dtype_passed
        and finite_passed
        and causal_gradient_passed
    ):

        print()

        print(
            "SUCCESS: Causal attention "
            "backward tests passed."
        )

    else:

        raise AssertionError(
            "Causal attention backward "
            "test failed."
        )


if __name__ == "__main__":
    main()