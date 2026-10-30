import numpy as np

from multi_head_attention import (
    MultiHeadAttention
)


BATCH_SIZE = 1
SEQUENCE_LENGTH = 3

D_MODEL = 4
NUM_HEADS = 2

EPSILON = 1e-3


def scalar_loss(
    layer,
    x,
    grad_output,
):
    """
    Artificial scalar loss:

        L =
        sum(
            MHA(x)
            * grad_output
        )
    """

    (
        output,
        _,
    ) = layer.forward(
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
    ] = original_value

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
        "Multi-Head Attention Backward Test"
    )

    print(
        "=================================="
    )

    rng = np.random.default_rng(
        123
    )

    attention = MultiHeadAttention(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        use_bias=True,
        seed=42,
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
    ) = attention.forward(
        x
    )

    grad_input = attention.backward(
        grad_output
    )

    analytical_dx = (
        grad_input.copy()
    )

    analytical_dwq = (
        attention
        .query_projection
        .dweight
        .copy()
    )

    analytical_dwk = (
        attention
        .key_projection
        .dweight
        .copy()
    )

    analytical_dwv = (
        attention
        .value_projection
        .dweight
        .copy()
    )

    analytical_dwo = (
        attention
        .output_projection
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
        f"{analytical_dx.shape}"
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
            attention,
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
    # Numerical dWq
    # ========================================================

    wq_index = (
        1,
        2,
    )

    numerical_dwq = numerical_gradient(
        attention
        .query_projection
        .weight,
        wq_index,
        lambda: scalar_loss(
            attention,
            x,
            grad_output,
        ),
    )

    expected_dwq = float(
        analytical_dwq[
            wq_index
        ]
    )

    # ========================================================
    # Numerical dWk
    # ========================================================

    wk_index = (
        2,
        1,
    )

    numerical_dwk = numerical_gradient(
        attention
        .key_projection
        .weight,
        wk_index,
        lambda: scalar_loss(
            attention,
            x,
            grad_output,
        ),
    )

    expected_dwk = float(
        analytical_dwk[
            wk_index
        ]
    )

    # ========================================================
    # Numerical dWv
    # ========================================================

    wv_index = (
        0,
        3,
    )

    numerical_dwv = numerical_gradient(
        attention
        .value_projection
        .weight,
        wv_index,
        lambda: scalar_loss(
            attention,
            x,
            grad_output,
        ),
    )

    expected_dwv = float(
        analytical_dwv[
            wv_index
        ]
    )

    # ========================================================
    # Numerical dWo
    # ========================================================

    wo_index = (
        3,
        0,
    )

    numerical_dwo = numerical_gradient(
        attention
        .output_projection
        .weight,
        wo_index,
        lambda: scalar_loss(
            attention,
            x,
            grad_output,
        ),
    )

    expected_dwo = float(
        analytical_dwo[
            wo_index
        ]
    )

    # ========================================================
    # Comparisons
    # ========================================================

    dx_passed = np.isclose(
        numerical_dx,
        expected_dx,
        atol=3e-3,
        rtol=3e-3,
    )

    dwq_passed = np.isclose(
        numerical_dwq,
        expected_dwq,
        atol=3e-3,
        rtol=3e-3,
    )

    dwk_passed = np.isclose(
        numerical_dwk,
        expected_dwk,
        atol=3e-3,
        rtol=3e-3,
    )

    dwv_passed = np.isclose(
        numerical_dwv,
        expected_dwv,
        atol=3e-3,
        rtol=3e-3,
    )

    dwo_passed = np.isclose(
        numerical_dwo,
        expected_dwo,
        atol=3e-3,
        rtol=3e-3,
    )

    shape_passed = (
        analytical_dx.shape
        == x.shape
    )

    dtype_passed = (
        analytical_dx.dtype
        == np.float32
        and
        analytical_dwq.dtype
        == np.float32
        and
        analytical_dwk.dtype
        == np.float32
        and
        analytical_dwv.dtype
        == np.float32
        and
        analytical_dwo.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                analytical_dx
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dwq
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dwk
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dwv
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dwo
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
            "dX",
            numerical_dx,
            expected_dx,
            dx_passed,
        ),

        (
            "dWq",
            numerical_dwq,
            expected_dwq,
            dwq_passed,
        ),

        (
            "dWk",
            numerical_dwk,
            expected_dwk,
            dwk_passed,
        ),

        (
            "dWv",
            numerical_dwv,
            expected_dwv,
            dwv_passed,
        ),

        (
            "dWo",
            numerical_dwo,
            expected_dwo,
            dwo_passed,
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

    all_gradients_passed = (
        dx_passed
        and dwq_passed
        and dwk_passed
        and dwv_passed
        and dwo_passed
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
    ):

        print()

        print(
            "SUCCESS: Multi-head attention "
            "backward tests passed."
        )

    else:

        raise AssertionError(
            "Multi-head attention backward "
            "test failed."
        )


if __name__ == "__main__":
    main()