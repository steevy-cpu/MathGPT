import numpy as np

from mlp import MLP


BATCH_SIZE = 1
SEQUENCE_LENGTH = 2

D_MODEL = 3
D_FF = 5

EPSILON = 1e-3


def scalar_loss(
    mlp,
    x,
    grad_output,
):
    """
    Artificial scalar loss:

        L =
        sum(
            MLP(x)
            * grad_output
        )
    """

    output = mlp.forward(
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
        "MLP Backward Test"
    )

    print(
        "================="
    )

    rng = np.random.default_rng(
        123
    )

    mlp = MLP(
        d_model=D_MODEL,
        d_ff=D_FF,
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

    output = mlp.forward(
        x
    )

    grad_input = mlp.backward(
        grad_output
    )

    analytical_dx = (
        grad_input.copy()
    )

    analytical_dw1 = (
        mlp
        .fc_in
        .dweight
        .copy()
    )

    analytical_db1 = (
        mlp
        .fc_in
        .dbias
        .copy()
    )

    analytical_dw2 = (
        mlp
        .fc_out
        .dweight
        .copy()
    )

    analytical_db2 = (
        mlp
        .fc_out
        .dbias
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
            mlp,
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
    # Numerical dW1
    # ========================================================

    w1_index = (
        1,
        3,
    )

    numerical_dw1 = numerical_gradient(
        mlp
        .fc_in
        .weight,
        w1_index,
        lambda: scalar_loss(
            mlp,
            x,
            grad_output,
        ),
    )

    expected_dw1 = float(
        analytical_dw1[
            w1_index
        ]
    )

    # ========================================================
    # Numerical db1
    # ========================================================

    b1_index = 2

    numerical_db1 = numerical_gradient(
        mlp
        .fc_in
        .bias,
        b1_index,
        lambda: scalar_loss(
            mlp,
            x,
            grad_output,
        ),
    )

    expected_db1 = float(
        analytical_db1[
            b1_index
        ]
    )

    # ========================================================
    # Numerical dW2
    # ========================================================

    w2_index = (
        4,
        1,
    )

    numerical_dw2 = numerical_gradient(
        mlp
        .fc_out
        .weight,
        w2_index,
        lambda: scalar_loss(
            mlp,
            x,
            grad_output,
        ),
    )

    expected_dw2 = float(
        analytical_dw2[
            w2_index
        ]
    )

    # ========================================================
    # Numerical db2
    # ========================================================

    b2_index = 1

    numerical_db2 = numerical_gradient(
        mlp
        .fc_out
        .bias,
        b2_index,
        lambda: scalar_loss(
            mlp,
            x,
            grad_output,
        ),
    )

    expected_db2 = float(
        analytical_db2[
            b2_index
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

    dw1_passed = np.isclose(
        numerical_dw1,
        expected_dw1,
        atol=3e-3,
        rtol=3e-3,
    )

    db1_passed = np.isclose(
        numerical_db1,
        expected_db1,
        atol=3e-3,
        rtol=3e-3,
    )

    dw2_passed = np.isclose(
        numerical_dw2,
        expected_dw2,
        atol=3e-3,
        rtol=3e-3,
    )

    db2_passed = np.isclose(
        numerical_db2,
        expected_db2,
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
        analytical_dw1.dtype
        == np.float32
        and
        analytical_db1.dtype
        == np.float32
        and
        analytical_dw2.dtype
        == np.float32
        and
        analytical_db2.dtype
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
                analytical_dw1
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_db1
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dw2
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_db2
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
            "dW1",
            numerical_dw1,
            expected_dw1,
            dw1_passed,
        ),

        (
            "db1",
            numerical_db1,
            expected_db1,
            db1_passed,
        ),

        (
            "dW2",
            numerical_dw2,
            expected_dw2,
            dw2_passed,
        ),

        (
            "db2",
            numerical_db2,
            expected_db2,
            db2_passed,
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
        and dw1_passed
        and db1_passed
        and dw2_passed
        and db2_passed
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
            "SUCCESS: MLP backward "
            "tests passed."
        )

    else:

        raise AssertionError(
            "MLP backward test failed."
        )


if __name__ == "__main__":
    main()