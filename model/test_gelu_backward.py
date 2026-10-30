import numpy as np

from gelu import (
    gelu,
    gelu_backward,
)


EPSILON = 1e-3


def scalar_loss(
    x,
    grad_output,
):
    """
    Artificial scalar loss:

        L = sum(
            GELU(x) * grad_output
        )

    This makes grad_output act as
    the upstream gradient.
    """

    output = gelu(
        x
    )

    return float(
        np.sum(
            output
            * grad_output
        )
    )


def main():

    print()
    print("GELU Backward Test")
    print("==================")

    x = np.array(
        [
            -2.0,
            -0.5,
            0.0,
            0.75,
            2.0,
        ],
        dtype=np.float32,
    )

    grad_output = np.array(
        [
            0.5,
            -1.0,
            0.25,
            1.5,
            -0.75,
        ],
        dtype=np.float32,
    )

    # ========================================================
    # Analytical backward
    # ========================================================

    analytical_gradient = (
        gelu_backward(
            x,
            grad_output,
        )
    )

    # ========================================================
    # Numerical gradients
    # ========================================================

    numerical_gradient = (
        np.zeros_like(
            x
        )
    )

    for index in range(
        x.size
    ):

        original_value = float(
            x[index]
        )

        # ---------------------------------------------
        # x + epsilon
        # ---------------------------------------------

        x[index] = (
            original_value
            + EPSILON
        )

        loss_plus = scalar_loss(
            x,
            grad_output,
        )

        # ---------------------------------------------
        # x - epsilon
        # ---------------------------------------------

        x[index] = (
            original_value
            - EPSILON
        )

        loss_minus = scalar_loss(
            x,
            grad_output,
        )

        # Restore original value.
        x[index] = (
            original_value
        )

        numerical_gradient[
            index
        ] = (
            loss_plus
            - loss_minus
        ) / (
            2.0
            * EPSILON
        )

    # ========================================================
    # Compare
    # ========================================================

    gradient_passed = np.allclose(
        numerical_gradient,
        analytical_gradient,
        atol=2e-3,
        rtol=2e-3,
    )

    shape_passed = (
        analytical_gradient.shape
        == x.shape
    )

    dtype_passed = (
        analytical_gradient.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                analytical_gradient
            )
        )
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Input:"
    )

    print(
        x
    )

    print()

    print(
        "Upstream gradient:"
    )

    print(
        grad_output
    )

    print()

    print(
        "Numerical gradient:"
    )

    print(
        numerical_gradient
    )

    print()

    print(
        "Analytical gradient:"
    )

    print(
        analytical_gradient
    )

    print()

    print(
        "Absolute differences:"
    )

    print(
        np.abs(
            numerical_gradient
            - analytical_gradient
        )
    )

    print()
    print("Final Checks")
    print("------------")

    print(
        f"Gradient check passed: "
        f"{gradient_passed}"
    )

    print(
        f"Shape passed: "
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

    if (
        gradient_passed
        and shape_passed
        and dtype_passed
        and finite_passed
    ):

        print()

        print(
            "SUCCESS: GELU backward "
            "tests passed."
        )

    else:

        raise AssertionError(
            "GELU backward test failed."
        )


if __name__ == "__main__":
    main()