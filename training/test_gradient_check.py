import numpy as np

from gradient_check import (
    numerical_gradient
)


def square(
    x,
):
    return x ** 2


def cubic(
    x,
):
    return x ** 3


def main():

    print()
    print("Gradient Fundamentals Test")
    print("==========================")

    # ========================================================
    # Test 1:
    # f(x) = x^2
    #
    # derivative:
    # f'(x) = 2x
    #
    # at x = 3:
    # derivative = 6
    # ========================================================

    x1 = np.float32(
        3.0
    )

    numerical_square = (
        numerical_gradient(
            square,
            x1,
        )
    )

    analytical_square = (
        np.float32(
            2.0
        )
        * x1
    )

    # ========================================================
    # Test 2:
    # f(x) = x^3
    #
    # derivative:
    # f'(x) = 3x^2
    #
    # at x = 2:
    # derivative = 12
    # ========================================================

    x2 = np.float32(
        2.0
    )

    numerical_cubic = (
        numerical_gradient(
            cubic,
            x2,
        )
    )

    analytical_cubic = (
        np.float32(
            3.0
        )
        * (
            x2 ** 2
        )
    )

    square_passed = (
        np.isclose(
            numerical_square,
            analytical_square,
            atol=1e-2,
        )
    )

    cubic_passed = (
        np.isclose(
            numerical_cubic,
            analytical_cubic,
            atol=1e-2,
        )
    )

    print()
    print(
        f"x^2 numerical gradient: "
        f"{numerical_square:.6f}"
    )

    print(
        f"x^2 analytical gradient: "
        f"{analytical_square:.6f}"
    )

    print()

    print(
        f"x^3 numerical gradient: "
        f"{numerical_cubic:.6f}"
    )

    print(
        f"x^3 analytical gradient: "
        f"{analytical_cubic:.6f}"
    )

    print()

    print(
        f"Square gradient passed: "
        f"{square_passed}"
    )

    print(
        f"Cubic gradient passed: "
        f"{cubic_passed}"
    )

    if (
        square_passed
        and cubic_passed
    ):

        print()
        print(
            "SUCCESS: Gradient "
            "fundamentals tests passed."
        )

    else:

        raise AssertionError(
            "Gradient fundamentals "
            "test failed."
        )


if __name__ == "__main__":
    main()