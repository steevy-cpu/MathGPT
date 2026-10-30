import numpy as np

from sgd import sgd_update


def main():

    print()
    print(
        "SGD Fundamentals Test"
    )

    print(
        "====================="
    )

    # ========================================================
    # Test 1:
    # Positive gradient
    #
    # parameter should move DOWN.
    # ========================================================

    parameter_1 = np.array(
        [
            0.50
        ],
        dtype=np.float32,
    )

    gradient_1 = np.array(
        [
            0.20
        ],
        dtype=np.float32,
    )

    sgd_update(
        parameter_1,
        gradient_1,
        learning_rate=0.01,
    )

    expected_1 = np.float32(
        0.498
    )

    positive_gradient_passed = (
        np.isclose(
            parameter_1[0],
            expected_1,
            atol=1e-7,
        )
    )

    # ========================================================
    # Test 2:
    # Negative gradient
    #
    # subtracting a negative gradient
    # should move parameter UP.
    # ========================================================

    parameter_2 = np.array(
        [
            0.50
        ],
        dtype=np.float32,
    )

    gradient_2 = np.array(
        [
            -0.20
        ],
        dtype=np.float32,
    )

    sgd_update(
        parameter_2,
        gradient_2,
        learning_rate=0.01,
    )

    expected_2 = np.float32(
        0.502
    )

    negative_gradient_passed = (
        np.isclose(
            parameter_2[0],
            expected_2,
            atol=1e-7,
        )
    )

    # ========================================================
    # Test 3:
    # Zero gradient
    #
    # parameter should not move.
    # ========================================================

    parameter_3 = np.array(
        [
            1.25
        ],
        dtype=np.float32,
    )

    gradient_3 = np.array(
        [
            0.0
        ],
        dtype=np.float32,
    )

    original_3 = (
        parameter_3.copy()
    )

    sgd_update(
        parameter_3,
        gradient_3,
        learning_rate=0.01,
    )

    zero_gradient_passed = (
        np.array_equal(
            parameter_3,
            original_3,
        )
    )

    # ========================================================
    # Test 4:
    # Matrix update
    # ========================================================

    parameter_matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    gradient_matrix = np.array(
        [
            [0.1, -0.2],
            [0.3, -0.4],
        ],
        dtype=np.float32,
    )

    expected_matrix = (
        parameter_matrix.copy()
        - np.float32(0.1)
        * gradient_matrix
    )

    sgd_update(
        parameter_matrix,
        gradient_matrix,
        learning_rate=0.1,
    )

    matrix_passed = np.allclose(
        parameter_matrix,
        expected_matrix,
        atol=1e-7,
    )

    # ========================================================
    # Test 5:
    # Float32 remains Float32
    # ========================================================

    dtype_passed = (
        parameter_matrix.dtype
        == np.float32
    )

    # ========================================================
    # Display
    # ========================================================

    print()

    print(
        "Positive gradient:"
    )

    print(
        f"Expected: 0.498000"
    )

    print(
        f"Actual:   "
        f"{parameter_1[0]:.6f}"
    )

    print()

    print(
        "Negative gradient:"
    )

    print(
        f"Expected: 0.502000"
    )

    print(
        f"Actual:   "
        f"{parameter_2[0]:.6f}"
    )

    print()

    print(
        "Matrix after update:"
    )

    print(
        parameter_matrix
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Positive gradient passed: "
        f"{positive_gradient_passed}"
    )

    print(
        f"Negative gradient passed: "
        f"{negative_gradient_passed}"
    )

    print(
        f"Zero gradient passed: "
        f"{zero_gradient_passed}"
    )

    print(
        f"Matrix update passed: "
        f"{matrix_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    if (
        positive_gradient_passed
        and
        negative_gradient_passed
        and
        zero_gradient_passed
        and
        matrix_passed
        and
        dtype_passed
    ):

        print()

        print(
            "SUCCESS: SGD fundamentals "
            "tests passed."
        )

    else:

        raise AssertionError(
            "SGD fundamentals "
            "test failed."
        )


if __name__ == "__main__":
    main()