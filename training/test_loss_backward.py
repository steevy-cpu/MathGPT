# Numerically differentiate individual logits and compare them agains our analytical gradient.
import numpy as np

from loss import (
    cross_entropy_loss,
    cross_entropy_backward,
)


EPSILON = 1e-3


def main():

    print()
    print(
        "Cross-Entropy Backward Test"
    )

    print(
        "==========================="
    )

    # Small test:
    #
    # batch = 2
    # sequence = 2
    # vocabulary = 3

    logits = np.array(
        [
            [
                [1.2, -0.3, 0.7],
                [0.1, 1.4, -0.8],
            ],

            [
                [-0.5, 0.2, 1.1],
                [2.0, 0.5, -1.0],
            ],
        ],
        dtype=np.float32,
    )

    targets = np.array(
        [
            [0, 1],
            [2, 0],
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Analytical gradient
    # ========================================================

    loss = cross_entropy_loss(
        logits,
        targets,
    )

    analytical_gradient = (
        cross_entropy_backward(
            logits,
            targets,
        )
    )

    print()
    print(
        f"Loss: "
        f"{loss:.6f}"
    )

    print(
        f"Logits shape: "
        f"{logits.shape}"
    )

    print(
        f"Gradient shape: "
        f"{analytical_gradient.shape}"
    )

    # ========================================================
    # Numerical gradient checks
    #
    # Check several logits rather than only one.
    # ========================================================

    test_indices = [
        (0, 0, 0),  # correct class
        (0, 0, 2),  # incorrect class
        (0, 1, 1),  # correct class
        (1, 0, 0),  # incorrect class
        (1, 1, 0),  # correct class
    ]

    gradient_checks = []

    print()
    print(
        "Numerical Gradient Checks"
    )

    print(
        "-------------------------"
    )

    for index in test_indices:

        original_value = float(
            logits[
                index
            ]
        )

        # ---------------------------------------------
        # + epsilon
        # ---------------------------------------------

        logits[
            index
        ] = (
            original_value
            + EPSILON
        )

        loss_plus = (
            cross_entropy_loss(
                logits,
                targets,
            )
        )

        # ---------------------------------------------
        # - epsilon
        # ---------------------------------------------

        logits[
            index
        ] = (
            original_value
            - EPSILON
        )

        loss_minus = (
            cross_entropy_loss(
                logits,
                targets,
            )
        )

        # Restore original logit.
        logits[
            index
        ] = original_value

        numerical_gradient = (
            float(
                loss_plus
                - loss_minus
            )
            / (
                2.0
                * EPSILON
            )
        )

        expected_gradient = float(
            analytical_gradient[
                index
            ]
        )

        passed = np.isclose(
            numerical_gradient,
            expected_gradient,
            atol=2e-3,
            rtol=2e-3,
        )

        gradient_checks.append(
            passed
        )

        print()

        print(
            f"Index {index}"
        )

        print(
            f"Numerical:  "
            f"{numerical_gradient:.6f}"
        )

        print(
            f"Analytical: "
            f"{expected_gradient:.6f}"
        )

        print(
            f"Passed:     "
            f"{passed}"
        )

    # ========================================================
    # Important structural properties
    # ========================================================

    # For every token position:
    #
    # sum of gradients over vocabulary = 0
    #
    # because:
    #
    # sum(probabilities) - sum(one_hot)
    # = 1 - 1
    # = 0

    gradient_sums = np.sum(
        analytical_gradient,
        axis=-1,
    )

    zero_sum_passed = (
        np.allclose(
            gradient_sums,
            0.0,
            atol=1e-6,
        )
    )

    # Correct target gradients should be negative.
    batch_indices = np.arange(
        targets.shape[0]
    )[:, None]

    sequence_indices = np.arange(
        targets.shape[1]
    )[None, :]

    correct_gradients = (
        analytical_gradient[
            batch_indices,
            sequence_indices,
            targets,
        ]
    )

    correct_negative_passed = (
        np.all(
            correct_gradients
            < 0
        )
    )

    shape_passed = (
        analytical_gradient.shape
        == logits.shape
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

    numerical_checks_passed = all(
        gradient_checks
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print(
        "Structural Checks"
    )

    print(
        "-----------------"
    )

    print()

    print(
        "Gradient sums across vocabulary:"
    )

    print(
        gradient_sums
    )

    print()

    print(
        f"All numerical checks passed: "
        f"{numerical_checks_passed}"
    )

    print(
        f"Gradient sums equal zero: "
        f"{zero_sum_passed}"
    )

    print(
        f"Correct-class gradients negative: "
        f"{correct_negative_passed}"
    )

    print(
        f"Gradient shape passed: "
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
        numerical_checks_passed
        and zero_sum_passed
        and correct_negative_passed
        and shape_passed
        and dtype_passed
        and finite_passed
    ):

        print()

        print(
            "SUCCESS: Cross-entropy "
            "backward tests passed."
        )

    else:

        raise AssertionError(
            "Cross-entropy backward "
            "test failed."
        )


if __name__ == "__main__":
    main()