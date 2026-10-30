import numpy as np

from loss import cross_entropy_loss


def main():

    print()
    print("Cross-Entropy Loss Test")
    print("=======================")

    # ========================================================
    # Test 1:
    # Good prediction should have low loss
    # ========================================================

    good_logits = np.array(
        [
            [
                [
                    0.0,
                    0.0,
                    10.0,
                ]
            ]
        ],
        dtype=np.float32,
    )

    targets = np.array(
        [
            [2]
        ],
        dtype=np.int64,
    )

    good_loss = (
        cross_entropy_loss(
            good_logits,
            targets,
        )
    )

    # ========================================================
    # Test 2:
    # Bad prediction should have high loss
    # ========================================================

    bad_logits = np.array(
        [
            [
                [
                    10.0,
                    0.0,
                    0.0,
                ]
            ]
        ],
        dtype=np.float32,
    )

    bad_loss = (
        cross_entropy_loss(
            bad_logits,
            targets,
        )
    )

    # ========================================================
    # Test 3:
    # Uniform logits
    #
    # For 3 equally likely classes:
    #
    # loss = ln(3)
    # ========================================================

    uniform_logits = np.zeros(
        (
            1,
            1,
            3,
        ),
        dtype=np.float32,
    )

    uniform_loss = (
        cross_entropy_loss(
            uniform_logits,
            targets,
        )
    )

    expected_uniform_loss = (
        np.log(
            3.0
        )
    )

    # ========================================================
    # Test 4:
    # Numerical stability
    # ========================================================

    huge_logits = np.array(
        [
            [
                [
                    10000.0,
                    10001.0,
                    10002.0,
                ]
            ]
        ],
        dtype=np.float32,
    )

    huge_loss = (
        cross_entropy_loss(
            huge_logits,
            targets,
        )
    )

    # ========================================================
    # Results
    # ========================================================

    good_vs_bad_passed = (
        good_loss
        < bad_loss
    )

    uniform_passed = (
        np.isclose(
            uniform_loss,
            expected_uniform_loss,
            atol=1e-6,
        )
    )

    finite_passed = (
        np.isfinite(
            huge_loss
        )
    )

    dtype_passed = (
        np.asarray(
            good_loss
        ).dtype
        == np.float32
    )

    print()
    print(
        f"Good prediction loss: "
        f"{good_loss:.6f}"
    )

    print(
        f"Bad prediction loss: "
        f"{bad_loss:.6f}"
    )

    print(
        f"Uniform prediction loss: "
        f"{uniform_loss:.6f}"
    )

    print(
        f"Expected ln(3): "
        f"{expected_uniform_loss:.6f}"
    )

    print(
        f"Huge-logit loss: "
        f"{huge_loss:.6f}"
    )

    print()

    print(
        f"Good loss < bad loss: "
        f"{good_vs_bad_passed}"
    )

    print(
        f"Uniform-loss test passed: "
        f"{uniform_passed}"
    )

    print(
        f"Numerical stability passed: "
        f"{finite_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    if (
        good_vs_bad_passed
        and uniform_passed
        and finite_passed
        and dtype_passed
    ):

        print()
        print(
            "SUCCESS: Cross-entropy "
            "loss tests passed."
        )

    else:

        raise AssertionError(
            "Cross-entropy loss test failed."
        )


if __name__ == "__main__":
    main()