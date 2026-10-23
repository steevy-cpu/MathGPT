import numpy as np

from softmax import softmax


def main():

    print()
    print("Softmax Test")
    print("============")

    # --------------------------------------------------------
    # Test 1:
    # Basic probability behavior
    # --------------------------------------------------------

    scores = np.array(
        [
            [2.0, 1.0, 0.1],
            [1.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )

    probabilities = softmax(
        scores,
        axis=-1,
    )

    print()
    print("Input scores:")
    print(scores)

    print()
    print("Probabilities:")
    print(probabilities)

    row_sums = np.sum(
        probabilities,
        axis=-1,
    )

    print()
    print(
        "Probability sums:"
    )

    print(
        row_sums
    )

    sums_passed = np.allclose(
        row_sums,
        1.0,
        atol=1e-6,
    )

    range_passed = (
        np.all(
            probabilities >= 0
        )
        and
        np.all(
            probabilities <= 1
        )
    )

    # --------------------------------------------------------
    # Test 2:
    # Largest score should receive
    # largest probability.
    # --------------------------------------------------------

    ranking_passed = (
        np.argmax(
            probabilities[0]
        )
        ==
        np.argmax(
            scores[0]
        )
    )

    # --------------------------------------------------------
    # Test 3:
    # Numerical stability
    # --------------------------------------------------------

    huge_scores = np.array(
        [
            [
                1000.0,
                1001.0,
                1002.0,
            ]
        ],
        dtype=np.float32,
    )

    huge_probabilities = softmax(
        huge_scores,
        axis=-1,
    )

    stability_passed = (
        np.all(
            np.isfinite(
                huge_probabilities
            )
        )
        and
        np.allclose(
            np.sum(
                huge_probabilities,
                axis=-1,
            ),
            1.0,
            atol=1e-6,
        )
    )

    # --------------------------------------------------------
    # Test 4:
    # Shift invariance
    #
    # Adding the same number to every
    # score must not change softmax.
    # --------------------------------------------------------

    shifted_scores = (
        scores
        + 500.0
    )

    shifted_probabilities = softmax(
        shifted_scores,
        axis=-1,
    )

    shift_passed = np.allclose(
        probabilities,
        shifted_probabilities,
        atol=1e-6,
    )

    print()
    print(
        f"Sums to 1 passed: "
        f"{sums_passed}"
    )

    print(
        f"Probability range passed: "
        f"{range_passed}"
    )

    print(
        f"Ranking test passed: "
        f"{ranking_passed}"
    )

    print(
        f"Numerical stability passed: "
        f"{stability_passed}"
    )

    print(
        f"Shift invariance passed: "
        f"{shift_passed}"
    )

    if (
        sums_passed
        and range_passed
        and ranking_passed
        and stability_passed
        and shift_passed
    ):

        print()
        print(
            "SUCCESS: Softmax tests passed."
        )

    else:

        raise AssertionError(
            "Softmax test failed."
        )


if __name__ == "__main__":
    main()