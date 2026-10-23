import numpy as np

from causal_attention import (
    causal_attention
)


BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
HEAD_DIM = 8


def main():

    rng = np.random.default_rng(
        42
    )

    query = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        )
    ).astype(
        np.float32
    )

    key = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        )
    ).astype(
        np.float32
    )

    value = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        )
    ).astype(
        np.float32
    )

    output, weights = causal_attention(
        query,
        key,
        value,
    )

    print()
    print("Causal Attention Test")
    print("=====================")

    print()

    print(
        f"Query shape: "
        f"{query.shape}"
    )

    print(
        f"Key shape: "
        f"{key.shape}"
    )

    print(
        f"Value shape: "
        f"{value.shape}"
    )

    print(
        f"Attention weights shape: "
        f"{weights.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print()

    print(
        "Attention matrix "
        "(batch 0):"
    )

    print(
        np.round(
            weights[0],
            4,
        )
    )

    # --------------------------------------------------------
    # Test 1:
    # Output shape
    # --------------------------------------------------------

    output_shape_passed = (
        output.shape
        ==
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        )
    )

    # --------------------------------------------------------
    # Test 2:
    # Attention matrix shape
    # --------------------------------------------------------

    weights_shape_passed = (
        weights.shape
        ==
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
        )
    )

    # --------------------------------------------------------
    # Test 3:
    # Every attention row must sum to 1.
    # --------------------------------------------------------

    row_sums = np.sum(
        weights,
        axis=-1,
    )

    row_sum_passed = np.allclose(
        row_sums,
        1.0,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 4:
    # Future attention must be exactly zero.
    # --------------------------------------------------------

    future_mask = np.triu(
        np.ones(
            (
                SEQUENCE_LENGTH,
                SEQUENCE_LENGTH,
            ),
            dtype=bool,
        ),
        k=1,
    )

    future_weights = weights[
        :,
        future_mask,
    ]

    causal_mask_passed = np.allclose(
        future_weights,
        0.0,
        atol=1e-7,
    )

    # --------------------------------------------------------
    # Test 5:
    #
    # Token 0 has no previous tokens.
    # Therefore it can only attend to itself.
    #
    # Its attention row must be:
    #
    # [1, 0, 0, 0]
    # --------------------------------------------------------

    first_token_expected = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    first_token_passed = np.allclose(
        weights[0, 0],
        first_token_expected,
        atol=1e-6,
    )

    # --------------------------------------------------------
    # Test 6:
    # Make sure everything is finite.
    # --------------------------------------------------------

    finite_passed = (
        np.all(
            np.isfinite(
                output
            )
        )
        and
        np.all(
            np.isfinite(
                weights
            )
        )
    )

    print()
    print(
        f"Output shape passed: "
        f"{output_shape_passed}"
    )

    print(
        f"Attention shape passed: "
        f"{weights_shape_passed}"
    )

    print(
        f"Attention rows sum to 1: "
        f"{row_sum_passed}"
    )

    print(
        f"Future tokens blocked: "
        f"{causal_mask_passed}"
    )

    print(
        f"First token self-only attention: "
        f"{first_token_passed}"
    )

    print(
        f"Finite values passed: "
        f"{finite_passed}"
    )

    if (
        output_shape_passed
        and weights_shape_passed
        and row_sum_passed
        and causal_mask_passed
        and first_token_passed
        and finite_passed
    ):

        print()
        print(
            "SUCCESS: Causal attention "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Causal attention test failed."
        )


if __name__ == "__main__":
    main()