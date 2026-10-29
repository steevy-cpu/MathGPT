import numpy as np

from multi_head_attention import (
    MultiHeadAttention
)


BATCH_SIZE = 2
SEQUENCE_LENGTH = 8

D_MODEL = 256
NUM_HEADS = 4
HEAD_DIM = 64


def main():

    attention = MultiHeadAttention(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        use_bias=True,
        seed=42,
    )

    rng = np.random.default_rng(
        123
    )

    x = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            D_MODEL,
        ),
    ).astype(
        np.float32
    )

    (
        output,
        weights,
    ) = attention.forward(
        x
    )

    print()
    print(
        "Multi-Head Attention Test"
    )

    print(
        "========================="
    )

    print()

    print(
        f"Input shape: "
        f"{x.shape}"
    )

    print(
        f"Number of heads: "
        f"{attention.num_heads}"
    )

    print(
        f"Head dimension: "
        f"{attention.head_dim}"
    )

    print(
        f"Attention weights shape: "
        f"{weights.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"Output dtype: "
        f"{output.dtype}"
    )

    print(
        f"Parameter count: "
        f"{attention.parameter_count():,}"
    )

    # --------------------------------------------------------
    # Test 1:
    # Correct head dimension
    # --------------------------------------------------------

    head_dimension_passed = (
        attention.head_dim
        == HEAD_DIM
    )

    # --------------------------------------------------------
    # Test 2:
    # Output shape
    # --------------------------------------------------------

    output_shape_passed = (
        output.shape
        ==
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            D_MODEL,
        )
    )

    # --------------------------------------------------------
    # Test 3:
    # Attention weights shape
    # --------------------------------------------------------

    weights_shape_passed = (
        weights.shape
        ==
        (
            BATCH_SIZE,
            NUM_HEADS,
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
        )
    )

    # --------------------------------------------------------
    # Test 4:
    # Every attention row sums to 1
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
    # Test 5:
    # Future tokens blocked
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

    future_weights = (
        weights[
            :,
            :,
            future_mask,
        ]
    )

    causal_mask_passed = (
        np.allclose(
            future_weights,
            0.0,
            atol=1e-7,
        )
    )

    # --------------------------------------------------------
    # Test 6:
    # Parameter count
    #
    # Each projection:
    #
    # 256 * 256 + 256
    # = 65,792
    #
    # Four projections:
    #
    # 65,792 × 4
    # = 263,168
    # --------------------------------------------------------

    expected_parameters = (
        4
        * (
            D_MODEL
            * D_MODEL
            + D_MODEL
        )
    )

    parameter_count_passed = (
        attention.parameter_count()
        == expected_parameters
    )

    # --------------------------------------------------------
    # Test 7:
    # Finite values
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

    # --------------------------------------------------------
    # Test 8:
    # Strong causality test
    #
    # Modify only the LAST token.
    #
    # Earlier positions should remain identical.
    # --------------------------------------------------------

    modified_x = x.copy()

    modified_x[
        :,
        -1,
        :
    ] += 100.0

    (
        modified_output,
        _,
    ) = attention.forward(
        modified_x
    )

    earlier_positions_same = (
        np.allclose(
            output[
                :,
                :-1,
                :
            ],
            modified_output[
                :,
                :-1,
                :
            ],
            atol=1e-5,
        )
    )

    # Final token can see itself,
    # so it SHOULD react.
    final_position_changed = (
        not np.allclose(
            output[
                :,
                -1,
                :
            ],
            modified_output[
                :,
                -1,
                :
            ],
            atol=1e-5,
        )
    )

    print()

    print(
        f"Head dimension passed: "
        f"{head_dimension_passed}"
    )

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
        f"Parameter count passed: "
        f"{parameter_count_passed}"
    )

    print(
        f"Finite values passed: "
        f"{finite_passed}"
    )

    print(
        f"Future-token causality passed: "
        f"{earlier_positions_same}"
    )

    print(
        f"Final token responds to change: "
        f"{final_position_changed}"
    )

    if (
        head_dimension_passed
        and output_shape_passed
        and weights_shape_passed
        and row_sum_passed
        and causal_mask_passed
        and parameter_count_passed
        and finite_passed
        and earlier_positions_same
        and final_position_changed
    ):

        print()
        print(
            "SUCCESS: Multi-head attention "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Multi-head attention test failed."
        )


if __name__ == "__main__":
    main()