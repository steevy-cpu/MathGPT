import numpy as np

from transformer_block import (
    TransformerBlock
)


BATCH_SIZE = 2
SEQUENCE_LENGTH = 8

D_MODEL = 256
NUM_HEADS = 4
D_FF = 1024


def main():

    block = TransformerBlock(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        use_bias=True,
        epsilon=1e-5,
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
        attention_weights,
    ) = block.forward(
        x
    )

    print()
    print(
        "Transformer Block Test"
    )

    print(
        "======================"
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
        f"Output dtype: "
        f"{output.dtype}"
    )

    print(
        f"Attention weights shape: "
        f"{attention_weights.shape}"
    )

    print(
        f"Parameter count: "
        f"{block.parameter_count():,}"
    )

    # --------------------------------------------------------
    # Test 1:
    # Output shape preserved
    # --------------------------------------------------------

    shape_passed = (
        output.shape
        == x.shape
    )

    # --------------------------------------------------------
    # Test 2:
    # Float32 preserved
    # --------------------------------------------------------

    dtype_passed = (
        output.dtype
        == np.float32
    )

    # --------------------------------------------------------
    # Test 3:
    # Attention shape
    # --------------------------------------------------------

    attention_shape_passed = (
        attention_weights.shape
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
    # Parameter count
    #
    # Attention:
    # 263,168
    #
    # MLP:
    # 525,568
    #
    # Two LayerNorms:
    # 512 + 512
    #
    # Total:
    # 789,760
    # --------------------------------------------------------

    expected_parameters = (
        263_168
        + 525_568
        + 512
        + 512
    )

    parameter_count_passed = (
        block.parameter_count()
        == expected_parameters
    )

    # --------------------------------------------------------
    # Test 5:
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
                attention_weights
            )
        )
    )

    # --------------------------------------------------------
    # Test 6:
    # Output should actually change
    #
    # If the Transformer block did nothing,
    # output would equal input.
    # --------------------------------------------------------

    transformation_passed = (
        not np.allclose(
            output,
            x,
        )
    )

    # --------------------------------------------------------
    # Test 7:
    # Strong causal behavior
    #
    # Change ONLY the last input token.
    #
    # Earlier output positions must remain unchanged.
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
    ) = block.forward(
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

    # --------------------------------------------------------
    # Test 8:
    # Attention rows still sum to 1
    # --------------------------------------------------------

    attention_row_sums = np.sum(
        attention_weights,
        axis=-1,
    )

    attention_sum_passed = (
        np.allclose(
            attention_row_sums,
            1.0,
            atol=1e-6,
        )
    )

    print()

    print(
        f"Shape passed: "
        f"{shape_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"Attention shape passed: "
        f"{attention_shape_passed}"
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
        f"Non-trivial transformation passed: "
        f"{transformation_passed}"
    )

    print(
        f"Future-token causality passed: "
        f"{earlier_positions_same}"
    )

    print(
        f"Final token responds to change: "
        f"{final_position_changed}"
    )

    print(
        f"Attention rows sum to 1: "
        f"{attention_sum_passed}"
    )

    if (
        shape_passed
        and dtype_passed
        and attention_shape_passed
        and parameter_count_passed
        and finite_passed
        and transformation_passed
        and earlier_positions_same
        and final_position_changed
        and attention_sum_passed
    ):

        print()

        print(
            "SUCCESS: Transformer block "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Transformer block test failed."
        )


if __name__ == "__main__":
    main()