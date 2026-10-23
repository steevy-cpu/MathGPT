import numpy as np

from linear import Linear


BATCH_SIZE = 8
SEQUENCE_LENGTH = 256

INPUT_DIM = 256
OUTPUT_DIM = 1024


def main():

    linear = Linear(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
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
            INPUT_DIM,
        ),
    ).astype(
        np.float32
    )

    output = linear.forward(
        x
    )

    print()
    print("Linear Layer Test")
    print("=================")

    print()

    print(
        f"Input shape: "
        f"{x.shape}"
    )

    print(
        f"Weight shape: "
        f"{linear.weight.shape}"
    )

    print(
        f"Bias shape: "
        f"{linear.bias.shape}"
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
        f"{linear.parameter_count():,}"
    )

    # --------------------------------------------------------
    # Expected shapes
    # --------------------------------------------------------

    expected_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        OUTPUT_DIM,
    )

    shape_passed = (
        output.shape
        == expected_shape
    )

    # 256 * 1024 weights
    # + 1024 biases
    expected_parameters = (
        INPUT_DIM
        * OUTPUT_DIM
        + OUTPUT_DIM
    )

    parameter_count_passed = (
        linear.parameter_count()
        == expected_parameters
    )

    # --------------------------------------------------------
    # Manual verification of one vector
    # --------------------------------------------------------

    manual_output = (
        x[0, 0]
        @ linear.weight
        + linear.bias
    )

    math_passed = np.allclose(
        output[0, 0],
        manual_output,
    )

    print()

    print(
        f"Shape test passed: "
        f"{shape_passed}"
    )

    print(
        f"Parameter count passed: "
        f"{parameter_count_passed}"
    )

    print(
        f"Matrix math passed: "
        f"{math_passed}"
    )

    if (
        shape_passed
        and parameter_count_passed
        and math_passed
    ):

        print()

        print(
            "SUCCESS: Linear layer "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Linear layer test failed."
        )


if __name__ == "__main__":
    main()