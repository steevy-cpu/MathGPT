import numpy as np

from layer_norm import LayerNorm


BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
D_MODEL = 256


def main():

    layer_norm = LayerNorm(
        d_model=D_MODEL,
        epsilon=1e-5,
    )

    rng = np.random.default_rng(
        123
    )

    # Give the values an intentionally
    # non-zero mean and large scale.
    x = rng.normal(
        loc=10.0,
        scale=5.0,
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            D_MODEL,
        ),
    ).astype(
        np.float32
    )

    output = layer_norm.forward(
        x
    )

    print()
    print("LayerNorm Test")
    print("==============")

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
        f"Gamma shape: "
        f"{layer_norm.gamma.shape}"
    )

    print(
        f"Beta shape: "
        f"{layer_norm.beta.shape}"
    )

    print(
        f"Parameter count: "
        f"{layer_norm.parameter_count():,}"
    )

    # --------------------------------------------------------
    # Examine one token vector.
    # --------------------------------------------------------

    first_vector = output[
        0,
        0,
    ]

    first_mean = np.mean(
        first_vector
    )

    first_variance = np.var(
        first_vector
    )

    print()
    print(
        f"First normalized vector mean: "
        f"{first_mean:.6f}"
    )

    print(
        f"First normalized vector variance: "
        f"{first_variance:.6f}"
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    shape_passed = (
        output.shape
        == x.shape
    )

    parameter_count_passed = (
        layer_norm.parameter_count()
        == 512
    )

    mean_passed = np.isclose(
        first_mean,
        0.0,
        atol=1e-5,
    )

    variance_passed = np.isclose(
        first_variance,
        1.0,
        atol=1e-4,
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
        f"Mean approximately zero: "
        f"{mean_passed}"
    )

    print(
        f"Variance approximately one: "
        f"{variance_passed}"
    )

    if (
        shape_passed
        and parameter_count_passed
        and mean_passed
        and variance_passed
    ):

        print()
        print(
            "SUCCESS: LayerNorm "
            "tests passed."
        )

    else:

        raise AssertionError(
            "LayerNorm test failed."
        )


if __name__ == "__main__":
    main()