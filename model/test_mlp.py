import numpy as np

from mlp import MLP


BATCH_SIZE = 2
SEQUENCE_LENGTH = 8

D_MODEL = 256
D_FF = 1024


def main():

    mlp = MLP(
        d_model=D_MODEL,
        d_ff=D_FF,
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

    output = mlp.forward(
        x
    )

    print()
    print("MLP Test")
    print("========")

    print()

    print(
        f"Input shape: "
        f"{x.shape}"
    )

    print(
        f"Expansion weight shape: "
        f"{mlp.fc_in.weight.shape}"
    )

    print(
        f"Compression weight shape: "
        f"{mlp.fc_out.weight.shape}"
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
        f"{mlp.parameter_count():,}"
    )

    # --------------------------------------------------------
    # Expected parameter count
    # --------------------------------------------------------

    expected_parameters = (
        (
            D_MODEL
            * D_FF
        )
        + D_FF
        + (
            D_FF
            * D_MODEL
        )
        + D_MODEL
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    shape_passed = (
        output.shape
        == x.shape
    )

    dtype_passed = (
        output.dtype
        == np.float32
    )

    parameter_count_passed = (
        mlp.parameter_count()
        == expected_parameters
    )

    finite_passed = (
        np.all(
            np.isfinite(
                output
            )
        )
    )

    nontrivial_output_passed = (
        not np.allclose(
            output,
            0.0,
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
        f"Parameter count passed: "
        f"{parameter_count_passed}"
    )

    print(
        f"Finite values passed: "
        f"{finite_passed}"
    )

    print(
        f"Non-trivial output passed: "
        f"{nontrivial_output_passed}"
    )

    if (
        shape_passed
        and dtype_passed
        and parameter_count_passed
        and finite_passed
        and nontrivial_output_passed
    ):

        print()
        print(
            "SUCCESS: MLP tests passed."
        )

    else:

        raise AssertionError(
            "MLP test failed."
        )


if __name__ == "__main__":
    main()