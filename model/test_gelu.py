import numpy as np

from gelu import gelu


def main():

    x = np.array(
        [
            -3.0,
            -1.0,
            0.0,
            1.0,
            3.0,
        ],
        dtype=np.float32,
    )

    output = gelu(
        x
    )

    print()
    print("GELU Test")
    print("=========")

    print()

    print(
        f"Input:"
    )

    print(
        x
    )

    print()

    print(
        f"Output:"
    )

    print(
        output
    )

    print()

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"Output dtype: "
        f"{output.dtype}"
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

    zero_passed = np.isclose(
        gelu(
            np.array(
                [0.0],
                dtype=np.float32,
            )
        )[0],
        0.0,
        atol=1e-7,
    )

    positive_behavior_passed = (
        output[-1]
        > output[-2]
        > 0
    )

    negative_behavior_passed = (
        output[0]
        < 0
        and
        output[1]
        < 0
    )

    finite_passed = (
        np.all(
            np.isfinite(
                output
            )
        )
    )

    print(
        f"Shape passed: "
        f"{shape_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"GELU(0) passed: "
        f"{zero_passed}"
    )

    print(
        f"Positive behavior passed: "
        f"{positive_behavior_passed}"
    )

    print(
        f"Negative behavior passed: "
        f"{negative_behavior_passed}"
    )

    print(
        f"Finite values passed: "
        f"{finite_passed}"
    )

    if (
        shape_passed
        and dtype_passed
        and zero_passed
        and positive_behavior_passed
        and negative_behavior_passed
        and finite_passed
    ):

        print()
        print(
            "SUCCESS: GELU tests passed."
        )

    else:

        raise AssertionError(
            "GELU test failed."
        )


if __name__ == "__main__":
    main()