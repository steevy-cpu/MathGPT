import numpy as np

from position_embedding import (
    PositionEmbedding
)


CONTEXT_LENGTH = 256
D_MODEL = 256


def main():

    position_embedding = (
        PositionEmbedding(
            context_length=(
                CONTEXT_LENGTH
            ),

            d_model=D_MODEL,

            seed=43,
        )
    )

    output = (
        position_embedding.forward(
            sequence_length=256
        )
    )

    print()
    print(
        "Position Embedding Test"
    )

    print(
        "======================="
    )

    print()

    print(
        f"Embedding matrix shape: "
        f"{position_embedding.weight.shape}"
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
        f"{position_embedding.parameter_count():,}"
    )

    shape_passed = (
        output.shape
        == (
            CONTEXT_LENGTH,
            D_MODEL,
        )
    )

    parameter_count_passed = (
        position_embedding.parameter_count()
        == 65_536
    )

    first_position_passed = (
        np.array_equal(
            output[0],
            position_embedding.weight[0],
        )
    )

    last_position_passed = (
        np.array_equal(
            output[-1],
            position_embedding.weight[255],
        )
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
        f"First position lookup passed: "
        f"{first_position_passed}"
    )

    print(
        f"Last position lookup passed: "
        f"{last_position_passed}"
    )

    if (
        shape_passed
        and parameter_count_passed
        and first_position_passed
        and last_position_passed
    ):

        print()
        print(
            "SUCCESS: Positional embedding "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Positional embedding test failed."
        )


if __name__ == "__main__":
    main()