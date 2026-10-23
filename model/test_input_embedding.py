import numpy as np

from input_embedding import InputEmbedding


VOCAB_SIZE = 16_000
CONTEXT_LENGTH = 256
D_MODEL = 256

BATCH_SIZE = 8


def main():

    input_embedding = InputEmbedding(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        seed=42,
    )

    rng = np.random.default_rng(
        123
    )

    token_ids = rng.integers(
        low=0,
        high=VOCAB_SIZE,
        size=(
            BATCH_SIZE,
            CONTEXT_LENGTH,
        ),
        dtype=np.int64,
    )

    output = input_embedding.forward(
        token_ids
    )

    print()
    print("Input Embedding Test")
    print("====================")

    print()

    print(
        f"Token IDs shape: "
        f"{token_ids.shape}"
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
        f"{input_embedding.parameter_count():,}"
    )

    expected_shape = (
        BATCH_SIZE,
        CONTEXT_LENGTH,
        D_MODEL,
    )

    shape_passed = (
        output.shape
        == expected_shape
    )

    parameter_count_passed = (
        input_embedding.parameter_count()
        == 4_161_536
    )

    # Manually reconstruct the first vector:
    #
    # token vector + position vector
    first_token_id = (
        token_ids[0, 0]
    )

    expected_first_vector = (
        input_embedding
        .token_embedding
        .weight[first_token_id]
        +
        input_embedding
        .position_embedding
        .weight[0]
    )

    addition_passed = np.allclose(
        output[0, 0],
        expected_first_vector,
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
        f"Token + position addition passed: "
        f"{addition_passed}"
    )

    if (
        shape_passed
        and parameter_count_passed
        and addition_passed
    ):

        print()

        print(
            "SUCCESS: Combined input "
            "embedding tests passed."
        )

    else:

        raise AssertionError(
            "Input embedding test failed."
        )


if __name__ == "__main__":
    main()