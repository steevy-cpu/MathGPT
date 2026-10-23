import numpy as np

from embedding import TokenEmbedding


VOCAB_SIZE = 16_000
D_MODEL = 256

BATCH_SIZE = 8
CONTEXT_LENGTH = 256


def main():

    embedding = TokenEmbedding(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        seed=42,
    )

    # Create fake token IDs.
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

    output = embedding.forward(
        token_ids
    )

    print()
    print("Token Embedding Test")
    print("====================")

    print()

    print(
        f"Token IDs shape: "
        f"{token_ids.shape}"
    )

    print(
        f"Embedding matrix shape: "
        f"{embedding.weight.shape}"
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
        f"{embedding.parameter_count():,}"
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
        embedding.parameter_count()
        == 4_096_000
    )

    # Verify that lookup really returns
    # the exact corresponding row.
    first_token_id = (
        token_ids[0, 0]
    )

    lookup_passed = np.array_equal(
        output[0, 0],
        embedding.weight[
            first_token_id
        ],
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
        f"Embedding lookup passed: "
        f"{lookup_passed}"
    )

    if (
        shape_passed
        and parameter_count_passed
        and lookup_passed
    ):

        print()
        print(
            "SUCCESS: Token embedding "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Token embedding test failed."
        )


if __name__ == "__main__":
    main()