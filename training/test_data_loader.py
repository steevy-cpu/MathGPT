from pathlib import Path

import numpy as np

from data_loader import TokenBatchLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent


TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_train.bin"
)


VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_validation.bin"
)


CONTEXT_LENGTH = 256

BATCH_SIZE = 8


def test_loader(
    name,
    binary_path,
):
    print()
    print("=" * 60)
    print(
        f"Testing {name}"
    )
    print("=" * 60)

    loader = TokenBatchLoader(
        binary_path=binary_path,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        seed=42,
    )

    X, Y = (
        loader.get_batch()
    )

    print()
    print(
        f"Total corpus tokens: "
        f"{loader.total_tokens:,}"
    )

    print(
        f"Possible windows: "
        f"{len(loader):,}"
    )

    print(
        f"X shape: "
        f"{X.shape}"
    )

    print(
        f"Y shape: "
        f"{Y.shape}"
    )

    print(
        f"X dtype: "
        f"{X.dtype}"
    )

    print(
        f"Y dtype: "
        f"{Y.dtype}"
    )

    print(
        f"Minimum X token: "
        f"{X.min()}"
    )

    print(
        f"Maximum X token: "
        f"{X.max()}"
    )

    # --------------------------------------------------------
    # Critical next-token alignment test
    #
    # Every Y token except the last one should equal
    # the next X token.
    #
    # X: A B C D
    # Y: B C D E
    #
    # X[1:] must equal Y[:-1]
    # --------------------------------------------------------

    alignment_passed = np.array_equal(
        X[:, 1:],
        Y[:, :-1],
    )

    print()
    print(
        f"Next-token alignment passed: "
        f"{alignment_passed}"
    )

    # --------------------------------------------------------
    # Show the first 20 IDs from example 0.
    # --------------------------------------------------------

    print()
    print(
        "First training example"
    )

    print(
        "----------------------"
    )

    print(
        "X first 20:"
    )

    print(
        X[0, :20]
    )

    print()

    print(
        "Y first 20:"
    )

    print(
        Y[0, :20]
    )

    if not alignment_passed:

        raise AssertionError(
            "X/Y next-token alignment failed."
        )

    if (
        X.shape
        != (
            BATCH_SIZE,
            CONTEXT_LENGTH,
        )
    ):

        raise AssertionError(
            "Incorrect X shape."
        )

    if (
        Y.shape
        != (
            BATCH_SIZE,
            CONTEXT_LENGTH,
        )
    ):

        raise AssertionError(
            "Incorrect Y shape."
        )

    if X.max() >= 16_000:

        raise AssertionError(
            "X contains invalid token ID."
        )

    if Y.max() >= 16_000:

        raise AssertionError(
            "Y contains invalid token ID."
        )

    print()
    print(
        "SUCCESS: Batch loader passed."
    )


def main():

    test_loader(
        "general_train",
        TRAIN_PATH,
    )

    test_loader(
        "general_validation",
        VALIDATION_PATH,
    )


if __name__ == "__main__":
    main()