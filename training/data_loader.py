from pathlib import Path

import numpy as np


class TokenBatchLoader:
    """
    Load random GPT training batches from a uint16
    tokenized binary corpus.

    Each training example contains:

        X = context tokens
        Y = the same sequence shifted one token forward

    Example:

        source = [10, 20, 30, 40, 50]

        X = [10, 20, 30, 40]
        Y = [20, 30, 40, 50]
    """

    def __init__(
        self,
        binary_path,
        context_length=256,
        batch_size=8,
        seed=42,
    ):
        self.binary_path = Path(
            binary_path
        )

        self.context_length = (
            context_length
        )

        self.batch_size = (
            batch_size
        )

        self.rng = (
            np.random.default_rng(
                seed
            )
        )

        if not self.binary_path.exists():

            raise FileNotFoundError(
                f"Binary corpus not found: "
                f"{self.binary_path}"
            )

        # ----------------------------------------------------
        # Memory-map the corpus.
        #
        # The whole binary file does NOT need to be copied
        # into RAM.
        # ----------------------------------------------------

        self.tokens = np.memmap(
            self.binary_path,
            dtype=np.uint16,
            mode="r",
        )

        self.total_tokens = len(
            self.tokens
        )

        # We need context_length + 1 tokens:
        #
        # [ t0 t1 t2 ... t256 ]
        #
        # X = t0 ... t255
        # Y = t1 ... t256
        # ----------------------------------------------------

        minimum_tokens = (
            self.context_length
            + 1
        )

        if (
            self.total_tokens
            < minimum_tokens
        ):

            raise ValueError(
                "Token corpus is too small "
                "for the requested context length."
            )

        self.maximum_start = (
            self.total_tokens
            - self.context_length
            - 1
        )

    def get_batch(self):
        """
        Return one random batch.

        Shapes:

            X: (batch_size, context_length)
            Y: (batch_size, context_length)

        dtype is int64 because these values will later
        be used as indices into the embedding matrix.
        """

        start_positions = (
            self.rng.integers(
                low=0,
                high=(
                    self.maximum_start
                    + 1
                ),
                size=self.batch_size,
            )
        )

        X = np.empty(
            (
                self.batch_size,
                self.context_length,
            ),
            dtype=np.int64,
        )

        Y = np.empty(
            (
                self.batch_size,
                self.context_length,
            ),
            dtype=np.int64,
        )

        for batch_index, start in enumerate(
            start_positions
        ):

            # Need 257 tokens to create
            # a 256-token input and target.
            chunk = np.asarray(
                self.tokens[
                    start:
                    start
                    + self.context_length
                    + 1
                ],
                dtype=np.int64,
            )

            X[
                batch_index
            ] = chunk[:-1]

            Y[
                batch_index
            ] = chunk[1:]

        return X, Y

    def __len__(self):
        """
        Number of possible context windows.

        Windows may overlap.
        """

        return (
            self.total_tokens
            - self.context_length
        )