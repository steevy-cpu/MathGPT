import copy
from pathlib import Path

import numpy as np


class TokenBatchLoader:
    """
    Load random GPT training batches from a uint16
    tokenized binary corpus.

    Each example contains:

        X = context tokens
        Y = X shifted one token forward
    """

    def __init__(
        self,
        binary_path,
        context_length=256,
        batch_size=8,
        seed=42,
    ):
        self.binary_path = Path(binary_path)
        self.context_length = context_length
        self.batch_size = batch_size
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"Binary corpus not found: "
                f"{self.binary_path}"
            )

        # Memory-map the corpus instead of loading
        # the whole binary into RAM.
        self.tokens = np.memmap(
            self.binary_path,
            dtype=np.uint16,
            mode="r",
        )

        self.total_tokens = len(self.tokens)

        minimum_tokens = (
            self.context_length
            + 1
        )

        if self.total_tokens < minimum_tokens:
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
        Return one random next-token batch.

        Shapes:

            X: (batch_size, context_length)
            Y: (batch_size, context_length)

        Both are int64 for embedding lookup.
        """

        start_positions = self.rng.integers(
            low=0,
            high=self.maximum_start + 1,
            size=self.batch_size,
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
            chunk = np.asarray(
                self.tokens[
                    start:
                    start
                    + self.context_length
                    + 1
                ],
                dtype=np.int64,
            )

            X[batch_index] = chunk[:-1]
            Y[batch_index] = chunk[1:]

        return X, Y

    def state_dict(self):
        """
        Return everything required to resume
        this loader deterministically.

        The critical piece is rng_state.

        We also store loader dimensions so a
        checkpoint cannot accidentally be restored
        into an incompatible loader.
        """

        return {
            "context_length":
                int(self.context_length),

            "batch_size":
                int(self.batch_size),

            "total_tokens":
                int(self.total_tokens),

            "maximum_start":
                int(self.maximum_start),

            "rng_state":
                copy.deepcopy(
                    self.rng.bit_generator.state
                ),
        }

    def load_state_dict(
        self,
        state,
    ):
        """
        Restore loader RNG state.

        After this call, the next get_batch()
        should return exactly the batch that
        the original loader would have returned.
        """

        required_keys = {
            "context_length",
            "batch_size",
            "total_tokens",
            "maximum_start",
            "rng_state",
        }

        missing_keys = (
            required_keys
            -
            set(state.keys())
        )

        if missing_keys:
            raise KeyError(
                "Loader state is missing keys: "
                f"{sorted(missing_keys)}"
            )

        if (
            int(state["context_length"])
            != self.context_length
        ):
            raise ValueError(
                "Checkpoint loader context_length "
                "does not match current loader."
            )

        if (
            int(state["batch_size"])
            != self.batch_size
        ):
            raise ValueError(
                "Checkpoint loader batch_size "
                "does not match current loader."
            )

        if (
            int(state["total_tokens"])
            != self.total_tokens
        ):
            raise ValueError(
                "Checkpoint token corpus size "
                "does not match current loader."
            )

        if (
            int(state["maximum_start"])
            != self.maximum_start
        ):
            raise ValueError(
                "Checkpoint loader window range "
                "does not match current loader."
            )

        self.rng.bit_generator.state = (
            copy.deepcopy(
                state["rng_state"]
            )
        )

    def __len__(self):
        """
        Number of possible context windows.
        """

        return (
            self.total_tokens
            - self.context_length
        )