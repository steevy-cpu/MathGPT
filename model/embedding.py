import numpy as np


class TokenEmbedding:
    """
    Trainable token embedding table.

    Weight shape:
        (vocab_size, d_model)

    Forward input:
        token_ids
        (batch, sequence)

    Forward output:
        (batch, sequence, d_model)
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        seed=42,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model

        rng = np.random.default_rng(
            seed
        )

        self.weight = rng.normal(
            loc=0.0,
            scale=0.02,
            size=(
                vocab_size,
                d_model,
            ),
        ).astype(
            np.float32
        )

        # Gradient for embedding matrix
        self.dweight = np.zeros_like(
            self.weight,
            dtype=np.float32,
        )

        # Forward cache
        self.token_ids_cache = None

    def forward(
        self,
        token_ids,
    ):
        """
        Look up embedding rows.
        """

        if token_ids.ndim != 2:

            raise ValueError(
                "token_ids must have shape "
                "(batch_size, sequence_length)"
            )

        if token_ids.min() < 0:

            raise ValueError(
                "Token IDs cannot be negative."
            )

        if (
            token_ids.max()
            >= self.vocab_size
        ):

            raise ValueError(
                "Token ID exceeds vocabulary size."
            )

        token_ids = np.asarray(
            token_ids,
            dtype=np.int64,
        )

        # Save IDs for backward.
        self.token_ids_cache = (
            token_ids.copy()
        )

        embeddings = (
            self.weight[
                token_ids
            ]
        )

        return embeddings.astype(
            np.float32,
            copy=False,
        )

    def backward(
        self,
        grad_output,
    ):
        """
        Backward through the embedding lookup.

        grad_output:
            (batch, sequence, d_model)

        Stores:
            dL/dEmbeddingWeight

        There is no gradient with respect
        to integer token IDs.
        """

        if (
            self.token_ids_cache
            is None
        ):

            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad_output = np.asarray(
            grad_output,
            dtype=np.float32,
        )

        expected_shape = (
            self.token_ids_cache.shape
            + (
                self.d_model,
            )
        )

        if (
            grad_output.shape
            != expected_shape
        ):

            raise ValueError(
                "grad_output has "
                "incorrect shape."
            )

        # Start fresh for this backward pass.
        self.dweight.fill(
            0.0
        )

        # ----------------------------------------------------
        # Scatter-add.
        #
        # If the same token ID appears multiple
        # times, all of its gradients accumulate
        # into the same embedding row.
        # ----------------------------------------------------

        np.add.at(
            self.dweight,
            self.token_ids_cache,
            grad_output,
        )

        # Token IDs themselves are discrete,
        # so there is no dTokenIDs to return.
        return None

    def zero_grad(
        self,
    ):
        """
        Reset embedding gradients.
        """

        self.dweight.fill(
            0.0
        )

    def parameter_count(
        self,
    ):
        return self.weight.size