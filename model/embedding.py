import numpy as np


class TokenEmbedding:
    """
    Convert token IDs into learned vector representations.

    Weight shape:
        (vocab_size, d_model)

    Input shape:
        (batch_size, sequence_length)

    Output shape:
        (batch_size, sequence_length, d_model)
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

        # Small random initialization.
        #
        # Each vocabulary token receives
        # its own d_model-dimensional vector.
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

    def forward(
        self,
        token_ids,
    ):
        """
        Look up embedding vectors for token IDs.
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

        if token_ids.max() >= self.vocab_size:

            raise ValueError(
                "Token ID exceeds vocabulary size."
            )

        # NumPy advanced indexing does the lookup.
        embeddings = self.weight[
            token_ids
        ]

        return embeddings

    def parameter_count(self):
        """
        Number of trainable embedding parameters.
        """

        return self.weight.size