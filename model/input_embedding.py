import numpy as np

try:
    from .embedding import TokenEmbedding
    from .position_embedding import PositionEmbedding
except ImportError:
    from embedding import TokenEmbedding
    from position_embedding import PositionEmbedding


class InputEmbedding:
    """
    Combine learned token embeddings with learned
    positional embeddings.

    Input:
        token_ids
        shape = (batch_size, sequence_length)

    Output:
        shape = (
            batch_size,
            sequence_length,
            d_model,
        )
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        seed=42,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            seed=seed,
        )

        self.position_embedding = PositionEmbedding(
            context_length=context_length,
            d_model=d_model,
            seed=seed + 1,
        )

    def forward(
        self,
        token_ids,
    ):
        """
        Convert token IDs into vectors containing
        both token identity and position information.
        """

        if token_ids.ndim != 2:

            raise ValueError(
                "token_ids must have shape "
                "(batch_size, sequence_length)"
            )

        sequence_length = (
            token_ids.shape[1]
        )

        if (
            sequence_length
            > self.context_length
        ):

            raise ValueError(
                "Sequence length exceeds "
                "maximum context length."
            )

        token_vectors = (
            self.token_embedding.forward(
                token_ids
            )
        )

        position_vectors = (
            self.position_embedding.forward(
                sequence_length
            )
        )

        # NumPy broadcasting:
        #
        # token_vectors:
        #     (batch, sequence, d_model)
        #
        # position_vectors:
        #     (sequence, d_model)
        #
        # result:
        #     (batch, sequence, d_model)

        output = (
            token_vectors
            + position_vectors
        )

        return output

    def parameter_count(self):

        return (
            self.token_embedding.parameter_count()
            + self.position_embedding.parameter_count()
        )