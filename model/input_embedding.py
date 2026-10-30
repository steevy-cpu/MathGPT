try:
    from .embedding import TokenEmbedding
    from .position_embedding import (
        PositionEmbedding
    )

except ImportError:
    from embedding import TokenEmbedding
    from position_embedding import (
        PositionEmbedding
    )


class InputEmbedding:
    """
    Token embedding + positional embedding.
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

        self.position_embedding = (
            PositionEmbedding(
                context_length=context_length,
                d_model=d_model,
                seed=seed + 1,
            )
        )

    def forward(
        self,
        token_ids,
    ):
        """
        Combine token and position vectors.
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

        output = (
            token_vectors
            + position_vectors
        )

        return output

    def backward(
        self,
        grad_output,
    ):
        """
        Backward through:

            token_embedding
            +
            position_embedding

        Both branches receive the same gradient.

        Token IDs are discrete, therefore this
        method does not return an input gradient.
        """

        self.token_embedding.backward(
            grad_output
        )

        self.position_embedding.backward(
            grad_output
        )

        return None

    def zero_grad(
        self,
    ):
        self.token_embedding.zero_grad()
        self.position_embedding.zero_grad()

    def parameter_count(
        self,
    ):
        return (
            self.token_embedding.parameter_count()
            + self.position_embedding.parameter_count()
        )