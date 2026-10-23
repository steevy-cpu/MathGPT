import numpy as np


class PositionEmbedding:
    """
    Learned positional embeddings.

    Weight shape:
        (context_length, d_model)

    Output shape:
        (sequence_length, d_model)
    """

    def __init__(
        self,
        context_length,
        d_model,
        seed=43,
    ):
        self.context_length = (
            context_length
        )

        self.d_model = (
            d_model
        )

        rng = np.random.default_rng(
            seed
        )

        self.weight = rng.normal(
            loc=0.0,
            scale=0.02,
            size=(
                context_length,
                d_model,
            ),
        ).astype(
            np.float32
        )

    def forward(
        self,
        sequence_length,
    ):
        """
        Return positional vectors for positions:

            0, 1, 2, ... sequence_length - 1
        """

        if sequence_length <= 0:

            raise ValueError(
                "sequence_length must be positive."
            )

        if (
            sequence_length
            > self.context_length
        ):

            raise ValueError(
                "sequence_length exceeds "
                "maximum context length."
            )

        return self.weight[
            :sequence_length
        ]

    def parameter_count(self):
        """
        Return number of trainable parameters.
        """

        return self.weight.size