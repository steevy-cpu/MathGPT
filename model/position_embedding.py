import numpy as np


class PositionEmbedding:
    """
    Learned positional embeddings.

    Weight shape:
        (context_length, d_model)
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

        self.dweight = np.zeros_like(
            self.weight,
            dtype=np.float32,
        )

        self.sequence_length_cache = None

    def forward(
        self,
        sequence_length,
    ):
        """
        Return positions:

            0 ... sequence_length - 1
        """

        if sequence_length <= 0:

            raise ValueError(
                "sequence_length must "
                "be positive."
            )

        if (
            sequence_length
            > self.context_length
        ):

            raise ValueError(
                "sequence_length exceeds "
                "maximum context length."
            )

        self.sequence_length_cache = (
            sequence_length
        )

        return (
            self.weight[
                :sequence_length
            ]
        )

    def backward(
        self,
        grad_output,
    ):
        """
        Backward through positional embeddings.

        grad_output may be:

            (sequence, d_model)

        or, through broadcasting:

            (batch, sequence, d_model)

        All batch contributions for a given
        position are summed together.
        """

        if (
            self.sequence_length_cache
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

        sequence_length = (
            self.sequence_length_cache
        )

        if (
            grad_output.shape[-2:]
            != (
                sequence_length,
                self.d_model,
            )
        ):

            raise ValueError(
                "grad_output has "
                "incorrect final dimensions."
            )

        # ----------------------------------------------------
        # Sum all leading dimensions.
        #
        # Typical case:
        #
        # (batch, sequence, d_model)
        #
        # ↓ sum batch
        #
        # (sequence, d_model)
        # ----------------------------------------------------

        reduction_axes = tuple(
            range(
                grad_output.ndim - 2
            )
        )

        if reduction_axes:

            position_gradient = np.sum(
                grad_output,
                axis=reduction_axes,
                dtype=np.float32,
            )

        else:

            position_gradient = (
                grad_output
            )

        self.dweight.fill(
            0.0
        )

        self.dweight[
            :sequence_length
        ] = position_gradient

        return None

    def zero_grad(
        self,
    ):
        self.dweight.fill(
            0.0
        )

    def parameter_count(
        self,
    ):
        return self.weight.size