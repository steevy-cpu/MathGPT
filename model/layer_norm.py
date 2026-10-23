import numpy as np


class LayerNorm:
    """
    Layer normalization over the final dimension.

    Input shape:
        (..., d_model)

    Output shape:
        (..., d_model)

    Learnable parameters:
        gamma: (d_model,)
        beta:  (d_model,)
    """

    def __init__(
        self,
        d_model,
        epsilon=1e-5,
    ):
        self.d_model = d_model
        self.epsilon = epsilon

        # Learnable scale.
        self.gamma = np.ones(
            d_model,
            dtype=np.float32,
        )

        # Learnable shift.
        self.beta = np.zeros(
            d_model,
            dtype=np.float32,
        )

    def forward(
        self,
        x,
    ):
        """
        Normalize each vector across
        its final dimension.
        """

        if x.shape[-1] != self.d_model:

            raise ValueError(
                "Last input dimension must "
                f"equal {self.d_model}, "
                f"got {x.shape[-1]}."
            )

        # Keep the final dimension so NumPy
        # broadcasting works naturally.
        mean = np.mean(
            x,
            axis=-1,
            keepdims=True,
        )

        variance = np.mean(
            (x - mean) ** 2,
            axis=-1,
            keepdims=True,
        )

        normalized = (
            (x - mean)
            / np.sqrt(
                variance
                + self.epsilon
            )
        )

        output = (
            self.gamma
            * normalized
            + self.beta
        )

        return output

    def parameter_count(
        self,
    ):
        """
        gamma + beta
        """

        return (
            self.gamma.size
            + self.beta.size
        )