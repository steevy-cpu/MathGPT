import numpy as np


class Linear:
    """
    Fully connected linear layer.

    Computes:

        Y = XW + b

    Weight shape:
        (input_dim, output_dim)

    Bias shape:
        (output_dim,)

    Input shape:
        (..., input_dim)

    Output shape:
        (..., output_dim)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        use_bias=True,
        seed=42,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        rng = np.random.default_rng(
            seed
        )

        # Small random initialization.
        self.weight = rng.normal(
            loc=0.0,
            scale=0.02,
            size=(
                input_dim,
                output_dim,
            ),
        ).astype(
            np.float32
        )

        if use_bias:
            self.bias = np.zeros(
                output_dim,
                dtype=np.float32,
            )

        else:
            self.bias = None

    def forward(
        self,
        x,
    ):
        """
        Apply:

            Y = XW + b
        """

        if x.shape[-1] != self.input_dim:

            raise ValueError(
                "Last input dimension must "
                f"equal {self.input_dim}, "
                f"got {x.shape[-1]}."
            )

        output = (
            x @ self.weight
        )

        if self.use_bias:

            output = (
                output
                + self.bias
            )

        return output

    def parameter_count(
        self,
    ):
        """
        Return number of trainable parameters.
        """

        count = (
            self.weight.size
        )

        if self.use_bias:

            count += (
                self.bias.size
            )

        return count