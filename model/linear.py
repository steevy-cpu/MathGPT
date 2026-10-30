import numpy as np


class Linear:
    """
    Fully connected linear layer.

    Forward:

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

        # ----------------------------------------------------
        # Gradient storage
        # ----------------------------------------------------

        self.dweight = np.zeros_like(
            self.weight,
            dtype=np.float32,
        )

        if use_bias:

            self.dbias = np.zeros_like(
                self.bias,
                dtype=np.float32,
            )

        else:

            self.dbias = None

        # ----------------------------------------------------
        # Forward-pass cache
        # ----------------------------------------------------

        self.input_cache = None

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

        # Save input for backward pass.
        self.input_cache = x

        output = (
            x
            @ self.weight
        )

        if self.use_bias:

            output = (
                output
                + self.bias
            )

        return output.astype(
            np.float32,
            copy=False,
        )

    def backward(
        self,
        grad_output,
    ):
        """
        Backpropagate through:

            Y = XW + b

        grad_output:
            dL/dY

        Returns:
            grad_input = dL/dX

        Also stores:

            self.dweight = dL/dW

            self.dbias = dL/db
        """

        if self.input_cache is None:

            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        if (
            grad_output.shape[-1]
            != self.output_dim
        ):

            raise ValueError(
                "Last grad_output dimension "
                f"must equal {self.output_dim}."
            )

        x = self.input_cache

        if (
            grad_output.shape[:-1]
            != x.shape[:-1]
        ):

            raise ValueError(
                "grad_output leading dimensions "
                "must match the cached input."
            )

        # ----------------------------------------------------
        # Flatten all leading dimensions.
        #
        # Example:
        #
        # x:
        # (batch, sequence, input_dim)
        #
        # becomes:
        #
        # (batch * sequence, input_dim)
        # ----------------------------------------------------

        x_flat = x.reshape(
            -1,
            self.input_dim,
        )

        grad_output_flat = (
            grad_output.reshape(
                -1,
                self.output_dim,
            )
        )

        # ----------------------------------------------------
        # Gradient with respect to weights
        #
        # dW = X^T @ dY
        #
        # (input_dim, N)
        # @
        # (N, output_dim)
        #
        # =
        #
        # (input_dim, output_dim)
        # ----------------------------------------------------

        self.dweight = (
            x_flat.T
            @ grad_output_flat
        ).astype(
            np.float32,
            copy=False,
        )

        # ----------------------------------------------------
        # Gradient with respect to bias
        #
        # Every example used the same bias,
        # so all contributions are summed.
        # ----------------------------------------------------

        if self.use_bias:

            self.dbias = np.sum(
                grad_output_flat,
                axis=0,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # Gradient with respect to input
        #
        # dX = dY @ W^T
        # ----------------------------------------------------

        grad_input = (
            grad_output
            @ self.weight.T
        )

        return grad_input.astype(
            np.float32,
            copy=False,
        )

    def zero_grad(
        self,
    ):
        """
        Reset parameter gradients to zero.
        """

        self.dweight.fill(
            0.0
        )

        if self.use_bias:

            self.dbias.fill(
                0.0
            )

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