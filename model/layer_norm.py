import numpy as np


class LayerNorm:
    """
    Layer normalization over the final dimension.

    Forward:

        mean = mean(x)

        variance = mean((x - mean)^2)

        x_hat =
            (x - mean)
            / sqrt(variance + epsilon)

        output =
            gamma * x_hat
            + beta
    """

    def __init__(
        self,
        d_model,
        epsilon=1e-5,
    ):
        self.d_model = d_model
        self.epsilon = np.float32(
            epsilon
        )

        # Trainable parameters
        self.gamma = np.ones(
            d_model,
            dtype=np.float32,
        )

        self.beta = np.zeros(
            d_model,
            dtype=np.float32,
        )

        # Gradient storage
        self.dgamma = np.zeros_like(
            self.gamma,
            dtype=np.float32,
        )

        self.dbeta = np.zeros_like(
            self.beta,
            dtype=np.float32,
        )

        # Forward cache
        self.x_hat_cache = None
        self.inv_std_cache = None

    def forward(
        self,
        x,
    ):
        """
        Normalize across the final dimension.
        """

        if x.shape[-1] != self.d_model:

            raise ValueError(
                "Last input dimension must "
                f"equal {self.d_model}, "
                f"got {x.shape[-1]}."
            )

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        mean = np.mean(
            x,
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )

        centered = (
            x
            - mean
        )

        variance = np.mean(
            centered ** 2,
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )

        inv_std = (
            np.float32(1.0)
            / np.sqrt(
                variance
                + self.epsilon
            )
        )

        x_hat = (
            centered
            * inv_std
        )

        output = (
            self.gamma
            * x_hat
            + self.beta
        )

        # Cache for backward
        self.x_hat_cache = x_hat
        self.inv_std_cache = inv_std

        return output.astype(
            np.float32,
            copy=False,
        )

    def backward(
        self,
        grad_output,
    ):
        """
        Backward pass through LayerNorm.

        grad_output:
            dL/dY

        Returns:
            dL/dX

        Stores:
            dL/dgamma
            dL/dbeta
        """

        if (
            self.x_hat_cache is None
            or self.inv_std_cache is None
        ):

            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad_output = np.asarray(
            grad_output,
            dtype=np.float32,
        )

        if (
            grad_output.shape
            != self.x_hat_cache.shape
        ):

            raise ValueError(
                "grad_output shape must "
                "match the forward output shape."
            )

        x_hat = (
            self.x_hat_cache
        )

        inv_std = (
            self.inv_std_cache
        )

        # Sum across every dimension
        # except the final feature dimension.
        reduction_axes = tuple(
            range(
                grad_output.ndim - 1
            )
        )

        # dgamma
        self.dgamma = np.sum(
            grad_output
            * x_hat,
            axis=reduction_axes,
            dtype=np.float32,
        )

        # dbeta
        self.dbeta = np.sum(
            grad_output,
            axis=reduction_axes,
            dtype=np.float32,
        )

        # Gradient through:
        # y = gamma * x_hat + beta
        grad_x_hat = (
            grad_output
            * self.gamma
        )

        n = np.float32(
            self.d_model
        )

        sum_grad = np.sum(
            grad_x_hat,
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )

        sum_grad_xhat = np.sum(
            grad_x_hat
            * x_hat,
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )

        # Compact LayerNorm derivative
        grad_input = (
            inv_std
            / n
            * (
                n
                * grad_x_hat
                - sum_grad
                - x_hat
                * sum_grad_xhat
            )
        )

        return grad_input.astype(
            np.float32,
            copy=False,
        )

    def zero_grad(
        self,
    ):
        """
        Reset parameter gradients.
        """

        self.dgamma.fill(
            0.0
        )

        self.dbeta.fill(
            0.0
        )

    def parameter_count(
        self,
    ):
        return (
            self.gamma.size
            + self.beta.size
        )