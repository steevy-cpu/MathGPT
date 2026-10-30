import numpy as np


try:
    from .parameters import named_parameters

except ImportError:
    from parameters import named_parameters


def should_apply_weight_decay(
    parameter,
):
    """
    Apply weight decay to matrix-like
    parameters.

    2D+:
        embeddings
        linear weights

    1D:
        biases
        LayerNorm parameters
    """

    return (
        parameter.ndim >= 2
    )


def adamw_update(
    parameter,
    gradient,
    first_moment,
    second_moment,
    step,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    apply_weight_decay=True,
):
    """
    Perform one AdamW update.

    AdamW separates:

        Adam adaptive update

    from:

        weight decay
    """

    if (
        parameter.shape
        != gradient.shape
    ):
        raise ValueError(
            "parameter and gradient "
            "must have identical shapes."
        )

    if (
        first_moment.shape
        != parameter.shape
    ):
        raise ValueError(
            "first_moment shape mismatch."
        )

    if (
        second_moment.shape
        != parameter.shape
    ):
        raise ValueError(
            "second_moment shape mismatch."
        )

    if step <= 0:

        raise ValueError(
            "step must be >= 1."
        )

    if learning_rate <= 0:

        raise ValueError(
            "learning_rate must be positive."
        )

    if weight_decay < 0:

        raise ValueError(
            "weight_decay cannot be negative."
        )

    if not (
        0.0 <= beta1 < 1.0
    ):

        raise ValueError(
            "beta1 must be in [0, 1)."
        )

    if not (
        0.0 <= beta2 < 1.0
    ):

        raise ValueError(
            "beta2 must be in [0, 1)."
        )

    # ========================================================
    # 1. Adam first moment
    # ========================================================

    first_moment *= np.float32(
        beta1
    )

    first_moment += (
        np.float32(
            1.0 - beta1
        )
        * gradient
    )

    # ========================================================
    # 2. Adam second moment
    # ========================================================

    second_moment *= np.float32(
        beta2
    )

    second_moment += (
        np.float32(
            1.0 - beta2
        )
        * (
            gradient
            * gradient
        )
    )

    # ========================================================
    # 3. Bias correction
    # ========================================================

    bias_correction_1 = (
        1.0
        - beta1 ** step
    )

    bias_correction_2 = (
        1.0
        - beta2 ** step
    )

    first_moment_hat = (
        first_moment
        / np.float32(
            bias_correction_1
        )
    )

    second_moment_hat = (
        second_moment
        / np.float32(
            bias_correction_2
        )
    )

    # ========================================================
    # 4. DECOUPLED weight decay
    #
    # Notice:
    #
    # weight decay was NOT added to gradient.
    #
    # Therefore m and v contain only information
    # about the loss gradient.
    # ========================================================

    if (
        apply_weight_decay
        and
        weight_decay > 0.0
    ):

        decay_factor = np.float32(
            1.0
            - learning_rate
            * weight_decay
        )

        parameter *= (
            decay_factor
        )

    # ========================================================
    # 5. Adam adaptive update
    # ========================================================

    adaptive_update = (
        np.float32(
            learning_rate
        )
        * first_moment_hat
        / (
            np.sqrt(
                second_moment_hat
            )
            + np.float32(
                epsilon
            )
        )
    )

    parameter -= adaptive_update.astype(
        np.float32,
        copy=False,
    )


class AdamW:
    """
    AdamW optimizer for MathGPT.

    Maintains Adam's:

        first moment  m
        second moment v

    while applying decoupled weight decay
    to matrix-like parameters.
    """

    def __init__(
        self,
        model,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    ):

        if learning_rate <= 0:

            raise ValueError(
                "learning_rate must be positive."
            )

        if weight_decay < 0:

            raise ValueError(
                "weight_decay cannot be negative."
            )

        if not (
            0.0 <= beta1 < 1.0
        ):

            raise ValueError(
                "beta1 must be in [0, 1)."
            )

        if not (
            0.0 <= beta2 < 1.0
        ):

            raise ValueError(
                "beta2 must be in [0, 1)."
            )

        self.model = model

        self.learning_rate = float(
            learning_rate
        )

        self.beta1 = float(
            beta1
        )

        self.beta2 = float(
            beta2
        )

        self.epsilon = float(
            epsilon
        )

        self.weight_decay = float(
            weight_decay
        )

        self.step_count = 0

        # ====================================================
        # Adam state
        # ====================================================

        self.first_moments = {}

        self.second_moments = {}

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            self.model
        ):

            self.first_moments[
                name
            ] = np.zeros_like(
                parameter,
                dtype=np.float32,
            )

            self.second_moments[
                name
            ] = np.zeros_like(
                parameter,
                dtype=np.float32,
            )

    def step(
        self,
    ):
        """
        Perform one AdamW optimization step.
        """

        self.step_count += 1

        for (
            name,
            parameter,
            gradient,
        ) in named_parameters(
            self.model
        ):

            if not np.all(
                np.isfinite(
                    gradient
                )
            ):

                raise FloatingPointError(
                    f"Non-finite gradient "
                    f"found in {name}."
                )

            adamw_update(
                parameter=parameter,
                gradient=gradient,
                first_moment=(
                    self.first_moments[
                        name
                    ]
                ),
                second_moment=(
                    self.second_moments[
                        name
                    ]
                ),
                step=self.step_count,
                learning_rate=(
                    self.learning_rate
                ),
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon,
                weight_decay=(
                    self.weight_decay
                ),
                apply_weight_decay=(
                    should_apply_weight_decay(
                        parameter
                    )
                ),
            )

    def zero_grad(
        self,
    ):
        """
        Reset gradients only.

        AdamW moment state survives.
        """

        self.model.zero_grad()