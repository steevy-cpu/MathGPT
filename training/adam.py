import numpy as np


try:
    from .parameters import named_parameters

except ImportError:
    from parameters import named_parameters


def adam_update(
    parameter,
    gradient,
    first_moment,
    second_moment,
    step,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """
    Apply one Adam update in place.

    parameter:
        Trainable parameter.

    gradient:
        Current gradient.

    first_moment:
        Exponential moving average
        of gradients.

    second_moment:
        Exponential moving average
        of squared gradients.

    step:
        Current optimization step,
        starting from 1.
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
    # 1. First moment
    #
    # Momentum-like memory of gradients.
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
    # 2. Second moment
    #
    # Memory of squared gradient magnitudes.
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
    # 4. Adam update
    # ========================================================

    update = (
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

    parameter -= update.astype(
        np.float32,
        copy=False,
    )


class Adam:
    """
    Adam optimizer for MathGPT.

    Every parameter gets:

        m = first-moment state
        v = second-moment state

    These states persist across training steps.
    """

    def __init__(
        self,
        model,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
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

        self.step_count = 0

        # ====================================================
        # Adam state
        #
        # Every parameter has its own m and v arrays.
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
        Perform one Adam optimization step.
        """

        # Adam's t begins at 1.
        self.step_count += 1

        for (
            name,
            parameter,
            gradient,
        ) in named_parameters(
            self.model
        ):

            if (
                parameter.shape
                != gradient.shape
            ):
                raise ValueError(
                    f"Gradient shape mismatch "
                    f"for {name}."
                )

            if not np.all(
                np.isfinite(
                    gradient
                )
            ):
                raise FloatingPointError(
                    f"Non-finite gradient "
                    f"found in {name}."
                )

            adam_update(
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
            )

    def zero_grad(
        self,
    ):
        """
        Reset model gradients.

        IMPORTANT:
        This does NOT reset Adam's
        m and v state.
        """

        self.model.zero_grad()