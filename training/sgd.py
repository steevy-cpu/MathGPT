import numpy as np

from parameters import named_parameters


def sgd_update(
    parameter,
    gradient,
    learning_rate,
):
    """
    Apply one SGD update in place:

        parameter -= learning_rate * gradient
    """

    if (
        parameter.shape
        != gradient.shape
    ):
        raise ValueError(
            "parameter and gradient "
            "must have identical shapes."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    parameter -= (
        np.float32(
            learning_rate
        )
        * gradient
    )


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Usage:

        optimizer = SGD(
            model,
            learning_rate=0.01,
        )

        loss.backward(...)
        optimizer.step()
        optimizer.zero_grad()
    """

    def __init__(
        self,
        model,
        learning_rate,
    ):
        if learning_rate <= 0:

            raise ValueError(
                "learning_rate must be positive."
            )

        self.model = model

        self.learning_rate = np.float32(
            learning_rate
        )

        self.step_count = 0

    def step(
        self,
    ):
        """
        Update every trainable parameter.
        """

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

            sgd_update(
                parameter=parameter,
                gradient=gradient,
                learning_rate=(
                    self.learning_rate
                ),
            )

        self.step_count += 1

    def zero_grad(
        self,
    ):
        """
        Reset every model gradient.
        """

        self.model.zero_grad()