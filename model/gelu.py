import numpy as np


def gelu(
    x,
):
    """
    GELU activation using the common
    tanh approximation.

    Input:
        NumPy array

    Output:
        Same shape as input.
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    coefficient = np.float32(
        np.sqrt(
            2.0
            / np.pi
        )
    )

    cubic_coefficient = np.float32(
        0.044715
    )

    inner = (
        coefficient
        * (
            x
            + cubic_coefficient
            * (x ** 3)
        )
    )

    output = (
        np.float32(0.5)
        * x
        * (
            np.float32(1.0)
            + np.tanh(
                inner
            )
        )
    )

    return output.astype(
        np.float32,
        copy=False,
    )


def gelu_backward(
    x,
    grad_output,
):
    """
    Backward pass for GELU.

    Parameters
    ----------
    x:
        Original input to GELU.

    grad_output:
        Gradient arriving from the next layer:

            dL/dY

    Returns
    -------
    grad_input:

        dL/dX
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    grad_output = np.asarray(
        grad_output,
        dtype=np.float32,
    )

    if (
        x.shape
        != grad_output.shape
    ):

        raise ValueError(
            "x and grad_output must "
            "have identical shapes."
        )

    coefficient = np.float32(
        np.sqrt(
            2.0
            / np.pi
        )
    )

    cubic_coefficient = np.float32(
        0.044715
    )

    # --------------------------------------------------------
    # u = sqrt(2/pi) * (x + 0.044715*x^3)
    # --------------------------------------------------------

    inner = (
        coefficient
        * (
            x
            + cubic_coefficient
            * (x ** 3)
        )
    )

    tanh_inner = np.tanh(
        inner
    )

    # --------------------------------------------------------
    # derivative of the inner expression:
    #
    # du/dx
    # =
    # sqrt(2/pi)
    # *
    # (1 + 3*0.044715*x^2)
    # --------------------------------------------------------

    inner_derivative = (
        coefficient
        * (
            np.float32(1.0)
            + np.float32(3.0)
            * cubic_coefficient
            * (x ** 2)
        )
    )

    # --------------------------------------------------------
    # derivative of GELU
    # --------------------------------------------------------

    gelu_derivative = (
        np.float32(0.5)
        * (
            np.float32(1.0)
            + tanh_inner
        )
        +
        np.float32(0.5)
        * x
        * (
            np.float32(1.0)
            - tanh_inner ** 2
        )
        * inner_derivative
    )

    # --------------------------------------------------------
    # Chain rule:
    #
    # dL/dX
    # =
    # dL/dY * dY/dX
    # --------------------------------------------------------

    grad_input = (
        grad_output
        * gelu_derivative
    )

    return grad_input.astype(
        np.float32,
        copy=False,
    )