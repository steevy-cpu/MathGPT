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

    cubic_coefficient = (
        np.float32(
            0.044715
        )
    )

    output = (
        np.float32(0.5)
        * x
        * (
            np.float32(1.0)
            + np.tanh(
                coefficient
                * (
                    x
                    + cubic_coefficient
                    * (x ** 3)
                )
            )
        )
    )

    return output.astype(
        np.float32,
        copy=False,
    )