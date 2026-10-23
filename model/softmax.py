import numpy as np


def softmax(
    x,
    axis=-1,
):
    """
    Numerically stable softmax.

    Converts raw scores into values
    between 0 and 1 that sum to 1
    along the selected axis.

    Parameters
    ----------
    x:
        NumPy array containing raw scores.

    axis:
        Dimension over which softmax
        should be calculated.

    Returns
    -------
    NumPy array with the same shape as x.
    """

    # --------------------------------------------------------
    # Numerical stability trick.
    #
    # Softmax(x) == Softmax(x - constant)
    #
    # Subtracting the maximum prevents exponentials
    # from becoming extremely large.
    # --------------------------------------------------------

    maximum = np.max(
        x,
        axis=axis,
        keepdims=True,
    )

    shifted = (
        x
        - maximum
    )

    exponentials = np.exp(
        shifted
    )

    denominator = np.sum(
        exponentials,
        axis=axis,
        keepdims=True,
    )

    probabilities = (
        exponentials
        / denominator
    )

    return probabilities