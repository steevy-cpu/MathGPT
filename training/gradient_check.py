import numpy as np


def numerical_gradient(
    function,
    x,
    epsilon=1e-4,
):
    """
    Estimate the derivative of a scalar function
    using a central finite difference.

    function:
        callable receiving one scalar

    x:
        scalar point to evaluate

    returns:
        approximate derivative
    """

    x = np.float32(
        x
    )

    epsilon = np.float32(
        epsilon
    )

    positive = function(
        x + epsilon
    )

    negative = function(
        x - epsilon
    )

    gradient = (
        positive
        - negative
    ) / (
        np.float32(2.0)
        * epsilon
    )

    return np.float32(
        gradient
    )