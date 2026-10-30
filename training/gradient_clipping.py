import numpy as np


try:
    from .parameters import named_parameters

except ImportError:
    from parameters import named_parameters


def global_norm_from_gradients(
    gradients,
):
    """
    Compute the global L2 norm of
    a collection of gradient arrays.

    Conceptually, all gradient arrays are
    flattened and joined into one giant vector:

        ||g|| =
        sqrt(
            sum(
                every gradient value squared
            )
        )

    Float64 is used for the accumulation to
    make the norm calculation more numerically
    stable.

    The gradients themselves remain float32.
    """

    total_squared = 0.0

    for gradient in gradients:

        if not np.all(
            np.isfinite(
                gradient
            )
        ):
            raise FloatingPointError(
                "Non-finite gradient detected."
            )

        gradient_64 = (
            gradient.astype(
                np.float64,
                copy=False,
            )
        )

        total_squared += float(
            np.sum(
                gradient_64
                * gradient_64
            )
        )

    return float(
        np.sqrt(
            total_squared
        )
    )


def clip_gradients(
    gradients,
    max_norm,
    epsilon=1e-6,
):
    """
    Clip a collection of gradients using
    global L2 norm clipping.

    Returns:

        original_norm,
        scale_factor

    Gradients are modified IN PLACE.
    """

    if max_norm <= 0:

        raise ValueError(
            "max_norm must be positive."
        )

    if epsilon < 0:

        raise ValueError(
            "epsilon cannot be negative."
        )

    # Convert to list because we need to:
    #
    # 1. inspect them
    # 2. potentially modify them afterward

    gradients = list(
        gradients
    )

    original_norm = (
        global_norm_from_gradients(
            gradients
        )
    )

    # ========================================================
    # No clipping needed
    # ========================================================

    if original_norm <= max_norm:

        return (
            original_norm,
            1.0,
        )

    # ========================================================
    # Calculate one GLOBAL scale factor
    # ========================================================

    scale_factor = (
        max_norm
        / (
            original_norm
            + epsilon
        )
    )

    scale_float32 = np.float32(
        scale_factor
    )

    # ========================================================
    # Scale every gradient by the SAME factor
    # ========================================================

    for gradient in gradients:

        gradient *= (
            scale_float32
        )

    return (
        original_norm,
        float(
            scale_factor
        ),
    )


def global_grad_norm(
    model,
):
    """
    Compute global gradient norm
    for an entire MathGPT model.
    """

    gradients = (
        gradient

        for (
            _,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    )

    return global_norm_from_gradients(
        gradients
    )


def clip_grad_norm(
    model,
    max_norm,
    epsilon=1e-6,
):
    """
    Clip every gradient in MathGPT using
    one global L2 norm.

    Returns:

        original_norm,
        scale_factor
    """

    gradients = (
        gradient

        for (
            _,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    )

    return clip_gradients(
        gradients=gradients,
        max_norm=max_norm,
        epsilon=epsilon,
    )