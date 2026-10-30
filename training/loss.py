import numpy as np


def _validate_inputs(
    logits,
    targets,
):
    """
    Shared validation for cross-entropy.
    """

    if logits.ndim != 3:

        raise ValueError(
            "logits must have shape "
            "(batch, sequence, vocab_size)."
        )

    if targets.ndim != 2:

        raise ValueError(
            "targets must have shape "
            "(batch, sequence)."
        )

    if (
        logits.shape[:2]
        != targets.shape
    ):

        raise ValueError(
            "logits batch/sequence dimensions "
            "must match targets."
        )

    vocab_size = (
        logits.shape[-1]
    )

    if targets.min() < 0:

        raise ValueError(
            "Target IDs cannot be negative."
        )

    if targets.max() >= vocab_size:

        raise ValueError(
            "Target ID exceeds vocabulary size."
        )


def cross_entropy_loss(
    logits,
    targets,
):
    """
    Mean next-token cross-entropy loss.

    logits:
        (batch, sequence, vocab_size)

    targets:
        (batch, sequence)

    Returns:
        scalar float32 loss
    """

    _validate_inputs(
        logits,
        targets,
    )

    logits = np.asarray(
        logits,
        dtype=np.float32,
    )

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    # Numerical stability
    max_logits = np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    shifted_logits = (
        logits
        - max_logits
    )

    log_sum_exp = np.log(
        np.sum(
            np.exp(
                shifted_logits
            ),
            axis=-1,
            keepdims=True,
        )
    )

    log_probs = (
        shifted_logits
        - log_sum_exp
    )

    batch_size = (
        targets.shape[0]
    )

    sequence_length = (
        targets.shape[1]
    )

    batch_indices = np.arange(
        batch_size
    )[:, None]

    sequence_indices = np.arange(
        sequence_length
    )[None, :]

    correct_log_probs = (
        log_probs[
            batch_indices,
            sequence_indices,
            targets,
        ]
    )

    loss = (
        -np.mean(
            correct_log_probs
        )
    )

    return np.float32(
        loss
    )


def cross_entropy_backward(
    logits,
    targets,
):
    """
    Gradient of mean cross-entropy loss
    with respect to logits.

    Returns:

        dL/dlogits

    Shape:
        same as logits
    """

    _validate_inputs(
        logits,
        targets,
    )

    logits = np.asarray(
        logits,
        dtype=np.float32,
    )

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Stable Softmax
    # --------------------------------------------------------

    max_logits = np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    shifted_logits = (
        logits
        - max_logits
    )

    exponentials = np.exp(
        shifted_logits
    )

    probabilities = (
        exponentials
        / np.sum(
            exponentials,
            axis=-1,
            keepdims=True,
        )
    )

    # --------------------------------------------------------
    # Gradient:
    #
    # softmax(logits) - one_hot(target)
    # --------------------------------------------------------

    grad_logits = (
        probabilities.copy()
    )

    batch_size = (
        targets.shape[0]
    )

    sequence_length = (
        targets.shape[1]
    )

    batch_indices = np.arange(
        batch_size
    )[:, None]

    sequence_indices = np.arange(
        sequence_length
    )[None, :]

    grad_logits[
        batch_indices,
        sequence_indices,
        targets,
    ] -= np.float32(
        1.0
    )

    # --------------------------------------------------------
    # Our loss uses MEAN rather than SUM.
    #
    # Number of predictions:
    #
    # batch × sequence
    # --------------------------------------------------------

    number_of_predictions = np.float32(
        batch_size
        * sequence_length
    )

    grad_logits /= (
        number_of_predictions
    )

    return grad_logits.astype(
        np.float32,
        copy=False,
    )


def cross_entropy_loss_and_backward(
    logits,
    targets,
):
    """
    Convenience function.

    Returns:

        loss,
        grad_logits
    """

    loss = cross_entropy_loss(
        logits,
        targets,
    )

    grad_logits = (
        cross_entropy_backward(
            logits,
            targets,
        )
    )

    return (
        loss,
        grad_logits,
    )