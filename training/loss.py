import numpy as np


def cross_entropy_loss(
    logits,
    targets,
):
    """
    Mean next-token cross-entropy loss.

    logits:
        shape:
            (batch, sequence, vocab_size)

    targets:
        shape:
            (batch, sequence)

    Returns:
        scalar float32 loss
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

    # --------------------------------------------------------
    # Numerical stability
    #
    # subtract max logit at each token position
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

    # --------------------------------------------------------
    # log(sum(exp(logits)))
    # --------------------------------------------------------

    log_sum_exp = np.log(
        np.sum(
            np.exp(
                shifted_logits
            ),
            axis=-1,
            keepdims=True,
        )
    )

    # --------------------------------------------------------
    # log-softmax
    # --------------------------------------------------------

    log_probs = (
        shifted_logits
        - log_sum_exp
    )

    # --------------------------------------------------------
    # Select the correct next-token log probability
    # --------------------------------------------------------

    batch_size = (
        targets.shape[0]
    )

    sequence_length = (
        targets.shape[1]
    )

    batch_indices = (
        np.arange(
            batch_size
        )[:, None]
    )

    sequence_indices = (
        np.arange(
            sequence_length
        )[None, :]
    )

    correct_log_probs = (
        log_probs[
            batch_indices,
            sequence_indices,
            targets,
        ]
    )

    # --------------------------------------------------------
    # Negative log likelihood
    # --------------------------------------------------------

    loss = (
        -np.mean(
            correct_log_probs
        )
    )

    return np.float32(
        loss
    )