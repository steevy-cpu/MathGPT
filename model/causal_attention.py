import numpy as np

try:
    from .softmax import softmax
except ImportError:
    from softmax import softmax


def causal_attention(
    query,
    key,
    value,
):
    """
    Scaled dot-product causal attention.

    Expected shapes:

        query:
            (batch, sequence, head_dim)

        key:
            (batch, sequence, head_dim)

        value:
            (batch, sequence, head_dim)

    Returns:

        output:
            (batch, sequence, head_dim)

        attention_weights:
            (batch, sequence, sequence)
    """

    # --------------------------------------------------------
    # Validate shapes.
    # --------------------------------------------------------

    if (
        query.ndim != 3
        or key.ndim != 3
        or value.ndim != 3
    ):
        raise ValueError(
            "query, key, and value must all "
            "have 3 dimensions."
        )

    if (
        query.shape
        != key.shape
        or query.shape
        != value.shape
    ):
        raise ValueError(
            "query, key, and value must "
            "have identical shapes."
        )

    batch_size, sequence_length, head_dim = (
        query.shape
    )

    # --------------------------------------------------------
    # Step 1:
    #
    # Compare every Query with every Key.
    #
    # Q:
    #   (batch, sequence, head_dim)
    #
    # K transpose:
    #   (batch, head_dim, sequence)
    #
    # scores:
    #   (batch, sequence, sequence)
    # --------------------------------------------------------

    key_transposed = np.transpose(
        key,
        (0, 2, 1),
    )

    scores = (
        query
        @ key_transposed
    )

    # --------------------------------------------------------
    # Step 2:
    # Scale by sqrt(head_dim).
    # --------------------------------------------------------

    scale = np.sqrt(
        head_dim
    )

    scores = (
        scores
        / scale
    )

    # --------------------------------------------------------
    # Step 3:
    # Build causal mask.
    #
    # np.triu(..., k=1) marks everything
    # ABOVE the diagonal.
    #
    # Those are future tokens.
    # --------------------------------------------------------

    causal_mask = np.triu(
        np.ones(
            (
                sequence_length,
                sequence_length,
            ),
            dtype=bool,
        ),
        k=1,
    )

    # Future positions receive -infinity.
    masked_scores = np.where(
        causal_mask,
        -np.inf,
        scores,
    )

    # --------------------------------------------------------
    # Step 4:
    # Convert scores into attention probabilities.
    #
    # Softmax operates across the Key dimension.
    # --------------------------------------------------------

    attention_weights = softmax(
        masked_scores,
        axis=-1,
    )

    # --------------------------------------------------------
    # Step 5:
    #
    # Use the attention probabilities to create
    # a weighted combination of Value vectors.
    #
    # attention_weights:
    #   (batch, sequence, sequence)
    #
    # value:
    #   (batch, sequence, head_dim)
    #
    # output:
    #   (batch, sequence, head_dim)
    # --------------------------------------------------------

    output = (
        attention_weights
        @ value
    )

    return (
        output,
        attention_weights,
    )