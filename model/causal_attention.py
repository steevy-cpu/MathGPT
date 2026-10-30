import numpy as np

try:
    from .softmax import softmax

except ImportError:
    from softmax import softmax


def build_causal_mask(
    sequence_length,
):
    """
    True values represent forbidden
    future-token positions.

    Example for sequence_length = 4:

        False  True   True   True
        False  False  True   True
        False  False  False  True
        False  False  False  False
    """

    return np.triu(
        np.ones(
            (
                sequence_length,
                sequence_length,
            ),
            dtype=bool,
        ),
        k=1,
    )


def causal_attention(
    query,
    key,
    value,
):
    """
    Scaled dot-product causal attention.

    Shapes:

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

    if (
        query.ndim != 3
        or key.ndim != 3
        or value.ndim != 3
    ):

        raise ValueError(
            "query, key, and value must "
            "all have 3 dimensions."
        )

    if (
        query.shape != key.shape
        or query.shape != value.shape
    ):

        raise ValueError(
            "query, key, and value must "
            "have identical shapes."
        )

    query = np.asarray(
        query,
        dtype=np.float32,
    )

    key = np.asarray(
        key,
        dtype=np.float32,
    )

    value = np.asarray(
        value,
        dtype=np.float32,
    )

    (
        batch_size,
        sequence_length,
        head_dim,
    ) = query.shape

    # ========================================================
    # QK^T
    # ========================================================

    key_transposed = np.transpose(
        key,
        (
            0,
            2,
            1,
        ),
    )

    scores = (
        query
        @ key_transposed
    )

    # ========================================================
    # Scale
    # ========================================================

    scale = np.float32(
        np.sqrt(
            np.float32(
                head_dim
            )
        )
    )

    scores = (
        scores
        / scale
    )

    # ========================================================
    # Causal mask
    # ========================================================

    causal_mask = (
        build_causal_mask(
            sequence_length
        )
    )

    masked_scores = np.where(
        causal_mask,
        np.float32(
            -np.inf
        ),
        scores,
    ).astype(
        np.float32,
        copy=False,
    )

    # ========================================================
    # Softmax
    # ========================================================

    attention_weights = softmax(
        masked_scores,
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )

    # ========================================================
    # Weighted Values
    # ========================================================

    output = (
        attention_weights
        @ value
    )

    return (
        output.astype(
            np.float32,
            copy=False,
        ),
        attention_weights,
    )


def causal_attention_backward(
    query,
    key,
    value,
    attention_weights,
    grad_output,
):
    """
    Backward pass for scaled dot-product
    causal self-attention.

    Forward:

        scores =
            Q @ K^T / sqrt(head_dim)

        A =
            softmax(causal_mask(scores))

        output =
            A @ V

    Parameters
    ----------
    query:
        (batch, sequence, head_dim)

    key:
        (batch, sequence, head_dim)

    value:
        (batch, sequence, head_dim)

    attention_weights:
        output of the forward Softmax

        (batch, sequence, sequence)

    grad_output:
        dL/dOutput

        (batch, sequence, head_dim)

    Returns
    -------
    grad_query:
        dL/dQ

    grad_key:
        dL/dK

    grad_value:
        dL/dV
    """

    query = np.asarray(
        query,
        dtype=np.float32,
    )

    key = np.asarray(
        key,
        dtype=np.float32,
    )

    value = np.asarray(
        value,
        dtype=np.float32,
    )

    attention_weights = np.asarray(
        attention_weights,
        dtype=np.float32,
    )

    grad_output = np.asarray(
        grad_output,
        dtype=np.float32,
    )

    if (
        query.shape != key.shape
        or query.shape != value.shape
        or query.shape != grad_output.shape
    ):

        raise ValueError(
            "query, key, value, and "
            "grad_output must have "
            "identical shapes."
        )

    (
        batch_size,
        sequence_length,
        head_dim,
    ) = query.shape

    expected_attention_shape = (
        batch_size,
        sequence_length,
        sequence_length,
    )

    if (
        attention_weights.shape
        != expected_attention_shape
    ):

        raise ValueError(
            "attention_weights has "
            "incorrect shape."
        )

    # ========================================================
    # 1. Backward through:
    #
    # output = A @ V
    #
    # dA = dOutput @ V^T
    #
    # dV = A^T @ dOutput
    # ========================================================

    value_transposed = np.transpose(
        value,
        (
            0,
            2,
            1,
        ),
    )

    grad_attention = (
        grad_output
        @ value_transposed
    )

    attention_transposed = np.transpose(
        attention_weights,
        (
            0,
            2,
            1,
        ),
    )

    grad_value = (
        attention_transposed
        @ grad_output
    )

    # ========================================================
    # 2. Softmax backward
    #
    # dScores =
    #
    # A * (
    #     dA
    #     - sum(dA * A)
    # )
    # ========================================================

    weighted_sum = np.sum(
        grad_attention
        * attention_weights,
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )

    grad_scores = (
        attention_weights
        * (
            grad_attention
            - weighted_sum
        )
    )

    # ========================================================
    # 3. Enforce causal mask in backward
    #
    # Future-token score gradients must
    # remain exactly zero.
    # ========================================================

    causal_mask = (
        build_causal_mask(
            sequence_length
        )
    )

    grad_scores = np.where(
        causal_mask,
        np.float32(
            0.0
        ),
        grad_scores,
    ).astype(
        np.float32,
        copy=False,
    )

    # ========================================================
    # 4. Backward through scaling
    #
    # scores =
    #     QK^T / sqrt(head_dim)
    # ========================================================

    scale = np.float32(
        np.sqrt(
            np.float32(
                head_dim
            )
        )
    )

    grad_scaled_scores = (
        grad_scores
        / scale
    )

    # ========================================================
    # 5. Backward through QK^T
    #
    # dQ =
    #     dScores @ K
    #
    # dK =
    #     dScores^T @ Q
    # ========================================================

    grad_query = (
        grad_scaled_scores
        @ key
    )

    grad_scores_transposed = (
        np.transpose(
            grad_scaled_scores,
            (
                0,
                2,
                1,
            ),
        )
    )

    grad_key = (
        grad_scores_transposed
        @ query
    )

    return (
        grad_query.astype(
            np.float32,
            copy=False,
        ),
        grad_key.astype(
            np.float32,
            copy=False,
        ),
        grad_value.astype(
            np.float32,
            copy=False,
        ),
    )