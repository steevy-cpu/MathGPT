import numpy as np

try:
    from .linear import Linear
    from .causal_attention import causal_attention

except ImportError:
    from linear import Linear
    from causal_attention import causal_attention


class MultiHeadAttention:
    """
    Multi-head causal self-attention.

    Input:
        x
        shape:
            (batch_size, sequence_length, d_model)

    Output:
        shape:
            (batch_size, sequence_length, d_model)

    Attention weights:
        shape:
            (
                batch_size,
                num_heads,
                sequence_length,
                sequence_length,
            )
    """

    def __init__(
        self,
        d_model,
        num_heads,
        use_bias=True,
        seed=42,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_bias = use_bias

        if (
            d_model
            % num_heads
            != 0
        ):
            raise ValueError(
                "d_model must be divisible "
                "by num_heads."
            )

        self.head_dim = (
            d_model
            // num_heads
        )

        # ----------------------------------------------------
        # Q, K, V projections
        # ----------------------------------------------------

        self.query_projection = Linear(
            input_dim=d_model,
            output_dim=d_model,
            use_bias=use_bias,
            seed=seed,
        )

        self.key_projection = Linear(
            input_dim=d_model,
            output_dim=d_model,
            use_bias=use_bias,
            seed=seed + 1,
        )

        self.value_projection = Linear(
            input_dim=d_model,
            output_dim=d_model,
            use_bias=use_bias,
            seed=seed + 2,
        )

        # ----------------------------------------------------
        # Final output projection
        # ----------------------------------------------------

        self.output_projection = Linear(
            input_dim=d_model,
            output_dim=d_model,
            use_bias=use_bias,
            seed=seed + 3,
        )

    def split_heads(
        self,
        x,
    ):
        """
        Convert:

            (batch, sequence, d_model)

        into:

            (batch, heads, sequence, head_dim)
        """

        batch_size = (
            x.shape[0]
        )

        sequence_length = (
            x.shape[1]
        )

        x = x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        x = np.transpose(
            x,
            (
                0,
                2,
                1,
                3,
            ),
        )

        return x

    def combine_heads(
        self,
        x,
    ):
        """
        Convert:

            (batch, heads, sequence, head_dim)

        back into:

            (batch, sequence, d_model)
        """

        batch_size = (
            x.shape[0]
        )

        sequence_length = (
            x.shape[2]
        )

        x = np.transpose(
            x,
            (
                0,
                2,
                1,
                3,
            ),
        )

        # transpose can create a non-contiguous view.
        x = np.ascontiguousarray(
            x
        )

        x = x.reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return x

    def forward(
        self,
        x,
    ):
        """
        Run multi-head causal self-attention.
        """

        if x.ndim != 3:

            raise ValueError(
                "Input must have shape "
                "(batch, sequence, d_model)."
            )

        if (
            x.shape[-1]
            != self.d_model
        ):

            raise ValueError(
                "Last input dimension must "
                f"equal d_model={self.d_model}."
            )

        batch_size = (
            x.shape[0]
        )

        sequence_length = (
            x.shape[1]
        )

        # ----------------------------------------------------
        # 1. Generate Q, K, V
        #
        # each:
        # (batch, sequence, d_model)
        # ----------------------------------------------------

        query = (
            self.query_projection.forward(
                x
            )
        )

        key = (
            self.key_projection.forward(
                x
            )
        )

        value = (
            self.value_projection.forward(
                x
            )
        )

        # ----------------------------------------------------
        # 2. Split into heads
        #
        # (B, T, D)
        #      ↓
        # (B, H, T, head_dim)
        # ----------------------------------------------------

        query = self.split_heads(
            query
        )

        key = self.split_heads(
            key
        )

        value = self.split_heads(
            value
        )

        # ----------------------------------------------------
        # 3. Flatten batch and head dimensions
        #
        # Our causal_attention() function expects:
        #
        # (batch, sequence, head_dim)
        #
        # So:
        #
        # B * H
        #
        # Example:
        #
        # 8 batches
        # × 4 heads
        # = 32 independent attention calculations
        # ----------------------------------------------------

        query_flat = query.reshape(
            batch_size
            * self.num_heads,
            sequence_length,
            self.head_dim,
        )

        key_flat = key.reshape(
            batch_size
            * self.num_heads,
            sequence_length,
            self.head_dim,
        )

        value_flat = value.reshape(
            batch_size
            * self.num_heads,
            sequence_length,
            self.head_dim,
        )

        # ----------------------------------------------------
        # 4. Run causal attention
        # ----------------------------------------------------

        (
            attention_output,
            attention_weights,
        ) = causal_attention(
            query_flat,
            key_flat,
            value_flat,
        )

        # ----------------------------------------------------
        # 5. Restore head dimension
        # ----------------------------------------------------

        attention_output = (
            attention_output.reshape(
                batch_size,
                self.num_heads,
                sequence_length,
                self.head_dim,
            )
        )

        attention_weights = (
            attention_weights.reshape(
                batch_size,
                self.num_heads,
                sequence_length,
                sequence_length,
            )
        )

        # ----------------------------------------------------
        # 6. Concatenate heads
        #
        # (B, H, T, head_dim)
        #          ↓
        # (B, T, D)
        # ----------------------------------------------------

        combined = (
            self.combine_heads(
                attention_output
            )
        )

        # ----------------------------------------------------
        # 7. Final output projection
        # ----------------------------------------------------

        output = (
            self.output_projection.forward(
                combined
            )
        )

        return (
            output,
            attention_weights,
        )

    def parameter_count(
        self,
    ):
        """
        Q + K + V + output projection
        """

        return (
            self.query_projection.parameter_count()
            + self.key_projection.parameter_count()
            + self.value_projection.parameter_count()
            + self.output_projection.parameter_count()
        )