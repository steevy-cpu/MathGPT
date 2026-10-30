import numpy as np

try:
    from .linear import Linear
    from .causal_attention import (
        causal_attention,
        causal_attention_backward,
    )

except ImportError:
    from linear import Linear
    from causal_attention import (
        causal_attention,
        causal_attention_backward,
    )


class MultiHeadAttention:
    """
    Multi-head causal self-attention.

    Forward input:
        (batch, sequence, d_model)

    Forward output:
        (batch, sequence, d_model)

    Backward input:
        dL/dOutput

    Backward output:
        dL/dInput
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

        # ====================================================
        # Q / K / V projections
        # ====================================================

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

        # ====================================================
        # Output projection
        # ====================================================

        self.output_projection = Linear(
            input_dim=d_model,
            output_dim=d_model,
            use_bias=use_bias,
            seed=seed + 3,
        )

        # ====================================================
        # Forward caches needed by backward
        # ====================================================

        self.query_flat_cache = None
        self.key_flat_cache = None
        self.value_flat_cache = None

        self.attention_weights_cache = None

        self.batch_size_cache = None
        self.sequence_length_cache = None

    def split_heads(
        self,
        x,
    ):
        """
        (B, T, D)
            ↓
        (B, H, T, head_dim)
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
        (B, H, T, head_dim)
            ↓
        (B, T, D)
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

        # ====================================================
        # 1. Q / K / V projections
        # ====================================================

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

        # ====================================================
        # 2. Split heads
        # ====================================================

        query = self.split_heads(
            query
        )

        key = self.split_heads(
            key
        )

        value = self.split_heads(
            value
        )

        # ====================================================
        # 3. Flatten batch × heads
        # ====================================================

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

        # ====================================================
        # 4. Causal attention
        # ====================================================

        (
            attention_output,
            attention_weights,
        ) = causal_attention(
            query_flat,
            key_flat,
            value_flat,
        )

        # Save for backward.
        self.query_flat_cache = (
            query_flat
        )

        self.key_flat_cache = (
            key_flat
        )

        self.value_flat_cache = (
            value_flat
        )

        self.attention_weights_cache = (
            attention_weights
        )

        self.batch_size_cache = (
            batch_size
        )

        self.sequence_length_cache = (
            sequence_length
        )

        # ====================================================
        # 5. Restore head dimension
        # ====================================================

        attention_output = (
            attention_output.reshape(
                batch_size,
                self.num_heads,
                sequence_length,
                self.head_dim,
            )
        )

        # Return weights in intuitive form:
        #
        # (B, H, T, T)
        returned_attention_weights = (
            attention_weights.reshape(
                batch_size,
                self.num_heads,
                sequence_length,
                sequence_length,
            )
        )

        # ====================================================
        # 6. Combine heads
        # ====================================================

        combined = (
            self.combine_heads(
                attention_output
            )
        )

        # ====================================================
        # 7. Output projection
        # ====================================================

        output = (
            self.output_projection.forward(
                combined
            )
        )

        return (
            output.astype(
                np.float32,
                copy=False,
            ),
            returned_attention_weights,
        )

    def backward(
        self,
        grad_output,
    ):
        """
        Backpropagate through the complete
        multi-head attention module.

        grad_output:
            dL/dOutput

        Returns:
            dL/dInput
        """

        if (
            self.query_flat_cache is None
            or self.key_flat_cache is None
            or self.value_flat_cache is None
            or self.attention_weights_cache is None
        ):

            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad_output = np.asarray(
            grad_output,
            dtype=np.float32,
        )

        batch_size = (
            self.batch_size_cache
        )

        sequence_length = (
            self.sequence_length_cache
        )

        expected_shape = (
            batch_size,
            sequence_length,
            self.d_model,
        )

        if (
            grad_output.shape
            != expected_shape
        ):

            raise ValueError(
                "grad_output has incorrect shape."
            )

        # ====================================================
        # 1. Backward through output projection Wo
        #
        # output = combined @ Wo + bo
        # ====================================================

        grad_combined = (
            self.output_projection.backward(
                grad_output
            )
        )

        # ====================================================
        # 2. Undo combine_heads()
        #
        # (B, T, D)
        #     ↓
        # (B, H, T, head_dim)
        # ====================================================

        grad_attention_heads = (
            self.split_heads(
                grad_combined
            )
        )

        grad_attention_flat = (
            grad_attention_heads.reshape(
                batch_size
                * self.num_heads,
                sequence_length,
                self.head_dim,
            )
        )

        # ====================================================
        # 3. Backward through causal attention
        #
        # Gives gradients for Q, K, V
        # ====================================================

        (
            grad_query_flat,
            grad_key_flat,
            grad_value_flat,
        ) = causal_attention_backward(
            self.query_flat_cache,
            self.key_flat_cache,
            self.value_flat_cache,
            self.attention_weights_cache,
            grad_attention_flat,
        )

        # ====================================================
        # 4. Restore heads
        # ====================================================

        grad_query_heads = (
            grad_query_flat.reshape(
                batch_size,
                self.num_heads,
                sequence_length,
                self.head_dim,
            )
        )

        grad_key_heads = (
            grad_key_flat.reshape(
                batch_size,
                self.num_heads,
                sequence_length,
                self.head_dim,
            )
        )

        grad_value_heads = (
            grad_value_flat.reshape(
                batch_size,
                self.num_heads,
                sequence_length,
                self.head_dim,
            )
        )

        # ====================================================
        # 5. Combine head gradients back into d_model
        # ====================================================

        grad_query = (
            self.combine_heads(
                grad_query_heads
            )
        )

        grad_key = (
            self.combine_heads(
                grad_key_heads
            )
        )

        grad_value = (
            self.combine_heads(
                grad_value_heads
            )
        )

        # ====================================================
        # 6. Backward through Q/K/V projections
        #
        # All three projections received the same x.
        # ====================================================

        grad_input_query = (
            self.query_projection.backward(
                grad_query
            )
        )

        grad_input_key = (
            self.key_projection.backward(
                grad_key
            )
        )

        grad_input_value = (
            self.value_projection.backward(
                grad_value
            )
        )

        # ====================================================
        # 7. Branch gradients ADD together
        #
        # x feeds Q, K, and V.
        #
        # Therefore:
        #
        # dX =
        # dX_Q + dX_K + dX_V
        # ====================================================

        grad_input = (
            grad_input_query
            + grad_input_key
            + grad_input_value
        )

        return grad_input.astype(
            np.float32,
            copy=False,
        )

    def zero_grad(
        self,
    ):
        """
        Reset gradients for all four
        Linear projections.
        """

        self.query_projection.zero_grad()
        self.key_projection.zero_grad()
        self.value_projection.zero_grad()
        self.output_projection.zero_grad()

    def parameter_count(
        self,
    ):
        return (
            self.query_projection.parameter_count()
            + self.key_projection.parameter_count()
            + self.value_projection.parameter_count()
            + self.output_projection.parameter_count()
        )