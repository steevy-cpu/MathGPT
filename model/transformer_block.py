try:
    from .layer_norm import LayerNorm
    from .multi_head_attention import MultiHeadAttention
    from .mlp import MLP

except ImportError:
    from layer_norm import LayerNorm
    from multi_head_attention import MultiHeadAttention
    from mlp import MLP


class TransformerBlock:
    """
    Pre-LayerNorm decoder-only Transformer block.

    Architecture:

        x
        ↓
        LayerNorm
        ↓
        Multi-Head Causal Attention
        ↓
        Residual Add

        ↓

        LayerNorm
        ↓
        MLP
        ↓
        Residual Add

    Input shape:
        (batch, sequence, d_model)

    Output shape:
        (batch, sequence, d_model)
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        use_bias=True,
        epsilon=1e-5,
        seed=42,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # ----------------------------------------------------
        # LayerNorm before attention
        # ----------------------------------------------------

        self.norm1 = LayerNorm(
            d_model=d_model,
            epsilon=epsilon,
        )

        # ----------------------------------------------------
        # Multi-head causal self-attention
        # ----------------------------------------------------

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_bias=use_bias,
            seed=seed,
        )

        # ----------------------------------------------------
        # LayerNorm before MLP
        # ----------------------------------------------------

        self.norm2 = LayerNorm(
            d_model=d_model,
            epsilon=epsilon,
        )

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------

        self.mlp = MLP(
            d_model=d_model,
            d_ff=d_ff,
            use_bias=use_bias,
            seed=seed + 10,
        )

    def forward(
        self,
        x,
    ):
        """
        Run one Transformer block.

        Returns:
            output
            attention_weights
        """

        if x.ndim != 3:

            raise ValueError(
                "Input must have shape "
                "(batch, sequence, d_model)."
            )

        if x.shape[-1] != self.d_model:

            raise ValueError(
                "Last input dimension must "
                f"equal d_model={self.d_model}."
            )

        # ====================================================
        # Attention sublayer
        # ====================================================

        normalized_x = (
            self.norm1.forward(
                x
            )
        )

        (
            attention_output,
            attention_weights,
        ) = self.attention.forward(
            normalized_x
        )

        # Residual connection:
        #
        # original input
        # +
        # attention result
        attention_residual = (
            x
            + attention_output
        )

        # ====================================================
        # MLP sublayer
        # ====================================================

        normalized_attention = (
            self.norm2.forward(
                attention_residual
            )
        )

        mlp_output = (
            self.mlp.forward(
                normalized_attention
            )
        )

        # Second residual connection
        output = (
            attention_residual
            + mlp_output
        )

        return (
            output,
            attention_weights,
        )

    def parameter_count(
        self,
    ):
        """
        Attention
        +
        MLP
        +
        two LayerNorms
        """

        return (
            self.norm1.parameter_count()
            + self.attention.parameter_count()
            + self.norm2.parameter_count()
            + self.mlp.parameter_count()
        )