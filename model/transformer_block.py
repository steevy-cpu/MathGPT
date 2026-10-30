try:
    from .layer_norm import LayerNorm
    from .multi_head_attention import (
        MultiHeadAttention
    )
    from .mlp import MLP

except ImportError:
    from layer_norm import LayerNorm
    from multi_head_attention import (
        MultiHeadAttention
    )
    from mlp import MLP


class TransformerBlock:
    """
    Pre-LayerNorm decoder-only Transformer block.

    Forward:

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

    Input / output shape:

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

        self.norm1 = LayerNorm(
            d_model=d_model,
            epsilon=epsilon,
        )

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_bias=use_bias,
            seed=seed,
        )

        self.norm2 = LayerNorm(
            d_model=d_model,
            epsilon=epsilon,
        )

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
        Forward through one Transformer block.
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

        # First residual connection
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

    def backward(
        self,
        grad_output,
    ):
        """
        Backpropagate through one Transformer block.

        grad_output:
            dL/dOutput

        Returns:
            dL/dInput
        """

        # ====================================================
        # SECOND RESIDUAL
        #
        # output =
        #     attention_residual
        #     + mlp_output
        #
        # Therefore grad_output branches into:
        #
        # 1. residual path
        # 2. MLP path
        # ====================================================

        grad_attention_residual_direct = (
            grad_output
        )

        grad_mlp_output = (
            grad_output
        )

        # ====================================================
        # MLP backward
        # ====================================================

        grad_normalized_attention = (
            self.mlp.backward(
                grad_mlp_output
            )
        )

        # ====================================================
        # LayerNorm 2 backward
        # ====================================================

        grad_attention_residual_norm = (
            self.norm2.backward(
                grad_normalized_attention
            )
        )

        # ====================================================
        # Merge second residual gradients
        # ====================================================

        grad_attention_residual = (
            grad_attention_residual_direct
            + grad_attention_residual_norm
        )

        # ====================================================
        # FIRST RESIDUAL
        #
        # attention_residual =
        #     x
        #     + attention_output
        #
        # grad_attention_residual again splits.
        # ====================================================

        grad_x_direct = (
            grad_attention_residual
        )

        grad_attention_output = (
            grad_attention_residual
        )

        # ====================================================
        # Multi-head attention backward
        # ====================================================

        grad_normalized_x = (
            self.attention.backward(
                grad_attention_output
            )
        )

        # ====================================================
        # LayerNorm 1 backward
        # ====================================================

        grad_x_norm = (
            self.norm1.backward(
                grad_normalized_x
            )
        )

        # ====================================================
        # Merge first residual gradients
        # ====================================================

        grad_input = (
            grad_x_direct
            + grad_x_norm
        )

        return grad_input

    def zero_grad(
        self,
    ):
        """
        Reset every parameter gradient
        in the Transformer block.
        """

        self.norm1.zero_grad()
        self.attention.zero_grad()
        self.norm2.zero_grad()
        self.mlp.zero_grad()

    def parameter_count(
        self,
    ):
        return (
            self.norm1.parameter_count()
            + self.attention.parameter_count()
            + self.norm2.parameter_count()
            + self.mlp.parameter_count()
        )