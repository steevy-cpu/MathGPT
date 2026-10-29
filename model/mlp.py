try:
    from .linear import Linear
    from .gelu import gelu

except ImportError:
    from linear import Linear
    from gelu import gelu


class MLP:
    """
    Transformer feed-forward network.

    Architecture:

        d_model
           ↓
        Linear
           ↓
          d_ff
           ↓
          GELU
           ↓
        Linear
           ↓
        d_model

    Input:
        (batch, sequence, d_model)

    Output:
        (batch, sequence, d_model)
    """

    def __init__(
        self,
        d_model,
        d_ff,
        use_bias=True,
        seed=42,
    ):
        self.d_model = (
            d_model
        )

        self.d_ff = (
            d_ff
        )

        self.use_bias = (
            use_bias
        )

        # ----------------------------------------------------
        # Expansion:
        #
        # 256 → 1024
        # ----------------------------------------------------

        self.fc_in = Linear(
            input_dim=d_model,
            output_dim=d_ff,
            use_bias=use_bias,
            seed=seed,
        )

        # ----------------------------------------------------
        # Compression:
        #
        # 1024 → 256
        # ----------------------------------------------------

        self.fc_out = Linear(
            input_dim=d_ff,
            output_dim=d_model,
            use_bias=use_bias,
            seed=seed + 1,
        )

    def forward(
        self,
        x,
    ):
        """
        Run the Transformer MLP.
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

        # ----------------------------------------------------
        # 1. Expand features
        # ----------------------------------------------------

        hidden = (
            self.fc_in.forward(
                x
            )
        )

        # ----------------------------------------------------
        # 2. Nonlinear activation
        # ----------------------------------------------------

        hidden = gelu(
            hidden
        )

        # ----------------------------------------------------
        # 3. Compress back to d_model
        # ----------------------------------------------------

        output = (
            self.fc_out.forward(
                hidden
            )
        )

        return output

    def parameter_count(
        self,
    ):
        """
        Count both Linear layers.
        """

        return (
            self.fc_in.parameter_count()
            + self.fc_out.parameter_count()
        )