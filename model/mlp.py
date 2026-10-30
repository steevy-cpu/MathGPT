try:
    from .linear import Linear
    from .gelu import (
        gelu,
        gelu_backward,
    )

except ImportError:
    from linear import Linear
    from gelu import (
        gelu,
        gelu_backward,
    )


class MLP:
    """
    Transformer feed-forward network.

    Forward:

        x
        ↓
        Linear
        ↓
        GELU
        ↓
        Linear
        ↓
        output

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
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias

        # ====================================================
        # Expansion projection
        #
        # d_model → d_ff
        # ====================================================

        self.fc_in = Linear(
            input_dim=d_model,
            output_dim=d_ff,
            use_bias=use_bias,
            seed=seed,
        )

        # ====================================================
        # Compression projection
        #
        # d_ff → d_model
        # ====================================================

        self.fc_out = Linear(
            input_dim=d_ff,
            output_dim=d_model,
            use_bias=use_bias,
            seed=seed + 1,
        )

        # ====================================================
        # Forward cache
        #
        # GELU backward needs the values that
        # entered GELU.
        # ====================================================

        self.pre_activation_cache = None

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

        # ====================================================
        # 1. Expansion Linear
        # ====================================================

        hidden = (
            self.fc_in.forward(
                x
            )
        )

        # Save the input to GELU.
        self.pre_activation_cache = (
            hidden
        )

        # ====================================================
        # 2. GELU
        # ====================================================

        activated = gelu(
            hidden
        )

        # ====================================================
        # 3. Compression Linear
        # ====================================================

        output = (
            self.fc_out.forward(
                activated
            )
        )

        return output

    def backward(
        self,
        grad_output,
    ):
        """
        Backpropagate through the MLP.

        grad_output:
            dL/dOutput

        Returns:
            dL/dInput
        """

        if (
            self.pre_activation_cache
            is None
        ):

            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        # ====================================================
        # 1. Backward through second Linear
        # ====================================================

        grad_activated = (
            self.fc_out.backward(
                grad_output
            )
        )

        # ====================================================
        # 2. Backward through GELU
        # ====================================================

        grad_hidden = gelu_backward(
            self.pre_activation_cache,
            grad_activated,
        )

        # ====================================================
        # 3. Backward through first Linear
        # ====================================================

        grad_input = (
            self.fc_in.backward(
                grad_hidden
            )
        )

        return grad_input

    def zero_grad(
        self,
    ):
        """
        Reset gradients in both Linear layers.
        """

        self.fc_in.zero_grad()
        self.fc_out.zero_grad()

    def parameter_count(
        self,
    ):
        return (
            self.fc_in.parameter_count()
            + self.fc_out.parameter_count()
        )