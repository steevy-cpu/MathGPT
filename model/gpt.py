import numpy as np

try:
    from .input_embedding import InputEmbedding
    from .transformer_block import TransformerBlock
    from .layer_norm import LayerNorm

except ImportError:
    from input_embedding import InputEmbedding
    from transformer_block import TransformerBlock
    from layer_norm import LayerNorm


class GPT:
    """
    Decoder-only GPT model.

    Architecture:

        Token IDs
            ↓
        Token + Position Embeddings
            ↓
        Transformer Blocks × N
            ↓
        Final LayerNorm
            ↓
        Tied token-embedding projection
            ↓
        Vocabulary logits

    Input:
        token_ids
        shape:
            (batch, sequence)

    Output:
        logits
        shape:
            (batch, sequence, vocab_size)
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        use_bias=True,
        epsilon=1e-5,
        tie_embeddings=True,
        seed=42,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.tie_embeddings = (
            tie_embeddings
        )

        if not self.tie_embeddings:

            raise ValueError(
                "This POC currently requires "
                "tie_embeddings=True."
            )

        # ====================================================
        # Input representation
        # ====================================================

        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            seed=seed,
        )

        # ====================================================
        # Transformer stack
        # ====================================================

        self.blocks = []

        for layer_index in range(
            num_layers
        ):

            # Different deterministic seed for
            # every Transformer block.
            block_seed = (
                seed
                + 100
                * (
                    layer_index
                    + 1
                )
            )

            block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_bias=use_bias,
                epsilon=epsilon,
                seed=block_seed,
            )

            self.blocks.append(
                block
            )

        # ====================================================
        # Final normalization
        # ====================================================

        self.final_norm = LayerNorm(
            d_model=d_model,
            epsilon=epsilon,
        )

    @classmethod
    def from_config(
        cls,
        config,
        seed=42,
    ):
        """
        Build GPT directly from our model config.
        """

        return cls(
            vocab_size=(
                config[
                    "vocab_size"
                ]
            ),
            context_length=(
                config[
                    "context_length"
                ]
            ),
            d_model=(
                config[
                    "d_model"
                ]
            ),
            num_layers=(
                config[
                    "num_layers"
                ]
            ),
            num_heads=(
                config[
                    "num_heads"
                ]
            ),
            d_ff=(
                config[
                    "d_ff"
                ]
            ),
            use_bias=(
                config.get(
                    "use_bias",
                    True,
                )
            ),
            tie_embeddings=(
                config.get(
                    "tie_embeddings",
                    True,
                )
            ),
            seed=seed,
        )

    def forward(
        self,
        token_ids,
        return_attention=False,
    ):
        """
        Run the full GPT forward pass.
        """

        if token_ids.ndim != 2:

            raise ValueError(
                "token_ids must have shape "
                "(batch, sequence)."
            )

        sequence_length = (
            token_ids.shape[1]
        )

        if (
            sequence_length
            > self.context_length
        ):

            raise ValueError(
                "Sequence length exceeds "
                "maximum context length."
            )

        # ====================================================
        # 1. Token + position embeddings
        # ====================================================

        x = (
            self.input_embedding.forward(
                token_ids
            )
        )

        # ====================================================
        # 2. Transformer stack
        # ====================================================

        attention_maps = (
            []
            if return_attention
            else None
        )

        for block in self.blocks:

            (
                x,
                attention_weights,
            ) = block.forward(
                x
            )

            if return_attention:

                attention_maps.append(
                    attention_weights
                )

        # ====================================================
        # 3. Final LayerNorm
        # ====================================================

        hidden_states = (
            self.final_norm.forward(
                x
            )
        )

        # ====================================================
        # 4. Tied vocabulary projection
        #
        # Hidden:
        #   (B, T, 256)
        #
        # Embedding weight:
        #   (16000, 256)
        #
        # Transpose:
        #   (256, 16000)
        #
        # Logits:
        #   (B, T, 16000)
        # ====================================================

        embedding_weight = (
            self.input_embedding
            .token_embedding
            .weight
        )

        logits = (
            hidden_states
            @ embedding_weight.T
        )

        logits = logits.astype(
            np.float32,
            copy=False,
        )

        if return_attention:

            return (
                logits,
                attention_maps,
            )

        return logits

    def parameter_count(
        self,
    ):
        """
        Count all unique trainable parameters.

        The output vocabulary projection does NOT
        add parameters because it reuses the token
        embedding matrix.
        """

        input_parameters = (
            self.input_embedding
            .parameter_count()
        )

        block_parameters = sum(
            block.parameter_count()
            for block
            in self.blocks
        )

        final_norm_parameters = (
            self.final_norm
            .parameter_count()
        )

        return (
            input_parameters
            + block_parameters
            + final_norm_parameters
        )