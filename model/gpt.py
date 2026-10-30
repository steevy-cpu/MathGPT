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

    Forward:

        token IDs
            ↓
        token + position embeddings
            ↓
        Transformer blocks
            ↓
        final LayerNorm
            ↓
        tied embedding projection
            ↓
        vocabulary logits

    Backward:

        dLogits
            ↓
        output projection
            ↓
        final LayerNorm
            ↓
        Transformer blocks in reverse
            ↓
        input embeddings
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
        # Input embeddings
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

        # ====================================================
        # Forward cache
        #
        # Needed because output projection backward uses
        # the hidden states that entered E^T.
        # ====================================================

        self.hidden_states_cache = None

    @classmethod
    def from_config(
        cls,
        config,
        seed=42,
    ):
        return cls(
            vocab_size=config[
                "vocab_size"
            ],

            context_length=config[
                "context_length"
            ],

            d_model=config[
                "d_model"
            ],

            num_layers=config[
                "num_layers"
            ],

            num_heads=config[
                "num_heads"
            ],

            d_ff=config[
                "d_ff"
            ],

            use_bias=config.get(
                "use_bias",
                True,
            ),

            tie_embeddings=config.get(
                "tie_embeddings",
                True,
            ),

            seed=seed,
        )

    def forward(
        self,
        token_ids,
        return_attention=False,
    ):
        """
        Full GPT forward pass.
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
        # 1. Input embeddings
        # ====================================================

        x = (
            self.input_embedding.forward(
                token_ids
            )
        )

        # ====================================================
        # 2. Transformer blocks
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

        # Cache for output projection backward.
        self.hidden_states_cache = (
            hidden_states
        )

        # ====================================================
        # 4. Weight-tied vocabulary projection
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

    def backward(
        self,
        grad_logits,
    ):
        """
        Full GPT backward pass.

        grad_logits:
            dL/dLogits

            shape:
                (
                    batch,
                    sequence,
                    vocab_size,
                )

        Returns:
            None

        Parameter gradients are stored
        inside the model components.
        """

        if (
            self.hidden_states_cache
            is None
        ):

            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad_logits = np.asarray(
            grad_logits,
            dtype=np.float32,
        )

        hidden_states = (
            self.hidden_states_cache
        )

        expected_shape = (
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.vocab_size,
        )

        if (
            grad_logits.shape
            != expected_shape
        ):

            raise ValueError(
                "grad_logits has "
                "incorrect shape."
            )

        embedding_weight = (
            self.input_embedding
            .token_embedding
            .weight
        )

        # ====================================================
        # 1. Backward through tied output projection
        #
        # logits =
        #     hidden @ E^T
        #
        # dHidden =
        #     dLogits @ E
        #
        # dE_output =
        #     dLogits^T @ hidden
        # ====================================================

        grad_hidden = (
            grad_logits
            @ embedding_weight
        )

        batch_size = (
            grad_logits.shape[0]
        )

        sequence_length = (
            grad_logits.shape[1]
        )

        grad_logits_flat = (
            grad_logits.reshape(
                batch_size
                * sequence_length,
                self.vocab_size,
            )
        )

        hidden_flat = (
            hidden_states.reshape(
                batch_size
                * sequence_length,
                self.d_model,
            )
        )

        grad_embedding_output = (
            grad_logits_flat.T
            @ hidden_flat
        ).astype(
            np.float32,
            copy=False,
        )

        # ====================================================
        # 2. Final LayerNorm backward
        # ====================================================

        grad_x = (
            self.final_norm.backward(
                grad_hidden
            )
        )

        # ====================================================
        # 3. Transformer stack backward
        #
        # Reverse order:
        #
        # block 5 → 4 → 3 → 2 → 1
        # ====================================================

        for block in reversed(
            self.blocks
        ):

            grad_x = (
                block.backward(
                    grad_x
                )
            )

        # ====================================================
        # 4. Input embedding backward
        #
        # This computes:
        #
        # dE_input
        # dPosition
        # ====================================================

        self.input_embedding.backward(
            grad_x
        )

        # ====================================================
        # 5. Weight tying
        #
        # E participates in TWO places:
        #
        # input lookup
        # +
        # output vocabulary projection
        #
        # Therefore gradients ADD.
        # ====================================================

        self.input_embedding.token_embedding.dweight += (
            grad_embedding_output
        )

        return None

    def zero_grad(
        self,
    ):
        """
        Reset every trainable gradient
        in the GPT model.
        """

        self.input_embedding.zero_grad()

        for block in self.blocks:

            block.zero_grad()

        self.final_norm.zero_grad()

    def parameter_count(
        self,
    ):
        """
        Count unique trainable parameters.

        Tied output projection adds no
        additional parameters.
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