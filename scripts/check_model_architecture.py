import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def main():

    config = load_config()

    vocab_size = config[
        "vocab_size"
    ]

    context_length = config[
        "context_length"
    ]

    d_model = config[
        "d_model"
    ]

    num_layers = config[
        "num_layers"
    ]

    num_heads = config[
        "num_heads"
    ]

    head_dim = config[
        "head_dim"
    ]

    d_ff = config[
        "d_ff"
    ]

    # ========================================================
    # Architecture validation
    # ========================================================

    if (
        num_heads * head_dim
        != d_model
    ):

        raise ValueError(
            "num_heads × head_dim "
            "must equal d_model."
        )

    # ========================================================
    # Embeddings
    # ========================================================

    token_embeddings = (
        vocab_size
        * d_model
    )

    position_embeddings = (
        context_length
        * d_model
    )

    # ========================================================
    # Attention parameters per block
    # ========================================================

    qkv_weights = (
        3
        * d_model
        * d_model
    )

    qkv_biases = (
        3
        * d_model
    )

    attention_output_weight = (
        d_model
        * d_model
    )

    attention_output_bias = (
        d_model
    )

    attention_parameters = (
        qkv_weights
        + qkv_biases
        + attention_output_weight
        + attention_output_bias
    )

    # ========================================================
    # MLP parameters per block
    # ========================================================

    mlp_first_weight = (
        d_model
        * d_ff
    )

    mlp_first_bias = (
        d_ff
    )

    mlp_second_weight = (
        d_ff
        * d_model
    )

    mlp_second_bias = (
        d_model
    )

    mlp_parameters = (
        mlp_first_weight
        + mlp_first_bias
        + mlp_second_weight
        + mlp_second_bias
    )

    # ========================================================
    # Two LayerNorms per block
    #
    # Each LayerNorm has:
    #
    # gamma = d_model
    # beta  = d_model
    #
    # Two norms:
    #
    # 4 × d_model
    # ========================================================

    layer_norm_parameters = (
        4
        * d_model
    )

    block_parameters = (
        attention_parameters
        + mlp_parameters
        + layer_norm_parameters
    )

    all_blocks = (
        block_parameters
        * num_layers
    )

    # ========================================================
    # Final LayerNorm
    # ========================================================

    final_layer_norm = (
        2
        * d_model
    )

    # ========================================================
    # Weight tying means there is no separate
    # vocabulary projection matrix.
    # ========================================================

    total_parameters = (
        token_embeddings
        + position_embeddings
        + all_blocks
        + final_layer_norm
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        "MathGPT Model Architecture"
    )

    print(
        "=========================="
    )

    print()

    print(
        f"Vocabulary size: "
        f"{vocab_size:,}"
    )

    print(
        f"Context length: "
        f"{context_length:,}"
    )

    print(
        f"Model dimension: "
        f"{d_model:,}"
    )

    print(
        f"Transformer layers: "
        f"{num_layers}"
    )

    print(
        f"Attention heads: "
        f"{num_heads}"
    )

    print(
        f"Head dimension: "
        f"{head_dim}"
    )

    print(
        f"Feed-forward dimension: "
        f"{d_ff:,}"
    )

    print()
    print(
        "Parameter Count"
    )

    print(
        "---------------"
    )

    print(
        f"Token embeddings: "
        f"{token_embeddings:,}"
    )

    print(
        f"Position embeddings: "
        f"{position_embeddings:,}"
    )

    print(
        f"Attention / block: "
        f"{attention_parameters:,}"
    )

    print(
        f"MLP / block: "
        f"{mlp_parameters:,}"
    )

    print(
        f"LayerNorm / block: "
        f"{layer_norm_parameters:,}"
    )

    print(
        f"Total / block: "
        f"{block_parameters:,}"
    )

    print(
        f"All Transformer blocks: "
        f"{all_blocks:,}"
    )

    print(
        f"Final LayerNorm: "
        f"{final_layer_norm:,}"
    )

    print()

    print(
        f"TOTAL PARAMETERS: "
        f"{total_parameters:,}"
    )

    print(
        f"TOTAL PARAMETERS: "
        f"{total_parameters / 1_000_000:.3f}M"
    )

    print()

    if (
        7_500_000
        <= total_parameters
        <= 8_500_000
    ):

        print(
            "SUCCESS: Model is within "
            "the ~8M POC parameter target."
        )

    else:

        print(
            "WARNING: Model is outside "
            "the intended ~8M parameter range."
        )


if __name__ == "__main__":
    main()