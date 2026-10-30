import json
import time
from pathlib import Path

import numpy as np

from gpt import GPT


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)

EXPECTED_TOTAL_PARAMETERS = (
    8_110_848
)

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def main():

    config = load_config()

    print()
    print("=" * 65)
    print(
        "MATHGPT STAGE 6 FINAL VALIDATION"
    )
    print("=" * 65)

    model = GPT.from_config(
        config,
        seed=42,
    )

    # ========================================================
    # Architecture report
    # ========================================================

    print()
    print("Architecture")
    print("------------")

    print(
        f"Vocabulary size: "
        f"{config['vocab_size']:,}"
    )

    print(
        f"Context length: "
        f"{config['context_length']:,}"
    )

    print(
        f"d_model: "
        f"{config['d_model']:,}"
    )

    print(
        f"Transformer blocks: "
        f"{config['num_layers']}"
    )

    print(
        f"Attention heads: "
        f"{config['num_heads']}"
    )

    print(
        f"Feed-forward dimension: "
        f"{config['d_ff']:,}"
    )

    # ========================================================
    # Parameter breakdown
    # ========================================================

    input_parameters = (
        model.input_embedding
        .parameter_count()
    )

    block_counts = [
        block.parameter_count()
        for block
        in model.blocks
    ]

    all_blocks_parameters = sum(
        block_counts
    )

    final_norm_parameters = (
        model.final_norm
        .parameter_count()
    )

    total_parameters = (
        model.parameter_count()
    )

    print()
    print("Parameter Audit")
    print("---------------")

    print(
        f"Input embeddings: "
        f"{input_parameters:,}"
    )

    for index, count in enumerate(
        block_counts,
        start=1,
    ):

        print(
            f"Transformer block {index}: "
            f"{count:,}"
        )

    print(
        f"All Transformer blocks: "
        f"{all_blocks_parameters:,}"
    )

    print(
        f"Final LayerNorm: "
        f"{final_norm_parameters:,}"
    )

    print()

    print(
        f"TOTAL PARAMETERS: "
        f"{total_parameters:,}"
    )

    # ========================================================
    # Expected parameter checks
    # ========================================================

    input_parameters_passed = (
        input_parameters
        == 4_161_536
    )

    block_parameters_passed = all(
        count == 789_760
        for count
        in block_counts
    )

    final_norm_passed = (
        final_norm_parameters
        == 512
    )

    total_parameters_passed = (
        total_parameters
        == EXPECTED_TOTAL_PARAMETERS
    )

    # ========================================================
    # Maximum context test
    #
    # Use batch size 1 to keep memory usage modest.
    # ========================================================

    rng = np.random.default_rng(
        123
    )

    token_ids = rng.integers(
        low=0,
        high=config[
            "vocab_size"
        ],
        size=(
            1,
            config[
                "context_length"
            ],
        ),
        dtype=np.int64,
    )

    print()
    print("Maximum Context Forward Pass")
    print("----------------------------")

    print(
        f"Input shape: "
        f"{token_ids.shape}"
    )

    start_time = (
        time.perf_counter()
    )

    logits = model.forward(
        token_ids
    )

    forward_time = (
        time.perf_counter()
        - start_time
    )

    expected_logits_shape = (
        1,
        config[
            "context_length"
        ],
        config[
            "vocab_size"
        ],
    )

    max_context_shape_passed = (
        logits.shape
        == expected_logits_shape
    )

    dtype_passed = (
        logits.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                logits
            )
        )
    )

    print(
        f"Logits shape: "
        f"{logits.shape}"
    )

    print(
        f"Logits dtype: "
        f"{logits.dtype}"
    )

    print(
        f"Forward time: "
        f"{forward_time:.3f} seconds"
    )

    print(
        f"Finite logits: "
        f"{finite_passed}"
    )

    # ========================================================
    # Vocabulary dimension check
    # ========================================================

    vocabulary_dimension_passed = (
        logits.shape[-1]
        == config[
            "vocab_size"
        ]
    )

    # ========================================================
    # Weight tying check
    # ========================================================

    embedding_weight = (
        model.input_embedding
        .token_embedding
        .weight
    )

    tied_weight_shape_passed = (
        embedding_weight.shape
        ==
        (
            config[
                "vocab_size"
            ],
            config[
                "d_model"
            ],
        )
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 65)
    print("FINAL CHECKS")
    print("=" * 65)

    print(
        f"Input parameter count: "
        f"{input_parameters_passed}"
    )

    print(
        f"Every block = 789,760 params: "
        f"{block_parameters_passed}"
    )

    print(
        f"Final LayerNorm = 512 params: "
        f"{final_norm_passed}"
    )

    print(
        f"Total = 8,110,848 params: "
        f"{total_parameters_passed}"
    )

    print(
        f"Maximum context shape passed: "
        f"{max_context_shape_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"Finite logits passed: "
        f"{finite_passed}"
    )

    print(
        f"Vocabulary output dimension passed: "
        f"{vocabulary_dimension_passed}"
    )

    print(
        f"Tied embedding shape passed: "
        f"{tied_weight_shape_passed}"
    )

    all_passed = (
        input_parameters_passed
        and block_parameters_passed
        and final_norm_passed
        and total_parameters_passed
        and max_context_shape_passed
        and dtype_passed
        and finite_passed
        and vocabulary_dimension_passed
        and tied_weight_shape_passed
    )

    print()
    print("=" * 65)

    if all_passed:

        print(
            "SUCCESS: STAGE 6 GPT ARCHITECTURE "
            "PASSED FINAL VALIDATION."
        )

    else:

        print(
            "FAILURE: One or more Stage 6 "
            "validation checks failed."
        )

    print("=" * 65)


if __name__ == "__main__":
    main()