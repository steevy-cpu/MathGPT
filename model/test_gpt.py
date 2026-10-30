import json
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


BATCH_SIZE = 2
SEQUENCE_LENGTH = 8


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

    model = GPT.from_config(
        config,
        seed=42,
    )

    rng = np.random.default_rng(
        123
    )

    token_ids = rng.integers(
        low=0,
        high=config[
            "vocab_size"
        ],
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=np.int64,
    )

    (
        logits,
        attention_maps,
    ) = model.forward(
        token_ids,
        return_attention=True,
    )

    print()
    print("Full GPT Forward Test")
    print("=====================")

    print()

    print(
        f"Token IDs shape: "
        f"{token_ids.shape}"
    )

    print(
        f"Transformer blocks: "
        f"{len(model.blocks)}"
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
        f"Attention maps returned: "
        f"{len(attention_maps)}"
    )

    print(
        f"Parameter count: "
        f"{model.parameter_count():,}"
    )

    # ========================================================
    # Test 1:
    # Correct number of Transformer blocks
    # ========================================================

    layer_count_passed = (
        len(
            model.blocks
        )
        == config[
            "num_layers"
        ]
    )

    # ========================================================
    # Test 2:
    # Logits shape
    # ========================================================

    expected_logits_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        config[
            "vocab_size"
        ],
    )

    logits_shape_passed = (
        logits.shape
        == expected_logits_shape
    )

    # ========================================================
    # Test 3:
    # Float32
    # ========================================================

    dtype_passed = (
        logits.dtype
        == np.float32
    )

    # ========================================================
    # Test 4:
    # Parameter count
    #
    # Input embeddings:
    # 4,161,536
    #
    # 5 blocks:
    # 3,948,800
    #
    # Final LayerNorm:
    # 512
    #
    # TOTAL:
    # 8,110,848
    # ========================================================

    expected_parameters = (
        8_110_848
    )

    parameter_count_passed = (
        model.parameter_count()
        == expected_parameters
    )

    # ========================================================
    # Test 5:
    # Attention maps
    # ========================================================

    attention_count_passed = (
        len(
            attention_maps
        )
        == config[
            "num_layers"
        ]
    )

    expected_attention_shape = (
        BATCH_SIZE,
        config[
            "num_heads"
        ],
        SEQUENCE_LENGTH,
        SEQUENCE_LENGTH,
    )

    attention_shapes_passed = all(
        attention.shape
        == expected_attention_shape
        for attention
        in attention_maps
    )

    # ========================================================
    # Test 6:
    # Finite logits
    # ========================================================

    finite_passed = (
        np.all(
            np.isfinite(
                logits
            )
        )
    )

    # ========================================================
    # Test 7:
    # Tied embedding projection
    #
    # Reconstruct the hidden states manually
    # and verify that logits are:
    #
    # hidden @ embedding_weight.T
    # ========================================================

    hidden = (
        model.input_embedding.forward(
            token_ids
        )
    )

    for block in model.blocks:

        (
            hidden,
            _,
        ) = block.forward(
            hidden
        )

    hidden = (
        model.final_norm.forward(
            hidden
        )
    )

    embedding_weight = (
        model.input_embedding
        .token_embedding
        .weight
    )

    manual_logits = (
        hidden
        @ embedding_weight.T
    )

    weight_tying_passed = (
        np.allclose(
            logits,
            manual_logits,
            atol=1e-5,
        )
    )

    # ========================================================
    # Test 8:
    # Full-model causality
    #
    # Change only the final token.
    #
    # Earlier logits must remain unchanged.
    # ========================================================

    modified_token_ids = (
        token_ids.copy()
    )

    original_last_token = (
        modified_token_ids[
            0,
            -1,
        ]
    )

    modified_token_ids[
        0,
        -1,
    ] = (
        original_last_token
        + 1
    ) % config[
        "vocab_size"
    ]

    modified_logits = (
        model.forward(
            modified_token_ids
        )
    )

    # Batch 0:
    # earlier positions must not change.
    earlier_positions_same = (
        np.allclose(
            logits[
                0,
                :-1,
                :
            ],
            modified_logits[
                0,
                :-1,
                :
            ],
            atol=1e-5,
        )
    )

    # The final position should respond.
    final_position_changed = (
        not np.allclose(
            logits[
                0,
                -1,
                :
            ],
            modified_logits[
                0,
                -1,
                :
            ],
            atol=1e-5,
        )
    )

    # Batch 1 was never modified.
    untouched_batch_same = (
        np.allclose(
            logits[
                1
            ],
            modified_logits[
                1
            ],
            atol=1e-5,
        )
    )

    print()

    print(
        f"Layer count passed: "
        f"{layer_count_passed}"
    )

    print(
        f"Logits shape passed: "
        f"{logits_shape_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"Parameter count passed: "
        f"{parameter_count_passed}"
    )

    print(
        f"Attention count passed: "
        f"{attention_count_passed}"
    )

    print(
        f"Attention shapes passed: "
        f"{attention_shapes_passed}"
    )

    print(
        f"Finite logits passed: "
        f"{finite_passed}"
    )

    print(
        f"Weight tying passed: "
        f"{weight_tying_passed}"
    )

    print(
        f"Full-model causality passed: "
        f"{earlier_positions_same}"
    )

    print(
        f"Final token responds to change: "
        f"{final_position_changed}"
    )

    print(
        f"Untouched batch unchanged: "
        f"{untouched_batch_same}"
    )

    if (
        layer_count_passed
        and logits_shape_passed
        and dtype_passed
        and parameter_count_passed
        and attention_count_passed
        and attention_shapes_passed
        and finite_passed
        and weight_tying_passed
        and earlier_positions_same
        and final_position_changed
        and untouched_batch_same
    ):

        print()

        print(
            "SUCCESS: Full GPT forward "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Full GPT test failed."
        )


if __name__ == "__main__":
    main()