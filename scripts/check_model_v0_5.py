import json
import sys

from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from model.gpt import GPT
from training.parameters import (
    named_parameters,
    parameter_count_from_traversal,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.5.json"
)


EXPECTED_PARAMETER_COUNT = 16_889_856


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


def main():

    print()
    print("=" * 76)
    print("GENERAL-v0.5 MODEL ARCHITECTURE CHECK")
    print("=" * 76)

    config = load_json(
        CONFIG_PATH
    )

    print()
    print("Config:")
    print(CONFIG_PATH)

    expected_fields = {
        "version": "0.5",
        "architecture": "decoder_only_gpt",
        "vocab_size": 16000,
        "context_length": 256,
        "d_model": 384,
        "num_layers": 6,
        "num_heads": 6,
        "head_dim": 64,
        "d_ff": 1536,
        "tie_embeddings": True,
        "use_bias": True,
        "dropout": 0.0,
        "dtype": "float32",
    }

    config_checks = {
        name:
            config.get(name) == expected_value

        for name, expected_value
        in expected_fields.items()
    }

    config_checks[
        "d_model_divisible_by_num_heads"
    ] = (
        config["d_model"]
        %
        config["num_heads"]
        ==
        0
    )

    config_checks[
        "derived_head_dim_matches_config"
    ] = (
        config["d_model"]
        //
        config["num_heads"]
        ==
        config["head_dim"]
    )

    print()
    print("Configuration checks")
    print("--------------------")

    for name, passed in config_checks.items():
        print(
            f"{name:<42} {passed}"
        )

    require(
        all(
            config_checks.values()
        ),
        "GENERAL-v0.5 configuration check failed.",
    )

    print()
    print("Instantiating NumPy GPT...")

    model = GPT.from_config(
        config,
        seed=42,
    )

    model_parameter_count = (
        model.parameter_count()
    )

    traversal_parameter_count = (
        parameter_count_from_traversal(
            model
        )
    )

    embedding_parameters = (
        config["vocab_size"]
        *
        config["d_model"]
        +
        config["context_length"]
        *
        config["d_model"]
    )

    final_norm_parameters = (
        2
        *
        config["d_model"]
    )

    block_parameters = (
        model_parameter_count
        -
        embedding_parameters
        -
        final_norm_parameters
    ) // config["num_layers"]

    parameter_checks = {
        "model_parameter_count":
            (
                model_parameter_count
                ==
                EXPECTED_PARAMETER_COUNT
            ),

        "traversal_parameter_count":
            (
                traversal_parameter_count
                ==
                EXPECTED_PARAMETER_COUNT
            ),

        "two_counting_methods_agree":
            (
                model_parameter_count
                ==
                traversal_parameter_count
            ),

        "embedding_parameters":
            (
                embedding_parameters
                ==
                6_242_304
            ),

        "parameters_per_block":
            (
                block_parameters
                ==
                1_774_464
            ),

        "final_norm_parameters":
            (
                final_norm_parameters
                ==
                768
            ),
    }

    print()
    print("Parameter counts")
    print("----------------")
    print(
        f"Embedding parameters: "
        f"{embedding_parameters:,}"
    )
    print(
        f"Parameters per block: "
        f"{block_parameters:,}"
    )
    print(
        f"Transformer blocks:   "
        f"{config['num_layers']}"
    )
    print(
        f"Final LayerNorm:       "
        f"{final_norm_parameters:,}"
    )
    print(
        f"Total parameters:      "
        f"{model_parameter_count:,}"
    )

    print()
    print("Parameter-count checks")
    print("----------------------")

    for name, passed in (
        parameter_checks.items()
    ):
        print(
            f"{name:<42} {passed}"
        )

    require(
        all(
            parameter_checks.values()
        ),
        "GENERAL-v0.5 parameter-count check failed.",
    )

    # ========================================================
    # Tiny forward pass
    # ========================================================

    token_ids = np.asarray(
        [
            [
                1,
                25,
                100,
                250,
                500,
                750,
                1000,
                2,
            ]
        ],
        dtype=np.int64,
    )

    logits = model.forward(
        token_ids
    )

    forward_checks = {
        "logit_shape":
            (
                logits.shape
                ==
                (
                    1,
                    8,
                    config["vocab_size"],
                )
            ),

        "logits_float32":
            (
                logits.dtype
                ==
                np.float32
            ),

        "logits_finite":
            bool(
                np.all(
                    np.isfinite(
                        logits
                    )
                )
            ),
    }

    print()
    print("Forward-pass checks")
    print("-------------------")

    for name, passed in (
        forward_checks.items()
    ):
        print(
            f"{name:<42} {passed}"
        )

    require(
        all(
            forward_checks.values()
        ),
        "GENERAL-v0.5 forward-pass check failed.",
    )

    # ========================================================
    # Tiny backward pass
    # ========================================================

    model.zero_grad()

    grad_logits = np.zeros_like(
        logits,
        dtype=np.float32,
    )

    grad_logits[
        0,
        -1,
        10,
    ] = np.float32(
        1.0
    )

    grad_logits[
        0,
        -1,
        20,
    ] = np.float32(
        -1.0
    )

    model.backward(
        grad_logits
    )

    parameter_tensors = []
    gradient_tensors = []

    all_parameters_finite = True
    all_gradients_finite = True
    shape_matches = True
    nonzero_gradient_tensor_count = 0

    for (
        name,
        parameter,
        gradient,
    ) in named_parameters(
        model
    ):

        parameter_tensors.append(
            name
        )

        gradient_tensors.append(
            name
        )

        if (
            parameter.shape
            !=
            gradient.shape
        ):
            shape_matches = False

        if not np.all(
            np.isfinite(
                parameter
            )
        ):
            all_parameters_finite = False

        if not np.all(
            np.isfinite(
                gradient
            )
        ):
            all_gradients_finite = False

        if np.any(
            gradient
            !=
            0.0
        ):
            nonzero_gradient_tensor_count += 1

    backward_checks = {
        "parameter_gradient_shapes_match":
            shape_matches,

        "all_parameters_finite":
            all_parameters_finite,

        "all_gradients_finite":
            all_gradients_finite,

        "backward_reached_parameters":
            (
                nonzero_gradient_tensor_count
                >
                0
            ),

        "parameter_tensor_count_matches":
            (
                len(
                    parameter_tensors
                )
                ==
                len(
                    gradient_tensors
                )
            ),
    }

    print()
    print("Backward-pass checks")
    print("--------------------")

    for name, passed in (
        backward_checks.items()
    ):
        print(
            f"{name:<42} {passed}"
        )

    print(
        f"Nonzero gradient tensors: "
        f"{nonzero_gradient_tensor_count}"
    )

    require(
        all(
            backward_checks.values()
        ),
        "GENERAL-v0.5 backward-pass check failed.",
    )

    print()
    print("Architecture")
    print("------------")
    print(
        f"vocab_size:      "
        f"{config['vocab_size']:,}"
    )
    print(
        f"context_length:  "
        f"{config['context_length']}"
    )
    print(
        f"d_model:         "
        f"{config['d_model']}"
    )
    print(
        f"layers:          "
        f"{config['num_layers']}"
    )
    print(
        f"heads:           "
        f"{config['num_heads']}"
    )
    print(
        f"head_dim:        "
        f"{config['head_dim']}"
    )
    print(
        f"d_ff:            "
        f"{config['d_ff']}"
    )
    print(
        f"parameters:      "
        f"{model_parameter_count:,}"
    )

    print()
    print("=" * 76)
    print(
        "SUCCESS: GENERAL-v0.5 MODEL "
        "ARCHITECTURE CHECK PASSED."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
