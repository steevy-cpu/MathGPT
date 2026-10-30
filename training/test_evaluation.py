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

from training.evaluation import (
    evaluate_model,
)

from training.parameters import (
    named_parameters,
)


MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.1.json"
)


# Keep the test short.
TEST_EVAL_BATCHES = 3


def load_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def main():

    print()
    print(
        "Real Validation System Test"
    )

    print(
        "==========================="
    )

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    validation_path = (
        PROJECT_ROOT
        /
        training_config[
            "data"
        ][
            "validation_binary"
        ]
    )

    batch_size = (
        training_config[
            "training"
        ][
            "batch_size"
        ]
    )

    context_length = (
        training_config[
            "training"
        ][
            "context_length"
        ]
    )

    # ========================================================
    # Full model
    # ========================================================

    model = GPT.from_config(
        model_config,
        seed=42,
    )

    # ========================================================
    # Save every parameter before evaluation
    # ========================================================

    parameters_before = {
        name: parameter.copy()

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    }

    # ========================================================
    # Evaluation #1
    # ========================================================

    result_1 = evaluate_model(
        model=model,
        binary_path=validation_path,
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=TEST_EVAL_BATCHES,
        seed=12345,
    )

    # ========================================================
    # Evaluation #2
    #
    # Same seed + unchanged model should produce
    # the exact same sampled validation windows.
    # ========================================================

    result_2 = evaluate_model(
        model=model,
        binary_path=validation_path,
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=TEST_EVAL_BATCHES,
        seed=12345,
    )

    # ========================================================
    # Checks
    # ========================================================

    expected_tokens = (
        TEST_EVAL_BATCHES
        *
        batch_size
        *
        context_length
    )

    token_count_passed = (
        result_1[
            "tokens"
        ]
        ==
        expected_tokens
    )

    batch_count_passed = (
        result_1[
            "batches"
        ]
        ==
        TEST_EVAL_BATCHES
    )

    loss_finite = (
        np.isfinite(
            result_1[
                "loss"
            ]
        )
    )

    perplexity_finite = (
        np.isfinite(
            result_1[
                "perplexity"
            ]
        )
    )

    positive_loss = (
        result_1[
            "loss"
        ]
        > 0.0
    )

    deterministic_loss = (
        np.isclose(
            result_1[
                "loss"
            ],
            result_2[
                "loss"
            ],
            atol=0.0,
            rtol=0.0,
        )
    )

    deterministic_batches = (
        np.array_equal(
            np.asarray(
                result_1[
                    "batch_losses"
                ]
            ),
            np.asarray(
                result_2[
                    "batch_losses"
                ]
            ),
        )
    )

    # ========================================================
    # Evaluation must NOT update parameters
    # ========================================================

    parameters_unchanged = all(
        np.array_equal(
            parameter,
            parameters_before[
                name
            ],
        )

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    )

    # ========================================================
    # Random-model baseline
    #
    # Uniform 16K prediction:
    #
    # loss = ln(16000)
    # ========================================================

    uniform_baseline = (
        np.log(
            model_config[
                "vocab_size"
            ]
        )
    )

    baseline_difference = (
        result_1[
            "loss"
        ]
        -
        uniform_baseline
    )

    # Don't require equality.
    #
    # Random initialization should merely
    # be somewhere near this neighborhood.
    baseline_sanity_passed = (
        abs(
            baseline_difference
        )
        < 1.0
    )

    # ========================================================
    # Display
    # ========================================================

    print()

    print(
        f"Validation batches: "
        f"{TEST_EVAL_BATCHES}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Context length: "
        f"{context_length}"
    )

    print(
        f"Validation tokens evaluated: "
        f"{result_1['tokens']:,}"
    )

    print()

    print(
        f"Validation loss: "
        f"{result_1['loss']:.8f}"
    )

    print(
        f"Perplexity: "
        f"{result_1['perplexity']:,.2f}"
    )

    print()

    print(
        f"Uniform 16K baseline loss: "
        f"{uniform_baseline:.8f}"
    )

    print(
        f"Difference from baseline: "
        f"{baseline_difference:+.8f}"
    )

    print()

    print(
        f"Evaluation time: "
        f"{result_1['seconds']:.4f}s"
    )

    print(
        f"Evaluation throughput: "
        f"{result_1['tokens_per_second']:,.1f} "
        f"tokens/s"
    )

    print()

    print(
        "Batch losses:"
    )

    for (
        index,
        batch_loss,
    ) in enumerate(
        result_1[
            "batch_losses"
        ],
        start=1,
    ):

        print(
            f"  Batch {index}: "
            f"{batch_loss:.8f}"
        )

    # ========================================================
    # Final checks
    # ========================================================

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Token count passed: "
        f"{token_count_passed}"
    )

    print(
        f"Batch count passed: "
        f"{batch_count_passed}"
    )

    print(
        f"Validation loss finite: "
        f"{loss_finite}"
    )

    print(
        f"Perplexity finite: "
        f"{perplexity_finite}"
    )

    print(
        f"Validation loss positive: "
        f"{positive_loss}"
    )

    print(
        f"Repeated evaluation deterministic: "
        f"{deterministic_loss}"
    )

    print(
        f"Batch losses deterministic: "
        f"{deterministic_batches}"
    )

    print(
        f"Evaluation changed no parameters: "
        f"{parameters_unchanged}"
    )

    print(
        f"Random-model baseline sane: "
        f"{baseline_sanity_passed}"
    )

    all_passed = (
        token_count_passed
        and
        batch_count_passed
        and
        loss_finite
        and
        perplexity_finite
        and
        positive_loss
        and
        deterministic_loss
        and
        deterministic_batches
        and
        parameters_unchanged
        and
        baseline_sanity_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: REAL VALIDATION "
            "SYSTEM VALIDATED."
        )

    else:

        raise AssertionError(
            "Validation system test failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()