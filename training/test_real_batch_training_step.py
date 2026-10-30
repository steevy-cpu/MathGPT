import json
import sys
import time

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

from training.data_loader import (
    TokenBatchLoader,
)

from training.loss import (
    cross_entropy_loss,
    cross_entropy_loss_and_backward,
)

from training.adamw import (
    AdamW,
)

from training.gradient_clipping import (
    global_grad_norm,
    clip_grad_norm,
)

from training.parameters import (
    named_parameters,
)


# ============================================================
# Paths
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


TRAIN_BINARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_train.bin"
)


TRAIN_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_train_metadata.json"
)


# ============================================================
# Smoke-test settings
# ============================================================

BATCH_SIZE = 1

SEED = 42

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0


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


def parameters_are_finite(
    model,
):

    return all(
        np.all(
            np.isfinite(
                parameter
            )
        )

        for (
            _,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    )


def main():

    print()
    print("=" * 70)

    print(
        "STAGE 9A — FULL-SIZE REAL-DATA "
        "TRAINING SMOKE TEST"
    )

    print("=" * 70)

    # ========================================================
    # Load configuration
    # ========================================================

    config = load_json(
        CONFIG_PATH
    )

    metadata = load_json(
        TRAIN_METADATA_PATH
    )

    context_length = config[
        "context_length"
    ]

    print()

    print(
        f"Tokenizer: "
        f"{metadata['tokenizer']}"
    )

    print(
        f"Training tokens: "
        f"{metadata['tokens']:,}"
    )

    print(
        f"Context length: "
        f"{context_length}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    # ========================================================
    # Data loader
    # ========================================================

    loader = TokenBatchLoader(
        binary_path=TRAIN_BINARY_PATH,
        context_length=context_length,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    X, Y = loader.get_batch()

    batch_passed = (
        X.shape
        ==
        (
            BATCH_SIZE,
            context_length,
        )
        and
        Y.shape
        ==
        (
            BATCH_SIZE,
            context_length,
        )
    )

    alignment_passed = (
        np.array_equal(
            X[:, 1:],
            Y[:, :-1],
        )
    )

    # ========================================================
    # Full model
    # ========================================================

    print()
    print(
        "Creating full MathGPT..."
    )

    model = GPT.from_config(
        config,
        seed=SEED,
    )

    parameter_count = (
        model.parameter_count()
    )

    parameter_count_passed = (
        parameter_count
        == 8_110_848
    )

    optimizer = AdamW(
        model=model,
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # Save selected parameter before update
    # ========================================================

    selected_parameter = (
        model.final_norm.gamma
    )

    selected_before = (
        selected_parameter.copy()
    )

    # ========================================================
    # FORWARD
    # ========================================================

    print()
    print(
        "Running forward pass..."
    )

    forward_start = (
        time.perf_counter()
    )

    logits = model.forward(
        X
    )

    forward_seconds = (
        time.perf_counter()
        - forward_start
    )

    expected_logits_shape = (
        BATCH_SIZE,
        context_length,
        config[
            "vocab_size"
        ],
    )

    logits_shape_passed = (
        logits.shape
        ==
        expected_logits_shape
    )

    logits_finite_passed = (
        np.all(
            np.isfinite(
                logits
            )
        )
    )

    # ========================================================
    # LOSS + dLOGITS
    # ========================================================

    loss_start = (
        time.perf_counter()
    )

    (
        loss_before,
        grad_logits,
    ) = (
        cross_entropy_loss_and_backward(
            logits,
            Y,
        )
    )

    loss_seconds = (
        time.perf_counter()
        - loss_start
    )

    loss_finite_passed = (
        np.isfinite(
            loss_before
        )
    )

    grad_logits_finite_passed = (
        np.all(
            np.isfinite(
                grad_logits
            )
        )
    )

    # ========================================================
    # BACKWARD
    # ========================================================

    print(
        "Running backward pass..."
    )

    model.zero_grad()

    backward_start = (
        time.perf_counter()
    )

    model.backward(
        grad_logits
    )

    backward_seconds = (
        time.perf_counter()
        - backward_start
    )

    gradient_norm_before = (
        global_grad_norm(
            model
        )
    )

    gradients_finite_passed = (
        np.isfinite(
            gradient_norm_before
        )
    )

    # ========================================================
    # GRADIENT CLIPPING
    # ========================================================

    (
        returned_gradient_norm,
        clip_scale,
    ) = clip_grad_norm(
        model=model,
        max_norm=MAX_GRAD_NORM,
    )

    gradient_norm_after = (
        global_grad_norm(
            model
        )
    )

    clipping_passed = (
        gradient_norm_after
        <=
        MAX_GRAD_NORM
        + 1e-5
    )

    # ========================================================
    # ADAMW STEP
    # ========================================================

    print(
        "Running AdamW step..."
    )

    optimizer_start = (
        time.perf_counter()
    )

    optimizer.step()

    optimizer_seconds = (
        time.perf_counter()
        - optimizer_start
    )

    parameter_changed = (
        not np.array_equal(
            selected_before,
            selected_parameter,
        )
    )

    optimizer_step_passed = (
        optimizer.step_count
        == 1
    )

    parameters_finite_passed = (
        parameters_are_finite(
            model
        )
    )

    # ========================================================
    # SECOND FORWARD
    #
    # Same exact batch.
    # ========================================================

    logits_after = model.forward(
        X
    )

    loss_after = (
        cross_entropy_loss(
            logits_after,
            Y,
        )
    )

    post_loss_finite_passed = (
        np.isfinite(
            loss_after
        )
    )

    # With a small single update on the exact
    # same batch, we normally expect a reduction.
    loss_decreased = (
        loss_after
        <
        loss_before
    )

    # ========================================================
    # Throughput
    # ========================================================

    tokens_per_batch = (
        BATCH_SIZE
        *
        context_length
    )

    training_compute_seconds = (
        forward_seconds
        +
        loss_seconds
        +
        backward_seconds
        +
        optimizer_seconds
    )

    if training_compute_seconds > 0:

        tokens_per_second = (
            tokens_per_batch
            /
            training_compute_seconds
        )

    else:

        tokens_per_second = 0.0

    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 70)

    print(
        "REAL-BATCH RESULT"
    )

    print("=" * 70)

    print()

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"X shape: "
        f"{X.shape}"
    )

    print(
        f"Y shape: "
        f"{Y.shape}"
    )

    print(
        f"Logits shape: "
        f"{logits.shape}"
    )

    print()

    print(
        f"Loss before: "
        f"{loss_before:.8f}"
    )

    print(
        f"Loss after:  "
        f"{loss_after:.8f}"
    )

    print()

    print(
        f"Gradient norm before clipping: "
        f"{gradient_norm_before:.8f}"
    )

    print(
        f"Clip scale: "
        f"{clip_scale:.8f}"
    )

    print(
        f"Gradient norm after clipping: "
        f"{gradient_norm_after:.8f}"
    )

    print()

    print(
        "Timing"
    )

    print(
        "------"
    )

    print(
        f"Forward:   "
        f"{forward_seconds:.4f}s"
    )

    print(
        f"Loss:      "
        f"{loss_seconds:.4f}s"
    )

    print(
        f"Backward:  "
        f"{backward_seconds:.4f}s"
    )

    print(
        f"AdamW:     "
        f"{optimizer_seconds:.4f}s"
    )

    print(
        f"Approx training throughput: "
        f"{tokens_per_second:,.1f} tokens/s"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Real batch shape passed: "
        f"{batch_passed}"
    )

    print(
        f"Next-token alignment passed: "
        f"{alignment_passed}"
    )

    print(
        f"8,110,848 parameters: "
        f"{parameter_count_passed}"
    )

    print(
        f"Logits shape passed: "
        f"{logits_shape_passed}"
    )

    print(
        f"Logits finite: "
        f"{logits_finite_passed}"
    )

    print(
        f"Loss finite: "
        f"{loss_finite_passed}"
    )

    print(
        f"dLogits finite: "
        f"{grad_logits_finite_passed}"
    )

    print(
        f"Gradients finite: "
        f"{gradients_finite_passed}"
    )

    print(
        f"Gradient clipping passed: "
        f"{clipping_passed}"
    )

    print(
        f"Parameter changed: "
        f"{parameter_changed}"
    )

    print(
        f"Optimizer step = 1: "
        f"{optimizer_step_passed}"
    )

    print(
        f"Parameters finite after update: "
        f"{parameters_finite_passed}"
    )

    print(
        f"Post-update loss finite: "
        f"{post_loss_finite_passed}"
    )

    print(
        f"Same-batch loss decreased: "
        f"{loss_decreased}"
    )

    all_passed = (
        batch_passed
        and
        alignment_passed
        and
        parameter_count_passed
        and
        logits_shape_passed
        and
        logits_finite_passed
        and
        loss_finite_passed
        and
        grad_logits_finite_passed
        and
        gradients_finite_passed
        and
        clipping_passed
        and
        parameter_changed
        and
        optimizer_step_passed
        and
        parameters_finite_passed
        and
        post_loss_finite_passed
        and
        loss_decreased
    )

    print()
    print("=" * 70)

    if all_passed:

        print(
            "SUCCESS: FULL MATHGPT COMPLETED "
            "A REAL-CORPUS TRAINING STEP."
        )

    else:

        print(
            "FAILURE: Stage 9A real-data "
            "smoke test failed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()