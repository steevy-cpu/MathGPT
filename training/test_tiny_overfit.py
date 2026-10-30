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

from training.loss import (
    cross_entropy_loss,
    cross_entropy_loss_and_backward,
)

from training.sgd import SGD


# ============================================================
# Experiment settings
# ============================================================

LEARNING_RATE = 0.01

TRAINING_STEPS = 300


def prediction_accuracy(
    logits,
    targets,
):
    """
    Calculate next-token accuracy.
    """

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = np.mean(
        predictions
        == targets
    )

    return (
        predictions,
        float(
            accuracy
        ),
    )


def main():

    print()
    print(
        "Tiny GPT Overfitting Experiment"
    )

    print(
        "==============================="
    )

    # ========================================================
    # Tiny GPT
    # ========================================================

    model = GPT(
        vocab_size=7,
        context_length=3,
        d_model=4,
        num_layers=2,
        num_heads=2,
        d_ff=6,
        use_bias=True,
        tie_embeddings=True,
        seed=42,
    )

    optimizer = SGD(
        model=model,
        learning_rate=LEARNING_RATE,
    )

    # ========================================================
    # One tiny sequence
    # ========================================================

    token_ids = np.array(
        [
            [
                2,
                4,
                2,
            ]
        ],
        dtype=np.int64,
    )

    targets = np.array(
        [
            [
                4,
                2,
                5,
            ]
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Initial evaluation
    # ========================================================

    initial_logits = model.forward(
        token_ids
    )

    initial_loss = cross_entropy_loss(
        initial_logits,
        targets,
    )

    (
        initial_predictions,
        initial_accuracy,
    ) = prediction_accuracy(
        initial_logits,
        targets,
    )

    print()
    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Training steps: "
        f"{TRAINING_STEPS}"
    )

    print()

    print(
        f"Initial loss: "
        f"{initial_loss:.6f}"
    )

    print(
        f"Initial predictions: "
        f"{initial_predictions}"
    )

    print(
        f"Targets:             "
        f"{targets}"
    )

    print(
        f"Initial accuracy: "
        f"{initial_accuracy:.2%}"
    )

    # ========================================================
    # Training loop
    # ========================================================

    losses = []

    for step in range(
        1,
        TRAINING_STEPS + 1,
    ):

        # ---------------------------------------------
        # Forward
        # ---------------------------------------------

        logits = model.forward(
            token_ids
        )

        # ---------------------------------------------
        # Loss + dLogits
        # ---------------------------------------------

        (
            loss,
            grad_logits,
        ) = cross_entropy_loss_and_backward(
            logits,
            targets,
        )

        losses.append(
            float(
                loss
            )
        )

        # ---------------------------------------------
        # Backward
        # ---------------------------------------------

        model.zero_grad()

        model.backward(
            grad_logits
        )

        # ---------------------------------------------
        # Update
        # ---------------------------------------------

        optimizer.step()

        # ---------------------------------------------
        # Progress report
        # ---------------------------------------------

        if (
            step == 1
            or step % 25 == 0
            or step == TRAINING_STEPS
        ):

            current_predictions = np.argmax(
                logits,
                axis=-1,
            )

            current_accuracy = np.mean(
                current_predictions
                == targets
            )

            print(
                f"Step "
                f"{step:>3} | "
                f"Loss "
                f"{loss:.6f} | "
                f"Accuracy "
                f"{current_accuracy:.2%}"
            )

    # ========================================================
    # Final evaluation
    # ========================================================

    final_logits = model.forward(
        token_ids
    )

    final_loss = cross_entropy_loss(
        final_logits,
        targets,
    )

    (
        final_predictions,
        final_accuracy,
    ) = prediction_accuracy(
        final_logits,
        targets,
    )

    # ========================================================
    # Checks
    # ========================================================

    loss_decreased = (
        final_loss
        < initial_loss
    )

    significant_learning = (
        final_loss
        < initial_loss
        * np.float32(
            0.5
        )
    )

    perfect_accuracy = (
        final_accuracy
        == 1.0
    )

    finite_loss = (
        np.isfinite(
            final_loss
        )
    )

    optimizer_steps_passed = (
        optimizer.step_count
        == TRAINING_STEPS
    )

    # We do NOT require every individual step
    # to decrease monotonically.
    #
    # Neural-network optimization can wiggle.
    #
    # We care that the final result is much better.

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        "=" * 65
    )

    print(
        "FINAL OVERFITTING RESULT"
    )

    print(
        "=" * 65
    )

    print()

    print(
        f"Initial loss: "
        f"{initial_loss:.6f}"
    )

    print(
        f"Final loss:   "
        f"{final_loss:.6f}"
    )

    print()

    print(
        f"Initial predictions: "
        f"{initial_predictions}"
    )

    print(
        f"Final predictions:   "
        f"{final_predictions}"
    )

    print(
        f"Targets:             "
        f"{targets}"
    )

    print()

    print(
        f"Initial accuracy: "
        f"{initial_accuracy:.2%}"
    )

    print(
        f"Final accuracy:   "
        f"{final_accuracy:.2%}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Loss decreased: "
        f"{loss_decreased}"
    )

    print(
        f"Loss reduced by >50%: "
        f"{significant_learning}"
    )

    print(
        f"Perfect next-token accuracy: "
        f"{perfect_accuracy}"
    )

    print(
        f"Loss finite: "
        f"{finite_loss}"
    )

    print(
        f"Optimizer step count passed: "
        f"{optimizer_steps_passed}"
    )

    all_passed = (
        loss_decreased
        and
        significant_learning
        and
        perfect_accuracy
        and
        finite_loss
        and
        optimizer_steps_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: MATHGPT OVERFIT "
            "THE TINY TRAINING EXAMPLE."
        )

    else:

        print(
            "WARNING: Tiny overfitting "
            "experiment did not fully converge."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()