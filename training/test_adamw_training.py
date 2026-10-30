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

from training.adamw import AdamW

from training.gradient_clipping import (
    global_grad_norm,
    clip_grad_norm,
)

from training.parameters import (
    named_parameters,
)


# ============================================================
# Training settings
# ============================================================

LEARNING_RATE = 0.005

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

TRAINING_STEPS = 300


def prediction_accuracy(
    logits,
    targets,
):
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
        "Final AdamW Training Validation"
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

    optimizer = AdamW(
        model=model,
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # Tiny next-token dataset
    # ========================================================

    token_ids = np.array(
        [
            [2, 4, 2]
        ],
        dtype=np.int64,
    )

    targets = np.array(
        [
            [4, 2, 5]
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
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Maximum gradient norm: "
        f"{MAX_GRAD_NORM}"
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
    # Statistics
    # ========================================================

    losses = []

    original_gradient_norms = []

    clipped_gradient_norms = []

    clip_scales = []

    clipping_events = 0

    all_losses_finite = True

    all_gradients_finite = True

    max_clipped_norm_seen = 0.0

    # ========================================================
    # TRAIN
    # ========================================================

    for step in range(
        1,
        TRAINING_STEPS + 1,
    ):

        # ----------------------------------------------------
        # 1. Forward
        # ----------------------------------------------------

        logits = model.forward(
            token_ids
        )

        # ----------------------------------------------------
        # 2. Loss + dLogits
        # ----------------------------------------------------

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

        if not np.isfinite(
            loss
        ):

            all_losses_finite = False

        # ----------------------------------------------------
        # 3. Clear old gradients
        # ----------------------------------------------------

        model.zero_grad()

        # ----------------------------------------------------
        # 4. Backpropagation
        # ----------------------------------------------------

        model.backward(
            grad_logits
        )

        # ----------------------------------------------------
        # 5. Global gradient norm
        # ----------------------------------------------------

        grad_norm_before = (
            global_grad_norm(
                model
            )
        )

        original_gradient_norms.append(
            grad_norm_before
        )

        if not np.isfinite(
            grad_norm_before
        ):

            all_gradients_finite = False

        # ----------------------------------------------------
        # 6. Gradient clipping
        # ----------------------------------------------------

        (
            returned_norm,
            clip_scale,
        ) = clip_grad_norm(
            model=model,
            max_norm=MAX_GRAD_NORM,
        )

        grad_norm_after = (
            global_grad_norm(
                model
            )
        )

        clipped_gradient_norms.append(
            grad_norm_after
        )

        clip_scales.append(
            clip_scale
        )

        max_clipped_norm_seen = max(
            max_clipped_norm_seen,
            grad_norm_after,
        )

        if (
            clip_scale
            < 1.0
        ):
            clipping_events += 1

        # ----------------------------------------------------
        # 7. AdamW
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # 8. Progress
        # ----------------------------------------------------

        if (
            step == 1
            or
            step % 25 == 0
            or
            step == TRAINING_STEPS
        ):

            predictions = np.argmax(
                logits,
                axis=-1,
            )

            accuracy = np.mean(
                predictions
                == targets
            )

            print(
                f"Step "
                f"{step:>3} | "
                f"Loss "
                f"{loss:.6f} | "
                f"Accuracy "
                f"{accuracy:.2%} | "
                f"Grad "
                f"{grad_norm_before:.4f} | "
                f"Clip "
                f"{clip_scale:.4f}"
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
    # Check AdamW state
    # ========================================================

    moment_state_nonzero = any(
        np.any(
            optimizer
            .first_moments[
                name
            ]
            != 0.0
        )

        for (
            name,
            _,
            _,
        ) in named_parameters(
            model
        )
    )

    second_moment_state_nonzero = any(
        np.any(
            optimizer
            .second_moments[
                name
            ]
            != 0.0
        )

        for (
            name,
            _,
            _,
        ) in named_parameters(
            model
        )
    )

    optimizer_state_finite = all(
        np.all(
            np.isfinite(
                optimizer
                .first_moments[
                    name
                ]
            )
        )
        and
        np.all(
            np.isfinite(
                optimizer
                .second_moments[
                    name
                ]
            )
        )

        for (
            name,
            _,
            _,
        ) in named_parameters(
            model
        )
    )

    # ========================================================
    # Final model checks
    # ========================================================

    model_parameters_finite = all(
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

    loss_decreased = (
        final_loss
        < initial_loss
    )

    significant_loss_reduction = (
        final_loss
        <
        initial_loss
        * 0.5
    )

    perfect_accuracy = (
        final_accuracy
        == 1.0
    )

    optimizer_step_count_passed = (
        optimizer.step_count
        == TRAINING_STEPS
    )

    clipping_bound_respected = (
        max_clipped_norm_seen
        <=
        MAX_GRAD_NORM
        + 1e-5
    )

    final_loss_finite = (
        np.isfinite(
            final_loss
        )
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        "=" * 65
    )

    print(
        "FINAL ADAMW TRAINING RESULT"
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
        f"Gradient clipping events: "
        f"{clipping_events}"
        f"/"
        f"{TRAINING_STEPS}"
    )

    print(
        f"Maximum clipped norm seen: "
        f"{max_clipped_norm_seen:.8f}"
    )

    print(
        f"Final optimizer step: "
        f"{optimizer.step_count}"
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
        f"{significant_loss_reduction}"
    )

    print(
        f"Perfect next-token accuracy: "
        f"{perfect_accuracy}"
    )

    print(
        f"All training losses finite: "
        f"{all_losses_finite}"
    )

    print(
        f"All gradient norms finite: "
        f"{all_gradients_finite}"
    )

    print(
        f"Gradient clipping bound respected: "
        f"{clipping_bound_respected}"
    )

    print(
        f"Adam first moments non-zero: "
        f"{moment_state_nonzero}"
    )

    print(
        f"Adam second moments non-zero: "
        f"{second_moment_state_nonzero}"
    )

    print(
        f"AdamW state finite: "
        f"{optimizer_state_finite}"
    )

    print(
        f"Model parameters finite: "
        f"{model_parameters_finite}"
    )

    print(
        f"Final loss finite: "
        f"{final_loss_finite}"
    )

    print(
        f"Optimizer step count passed: "
        f"{optimizer_step_count_passed}"
    )

    all_passed = (
        loss_decreased
        and
        significant_loss_reduction
        and
        perfect_accuracy
        and
        all_losses_finite
        and
        all_gradients_finite
        and
        clipping_bound_respected
        and
        moment_state_nonzero
        and
        second_moment_state_nonzero
        and
        optimizer_state_finite
        and
        model_parameters_finite
        and
        final_loss_finite
        and
        optimizer_step_count_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: COMPLETE ADAMW "
            "TRAINING PIPELINE VALIDATED."
        )

    else:

        print(
            "WARNING: Final AdamW training "
            "validation did not fully pass."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()