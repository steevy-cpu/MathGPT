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

from training.parameters import (
    named_parameters,
)


LEARNING_RATE = 0.001


def main():

    print()
    print(
        "Single GPT Training Step Test"
    )

    print(
        "============================="
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
    # Tiny next-token training example
    #
    # X:
    #     [2, 4, 2]
    #
    # Y:
    #     [4, 2, 5]
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
    # 1. LOSS BEFORE TRAINING
    # ========================================================

    logits_before = (
        model.forward(
            token_ids
        )
    )

    loss_before = (
        cross_entropy_loss(
            logits_before,
            targets,
        )
    )

    # ========================================================
    # Save parameters before update
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
    # 2. Forward + backward
    # ========================================================

    (
        loss_for_backward,
        grad_logits,
    ) = (
        cross_entropy_loss_and_backward(
            logits_before,
            targets,
        )
    )

    model.zero_grad()

    model.backward(
        grad_logits
    )

    # ========================================================
    # Compute global gradient norm
    # ========================================================

    gradient_squared_sum = 0.0

    for (
        _,
        _,
        gradient,
    ) in named_parameters(
        model
    ):

        gradient_squared_sum += float(
            np.sum(
                gradient.astype(
                    np.float64
                ) ** 2
            )
        )

    gradient_norm = np.sqrt(
        gradient_squared_sum
    )

    # ========================================================
    # 3. SGD UPDATE
    # ========================================================

    optimizer.step()

    # ========================================================
    # 4. LOSS AFTER TRAINING
    # ========================================================

    logits_after = (
        model.forward(
            token_ids
        )
    )

    loss_after = (
        cross_entropy_loss(
            logits_after,
            targets,
        )
    )

    # ========================================================
    # Measure parameter movement
    # ========================================================

    changed_arrays = 0

    total_parameter_change_squared = 0.0

    for (
        name,
        parameter,
        _,
    ) in named_parameters(
        model
    ):

        before = (
            parameters_before[
                name
            ]
        )

        difference = (
            parameter
            - before
        )

        if np.any(
            difference != 0
        ):

            changed_arrays += 1

        total_parameter_change_squared += float(
            np.sum(
                difference.astype(
                    np.float64
                ) ** 2
            )
        )

    parameter_change_norm = (
        np.sqrt(
            total_parameter_change_squared
        )
    )

    # ========================================================
    # Calculate improvement
    # ========================================================

    loss_change = (
        float(
            loss_after
        )
        - float(
            loss_before
        )
    )

    loss_reduction = (
        float(
            loss_before
        )
        - float(
            loss_after
        )
    )

    percentage_reduction = (
        loss_reduction
        / float(
            loss_before
        )
        * 100.0
    )

    # ========================================================
    # Tests
    # ========================================================

    backward_loss_passed = (
        np.isclose(
            loss_before,
            loss_for_backward,
            atol=1e-7,
        )
    )

    loss_decreased_passed = (
        loss_after
        < loss_before
    )

    parameters_changed_passed = (
        changed_arrays
        > 0
        and
        parameter_change_norm
        > 0.0
    )

    gradient_nonzero_passed = (
        gradient_norm
        > 0.0
    )

    finite_passed = (
        np.isfinite(
            loss_before
        )
        and
        np.isfinite(
            loss_after
        )
        and
        np.isfinite(
            gradient_norm
        )
        and
        np.isfinite(
            parameter_change_norm
        )
    )

    step_count_passed = (
        optimizer.step_count
        == 1
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print()

    print(
        f"Loss BEFORE update: "
        f"{loss_before:.8f}"
    )

    print(
        f"Loss AFTER update:  "
        f"{loss_after:.8f}"
    )

    print()

    print(
        f"Loss change: "
        f"{loss_change:+.8f}"
    )

    print(
        f"Loss reduction: "
        f"{loss_reduction:.8f}"
    )

    print(
        f"Percentage reduction: "
        f"{percentage_reduction:.4f}%"
    )

    print()

    print(
        f"Global gradient norm: "
        f"{gradient_norm:.8f}"
    )

    print(
        f"Parameter change norm: "
        f"{parameter_change_norm:.8f}"
    )

    print(
        f"Parameter arrays changed: "
        f"{changed_arrays}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Forward/backward loss match: "
        f"{backward_loss_passed}"
    )

    print(
        f"Loss decreased: "
        f"{loss_decreased_passed}"
    )

    print(
        f"Parameters changed: "
        f"{parameters_changed_passed}"
    )

    print(
        f"Gradient non-zero: "
        f"{gradient_nonzero_passed}"
    )

    print(
        f"Values finite: "
        f"{finite_passed}"
    )

    print(
        f"Optimizer step count = 1: "
        f"{step_count_passed}"
    )

    all_passed = (
        backward_loss_passed
        and
        loss_decreased_passed
        and
        parameters_changed_passed
        and
        gradient_nonzero_passed
        and
        finite_passed
        and
        step_count_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: MATHGPT COMPLETED "
            "ITS FIRST LEARNING STEP."
        )

    else:

        print(
            "FAILURE: Single-step "
            "learning validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()