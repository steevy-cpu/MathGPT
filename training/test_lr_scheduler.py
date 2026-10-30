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


from training.lr_scheduler import (
    warmup_cosine_learning_rate,
    set_optimizer_learning_rate,
)


# ============================================================
# Real MathGPT schedule
# ============================================================

MAX_LR = 3e-4

MIN_LR = 3e-5

WARMUP_STEPS = 400

TOTAL_STEPS = 8197


class DummyOptimizer:
    """
    Tiny optimizer stand-in.

    We only need a learning_rate attribute
    to test scheduler integration.
    """

    def __init__(
        self,
    ):
        self.learning_rate = 0.0


def get_lr(
    step,
):

    return warmup_cosine_learning_rate(
        step=step,
        max_learning_rate=MAX_LR,
        min_learning_rate=MIN_LR,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS,
    )


def main():

    print()
    print(
        "Warmup + Cosine LR Scheduler Test"
    )

    print(
        "================================="
    )

    # ========================================================
    # Important schedule points
    # ========================================================

    lr_step_1 = get_lr(
        1
    )

    lr_step_200 = get_lr(
        200
    )

    lr_step_400 = get_lr(
        400
    )

    lr_step_401 = get_lr(
        401
    )

    lr_middle = get_lr(
        4298
    )

    lr_final = get_lr(
        TOTAL_STEPS
    )

    lr_after_final = get_lr(
        TOTAL_STEPS
        + 1
    )

    # ========================================================
    # Exact warmup expectations
    # ========================================================

    expected_step_1 = (
        MAX_LR
        /
        WARMUP_STEPS
    )

    expected_step_200 = (
        MAX_LR
        *
        (
            200
            /
            WARMUP_STEPS
        )
    )

    expected_step_400 = (
        MAX_LR
    )

    step_1_passed = np.isclose(
        lr_step_1,
        expected_step_1,
        atol=1e-12,
    )

    step_200_passed = np.isclose(
        lr_step_200,
        expected_step_200,
        atol=1e-12,
    )

    warmup_peak_passed = np.isclose(
        lr_step_400,
        MAX_LR,
        atol=1e-12,
    )

    # ========================================================
    # Cosine decay begins after warmup
    # ========================================================

    decay_begins_passed = (
        lr_step_401
        <
        MAX_LR
    )

    final_lr_passed = np.isclose(
        lr_final,
        MIN_LR,
        atol=1e-12,
    )

    after_final_passed = np.isclose(
        lr_after_final,
        MIN_LR,
        atol=1e-12,
    )

    # ========================================================
    # Warmup should increase monotonically
    # ========================================================

    warmup_rates = [
        get_lr(
            step
        )

        for step in range(
            1,
            WARMUP_STEPS + 1,
        )
    ]

    warmup_monotonic = all(
        warmup_rates[index]
        <=
        warmup_rates[
            index + 1
        ]

        for index in range(
            len(
                warmup_rates
            )
            - 1
        )
    )

    # ========================================================
    # Decay should decrease monotonically
    # ========================================================

    decay_rates = [
        get_lr(
            step
        )

        for step in range(
            WARMUP_STEPS,
            TOTAL_STEPS + 1,
        )
    ]

    decay_monotonic = all(
        decay_rates[index]
        >=
        decay_rates[
            index + 1
        ]

        for index in range(
            len(
                decay_rates
            )
            - 1
        )
    )

    # ========================================================
    # Bounds
    # ========================================================

    all_rates = (
        warmup_rates
        +
        decay_rates
    )

    bounds_passed = all(
        0.0
        <
        learning_rate
        <=
        MAX_LR

        for learning_rate
        in all_rates
    )

    minimum_bound_passed = (
        min(
            decay_rates
        )
        >=
        MIN_LR
        -
        1e-12
    )

    # ========================================================
    # Optimizer integration
    # ========================================================

    optimizer = (
        DummyOptimizer()
    )

    applied_lr = (
        set_optimizer_learning_rate(
            optimizer=optimizer,
            step=200,
            max_learning_rate=MAX_LR,
            min_learning_rate=MIN_LR,
            warmup_steps=WARMUP_STEPS,
            total_steps=TOTAL_STEPS,
        )
    )

    optimizer_integration_passed = (
        np.isclose(
            optimizer.learning_rate,
            applied_lr,
            atol=0.0,
        )
        and
        np.isclose(
            applied_lr,
            lr_step_200,
            atol=1e-12,
        )
    )

    # ========================================================
    # Report
    # ========================================================

    print()

    print(
        f"Maximum LR: "
        f"{MAX_LR:.8f}"
    )

    print(
        f"Minimum LR: "
        f"{MIN_LR:.8f}"
    )

    print(
        f"Warmup steps: "
        f"{WARMUP_STEPS:,}"
    )

    print(
        f"Total steps: "
        f"{TOTAL_STEPS:,}"
    )

    print()

    print(
        "Important Steps"
    )

    print(
        "---------------"
    )

    print(
        f"Step 1:     "
        f"{lr_step_1:.10f}"
    )

    print(
        f"Step 200:   "
        f"{lr_step_200:.10f}"
    )

    print(
        f"Step 400:   "
        f"{lr_step_400:.10f}"
    )

    print(
        f"Step 401:   "
        f"{lr_step_401:.10f}"
    )

    print(
        f"Step 4298:  "
        f"{lr_middle:.10f}"
    )

    print(
        f"Step 8197:  "
        f"{lr_final:.10f}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Step 1 warmup correct: "
        f"{step_1_passed}"
    )

    print(
        f"Step 200 warmup correct: "
        f"{step_200_passed}"
    )

    print(
        f"Warmup reaches max LR: "
        f"{warmup_peak_passed}"
    )

    print(
        f"Cosine decay begins: "
        f"{decay_begins_passed}"
    )

    print(
        f"Final LR = minimum LR: "
        f"{final_lr_passed}"
    )

    print(
        f"After-final LR stays minimum: "
        f"{after_final_passed}"
    )

    print(
        f"Warmup monotonic increasing: "
        f"{warmup_monotonic}"
    )

    print(
        f"Cosine decay monotonic decreasing: "
        f"{decay_monotonic}"
    )

    print(
        f"Learning-rate bounds passed: "
        f"{bounds_passed}"
    )

    print(
        f"Minimum LR bound passed: "
        f"{minimum_bound_passed}"
    )

    print(
        f"Optimizer integration passed: "
        f"{optimizer_integration_passed}"
    )

    all_passed = (
        step_1_passed
        and
        step_200_passed
        and
        warmup_peak_passed
        and
        decay_begins_passed
        and
        final_lr_passed
        and
        after_final_passed
        and
        warmup_monotonic
        and
        decay_monotonic
        and
        bounds_passed
        and
        minimum_bound_passed
        and
        optimizer_integration_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: WARMUP + COSINE "
            "LEARNING-RATE SCHEDULE VALIDATED."
        )

    else:

        raise AssertionError(
            "Learning-rate scheduler "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()