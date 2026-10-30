import math


def warmup_cosine_learning_rate(
    step,
    *,
    max_learning_rate,
    min_learning_rate,
    warmup_steps,
    total_steps,
):
    """
    Warmup + cosine learning-rate schedule.

    Training steps are 1-based:

        first optimizer update:
            step = 1

        final optimizer update:
            step = total_steps


    Phase 1 — Linear warmup

        step 1 ... warmup_steps

        LR gradually increases from a small
        value to max_learning_rate.


    Phase 2 — Cosine decay

        warmup_steps ... total_steps

        LR gradually decreases from
        max_learning_rate to
        min_learning_rate.
    """

    # ========================================================
    # Validation
    # ========================================================

    if step < 1:
        raise ValueError(
            "step must be >= 1."
        )

    if max_learning_rate <= 0:
        raise ValueError(
            "max_learning_rate must "
            "be positive."
        )

    if min_learning_rate < 0:
        raise ValueError(
            "min_learning_rate cannot "
            "be negative."
        )

    if (
        min_learning_rate
        >
        max_learning_rate
    ):
        raise ValueError(
            "min_learning_rate cannot "
            "exceed max_learning_rate."
        )

    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps cannot "
            "be negative."
        )

    if total_steps <= 0:
        raise ValueError(
            "total_steps must "
            "be positive."
        )

    if warmup_steps >= total_steps:
        raise ValueError(
            "warmup_steps must be "
            "smaller than total_steps."
        )

    # ========================================================
    # After training is technically finished,
    # remain at minimum LR.
    #
    # This makes the function safe if queried
    # one step beyond the intended schedule.
    # ========================================================

    if step >= total_steps:

        return float(
            min_learning_rate
        )

    # ========================================================
    # Phase 1:
    # Linear warmup
    # ========================================================

    if (
        warmup_steps > 0
        and
        step <= warmup_steps
    ):

        warmup_fraction = (
            step
            /
            warmup_steps
        )

        learning_rate = (
            max_learning_rate
            *
            warmup_fraction
        )

        return float(
            learning_rate
        )

    # ========================================================
    # Phase 2:
    # Cosine decay
    # ========================================================

    decay_steps = (
        total_steps
        -
        warmup_steps
    )

    decay_progress = (
        step
        -
        warmup_steps
    ) / decay_steps

    # Clamp for numerical safety.
    decay_progress = min(
        max(
            decay_progress,
            0.0,
        ),
        1.0,
    )

    cosine_factor = (
        0.5
        *
        (
            1.0
            +
            math.cos(
                math.pi
                *
                decay_progress
            )
        )
    )

    learning_rate = (
        min_learning_rate
        +
        (
            max_learning_rate
            -
            min_learning_rate
        )
        *
        cosine_factor
    )

    return float(
        learning_rate
    )


def set_optimizer_learning_rate(
    optimizer,
    step,
    *,
    max_learning_rate,
    min_learning_rate,
    warmup_steps,
    total_steps,
):
    """
    Calculate the learning rate for a step
    and apply it directly to an optimizer.

    Returns the learning rate that was set.
    """

    learning_rate = (
        warmup_cosine_learning_rate(
            step=step,
            max_learning_rate=(
                max_learning_rate
            ),
            min_learning_rate=(
                min_learning_rate
            ),
            warmup_steps=(
                warmup_steps
            ),
            total_steps=(
                total_steps
            ),
        )
    )

    optimizer.learning_rate = float(
        learning_rate
    )

    return learning_rate