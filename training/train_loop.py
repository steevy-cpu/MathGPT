import time

import numpy as np


try:
    from .loss import (
        cross_entropy_loss_and_backward,
    )

    from .gradient_clipping import (
        clip_grad_norm,
    )

    from .lr_scheduler import (
        set_optimizer_learning_rate,
    )

except ImportError:
    from loss import (
        cross_entropy_loss_and_backward,
    )

    from gradient_clipping import (
        clip_grad_norm,
    )

    from lr_scheduler import (
        set_optimizer_learning_rate,
    )


def train_one_step(
    *,
    model,
    optimizer,
    train_loader,
    max_learning_rate,
    min_learning_rate,
    warmup_steps,
    total_steps,
    max_grad_norm,
):
    """
    Run one complete GPT training step.

    Returns a dictionary of metrics.

    Order matters:

        batch
        forward
        loss
        zero_grad
        backward
        clip
        set LR
        optimizer step
    """

    # ========================================================
    # Determine NEXT optimizer step.
    #
    # AdamW step_count increments inside optimizer.step().
    # ========================================================

    next_step = (
        optimizer.step_count
        + 1
    )

    if next_step > total_steps:

        raise RuntimeError(
            "Training attempted to go "
            "past total_steps."
        )

    # ========================================================
    # Batch
    # ========================================================

    X, Y = train_loader.get_batch()

    batch_tokens = int(
        Y.size
    )

    # ========================================================
    # Forward
    # ========================================================

    step_start = (
        time.perf_counter()
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

    # ========================================================
    # Loss + dLogits
    # ========================================================

    loss_start = (
        time.perf_counter()
    )

    (
        loss,
        grad_logits,
    ) = cross_entropy_loss_and_backward(
        logits,
        Y,
    )

    loss_seconds = (
        time.perf_counter()
        - loss_start
    )

    if not np.isfinite(
        loss
    ):

        raise FloatingPointError(
            "Non-finite training loss."
        )

    # ========================================================
    # Clear old gradients
    # ========================================================

    optimizer.zero_grad()

    # ========================================================
    # Backward
    # ========================================================

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

    # ========================================================
    # Global gradient clipping
    # ========================================================

    clipping_start = (
        time.perf_counter()
    )

    (
        gradient_norm,
        clip_scale,
    ) = clip_grad_norm(
        model=model,
        max_norm=max_grad_norm,
    )

    clipping_seconds = (
        time.perf_counter()
        - clipping_start
    )

    if not np.isfinite(
        gradient_norm
    ):

        raise FloatingPointError(
            "Non-finite gradient norm."
        )

    # ========================================================
    # Learning-rate scheduler
    #
    # Set LR BEFORE optimizer.step().
    # ========================================================

    learning_rate = (
        set_optimizer_learning_rate(
            optimizer=optimizer,
            step=next_step,
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

    # ========================================================
    # AdamW
    # ========================================================

    optimizer_start = (
        time.perf_counter()
    )

    optimizer.step()

    optimizer_seconds = (
        time.perf_counter()
        - optimizer_start
    )

    # ========================================================
    # Timing
    # ========================================================

    total_seconds = (
        time.perf_counter()
        - step_start
    )

    tokens_per_second = (
        batch_tokens
        /
        total_seconds
        if total_seconds > 0
        else 0.0
    )

    # ========================================================
    # Metrics
    # ========================================================

    return {
        "step":
            int(
                optimizer.step_count
            ),

        "loss":
            float(
                loss
            ),

        "learning_rate":
            float(
                learning_rate
            ),

        "gradient_norm":
            float(
                gradient_norm
            ),

        "clip_scale":
            float(
                clip_scale
            ),

        "batch_tokens":
            batch_tokens,

        "forward_seconds":
            float(
                forward_seconds
            ),

        "loss_seconds":
            float(
                loss_seconds
            ),

        "backward_seconds":
            float(
                backward_seconds
            ),

        "clipping_seconds":
            float(
                clipping_seconds
            ),

        "optimizer_seconds":
            float(
                optimizer_seconds
            ),

        "total_seconds":
            float(
                total_seconds
            ),

        "tokens_per_second":
            float(
                tokens_per_second
            ),
    }


def run_training_steps(
    *,
    model,
    optimizer,
    train_loader,
    num_steps,
    max_learning_rate,
    min_learning_rate,
    warmup_steps,
    total_steps,
    max_grad_norm,
    starting_tokens_seen=0,
    progress_every=1,
):
    """
    Run several training steps.

    This is the reusable core loop that the
    final trainer will eventually call.

    Returns:

        history,
        tokens_seen
    """

    if num_steps <= 0:

        raise ValueError(
            "num_steps must be positive."
        )

    if progress_every <= 0:

        raise ValueError(
            "progress_every must be positive."
        )

    tokens_seen = int(
        starting_tokens_seen
    )

    history = []

    for _ in range(
        num_steps
    ):

        metrics = train_one_step(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
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
            max_grad_norm=(
                max_grad_norm
            ),
        )

        tokens_seen += (
            metrics[
                "batch_tokens"
            ]
        )

        metrics[
            "tokens_seen"
        ] = tokens_seen

        history.append(
            metrics
        )

        step = metrics[
            "step"
        ]

        if (
            step == 1
            or
            step % progress_every == 0
        ):

            print(
                f"Step "
                f"{step:>5} | "
                f"Loss "
                f"{metrics['loss']:.6f} | "
                f"LR "
                f"{metrics['learning_rate']:.8f} | "
                f"Grad "
                f"{metrics['gradient_norm']:.4f} | "
                f"Clip "
                f"{metrics['clip_scale']:.4f} | "
                f"{metrics['tokens_per_second']:,.0f} tok/s"
            )

    return (
        history,
        tokens_seen,
    )