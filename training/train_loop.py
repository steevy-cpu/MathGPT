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

    from .checkpoint_manager import (
        save_training_checkpoint,
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

    from checkpoint_manager import (
        save_training_checkpoint,
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
    """

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

    step_start = (
        time.perf_counter()
    )

    # ========================================================
    # Forward
    # ========================================================

    forward_start = (
        time.perf_counter()
    )

    logits = model.forward(X)

    forward_seconds = (
        time.perf_counter()
        -
        forward_start
    )

    # ========================================================
    # Loss
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
        -
        loss_start
    )

    if not np.isfinite(loss):
        raise FloatingPointError(
            "Non-finite training loss."
        )

    # ========================================================
    # Backward
    # ========================================================

    optimizer.zero_grad()

    backward_start = (
        time.perf_counter()
    )

    model.backward(
        grad_logits
    )

    backward_seconds = (
        time.perf_counter()
        -
        backward_start
    )

    # ========================================================
    # Gradient clipping
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
        -
        clipping_start
    )

    if not np.isfinite(
        gradient_norm
    ):
        raise FloatingPointError(
            "Non-finite gradient norm."
        )

    # ========================================================
    # Learning rate
    # ========================================================

    learning_rate = (
        set_optimizer_learning_rate(
            optimizer=optimizer,
            step=next_step,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer_start = (
        time.perf_counter()
    )

    optimizer.step()

    optimizer_seconds = (
        time.perf_counter()
        -
        optimizer_start
    )

    total_seconds = (
        time.perf_counter()
        -
        step_start
    )

    tokens_per_second = (
        batch_tokens
        /
        total_seconds
        if total_seconds > 0
        else 0.0
    )

    return {
        "step":
            int(
                optimizer.step_count
            ),

        "loss":
            float(loss),

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
    logger=None,
    log_every=1,
    checkpoint_every=None,
    checkpoint_directory=None,
    checkpoint_prefix="mathgpt",
):
    """
    Run several GPT training steps.

    Supports:

        persistent logging
        periodic checkpoints
        deterministic loader-state checkpointing
    """

    if num_steps <= 0:
        raise ValueError(
            "num_steps must be positive."
        )

    if progress_every <= 0:
        raise ValueError(
            "progress_every must be positive."
        )

    if log_every <= 0:
        raise ValueError(
            "log_every must be positive."
        )

    checkpointing_enabled = (
        checkpoint_every
        is not None
    )

    if checkpointing_enabled:

        if checkpoint_every <= 0:
            raise ValueError(
                "checkpoint_every must "
                "be positive."
            )

        if checkpoint_directory is None:
            raise ValueError(
                "checkpoint_directory is required "
                "when checkpointing is enabled."
            )

    tokens_seen = int(
        starting_tokens_seen
    )

    history = []

    saved_checkpoints = []

    for _ in range(
        num_steps
    ):

        metrics = train_one_step(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_grad_norm=max_grad_norm,
        )

        # ====================================================
        # Token accounting
        # ====================================================

        tokens_seen += (
            metrics[
                "batch_tokens"
            ]
        )

        metrics[
            "tokens_seen"
        ] = tokens_seen

        step = metrics[
            "step"
        ]

        history.append(
            metrics
        )

        # ====================================================
        # Persistent logging
        # ====================================================

        if (
            logger is not None
            and
            (
                step == 1
                or
                step % log_every == 0
            )
        ):
            logger.log_training(
                metrics
            )

        # ====================================================
        # Periodic checkpoint
        # ====================================================

        if (
            checkpointing_enabled
            and
            step % checkpoint_every == 0
        ):

            (
                checkpoint_path,
                _,
            ) = save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                directory=checkpoint_directory,
                step=step,
                tokens_seen=tokens_seen,

                # Critical Stage 9H addition:
                train_loader=train_loader,

                logger=logger,
                prefix=checkpoint_prefix,
            )

            saved_checkpoints.append(
                checkpoint_path
            )

        # ====================================================
        # Terminal progress
        # ====================================================

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
                f"{metrics['tokens_per_second']:,.0f} "
                f"tok/s"
            )

    return (
        history,
        tokens_seen,
        saved_checkpoints,
    )