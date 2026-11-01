from pathlib import Path


try:
    from .checkpoint import (
        save_checkpoint,
        load_checkpoint,
    )

except ImportError:
    from checkpoint import (
        save_checkpoint,
        load_checkpoint,
    )


def checkpoint_path_for_step(
    directory,
    step,
    prefix="mathgpt",
):
    """
    Build a deterministic checkpoint filename.

    Example:

        mathgpt_step_001000.npz
    """

    if step <= 0:
        raise ValueError(
            "step must be positive."
        )

    directory = Path(directory)

    return (
        directory
        /
        (
            f"{prefix}_step_"
            f"{step:06d}.npz"
        )
    )


def save_training_checkpoint(
    *,
    model,
    optimizer,
    directory,
    step,
    tokens_seen,
    train_loader=None,
    logger=None,
    prefix="mathgpt",
    extra_state=None,
):
    """
    Save a resumable training checkpoint.

    Includes:

        model parameters
        AdamW m/v
        optimizer step
        tokens_seen
        training step
        training-loader RNG state
    """

    if step <= 0:
        raise ValueError(
            "step must be positive."
        )

    if tokens_seen < 0:
        raise ValueError(
            "tokens_seen cannot be negative."
        )

    if optimizer.step_count != step:
        raise ValueError(
            "Checkpoint step does not match "
            "optimizer.step_count."
        )

    path = checkpoint_path_for_step(
        directory=directory,
        step=step,
        prefix=prefix,
    )

    state = {
        "training_step":
            int(step),

        "tokens_seen":
            int(tokens_seen),
    }

    # ========================================================
    # Data loader RNG state
    # ========================================================

    if train_loader is not None:
        state[
            "train_loader_state"
        ] = train_loader.state_dict()

    # ========================================================
    # Optional extra trainer state
    # ========================================================

    if extra_state is not None:
        state.update(extra_state)

    metadata = save_checkpoint(
        model=model,
        optimizer=optimizer,
        path=path,
        extra_state=state,
    )

    # ========================================================
    # Log checkpoint event
    # ========================================================

    if logger is not None:
        logger.log_event(
            "checkpoint_saved",
            step=int(step),
            tokens_seen=int(tokens_seen),
            path=str(path),
        )

    return (
        path,
        metadata,
    )


def restore_training_checkpoint(
    *,
    model,
    optimizer,
    path,
    train_loader=None,
    strict=True,
):
    """
    Restore a complete training checkpoint.

    Restores:

        model parameters
        AdamW state
        optimizer step
        loader RNG state

    Returns:

        metadata
        training_step
        tokens_seen
    """

    metadata = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=path,
        strict=strict,
    )

    extra_state = metadata.get(
        "extra_state",
        {},
    )

    if "training_step" not in extra_state:
        raise KeyError(
            "Checkpoint is missing "
            "training_step."
        )

    if "tokens_seen" not in extra_state:
        raise KeyError(
            "Checkpoint is missing "
            "tokens_seen."
        )

    training_step = int(
        extra_state[
            "training_step"
        ]
    )

    tokens_seen = int(
        extra_state[
            "tokens_seen"
        ]
    )

    # ========================================================
    # Consistency check
    # ========================================================

    if (
        optimizer.step_count
        != training_step
    ):
        raise ValueError(
            "Restored optimizer step does "
            "not match training_step."
        )

    # ========================================================
    # Restore loader RNG
    # ========================================================

    if train_loader is not None:

        loader_state = extra_state.get(
            "train_loader_state"
        )

        if loader_state is None:

            if strict:
                raise KeyError(
                    "Checkpoint does not contain "
                    "training loader state."
                )

        else:
            train_loader.load_state_dict(
                loader_state
            )

    return (
        metadata,
        training_step,
        tokens_seen,
    )