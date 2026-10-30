import math
import time

import numpy as np


try:
    from .data_loader import TokenBatchLoader
    from .loss import cross_entropy_loss

except ImportError:
    from data_loader import TokenBatchLoader
    from loss import cross_entropy_loss


def evaluate_model(
    model,
    binary_path,
    *,
    context_length,
    batch_size,
    eval_batches,
    seed=12345,
):
    """
    Evaluate a GPT model on deterministic
    validation batches.

    IMPORTANT:
        This function performs forward passes
        only.

        No gradients are calculated.
        No parameters are updated.

    A fresh loader with a fixed seed is created
    for every evaluation call, which means the
    same validation windows are evaluated each
    time.

    This makes validation measurements directly
    comparable across training.
    """

    if eval_batches <= 0:

        raise ValueError(
            "eval_batches must be positive."
        )

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be positive."
        )

    # ========================================================
    # Fixed validation sampler
    # ========================================================

    loader = TokenBatchLoader(
        binary_path=binary_path,
        context_length=context_length,
        batch_size=batch_size,
        seed=seed,
    )

    total_weighted_loss = 0.0

    total_tokens = 0

    batch_losses = []

    start_time = (
        time.perf_counter()
    )

    # ========================================================
    # Evaluation loop
    # ========================================================

    for _ in range(
        eval_batches
    ):

        X, Y = (
            loader.get_batch()
        )

        logits = model.forward(
            X
        )

        loss = cross_entropy_loss(
            logits,
            Y,
        )

        if not np.isfinite(
            loss
        ):

            raise FloatingPointError(
                "Non-finite validation loss."
            )

        batch_token_count = (
            Y.size
        )

        total_weighted_loss += (
            float(
                loss
            )
            *
            batch_token_count
        )

        total_tokens += (
            batch_token_count
        )

        batch_losses.append(
            float(
                loss
            )
        )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    # ========================================================
    # Token-weighted average
    # ========================================================

    average_loss = (
        total_weighted_loss
        /
        total_tokens
    )

    # ========================================================
    # Perplexity
    #
    # perplexity = e^(cross entropy)
    # ========================================================

    try:

        perplexity = math.exp(
            average_loss
        )

    except OverflowError:

        perplexity = float(
            "inf"
        )

    tokens_per_second = (
        total_tokens
        /
        elapsed
        if elapsed > 0
        else 0.0
    )

    return {
        "loss":
            float(
                average_loss
            ),

        "perplexity":
            float(
                perplexity
            ),

        "batches":
            int(
                eval_batches
            ),

        "tokens":
            int(
                total_tokens
            ),

        "seconds":
            float(
                elapsed
            ),

        "tokens_per_second":
            float(
                tokens_per_second
            ),

        "batch_losses":
            batch_losses,
    }