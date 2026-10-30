import json
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

from training.data_loader import (
    TokenBatchLoader,
)

from training.adamw import (
    AdamW,
)

from training.parameters import (
    named_parameters,
)

from training.lr_scheduler import (
    warmup_cosine_learning_rate,
)

from training.train_loop import (
    run_training_steps,
)


MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.1.json"
)


TEST_STEPS = 10


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


def main():

    print()
    print(
        "Core Real Training Loop Test"
    )

    print(
        "============================"
    )

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    # ========================================================
    # Config values
    # ========================================================

    batch_size = (
        training_config[
            "training"
        ][
            "batch_size"
        ]
    )

    context_length = (
        training_config[
            "training"
        ][
            "context_length"
        ]
    )

    total_steps = (
        training_config[
            "training"
        ][
            "max_steps"
        ]
    )

    seed = (
        training_config[
            "training"
        ][
            "seed"
        ]
    )

    max_lr = (
        training_config[
            "optimizer"
        ][
            "learning_rate"
        ]
    )

    beta1 = (
        training_config[
            "optimizer"
        ][
            "beta1"
        ]
    )

    beta2 = (
        training_config[
            "optimizer"
        ][
            "beta2"
        ]
    )

    epsilon = (
        training_config[
            "optimizer"
        ][
            "epsilon"
        ]
    )

    weight_decay = (
        training_config[
            "optimizer"
        ][
            "weight_decay"
        ]
    )

    warmup_steps = (
        training_config[
            "scheduler"
        ][
            "warmup_steps"
        ]
    )

    min_lr = (
        training_config[
            "scheduler"
        ][
            "min_learning_rate"
        ]
    )

    max_grad_norm = (
        training_config[
            "gradient_clipping"
        ][
            "max_norm"
        ]
    )

    train_path = (
        PROJECT_ROOT
        /
        training_config[
            "data"
        ][
            "train_binary"
        ]
    )

    # ========================================================
    # Model
    # ========================================================

    model = GPT.from_config(
        model_config,
        seed=seed,
    )

    # Save one weight matrix before training.
    embedding_before = (
        model
        .input_embedding
        .token_embedding
        .weight
        .copy()
    )

    # ========================================================
    # Loader
    # ========================================================

    train_loader = TokenBatchLoader(
        binary_path=train_path,
        context_length=context_length,
        batch_size=batch_size,
        seed=seed,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = AdamW(
        model=model,
        learning_rate=max_lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )

    # ========================================================
    # Run real training
    # ========================================================

    print()

    (
        history,
        tokens_seen,
    ) = run_training_steps(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        num_steps=TEST_STEPS,
        max_learning_rate=max_lr,
        min_learning_rate=min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_grad_norm=max_grad_norm,
        starting_tokens_seen=0,
        progress_every=1,
    )

    # ========================================================
    # Expected values
    # ========================================================

    tokens_per_step = (
        batch_size
        *
        context_length
    )

    expected_tokens_seen = (
        TEST_STEPS
        *
        tokens_per_step
    )

    expected_final_lr = (
        warmup_cosine_learning_rate(
            step=TEST_STEPS,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    )

    # ========================================================
    # Checks
    # ========================================================

    history_length_passed = (
        len(
            history
        )
        ==
        TEST_STEPS
    )

    optimizer_steps_passed = (
        optimizer.step_count
        ==
        TEST_STEPS
    )

    tokens_seen_passed = (
        tokens_seen
        ==
        expected_tokens_seen
    )

    steps_sequential = (
        [
            metrics[
                "step"
            ]
            for metrics
            in history
        ]
        ==
        list(
            range(
                1,
                TEST_STEPS + 1,
            )
        )
    )

    all_losses_finite = all(
        np.isfinite(
            metrics[
                "loss"
            ]
        )

        for metrics
        in history
    )

    all_gradient_norms_finite = all(
        np.isfinite(
            metrics[
                "gradient_norm"
            ]
        )

        for metrics
        in history
    )

    clipping_bound_passed = all(
        (
            metrics[
                "clip_scale"
            ]
            > 0.0
            and
            metrics[
                "clip_scale"
            ]
            <= 1.0
        )

        for metrics
        in history
    )

    learning_rates_positive = all(
        metrics[
            "learning_rate"
        ]
        > 0.0

        for metrics
        in history
    )

    warmup_increasing = all(
        history[
            index
        ][
            "learning_rate"
        ]
        <
        history[
            index + 1
        ][
            "learning_rate"
        ]

        for index in range(
            len(
                history
            )
            - 1
        )
    )

    final_lr_passed = np.isclose(
        optimizer.learning_rate,
        expected_final_lr,
        atol=1e-12,
    )

    parameters_changed = (
        not np.array_equal(
            model
            .input_embedding
            .token_embedding
            .weight,
            embedding_before,
        )
    )

    parameters_finite = all(
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

    throughput_positive = all(
        metrics[
            "tokens_per_second"
        ]
        > 0.0

        for metrics
        in history
    )

    # ========================================================
    # Summary
    # ========================================================

    losses = [
        metrics[
            "loss"
        ]
        for metrics
        in history
    ]

    average_loss = float(
        np.mean(
            losses
        )
    )

    average_throughput = float(
        np.mean(
            [
                metrics[
                    "tokens_per_second"
                ]

                for metrics
                in history
            ]
        )
    )

    print()
    print(
        "=" * 65
    )

    print(
        "TRAINING LOOP SUMMARY"
    )

    print(
        "=" * 65
    )

    print()

    print(
        f"Steps completed: "
        f"{optimizer.step_count}"
    )

    print(
        f"Tokens per step: "
        f"{tokens_per_step:,}"
    )

    print(
        f"Total tokens seen: "
        f"{tokens_seen:,}"
    )

    print(
        f"Average loss: "
        f"{average_loss:.6f}"
    )

    print(
        f"Final learning rate: "
        f"{optimizer.learning_rate:.10f}"
    )

    print(
        f"Average throughput: "
        f"{average_throughput:,.1f} "
        f"tokens/s"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"History length passed: "
        f"{history_length_passed}"
    )

    print(
        f"Optimizer completed 10 steps: "
        f"{optimizer_steps_passed}"
    )

    print(
        f"Token accounting passed: "
        f"{tokens_seen_passed}"
    )

    print(
        f"Step numbers sequential: "
        f"{steps_sequential}"
    )

    print(
        f"All losses finite: "
        f"{all_losses_finite}"
    )

    print(
        f"All gradient norms finite: "
        f"{all_gradient_norms_finite}"
    )

    print(
        f"Clip scales valid: "
        f"{clipping_bound_passed}"
    )

    print(
        f"Learning rates positive: "
        f"{learning_rates_positive}"
    )

    print(
        f"Warmup learning rates increase: "
        f"{warmup_increasing}"
    )

    print(
        f"Final scheduled LR correct: "
        f"{final_lr_passed}"
    )

    print(
        f"Model parameters changed: "
        f"{parameters_changed}"
    )

    print(
        f"Model parameters finite: "
        f"{parameters_finite}"
    )

    print(
        f"Throughput positive: "
        f"{throughput_positive}"
    )

    all_passed = (
        history_length_passed
        and
        optimizer_steps_passed
        and
        tokens_seen_passed
        and
        steps_sequential
        and
        all_losses_finite
        and
        all_gradient_norms_finite
        and
        clipping_bound_passed
        and
        learning_rates_positive
        and
        warmup_increasing
        and
        final_lr_passed
        and
        parameters_changed
        and
        parameters_finite
        and
        throughput_positive
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: CORE REAL TRAINING "
            "LOOP VALIDATED."
        )

    else:

        raise AssertionError(
            "Core training loop validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()