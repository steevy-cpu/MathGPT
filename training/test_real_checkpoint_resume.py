import json
import sys

from pathlib import Path

import numpy as np


# ============================================================
# Project root
# ============================================================

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


# ============================================================
# Imports
# ============================================================

from model.gpt import GPT

from training.adamw import (
    AdamW,
)

from training.data_loader import (
    TokenBatchLoader,
)

from training.evaluation import (
    evaluate_model,
)

from training.parameters import (
    named_parameters,
)

from training.train_loop import (
    run_training_steps,
)

from training.checkpoint_manager import (
    restore_training_checkpoint,
)

from training.lr_scheduler import (
    warmup_cosine_learning_rate,
)


# ============================================================
# Paths
# ============================================================

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


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "general_pretraining"
    / "general_pretraining_short_v0.1_20261105_112710"
    / "mathgpt_step_001000.npz"
)


# ============================================================
# Known Stage 9I facts
# ============================================================

EXPECTED_RESTORED_STEP = 1_000

EXPECTED_RESTORED_TOKENS = 1_024_000

EXPECTED_PARAMETER_COUNT = 8_110_848

RESUME_STEPS = 5

EXPECTED_FINAL_STEP = (
    EXPECTED_RESTORED_STEP
    +
    RESUME_STEPS
)

EXPECTED_FINAL_TOKENS = (
    EXPECTED_RESTORED_TOKENS
    +
    (
        RESUME_STEPS
        *
        4
        *
        256
    )
)


# Validation result produced on the Mac at step 1000.
MAC_STAGE9I_VALIDATION_LOSS = (
    8.16890445
)


# Cross-machine floating-point operations may differ
# slightly between macOS ARM and Linux AMD64.
#
# We therefore use a small tolerance instead of
# demanding exact floating-point identity.
CROSS_MACHINE_VALIDATION_TOLERANCE = (
    1e-3
)


# ============================================================
# Helpers
# ============================================================

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


def parameters_finite(
    model,
):

    return all(
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


def gradients_finite(
    model,
):

    return all(
        np.all(
            np.isfinite(
                gradient
            )
        )

        for (
            _,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    )


def optimizer_moments_finite(
    optimizer,
):

    first_finite = all(
        np.all(
            np.isfinite(
                moment
            )
        )

        for moment
        in optimizer.first_moments.values()
    )

    second_finite = all(
        np.all(
            np.isfinite(
                moment
            )
        )

        for moment
        in optimizer.second_moments.values()
    )

    return (
        first_finite
        and
        second_finite
    )


def optimizer_moments_nonzero(
    optimizer,
):

    first_nonzero = any(
        np.any(
            moment != 0
        )

        for moment
        in optimizer.first_moments.values()
    )

    second_nonzero = any(
        np.any(
            moment != 0
        )

        for moment
        in optimizer.second_moments.values()
    )

    return (
        first_nonzero
        and
        second_nonzero
    )


def parameter_count(
    model,
):

    return int(
        sum(
            parameter.size

            for (
                _,
                parameter,
                _,
            ) in named_parameters(
                model
            )
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 9J.1 — REAL CROSS-MACHINE "
        "CHECKPOINT RESUME TEST"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # Load configs
    # ========================================================

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    # ========================================================
    # Read training configuration
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

    seed = (
        training_config[
            "training"
        ][
            "seed"
        ]
    )

    total_steps = (
        training_config[
            "training"
        ][
            "max_steps"
        ]
    )

    # --------------------------------------------------------

    max_learning_rate = (
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

    # --------------------------------------------------------

    warmup_steps = (
        training_config[
            "scheduler"
        ][
            "warmup_steps"
        ]
    )

    min_learning_rate = (
        training_config[
            "scheduler"
        ][
            "min_learning_rate"
        ]
    )

    # --------------------------------------------------------

    max_grad_norm = (
        training_config[
            "gradient_clipping"
        ][
            "max_norm"
        ]
    )

    eval_batches = (
        training_config[
            "validation"
        ][
            "eval_batches"
        ]
    )

    # ========================================================
    # Data
    # ========================================================

    train_path = (
        PROJECT_ROOT
        /
        training_config[
            "data"
        ][
            "train_binary"
        ]
    )

    validation_path = (
        PROJECT_ROOT
        /
        training_config[
            "data"
        ][
            "validation_binary"
        ]
    )

    # ========================================================
    # Preflight
    # ========================================================

    print()

    print(
        f"Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Real Stage 9I checkpoint "
            f"not found: "
            f"{CHECKPOINT_PATH}"
        )

    if not train_path.exists():

        raise FileNotFoundError(
            f"Training binary missing: "
            f"{train_path}"
        )

    if not validation_path.exists():

        raise FileNotFoundError(
            f"Validation binary missing: "
            f"{validation_path}"
        )

    print(
        f"Checkpoint size: "
        f"{CHECKPOINT_PATH.stat().st_size / (1024 ** 2):.2f} MiB"
    )

    print(
        f"Train corpus: "
        f"{train_path}"
    )

    print(
        f"Validation corpus: "
        f"{validation_path}"
    )

    # ========================================================
    # Create FRESH model
    #
    # Deliberately use a different seed.
    #
    # The checkpoint must overwrite the initial
    # random model parameters.
    # ========================================================

    print()
    print(
        "Creating fresh full-size MathGPT..."
    )

    model = GPT.from_config(
        model_config,
        seed=999,
    )

    model_parameter_count = (
        parameter_count(
            model
        )
    )

    print(
        f"Model parameter count: "
        f"{model_parameter_count:,}"
    )

    # ========================================================
    # Fresh optimizer
    # ========================================================

    optimizer = AdamW(
        model=model,
        learning_rate=max_learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )

    # ========================================================
    # Fresh training loader
    #
    # Again, deliberately use a different seed.
    # The checkpoint must replace its RNG state.
    # ========================================================

    train_loader = TokenBatchLoader(
        binary_path=train_path,
        context_length=context_length,
        batch_size=batch_size,
        seed=999,
    )

    fresh_loader_state = (
        train_loader.state_dict()
    )

    # ========================================================
    # Restore real Stage 9I checkpoint
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "RESTORING REAL STEP-1000 CHECKPOINT"
    )

    print(
        "=" * 72
    )

    (
        metadata,
        restored_step,
        restored_tokens_seen,
    ) = restore_training_checkpoint(
        model=model,
        optimizer=optimizer,
        path=CHECKPOINT_PATH,
        train_loader=train_loader,
        strict=True,
    )

    restored_loader_state = (
        train_loader.state_dict()
    )

    checkpoint_extra_state = (
        metadata.get(
            "extra_state",
            {},
        )
    )

    checkpoint_loader_state = (
        checkpoint_extra_state.get(
            "train_loader_state"
        )
    )

    # ========================================================
    # Restore checks
    # ========================================================

    restored_step_passed = (
        restored_step
        ==
        EXPECTED_RESTORED_STEP
    )

    restored_tokens_passed = (
        restored_tokens_seen
        ==
        EXPECTED_RESTORED_TOKENS
    )

    parameter_count_passed = (
        metadata[
            "parameter_count"
        ]
        ==
        EXPECTED_PARAMETER_COUNT
        ==
        model_parameter_count
    )

    optimizer_class_passed = (
        metadata[
            "optimizer_class"
        ]
        ==
        "AdamW"
    )

    optimizer_step_passed = (
        optimizer.step_count
        ==
        EXPECTED_RESTORED_STEP
    )

    moment_state_present = (
        metadata.get(
            "has_moment_state",
            False,
        )
    )

    moment_state_finite = (
        optimizer_moments_finite(
            optimizer
        )
    )

    moment_state_nonzero = (
        optimizer_moments_nonzero(
            optimizer
        )
    )

    loader_state_present = (
        checkpoint_loader_state
        is not None
    )

    loader_changed_from_fresh = (
        json.dumps(
            fresh_loader_state,
            sort_keys=True,
        )
        !=
        json.dumps(
            restored_loader_state,
            sort_keys=True,
        )
    )

    loader_state_matches_checkpoint = (
        loader_state_present
        and
        json.dumps(
            restored_loader_state,
            sort_keys=True,
        )
        ==
        json.dumps(
            checkpoint_loader_state,
            sort_keys=True,
        )
    )

    restored_parameters_finite = (
        parameters_finite(
            model
        )
    )

    # ========================================================
    # Validate restored model BEFORE further training
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "VALIDATING RESTORED STEP-1000 MODEL"
    )

    print(
        "=" * 72
    )

    restored_validation = (
        evaluate_model(
            model=model,
            binary_path=validation_path,
            context_length=context_length,
            batch_size=batch_size,
            eval_batches=eval_batches,
            seed=12345,
        )
    )

    restored_validation_loss = (
        restored_validation[
            "loss"
        ]
    )

    validation_difference = abs(
        restored_validation_loss
        -
        MAC_STAGE9I_VALIDATION_LOSS
    )

    cross_machine_validation_passed = (
        validation_difference
        <=
        CROSS_MACHINE_VALIDATION_TOLERANCE
    )

    print()

    print(
        f"Mac Stage 9I validation loss: "
        f"{MAC_STAGE9I_VALIDATION_LOSS:.8f}"
    )

    print(
        f"Linux restored validation loss: "
        f"{restored_validation_loss:.8f}"
    )

    print(
        f"Absolute difference: "
        f"{validation_difference:.10f}"
    )

    print(
        f"Tolerance: "
        f"{CROSS_MACHINE_VALIDATION_TOLERANCE}"
    )

    # ========================================================
    # Expected scheduler values
    # ========================================================

    expected_lr_step_1001 = (
        warmup_cosine_learning_rate(
            step=1001,
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

    expected_lr_step_1005 = (
        warmup_cosine_learning_rate(
            step=1005,
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
    # Continue real training
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "CONTINUING REAL TRAINING: "
        "STEPS 1001–1005"
    )

    print(
        "=" * 72
    )

    (
        history,
        final_tokens_seen,
        saved_checkpoints,
    ) = run_training_steps(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        num_steps=RESUME_STEPS,
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
        starting_tokens_seen=(
            restored_tokens_seen
        ),
        progress_every=1,

        # Deliberately no checkpointing.
        #
        # This is a validation test only.
        checkpoint_every=None,
    )

    # ========================================================
    # Post-resume validation
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "VALIDATION AFTER STEP 1005"
    )

    print(
        "=" * 72
    )

    post_resume_validation = (
        evaluate_model(
            model=model,
            binary_path=validation_path,
            context_length=context_length,
            batch_size=batch_size,
            eval_batches=eval_batches,
            seed=12345,
        )
    )

    print()

    print(
        f"Validation loss after resume: "
        f"{post_resume_validation['loss']:.8f}"
    )

    print(
        f"Validation perplexity after resume: "
        f"{post_resume_validation['perplexity']:,.2f}"
    )

    # ========================================================
    # Continued-training checks
    # ========================================================

    expected_steps = list(
        range(
            1001,
            1006,
        )
    )

    actual_steps = [
        metrics[
            "step"
        ]

        for metrics
        in history
    ]

    steps_sequential = (
        actual_steps
        ==
        expected_steps
    )

    final_step_passed = (
        optimizer.step_count
        ==
        EXPECTED_FINAL_STEP
    )

    final_tokens_passed = (
        final_tokens_seen
        ==
        EXPECTED_FINAL_TOKENS
    )

    no_test_checkpoints_created = (
        saved_checkpoints
        ==
        []
    )

    training_losses_finite = all(
        np.isfinite(
            metrics[
                "loss"
            ]
        )

        for metrics
        in history
    )

    training_gradients_finite = all(
        np.isfinite(
            metrics[
                "gradient_norm"
            ]
        )

        for metrics
        in history
    )

    clip_scales_valid = all(
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

    first_resume_lr_passed = (
        np.isclose(
            history[
                0
            ][
                "learning_rate"
            ],
            expected_lr_step_1001,
            atol=1e-15,
            rtol=0.0,
        )
    )

    final_resume_lr_passed = (
        np.isclose(
            history[
                -1
            ][
                "learning_rate"
            ],
            expected_lr_step_1005,
            atol=1e-15,
            rtol=0.0,
        )
    )

    final_parameters_finite = (
        parameters_finite(
            model
        )
    )

    final_gradients_finite = (
        gradients_finite(
            model
        )
    )

    final_moments_finite = (
        optimizer_moments_finite(
            optimizer
        )
    )

    post_validation_finite = (
        np.isfinite(
            post_resume_validation[
                "loss"
            ]
        )
        and
        np.isfinite(
            post_resume_validation[
                "perplexity"
            ]
        )
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 9J.1 RESULTS"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Restored step: "
        f"{restored_step:,}"
    )

    print(
        f"Restored tokens: "
        f"{restored_tokens_seen:,}"
    )

    print(
        f"Restored optimizer step: "
        f"{optimizer.step_count - RESUME_STEPS:,}"
    )

    print(
        f"Checkpoint parameter count: "
        f"{metadata['parameter_count']:,}"
    )

    print()

    print(
        "Resumed Training"
    )

    print(
        "----------------"
    )

    for metrics in history:

        print(
            f"Step "
            f"{metrics['step']} | "
            f"Loss "
            f"{metrics['loss']:.8f} | "
            f"LR "
            f"{metrics['learning_rate']:.10f} | "
            f"Grad "
            f"{metrics['gradient_norm']:.6f} | "
            f"Clip "
            f"{metrics['clip_scale']:.6f}"
        )

    print()

    print(
        f"Final step: "
        f"{optimizer.step_count:,}"
    )

    print(
        f"Final tokens seen: "
        f"{final_tokens_seen:,}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Checkpoint step restored: "
        f"{restored_step_passed}"
    )

    print(
        f"Checkpoint token count restored: "
        f"{restored_tokens_passed}"
    )

    print(
        f"8,110,848 parameters restored: "
        f"{parameter_count_passed}"
    )

    print(
        f"Optimizer class is AdamW: "
        f"{optimizer_class_passed}"
    )

    print(
        f"Optimizer step restored: "
        f"{optimizer_step_passed}"
    )

    print(
        f"AdamW moment state present: "
        f"{moment_state_present}"
    )

    print(
        f"AdamW moments finite: "
        f"{moment_state_finite}"
    )

    print(
        f"AdamW moments non-zero: "
        f"{moment_state_nonzero}"
    )

    print(
        f"Loader RNG state present: "
        f"{loader_state_present}"
    )

    print(
        f"Fresh loader changed by restore: "
        f"{loader_changed_from_fresh}"
    )

    print(
        f"Loader RNG matches checkpoint: "
        f"{loader_state_matches_checkpoint}"
    )

    print(
        f"Restored model parameters finite: "
        f"{restored_parameters_finite}"
    )

    print(
        f"Cross-machine validation matches: "
        f"{cross_machine_validation_passed}"
    )

    print(
        f"Steps 1001-1005 sequential: "
        f"{steps_sequential}"
    )

    print(
        f"Reached optimizer step 1005: "
        f"{final_step_passed}"
    )

    print(
        f"Token accounting reached 1,029,120: "
        f"{final_tokens_passed}"
    )

    print(
        f"Step 1001 LR correct: "
        f"{first_resume_lr_passed}"
    )

    print(
        f"Step 1005 LR correct: "
        f"{final_resume_lr_passed}"
    )

    print(
        f"All resumed losses finite: "
        f"{training_losses_finite}"
    )

    print(
        f"All resumed gradient norms finite: "
        f"{training_gradients_finite}"
    )

    print(
        f"All clip scales valid: "
        f"{clip_scales_valid}"
    )

    print(
        f"Final model parameters finite: "
        f"{final_parameters_finite}"
    )

    print(
        f"Final gradients finite: "
        f"{final_gradients_finite}"
    )

    print(
        f"Final AdamW moments finite: "
        f"{final_moments_finite}"
    )

    print(
        f"Post-resume validation finite: "
        f"{post_validation_finite}"
    )

    print(
        f"No test checkpoint created: "
        f"{no_test_checkpoints_created}"
    )

    # ========================================================
    # Overall result
    # ========================================================

    all_passed = (
        restored_step_passed
        and
        restored_tokens_passed
        and
        parameter_count_passed
        and
        optimizer_class_passed
        and
        optimizer_step_passed
        and
        moment_state_present
        and
        moment_state_finite
        and
        moment_state_nonzero
        and
        loader_state_present
        and
        loader_changed_from_fresh
        and
        loader_state_matches_checkpoint
        and
        restored_parameters_finite
        and
        cross_machine_validation_passed
        and
        steps_sequential
        and
        final_step_passed
        and
        final_tokens_passed
        and
        first_resume_lr_passed
        and
        final_resume_lr_passed
        and
        training_losses_finite
        and
        training_gradients_finite
        and
        clip_scales_valid
        and
        final_parameters_finite
        and
        final_gradients_finite
        and
        final_moments_finite
        and
        post_validation_finite
        and
        no_test_checkpoints_created
    )

    print()
    print(
        "=" * 72
    )

    if all_passed:

        print(
            "SUCCESS: REAL STAGE 9I CHECKPOINT "
            "RESTORED AND RESUMED ACROSS MACHINES."
        )

    else:

        raise AssertionError(
            "Stage 9J.1 real checkpoint "
            "resume validation failed."
        )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()