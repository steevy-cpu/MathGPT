import argparse
import json
import sys

from datetime import (
    datetime,
    timezone,
)

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

from training.logger import (
    TrainingLogger,
)

from training.parameters import (
    named_parameters,
)

from training.train_loop import (
    run_training_steps,
)

from training.checkpoint_manager import (
    restore_training_checkpoint,
    save_training_checkpoint,
)


# ============================================================
# Config paths
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


def resolve_project_path(
    path,
):

    path = Path(
        path
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        /
        path
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


def optimizer_state_finite(
    optimizer,
):

    first_finite = all(
        np.all(
            np.isfinite(
                value
            )
        )

        for value
        in optimizer.first_moments.values()
    )

    second_finite = all(
        np.all(
            np.isfinite(
                value
            )
        )

        for value
        in optimizer.second_moments.values()
    )

    return (
        first_finite
        and
        second_finite
    )


def metadata_path_for_binary(
    binary_path,
):

    return binary_path.with_name(
        binary_path.stem
        +
        "_metadata.json"
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Run or resume MathGPT "
            "general pretraining."
        )
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint to resume from. "
            "If omitted, training begins "
            "from a fresh model."
        ),
    )

    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help=(
            "Absolute optimizer step at "
            "which this run should stop. "
            "Defaults to config max_steps."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional explicit run name."
        ),
    )

    return parser.parse_args()


# ============================================================
# Validation helper
# ============================================================

def run_validation(
    *,
    model,
    validation_path,
    context_length,
    batch_size,
    eval_batches,
    step,
    tokens_seen,
    logger,
):

    print()
    print(
        "=" * 76
    )

    print(
        f"VALIDATION — STEP "
        f"{step:,}"
    )

    print(
        "=" * 76
    )

    result = evaluate_model(
        model=model,
        binary_path=validation_path,
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=eval_batches,
        seed=12345,
    )

    logger.log_validation(
        step=step,
        tokens_seen=tokens_seen,
        result=result,
    )

    print()

    print(
        f"Validation loss: "
        f"{result['loss']:.8f}"
    )

    print(
        f"Validation perplexity: "
        f"{result['perplexity']:,.2f}"
    )

    print(
        f"Validation tokens: "
        f"{result['tokens']:,}"
    )

    print(
        f"Validation throughput: "
        f"{result['tokens_per_second']:,.1f} "
        f"tokens/s"
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print()
    print(
        "=" * 76
    )

    print(
        "MATHGPT — GENERAL PRETRAINING"
    )

    print(
        "=" * 76
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
    # Training config
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

    configured_max_steps = (
        training_config[
            "training"
        ][
            "max_steps"
        ]
    )

    # ========================================================
    # Optimizer
    # ========================================================

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

    # ========================================================
    # Scheduler
    # ========================================================

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

    # ========================================================
    # Gradient clipping
    # ========================================================

    max_grad_norm = (
        training_config[
            "gradient_clipping"
        ][
            "max_norm"
        ]
    )

    # ========================================================
    # Logging
    # ========================================================

    log_every = (
        training_config[
            "logging"
        ][
            "log_every_steps"
        ]
    )

    # ========================================================
    # Validation
    # ========================================================

    eval_every = (
        training_config[
            "validation"
        ][
            "eval_every_steps"
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
    # Checkpointing
    # ========================================================

    checkpoint_every = (
        training_config[
            "checkpointing"
        ][
            "save_every_steps"
        ]
    )

    checkpoint_root = (
        resolve_project_path(
            training_config[
                "checkpointing"
            ][
                "directory"
            ]
        )
    )

    # ========================================================
    # Data
    # ========================================================

    train_path = (
        resolve_project_path(
            training_config[
                "data"
            ][
                "train_binary"
            ]
        )
    )

    validation_path = (
        resolve_project_path(
            training_config[
                "data"
            ][
                "validation_binary"
            ]
        )
    )

    expected_tokenizer = (
        training_config[
            "data"
        ][
            "tokenizer"
        ]
    )

    # ========================================================
    # Preflight: files
    # ========================================================

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

    train_metadata_path = (
        metadata_path_for_binary(
            train_path
        )
    )

    validation_metadata_path = (
        metadata_path_for_binary(
            validation_path
        )
    )

    if not train_metadata_path.exists():

        raise FileNotFoundError(
            f"Training metadata missing: "
            f"{train_metadata_path}"
        )

    if not validation_metadata_path.exists():

        raise FileNotFoundError(
            f"Validation metadata missing: "
            f"{validation_metadata_path}"
        )

    train_metadata = load_json(
        train_metadata_path
    )

    validation_metadata = load_json(
        validation_metadata_path
    )

    # ========================================================
    # Preflight: dataset identity
    # ========================================================

    train_tokenizer = (
        train_metadata.get(
            "tokenizer"
        )
    )

    validation_tokenizer = (
        validation_metadata.get(
            "tokenizer"
        )
    )

    if (
        train_tokenizer
        != expected_tokenizer
    ):

        raise RuntimeError(
            "Training corpus tokenizer "
            "does not match config."
        )

    if (
        validation_tokenizer
        != expected_tokenizer
    ):

        raise RuntimeError(
            "Validation corpus tokenizer "
            "does not match config."
        )

    if (
        train_metadata.get(
            "stage"
        )
        !=
        "5M.10"
    ):

        raise RuntimeError(
            "Training corpus is not "
            "Stage 5M.10."
        )

    if (
        validation_metadata.get(
            "stage"
        )
        !=
        "5M.10"
    ):

        raise RuntimeError(
            "Validation corpus is not "
            "Stage 5M.10."
        )

    # ========================================================
    # Target step
    #
    # This controls where THIS invocation stops.
    #
    # It does NOT alter the LR schedule.
    #
    # LR always follows the full configured
    # 8197-step schedule.
    # ========================================================

    if args.target_step is None:

        target_step = (
            configured_max_steps
        )

    else:

        target_step = int(
            args.target_step
        )

    if target_step <= 0:

        raise ValueError(
            "target-step must be positive."
        )

    if (
        target_step
        >
        configured_max_steps
    ):

        raise ValueError(
            "target-step cannot exceed "
            "configured max_steps."
        )

    # ========================================================
    # Run identity
    # ========================================================

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    if args.run_name is not None:

        run_name = (
            args.run_name
        )

    elif args.resume is not None:

        run_name = (
            "general_pretraining_resume_"
            f"{timestamp}"
        )

    else:

        run_name = (
            "general_pretraining_fresh_"
            f"{timestamp}"
        )

    log_path = (
        PROJECT_ROOT
        / "logs"
        / (
            f"{run_name}.jsonl"
        )
    )

    summary_path = (
        PROJECT_ROOT
        / "logs"
        / (
            f"{run_name}_summary.json"
        )
    )

    checkpoint_directory = (
        checkpoint_root
        /
        run_name
    )

    # ========================================================
    # Create model
    # ========================================================

    print()

    print(
        "Creating MathGPT..."
    )

    model = GPT.from_config(
        model_config,
        seed=seed,
    )

    count = parameter_count(
        model
    )

    if count != 8_110_848:

        raise RuntimeError(
            "Unexpected model parameter count: "
            f"{count:,}"
        )

    # ========================================================
    # Optimizer
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
    # Training loader
    # ========================================================

    train_loader = TokenBatchLoader(
        binary_path=train_path,
        context_length=context_length,
        batch_size=batch_size,
        seed=seed,
    )

    # ========================================================
    # Logger
    # ========================================================

    logger = TrainingLogger(
        log_path
    )

    # ========================================================
    # Fresh start OR resume
    # ========================================================

    resume_path = None

    if args.resume is not None:

        resume_path = resolve_project_path(
            args.resume
        )

        if not resume_path.exists():

            raise FileNotFoundError(
                f"Resume checkpoint missing: "
                f"{resume_path}"
            )

        print()
        print(
            "=" * 76
        )

        print(
            "RESTORING CHECKPOINT"
        )

        print(
            "=" * 76
        )

        print()

        print(
            resume_path
        )

        (
            checkpoint_metadata,
            starting_step,
            tokens_seen,
        ) = restore_training_checkpoint(
            model=model,
            optimizer=optimizer,
            path=resume_path,
            train_loader=train_loader,
            strict=True,
        )

        logger.log_event(
            "training_resumed",
            checkpoint=str(
                resume_path
            ),
            restored_step=int(
                starting_step
            ),
            restored_tokens_seen=int(
                tokens_seen
            ),
        )

    else:

        checkpoint_metadata = None

        starting_step = 0

        tokens_seen = 0

        logger.log_event(
            "training_started_fresh",
            step=0,
            tokens_seen=0,
        )

    # ========================================================
    # Resume safety
    # ========================================================

    if (
        starting_step
        >=
        target_step
    ):

        raise ValueError(
            f"Checkpoint is already at step "
            f"{starting_step:,}, but target-step "
            f"is {target_step:,}."
        )

    tokens_per_step = (
        batch_size
        *
        context_length
    )

    expected_final_tokens = (
        tokens_seen
        +
        (
            target_step
            -
            starting_step
        )
        *
        tokens_per_step
    )

    # ========================================================
    # Display final launch configuration
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PRETRAINING CONFIGURATION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Run name: "
        f"{run_name}"
    )

    print(
        f"Mode: "
        f"{'RESUME' if resume_path else 'FRESH'}"
    )

    print()

    print(
        f"Model parameters: "
        f"{count:,}"
    )

    print(
        f"Tokenizer: "
        f"{expected_tokenizer}"
    )

    print()

    print(
        f"Train tokens: "
        f"{train_metadata['tokens']:,}"
    )

    print(
        f"Validation tokens: "
        f"{validation_metadata['tokens']:,}"
    )

    print()

    print(
        f"Starting step: "
        f"{starting_step:,}"
    )

    print(
        f"Target step: "
        f"{target_step:,}"
    )

    print(
        f"Configured schedule end: "
        f"{configured_max_steps:,}"
    )

    print()

    print(
        f"Starting tokens seen: "
        f"{tokens_seen:,}"
    )

    print(
        f"Tokens per step: "
        f"{tokens_per_step:,}"
    )

    print(
        f"Expected final tokens seen: "
        f"{expected_final_tokens:,}"
    )

    print()

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Context length: "
        f"{context_length}"
    )

    print()

    print(
        f"Max LR: "
        f"{max_learning_rate}"
    )

    print(
        f"Min LR: "
        f"{min_learning_rate}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps}"
    )

    print()

    print(
        f"Log every: "
        f"{log_every} steps"
    )

    print(
        f"Validate every: "
        f"{eval_every} steps"
    )

    print(
        f"Checkpoint every: "
        f"{checkpoint_every:,} steps"
    )

    print()

    print(
        f"Log path:"
    )

    print(
        log_path
    )

    print()

    print(
        f"Checkpoint directory:"
    )

    print(
        checkpoint_directory
    )

    # ========================================================
    # Initial / resumed validation
    # ========================================================

    initial_validation = (
        run_validation(
            model=model,
            validation_path=(
                validation_path
            ),
            context_length=(
                context_length
            ),
            batch_size=(
                batch_size
            ),
            eval_batches=(
                eval_batches
            ),
            step=(
                starting_step
            ),
            tokens_seen=(
                tokens_seen
            ),
            logger=logger,
        )
    )

    validation_history = [
        {
            "step":
                int(
                    starting_step
                ),

            "tokens_seen":
                int(
                    tokens_seen
                ),

            "loss":
                float(
                    initial_validation[
                        "loss"
                    ]
                ),

            "perplexity":
                float(
                    initial_validation[
                        "perplexity"
                    ]
                ),
        }
    ]

    # ========================================================
    # Main training
    #
    # Training is broken into chunks ending on
    # absolute validation boundaries:
    #
    #   1200
    #   1400
    #   1600
    #   ...
    #
    # when resuming from step 1000.
    #
    # Validation therefore never happens between
    # forward and backward passes.
    # ========================================================

    all_history = []

    all_checkpoints = []

    try:

        while (
            optimizer.step_count
            <
            target_step
        ):

            current_step = (
                optimizer.step_count
            )

            next_validation_boundary = (
                (
                    current_step
                    //
                    eval_every
                )
                +
                1
            ) * eval_every

            chunk_end_step = min(
                next_validation_boundary,
                target_step,
            )

            chunk_steps = (
                chunk_end_step
                -
                current_step
            )

            print()
            print(
                "=" * 76
            )

            print(
                f"TRAINING "
                f"{current_step + 1:,}"
                f"–"
                f"{chunk_end_step:,}"
            )

            print(
                "=" * 76
            )

            (
                history,
                tokens_seen,
                saved_checkpoints,
            ) = run_training_steps(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                num_steps=chunk_steps,
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
                    configured_max_steps
                ),
                max_grad_norm=(
                    max_grad_norm
                ),
                starting_tokens_seen=(
                    tokens_seen
                ),
                progress_every=(
                    log_every
                ),
                logger=logger,
                log_every=(
                    log_every
                ),
                checkpoint_every=(
                    checkpoint_every
                ),
                checkpoint_directory=(
                    checkpoint_directory
                ),
                checkpoint_prefix=(
                    "mathgpt"
                ),
            )

            all_history.extend(
                history
            )

            all_checkpoints.extend(
                saved_checkpoints
            )

            # =================================================
            # Validation
            # =================================================

            validation_result = (
                run_validation(
                    model=model,
                    validation_path=(
                        validation_path
                    ),
                    context_length=(
                        context_length
                    ),
                    batch_size=(
                        batch_size
                    ),
                    eval_batches=(
                        eval_batches
                    ),
                    step=(
                        optimizer.step_count
                    ),
                    tokens_seen=(
                        tokens_seen
                    ),
                    logger=logger,
                )
            )

            validation_history.append(
                {
                    "step":
                        int(
                            optimizer.step_count
                        ),

                    "tokens_seen":
                        int(
                            tokens_seen
                        ),

                    "loss":
                        float(
                            validation_result[
                                "loss"
                            ]
                        ),

                    "perplexity":
                        float(
                            validation_result[
                                "perplexity"
                            ]
                        ),
                }
            )

    except KeyboardInterrupt:

        print()
        print()
        print(
            "=" * 76
        )

        print(
            "TRAINING INTERRUPTED"
        )

        print(
            "=" * 76
        )

        print()

        print(
            "No emergency checkpoint was written "
            "because interruption may have happened "
            "inside an incomplete training step."
        )

        print(
            "Resume from the most recent completed "
            "periodic checkpoint."
        )

        logger.log_event(
            "training_interrupted",
            last_reported_step=int(
                optimizer.step_count
            ),
            tokens_seen=int(
                tokens_seen
            ),
        )

        raise

    # ========================================================
    # Final checkpoint
    #
    # Multiples of checkpoint_every are already
    # saved by run_training_steps().
    #
    # Step 8197 is not a multiple of 1000, so
    # save a final checkpoint explicitly.
    # ========================================================

    final_step = (
        optimizer.step_count
    )

    if (
        final_step
        %
        checkpoint_every
        !=
        0
    ):

        print()
        print(
            "Saving final checkpoint..."
        )

        (
            final_checkpoint,
            _,
        ) = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            directory=(
                checkpoint_directory
            ),
            step=(
                final_step
            ),
            tokens_seen=(
                tokens_seen
            ),
            train_loader=(
                train_loader
            ),
            logger=logger,
            prefix="mathgpt",
        )

        all_checkpoints.append(
            final_checkpoint
        )

    # ========================================================
    # Final integrity checks
    # ========================================================

    step_count_passed = (
        optimizer.step_count
        ==
        target_step
    )

    token_accounting_passed = (
        tokens_seen
        ==
        expected_final_tokens
    )

    losses_finite = all(
        np.isfinite(
            metrics[
                "loss"
            ]
        )

        for metrics
        in all_history
    )

    gradients_finite = all(
        np.isfinite(
            metrics[
                "gradient_norm"
            ]
        )

        for metrics
        in all_history
    )

    parameters_are_finite = (
        parameters_finite(
            model
        )
    )

    optimizer_is_finite = (
        optimizer_state_finite(
            optimizer
        )
    )

    final_validation = (
        validation_history[
            -1
        ]
    )

    # ========================================================
    # Session statistics
    # ========================================================

    if all_history:

        recent_history = (
            all_history[
                -50:
            ]
        )

        recent_train_loss = float(
            np.mean(
                [
                    metrics[
                        "loss"
                    ]

                    for metrics
                    in recent_history
                ]
            )
        )

        average_throughput = float(
            np.mean(
                [
                    metrics[
                        "tokens_per_second"
                    ]

                    for metrics
                    in all_history
                ]
            )
        )

    else:

        recent_train_loss = None

        average_throughput = None

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "run_name":
            run_name,

        "mode":
            (
                "resume"
                if resume_path
                else
                "fresh"
            ),

        "resume_checkpoint":
            (
                None
                if resume_path is None
                else str(
                    resume_path
                )
            ),

        "starting_step":
            int(
                starting_step
            ),

        "final_step":
            int(
                optimizer.step_count
            ),

        "target_step":
            int(
                target_step
            ),

        "configured_max_steps":
            int(
                configured_max_steps
            ),

        "starting_tokens_seen":
            int(
                (
                    EXPECTED_STARTING_TOKENS
                    if False
                    else (
                        validation_history[
                            0
                        ][
                            "tokens_seen"
                        ]
                    )
                )
            ),

        "final_tokens_seen":
            int(
                tokens_seen
            ),

        "tokens_per_step":
            int(
                tokens_per_step
            ),

        "session_steps":
            int(
                target_step
                -
                starting_step
            ),

        "session_token_exposures":
            int(
                (
                    target_step
                    -
                    starting_step
                )
                *
                tokens_per_step
            ),

        "initial_validation_loss":
            float(
                validation_history[
                    0
                ][
                    "loss"
                ]
            ),

        "final_validation_loss":
            float(
                final_validation[
                    "loss"
                ]
            ),

        "initial_validation_perplexity":
            float(
                validation_history[
                    0
                ][
                    "perplexity"
                ]
            ),

        "final_validation_perplexity":
            float(
                final_validation[
                    "perplexity"
                ]
            ),

        "session_recent_50_train_loss":
            recent_train_loss,

        "average_tokens_per_second":
            average_throughput,

        "final_learning_rate":
            float(
                optimizer.learning_rate
            ),

        "validation_history":
            validation_history,

        "checkpoints_created":
            [
                str(
                    path
                )

                for path
                in all_checkpoints
            ],

        "checks": {
            "step_count_passed":
                step_count_passed,

            "token_accounting_passed":
                token_accounting_passed,

            "training_losses_finite":
                losses_finite,

            "gradient_norms_finite":
                gradients_finite,

            "model_parameters_finite":
                parameters_are_finite,

            "optimizer_state_finite":
                optimizer_is_finite,
        },
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    logger.log_event(
        "training_completed",
        step=int(
            optimizer.step_count
        ),
        tokens_seen=int(
            tokens_seen
        ),
        final_validation_loss=float(
            final_validation[
                "loss"
            ]
        ),
        summary_path=str(
            summary_path
        ),
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL PRETRAINING RUN COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Starting step: "
        f"{starting_step:,}"
    )

    print(
        f"Final step: "
        f"{optimizer.step_count:,}"
    )

    print(
        f"Target step: "
        f"{target_step:,}"
    )

    print()

    print(
        f"Final tokens seen: "
        f"{tokens_seen:,}"
    )

    print()

    if recent_train_loss is not None:

        print(
            f"Recent 50-step train loss: "
            f"{recent_train_loss:.8f}"
        )

    print(
        f"Initial validation loss: "
        f"{validation_history[0]['loss']:.8f}"
    )

    print(
        f"Final validation loss:   "
        f"{final_validation['loss']:.8f}"
    )

    print()

    print(
        f"Initial validation PPL: "
        f"{validation_history[0]['perplexity']:,.2f}"
    )

    print(
        f"Final validation PPL:   "
        f"{final_validation['perplexity']:,.2f}"
    )

    print()

    print(
        f"Final LR: "
        f"{optimizer.learning_rate:.10f}"
    )

    if (
        average_throughput
        is not None
    ):

        print(
            f"Average throughput: "
            f"{average_throughput:,.1f} "
            f"tokens/s"
        )

    print()

    print(
        f"Checkpoints created "
        f"during this invocation: "
        f"{len(all_checkpoints)}"
    )

    for path in all_checkpoints:

        print(
            f"  {path.name}"
        )

    print()

    print(
        "Validation History"
    )

    print(
        "------------------"
    )

    for result in validation_history:

        print(
            f"Step "
            f"{result['step']:>5,} | "
            f"Tokens "
            f"{result['tokens_seen']:>9,} | "
            f"Loss "
            f"{result['loss']:.8f} | "
            f"PPL "
            f"{result['perplexity']:,.2f}"
        )

    print()

    print(
        "Integrity Checks"
    )

    print(
        "----------------"
    )

    print(
        f"Target step reached: "
        f"{step_count_passed}"
    )

    print(
        f"Token accounting correct: "
        f"{token_accounting_passed}"
    )

    print(
        f"All training losses finite: "
        f"{losses_finite}"
    )

    print(
        f"All gradient norms finite: "
        f"{gradients_finite}"
    )

    print(
        f"Model parameters finite: "
        f"{parameters_are_finite}"
    )

    print(
        f"AdamW state finite: "
        f"{optimizer_is_finite}"
    )

    print()

    print(
        f"Training log:"
    )

    print(
        log_path
    )

    print()

    print(
        f"Summary:"
    )

    print(
        summary_path
    )

    print()

    print(
        f"Checkpoint directory:"
    )

    print(
        checkpoint_directory
    )

    # ========================================================
    # Final acceptance
    # ========================================================

    all_passed = (
        step_count_passed
        and
        token_accounting_passed
        and
        losses_finite
        and
        gradients_finite
        and
        parameters_are_finite
        and
        optimizer_is_finite
    )

    print()
    print(
        "=" * 76
    )

    if all_passed:

        print(
            "SUCCESS: GENERAL PRETRAINING "
            "LAUNCHER COMPLETED SAFELY."
        )

    else:

        raise AssertionError(
            "General pretraining launcher "
            "integrity checks failed."
        )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()