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
# Experiment identity
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.2_matched8197.json"
)


EXPECTED_EXPERIMENT = (
    "general_pretraining_v0.2_matched_v0.1_budget"
)


EXPECTED_DATASET_VERSION = (
    "general_pretraining_v0.2"
)


EXPECTED_DATA_STAGE = (
    "12A.6"
)


EXPECTED_TOKENIZER = (
    "tokenizer_50m_v0.2"
)


EXPECTED_PARAMETER_COUNT = (
    8_110_848
)


# Full v0.2 corpus sizes.
EXPECTED_CORPUS_TRAIN_TOKENS = (
    51_776_327
)


EXPECTED_CORPUS_VALIDATION_TOKENS = (
    5_778_078
)


# Matched v0.1 training budget.
EXPECTED_MAX_STEPS = (
    8_197
)


EXPECTED_TOKENS_PER_STEP = (
    1_024
)


EXPECTED_BUDGET_EXPOSURES = (
    8_393_728
)


EXPECTED_WARMUP_STEPS = (
    400
)


EXPECTED_BATCH_SIZE = (
    4
)


EXPECTED_CONTEXT_LENGTH = (
    256
)


# ============================================================
# Helpers
# ============================================================

def load_json(
    path,
):

    if not Path(
        path
    ).exists():

        raise FileNotFoundError(
            path
        )

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


def metadata_path_for_binary(
    binary_path,
):

    return binary_path.with_name(
        binary_path.stem
        +
        "_metadata.json"
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


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "General-v0.2 matched-budget "
            "experiment using the exact "
            "General-v0.1 training schedule."
        )
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Configuration validation
# ============================================================

def validate_config(
    config,
):

    data = (
        config[
            "data"
        ]
    )

    training = (
        config[
            "training"
        ]
    )

    optimizer = (
        config[
            "optimizer"
        ]
    )

    scheduler = (
        config[
            "scheduler"
        ]
    )

    checks = {
        "experiment":
            (
                config.get(
                    "experiment"
                )
                ==
                EXPECTED_EXPERIMENT
            ),

        "tokenizer":
            (
                data[
                    "tokenizer"
                ]
                ==
                EXPECTED_TOKENIZER
            ),

        "corpus train tokens":
            (
                int(
                    data[
                        "corpus_training_tokens"
                    ]
                )
                ==
                EXPECTED_CORPUS_TRAIN_TOKENS
            ),

        "corpus validation tokens":
            (
                int(
                    data[
                        "corpus_validation_tokens"
                    ]
                )
                ==
                EXPECTED_CORPUS_VALIDATION_TOKENS
            ),

        "batch size":
            (
                int(
                    training[
                        "batch_size"
                    ]
                )
                ==
                EXPECTED_BATCH_SIZE
            ),

        "context length":
            (
                int(
                    training[
                        "context_length"
                    ]
                )
                ==
                EXPECTED_CONTEXT_LENGTH
            ),

        "tokens per step":
            (
                int(
                    training[
                        "tokens_per_step"
                    ]
                )
                ==
                EXPECTED_TOKENS_PER_STEP
            ),

        "seed":
            (
                int(
                    training[
                        "seed"
                    ]
                )
                ==
                42
            ),

        "max steps":
            (
                int(
                    training[
                        "max_steps"
                    ]
                )
                ==
                EXPECTED_MAX_STEPS
            ),

        "budget exposures":
            (
                int(
                    training[
                        "target_token_exposures"
                    ]
                )
                ==
                EXPECTED_BUDGET_EXPOSURES
            ),

        "warmup":
            (
                int(
                    scheduler[
                        "warmup_steps"
                    ]
                )
                ==
                EXPECTED_WARMUP_STEPS
            ),

        "max LR":
            (
                float(
                    optimizer[
                        "learning_rate"
                    ]
                )
                ==
                0.0003
            ),

        "min LR":
            (
                float(
                    scheduler[
                        "min_learning_rate"
                    ]
                )
                ==
                0.00003
            ),

        "beta1":
            (
                float(
                    optimizer[
                        "beta1"
                    ]
                )
                ==
                0.9
            ),

        "beta2":
            (
                float(
                    optimizer[
                        "beta2"
                    ]
                )
                ==
                0.999
            ),

        "epsilon":
            (
                float(
                    optimizer[
                        "epsilon"
                    ]
                )
                ==
                1e-8
            ),

        "weight decay":
            (
                float(
                    optimizer[
                        "weight_decay"
                    ]
                )
                ==
                0.01
            ),
    }

    calculated_tokens_per_step = (
        int(
            training[
                "batch_size"
            ]
        )
        *
        int(
            training[
                "context_length"
            ]
        )
    )

    calculated_exposures = (
        calculated_tokens_per_step
        *
        int(
            training[
                "max_steps"
            ]
        )
    )

    checks[
        "batch × context = 1,024"
    ] = (
        calculated_tokens_per_step
        ==
        EXPECTED_TOKENS_PER_STEP
    )

    checks[
        "8,197 × 1,024 = 8,393,728"
    ] = (
        calculated_exposures
        ==
        EXPECTED_BUDGET_EXPOSURES
    )

    print()
    print(
        "=" * 76
    )

    print(
        "MATCHED-BUDGET CONFIG PREFLIGHT"
    )

    print(
        "=" * 76
    )

    print()

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    failed = [
        name

        for (
            name,
            passed,
        ) in checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "Matched-budget config "
            f"failed: {failed}"
        )

    print()
    print(
        "SUCCESS: matched v0.1 "
        "training budget verified."
    )


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    *,
    binary_path,
    metadata,
    expected_split,
    expected_tokens,
):

    tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    checks = {
        "stage":
            (
                metadata.get(
                    "stage"
                )
                ==
                EXPECTED_DATA_STAGE
            ),

        "dataset version":
            (
                metadata.get(
                    "dataset_version"
                )
                ==
                EXPECTED_DATASET_VERSION
            ),

        "split":
            (
                metadata.get(
                    "split"
                )
                ==
                expected_split
            ),

        "tokenizer":
            (
                metadata.get(
                    "tokenizer"
                )
                ==
                EXPECTED_TOKENIZER
            ),

        "metadata tokens":
            (
                int(
                    metadata.get(
                        "tokens"
                    )
                )
                ==
                expected_tokens
            ),

        "binary tokens":
            (
                len(
                    tokens
                )
                ==
                expected_tokens
            ),

        "dtype":
            (
                metadata.get(
                    "dtype"
                )
                ==
                "uint16"
            ),

        "vocab":
            (
                metadata.get(
                    "vocab_size"
                )
                ==
                16_000
            ),
    }

    failed = [
        name

        for (
            name,
            passed,
        ) in checks.items()

        if not passed
    ]

    if failed:

        raise RuntimeError(
            "Dataset validation failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Validation
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
        "STAGE 13C — GENERAL-v0.2 "
        "MATCHED-v0.1-BUDGET TRAINING"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Control variable:"
    )

    print(
        "same 8,197-step / 8.393M-token "
        "training budget as General-v0.1."
    )

    print()

    print(
        "Changed variable:"
    )

    print(
        "training samples come from the "
        "larger General-v0.2 corpus."
    )

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    config = load_json(
        TRAINING_CONFIG_PATH
    )

    validate_config(
        config
    )

    # ========================================================
    # Paths
    # ========================================================

    train_path = resolve_project_path(
        config[
            "data"
        ][
            "train_binary"
        ]
    )

    validation_path = resolve_project_path(
        config[
            "data"
        ][
            "validation_binary"
        ]
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

    train_metadata = load_json(
        train_metadata_path
    )

    validation_metadata = load_json(
        validation_metadata_path
    )

    train_checks = validate_dataset(
        binary_path=train_path,
        metadata=train_metadata,
        expected_split=(
            "general_v0_2_train"
        ),
        expected_tokens=(
            EXPECTED_CORPUS_TRAIN_TOKENS
        ),
    )

    validation_checks = validate_dataset(
        binary_path=validation_path,
        metadata=validation_metadata,
        expected_split=(
            "general_v0_2_validation"
        ),
        expected_tokens=(
            EXPECTED_CORPUS_VALIDATION_TOKENS
        ),
    )

    print()
    print(
        "Dataset checks"
    )

    print(
        "--------------"
    )

    print(
        f"Training dataset: "
        f"{all(train_checks.values())}"
    )

    print(
        f"Validation dataset: "
        f"{all(validation_checks.values())}"
    )

    # ========================================================
    # Training values
    # ========================================================

    training = (
        config[
            "training"
        ]
    )

    optimizer_config = (
        config[
            "optimizer"
        ]
    )

    scheduler = (
        config[
            "scheduler"
        ]
    )

    batch_size = int(
        training[
            "batch_size"
        ]
    )

    context_length = int(
        training[
            "context_length"
        ]
    )

    seed = int(
        training[
            "seed"
        ]
    )

    configured_max_steps = int(
        training[
            "max_steps"
        ]
    )

    tokens_per_step = (
        batch_size
        *
        context_length
    )

    max_learning_rate = float(
        optimizer_config[
            "learning_rate"
        ]
    )

    min_learning_rate = float(
        scheduler[
            "min_learning_rate"
        ]
    )

    warmup_steps = int(
        scheduler[
            "warmup_steps"
        ]
    )

    max_grad_norm = float(
        config[
            "gradient_clipping"
        ][
            "max_norm"
        ]
    )

    log_every = int(
        config[
            "logging"
        ][
            "log_every_steps"
        ]
    )

    eval_every = int(
        config[
            "validation"
        ][
            "eval_every_steps"
        ]
    )

    eval_batches = int(
        config[
            "validation"
        ][
            "eval_batches"
        ]
    )

    checkpoint_every = int(
        config[
            "checkpointing"
        ][
            "save_every_steps"
        ]
    )

    checkpoint_root = resolve_project_path(
        config[
            "checkpointing"
        ][
            "directory"
        ]
    )

    # ========================================================
    # Target step
    # ========================================================

    if args.target_step is None:

        target_step = (
            configured_max_steps
        )

    else:

        target_step = int(
            args.target_step
        )

    if not (
        1
        <=
        target_step
        <=
        configured_max_steps
    ):

        raise ValueError(
            "target-step must be between "
            "1 and 8,197."
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
            "general_v0_2_matched8197_resume_"
            f"{timestamp}"
        )

    else:

        run_name = (
            "general_v0_2_matched8197_"
            f"{timestamp}"
        )

    log_path = (
        PROJECT_ROOT
        / "logs"
        / f"{run_name}.jsonl"
    )

    summary_path = (
        PROJECT_ROOT
        / "logs"
        / f"{run_name}_summary.json"
    )

    checkpoint_directory = (
        checkpoint_root
        /
        run_name
    )

    if (
        args.resume is None
        and
        (
            log_path.exists()
            or
            summary_path.exists()
            or
            (
                checkpoint_directory.exists()
                and
                any(
                    checkpoint_directory.iterdir()
                )
            )
        )
    ):

        raise FileExistsError(
            "Fresh run artifacts already exist. "
            "Choose a different --run-name."
        )

    # ========================================================
    # Model + optimizer + loader
    # ========================================================

    model = GPT.from_config(
        model_config,
        seed=seed,
    )

    count = parameter_count(
        model
    )

    if (
        count
        !=
        EXPECTED_PARAMETER_COUNT
    ):

        raise RuntimeError(
            "Unexpected model parameter count."
        )

    optimizer = AdamW(
        model=model,
        learning_rate=max_learning_rate,
        beta1=float(
            optimizer_config[
                "beta1"
            ]
        ),
        beta2=float(
            optimizer_config[
                "beta2"
            ]
        ),
        epsilon=float(
            optimizer_config[
                "epsilon"
            ]
        ),
        weight_decay=float(
            optimizer_config[
                "weight_decay"
            ]
        ),
    )

    train_loader = TokenBatchLoader(
        binary_path=train_path,
        context_length=context_length,
        batch_size=batch_size,
        seed=seed,
    )

    if (
        train_loader.total_tokens
        !=
        EXPECTED_CORPUS_TRAIN_TOKENS
    ):

        raise RuntimeError(
            "Loader corpus size mismatch."
        )

    print()

    print(
        f"Model parameters: "
        f"{count:,}"
    )

    print(
        f"Full v0.2 corpus: "
        f"{train_loader.total_tokens:,} tokens"
    )

    print(
        f"Matched training budget: "
        f"{EXPECTED_BUDGET_EXPOSURES:,} exposures"
    )

    print(
        f"Schedule: "
        f"{EXPECTED_MAX_STEPS:,} steps / "
        f"{EXPECTED_WARMUP_STEPS} warmup"
    )

    if args.preflight_only:

        print()
        print(
            "=" * 76
        )

        print(
            "PREFLIGHT-ONLY SUCCESS"
        )

        print(
            "=" * 76
        )

        print(
            "No training was executed."
        )

        return

    logger = TrainingLogger(
        log_path
    )

    # ========================================================
    # Fresh or resume
    # ========================================================

    if args.resume is not None:

        resume_path = resolve_project_path(
            args.resume
        )

        (
            _,
            starting_step,
            tokens_seen,
        ) = restore_training_checkpoint(
            model=model,
            optimizer=optimizer,
            path=resume_path,
            train_loader=train_loader,
            strict=True,
        )

        expected_resume_tokens = (
            starting_step
            *
            tokens_per_step
        )

        if (
            tokens_seen
            !=
            expected_resume_tokens
        ):

            raise RuntimeError(
                "Resume token accounting mismatch."
            )

        logger.log_event(
            "training_resumed",
            experiment=EXPECTED_EXPERIMENT,
            checkpoint=str(
                resume_path
            ),
            restored_step=(
                starting_step
            ),
            restored_tokens_seen=(
                tokens_seen
            ),
        )

    else:

        resume_path = None

        starting_step = 0

        tokens_seen = 0

        logger.log_event(
            "training_started_fresh",
            experiment=EXPECTED_EXPERIMENT,
            step=0,
            tokens_seen=0,
        )

    if (
        starting_step
        >=
        target_step
    ):

        raise ValueError(
            "Checkpoint is already at or "
            "beyond target step."
        )

    expected_final_tokens = (
        target_step
        *
        tokens_per_step
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "MATCHED-BUDGET TRAINING CONFIGURATION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Run: {run_name}"
    )

    print(
        f"Mode: "
        f"{'RESUME' if resume_path else 'FRESH'}"
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
        f"Expected final exposures: "
        f"{expected_final_tokens:,}"
    )

    print()

    print(
        f"Warmup: "
        f"{warmup_steps}"
    )

    print(
        f"Max LR: "
        f"{max_learning_rate}"
    )

    print(
        f"Min LR: "
        f"{min_learning_rate}"
    )

    # ========================================================
    # Initial validation
    # ========================================================

    initial_validation = run_validation(
        model=model,
        validation_path=validation_path,
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=eval_batches,
        step=starting_step,
        tokens_seen=tokens_seen,
        logger=logger,
    )

    validation_history = [
        {
            "step":
                starting_step,

            "tokens_seen":
                tokens_seen,

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

    all_history = []

    all_checkpoints = []

    # ========================================================
    # Training loop
    # ========================================================

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

        chunk_end = min(
            next_validation_boundary,
            target_step,
        )

        chunk_steps = (
            chunk_end
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
            f"{chunk_end:,}"
        )

        print(
            "=" * 76
        )

        (
            history,
            tokens_seen,
            checkpoints,
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
            checkpoint_prefix="mathgpt",
        )

        all_history.extend(
            history
        )

        all_checkpoints.extend(
            checkpoints
        )

        result = run_validation(
            model=model,
            validation_path=validation_path,
            context_length=context_length,
            batch_size=batch_size,
            eval_batches=eval_batches,
            step=optimizer.step_count,
            tokens_seen=tokens_seen,
            logger=logger,
        )

        validation_history.append(
            {
                "step":
                    optimizer.step_count,

                "tokens_seen":
                    tokens_seen,

                "loss":
                    float(
                        result[
                            "loss"
                        ]
                    ),

                "perplexity":
                    float(
                        result[
                            "perplexity"
                        ]
                    ),
            }
        )

    # ========================================================
    # Final checkpoint
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

        (
            final_checkpoint,
            _,
        ) = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            directory=(
                checkpoint_directory
            ),
            step=final_step,
            tokens_seen=tokens_seen,
            train_loader=train_loader,
            logger=logger,
            prefix="mathgpt",
        )

        all_checkpoints.append(
            final_checkpoint
        )

    # ========================================================
    # Integrity
    # ========================================================

    checks = {
        "step_count":
            (
                optimizer.step_count
                ==
                target_step
            ),

        "token_accounting":
            (
                tokens_seen
                ==
                expected_final_tokens
            ),

        "training_losses_finite":
            all(
                np.isfinite(
                    metric[
                        "loss"
                    ]
                )

                for metric
                in all_history
            ),

        "gradients_finite":
            all(
                np.isfinite(
                    metric[
                        "gradient_norm"
                    ]
                )

                for metric
                in all_history
            ),

        "parameters_finite":
            parameters_finite(
                model
            ),

        "optimizer_finite":
            optimizer_state_finite(
                optimizer
            ),
    }

    final_validation = (
        validation_history[
            -1
        ]
    )

    recent_loss = float(
        np.mean(
            [
                metric[
                    "loss"
                ]

                for metric
                in all_history[
                    -50:
                ]
            ]
        )
    )

    summary = {
        "experiment":
            EXPECTED_EXPERIMENT,

        "purpose":
            (
                "Data-scaling control: "
                "General-v0.2 corpus with "
                "General-v0.1 training budget."
            ),

        "run_name":
            run_name,

        "starting_step":
            starting_step,

        "final_step":
            optimizer.step_count,

        "final_tokens_seen":
            tokens_seen,

        "corpus_training_tokens":
            EXPECTED_CORPUS_TRAIN_TOKENS,

        "training_budget_exposures":
            EXPECTED_BUDGET_EXPOSURES,

        "warmup_steps":
            warmup_steps,

        "initial_validation_loss":
            validation_history[
                0
            ][
                "loss"
            ],

        "final_validation_loss":
            final_validation[
                "loss"
            ],

        "recent_50_train_loss":
            recent_loss,

        "final_learning_rate":
            optimizer.learning_rate,

        "validation_history":
            validation_history,

        "checkpoints_created":
            [
                str(
                    checkpoint
                )

                for checkpoint
                in all_checkpoints
            ],

        "checks":
            checks,
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

    print()
    print(
        "=" * 76
    )

    print(
        "MATCHED-BUDGET RUN COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Final step: "
        f"{optimizer.step_count:,}"
    )

    print(
        f"Final token exposures: "
        f"{tokens_seen:,}"
    )

    print(
        f"Recent-50 train loss: "
        f"{recent_loss:.8f}"
    )

    print(
        f"Initial validation loss: "
        f"{validation_history[0]['loss']:.8f}"
    )

    print(
        f"Final validation loss: "
        f"{final_validation['loss']:.8f}"
    )

    print()

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    if not all(
        checks.values()
    ):

        raise RuntimeError(
            "Matched-budget experiment "
            "failed integrity checks."
        )

    print()
    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: STAGE 13C "
        "MATCHED-BUDGET TRAINING PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()