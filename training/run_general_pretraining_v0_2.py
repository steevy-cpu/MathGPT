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
# Fixed General-v0.2 experiment identity
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.2.json"
)


EXPECTED_CONFIG_VERSION = (
    "0.2"
)


EXPECTED_EXPERIMENT = (
    "general_pretraining_v0.2_50m"
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


EXPECTED_TRAIN_TOKENS = (
    51_776_327
)


EXPECTED_VALIDATION_TOKENS = (
    5_778_078
)


EXPECTED_BATCH_SIZE = (
    4
)


EXPECTED_CONTEXT_LENGTH = (
    256
)


EXPECTED_TOKENS_PER_STEP = (
    1_024
)


EXPECTED_MAX_STEPS = (
    50_563
)


EXPECTED_FINAL_EXPOSURES = (
    51_776_512
)


EXPECTED_WARMUP_STEPS = (
    2_467
)


EXPECTED_TRAIN_SPLIT = (
    "general_v0_2_train"
)


EXPECTED_VALIDATION_SPLIT = (
    "general_v0_2_validation"
)


# ============================================================
# Helpers
# ============================================================

def load_json(
    path,
):

    if not path.exists():

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
            "Train MathGPT-General-v0.2 "
            "on the fixed ~51.8M-token "
            "General-v0.2 corpus."
        )
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a complete "
            "General-v0.2 checkpoint."
        ),
    )

    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help=(
            "Absolute step at which this "
            "invocation should stop. "
            "The LR schedule always remains "
            "the full 50,563-step schedule."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Explicit run name."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate configuration, dataset, "
            "model, and loader without training."
        ),
    )

    return parser.parse_args()


# ============================================================
# Dataset metadata validation
# ============================================================

def validate_dataset_metadata(
    *,
    metadata,
    binary_path,
    expected_split,
    expected_tokens,
):

    checks = {
        "stage":
            (
                metadata.get(
                    "stage"
                )
                ==
                EXPECTED_DATA_STAGE
            ),

        "dataset_version":
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

        "vocab_size":
            (
                metadata.get(
                    "vocab_size"
                )
                ==
                16_000
            ),

        "dtype":
            (
                metadata.get(
                    "dtype"
                )
                ==
                "uint16"
            ),

        "eos_token_id":
            (
                metadata.get(
                    "eos_token_id"
                )
                ==
                2
            ),

        "token_count":
            (
                metadata.get(
                    "tokens"
                )
                ==
                expected_tokens
            ),

        "binary_size":
            (
                binary_path.stat().st_size
                ==
                expected_tokens
                *
                2
            ),
    }

    # Fast independent binary token count.
    binary_tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    checks[
        "binary_token_count"
    ] = (
        len(
            binary_tokens
        )
        ==
        expected_tokens
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
            "General-v0.2 dataset "
            "preflight failed: "
            f"{failed}"
        )

    return checks


# ============================================================
# Full v0.2 configuration preflight
# ============================================================

def preflight_config(
    training_config,
):

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.2 CONFIG PREFLIGHT"
    )

    print(
        "=" * 76
    )

    data = (
        training_config[
            "data"
        ]
    )

    training = (
        training_config[
            "training"
        ]
    )

    scheduler = (
        training_config[
            "scheduler"
        ]
    )

    optimizer = (
        training_config[
            "optimizer"
        ]
    )

    gradient_clipping = (
        training_config[
            "gradient_clipping"
        ]
    )

    # ========================================================
    # Exact experiment identity
    # ========================================================

    checks = {
        "Config version = 0.2":
            (
                str(
                    training_config.get(
                        "version"
                    )
                )
                ==
                EXPECTED_CONFIG_VERSION
            ),

        "Experiment identity":
            (
                training_config.get(
                    "experiment"
                )
                ==
                EXPECTED_EXPERIMENT
            ),

        "Tokenizer unchanged":
            (
                data[
                    "tokenizer"
                ]
                ==
                EXPECTED_TOKENIZER
            ),

        "Training tokens":
            (
                int(
                    data[
                        "training_tokens"
                    ]
                )
                ==
                EXPECTED_TRAIN_TOKENS
            ),

        "Validation tokens":
            (
                int(
                    data[
                        "validation_tokens"
                    ]
                )
                ==
                EXPECTED_VALIDATION_TOKENS
            ),

        "Batch size":
            (
                int(
                    training[
                        "batch_size"
                    ]
                )
                ==
                EXPECTED_BATCH_SIZE
            ),

        "Context length":
            (
                int(
                    training[
                        "context_length"
                    ]
                )
                ==
                EXPECTED_CONTEXT_LENGTH
            ),

        "Tokens per step":
            (
                int(
                    training[
                        "tokens_per_step"
                    ]
                )
                ==
                EXPECTED_TOKENS_PER_STEP
            ),

        "Max steps":
            (
                int(
                    training[
                        "max_steps"
                    ]
                )
                ==
                EXPECTED_MAX_STEPS
            ),

        "Target exposures":
            (
                int(
                    training[
                        "target_token_exposures"
                    ]
                )
                ==
                EXPECTED_FINAL_EXPOSURES
            ),

        "Seed = 42":
            (
                int(
                    training[
                        "seed"
                    ]
                )
                ==
                42
            ),

        "Warmup steps":
            (
                int(
                    scheduler[
                        "warmup_steps"
                    ]
                )
                ==
                EXPECTED_WARMUP_STEPS
            ),

        "Max LR = 3e-4":
            (
                float(
                    optimizer[
                        "learning_rate"
                    ]
                )
                ==
                0.0003
            ),

        "Min LR = 3e-5":
            (
                float(
                    scheduler[
                        "min_learning_rate"
                    ]
                )
                ==
                0.00003
            ),

        "AdamW beta1":
            (
                float(
                    optimizer[
                        "beta1"
                    ]
                )
                ==
                0.9
            ),

        "AdamW beta2":
            (
                float(
                    optimizer[
                        "beta2"
                    ]
                )
                ==
                0.999
            ),

        "AdamW epsilon":
            (
                float(
                    optimizer[
                        "epsilon"
                    ]
                )
                ==
                1e-8
            ),

        "Weight decay":
            (
                float(
                    optimizer[
                        "weight_decay"
                    ]
                )
                ==
                0.01
            ),

        "Gradient clipping enabled":
            bool(
                gradient_clipping[
                    "enabled"
                ]
            ),

        "Gradient clip = 1.0":
            (
                float(
                    gradient_clipping[
                        "max_norm"
                    ]
                )
                ==
                1.0
            ),
    }

    # ========================================================
    # Mathematical schedule checks
    # ========================================================

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
        int(
            training[
                "max_steps"
            ]
        )
        *
        calculated_tokens_per_step
    )

    expected_steps = (
        EXPECTED_TRAIN_TOKENS
        +
        calculated_tokens_per_step
        -
        1
    ) // calculated_tokens_per_step

    exposure_overage = (
        calculated_exposures
        -
        EXPECTED_TRAIN_TOKENS
    )

    checks[
        "Batch × context = 1,024"
    ] = (
        calculated_tokens_per_step
        ==
        EXPECTED_TOKENS_PER_STEP
    )

    checks[
        "Corpus-equivalent ceiling = 50,563"
    ] = (
        expected_steps
        ==
        EXPECTED_MAX_STEPS
    )

    checks[
        "Calculated exposures = 51,776,512"
    ] = (
        calculated_exposures
        ==
        EXPECTED_FINAL_EXPOSURES
    )

    checks[
        "Exposure overage = 185"
    ] = (
        exposure_overage
        ==
        185
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
            "General-v0.2 config "
            f"preflight failed: {failed}"
        )

    print()
    print(
        "SUCCESS: General-v0.2 "
        "configuration is locked and valid."
    )


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
        "MATHGPT — GENERAL-v0.2 PRETRAINING"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "This launcher is dedicated to "
        "MathGPT-General-v0.2."
    )

    print(
        "General-v0.1 artifacts are not used "
        "or modified."
    )

    # ========================================================
    # Load configuration
    # ========================================================

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    preflight_config(
        training_config
    )

    # ========================================================
    # Training config values
    # ========================================================

    batch_size = int(
        training_config[
            "training"
        ][
            "batch_size"
        ]
    )

    context_length = int(
        training_config[
            "training"
        ][
            "context_length"
        ]
    )

    seed = int(
        training_config[
            "training"
        ][
            "seed"
        ]
    )

    configured_max_steps = int(
        training_config[
            "training"
        ][
            "max_steps"
        ]
    )

    tokens_per_step = (
        batch_size
        *
        context_length
    )

    # ========================================================
    # Optimizer
    # ========================================================

    max_learning_rate = float(
        training_config[
            "optimizer"
        ][
            "learning_rate"
        ]
    )

    beta1 = float(
        training_config[
            "optimizer"
        ][
            "beta1"
        ]
    )

    beta2 = float(
        training_config[
            "optimizer"
        ][
            "beta2"
        ]
    )

    epsilon = float(
        training_config[
            "optimizer"
        ][
            "epsilon"
        ]
    )

    weight_decay = float(
        training_config[
            "optimizer"
        ][
            "weight_decay"
        ]
    )

    # ========================================================
    # Scheduler
    # ========================================================

    warmup_steps = int(
        training_config[
            "scheduler"
        ][
            "warmup_steps"
        ]
    )

    min_learning_rate = float(
        training_config[
            "scheduler"
        ][
            "min_learning_rate"
        ]
    )

    # ========================================================
    # Gradient clipping
    # ========================================================

    max_grad_norm = float(
        training_config[
            "gradient_clipping"
        ][
            "max_norm"
        ]
    )

    # ========================================================
    # Logging / validation / checkpoints
    # ========================================================

    log_every = int(
        training_config[
            "logging"
        ][
            "log_every_steps"
        ]
    )

    eval_every = int(
        training_config[
            "validation"
        ][
            "eval_every_steps"
        ]
    )

    eval_batches = int(
        training_config[
            "validation"
        ][
            "eval_batches"
        ]
    )

    checkpoint_every = int(
        training_config[
            "checkpointing"
        ][
            "save_every_steps"
        ]
    )

    checkpoint_root = resolve_project_path(
        training_config[
            "checkpointing"
        ][
            "directory"
        ]
    )

    # ========================================================
    # Dataset paths
    # ========================================================

    train_path = resolve_project_path(
        training_config[
            "data"
        ][
            "train_binary"
        ]
    )

    validation_path = resolve_project_path(
        training_config[
            "data"
        ][
            "validation_binary"
        ]
    )

    for path in (
        train_path,
        validation_path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                path
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

    # ========================================================
    # Dataset identity checks
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.2 DATASET PREFLIGHT"
    )

    print(
        "=" * 76
    )

    train_checks = (
        validate_dataset_metadata(
            metadata=train_metadata,
            binary_path=train_path,
            expected_split=EXPECTED_TRAIN_SPLIT,
            expected_tokens=EXPECTED_TRAIN_TOKENS,
        )
    )

    validation_checks = (
        validate_dataset_metadata(
            metadata=validation_metadata,
            binary_path=validation_path,
            expected_split=EXPECTED_VALIDATION_SPLIT,
            expected_tokens=EXPECTED_VALIDATION_TOKENS,
        )
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
        "Training binary checks:"
    )

    for (
        name,
        passed,
    ) in train_checks.items():

        print(
            f"  {name}: "
            f"{passed}"
        )

    print()

    print(
        "Validation binary checks:"
    )

    for (
        name,
        passed,
    ) in validation_checks.items():

        print(
            f"  {name}: "
            f"{passed}"
        )

    print()

    print(
        "SUCCESS: General-v0.2 dataset "
        "identity confirmed."
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
            f"{configured_max_steps:,}."
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
            "general_pretraining_v0_2_resume_"
            f"{timestamp}"
        )

    else:

        run_name = (
            "general_pretraining_v0_2_fresh_"
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

    # ========================================================
    # Fresh-run overwrite protection
    # ========================================================

    if args.resume is None:

        existing_artifacts = []

        if log_path.exists():

            existing_artifacts.append(
                log_path
            )

        if summary_path.exists():

            existing_artifacts.append(
                summary_path
            )

        if (
            checkpoint_directory.exists()
            and
            any(
                checkpoint_directory.iterdir()
            )
        ):

            existing_artifacts.append(
                checkpoint_directory
            )

        if existing_artifacts:

            raise FileExistsError(
                "Fresh run would reuse existing "
                "artifacts. Choose a new --run-name. "
                f"Existing: {existing_artifacts}"
            )

    # ========================================================
    # Create fresh model
    # ========================================================

    print()
    print(
        "Creating fresh MathGPT model..."
    )

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
            "Unexpected parameter count: "
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

    if (
        train_loader.total_tokens
        !=
        EXPECTED_TRAIN_TOKENS
    ):

        raise RuntimeError(
            "Training loader token count "
            "does not match General-v0.2."
        )

    print(
        f"Model parameters: "
        f"{count:,}"
    )

    print(
        f"Loader tokens: "
        f"{train_loader.total_tokens:,}"
    )

    # ========================================================
    # Preflight-only mode
    # ========================================================

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

        print()

        print(
            "Model, configuration, dataset, "
            "metadata, and loader all passed."
        )

        print()

        print(
            "No training steps were executed."
        )

        return

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
            "RESTORING GENERAL-v0.2 CHECKPOINT"
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
                "Resume checkpoint token "
                "accounting does not match "
                "General-v0.2 schedule."
            )

        logger.log_event(
            "training_resumed",
            experiment=EXPECTED_EXPERIMENT,
            dataset_version=EXPECTED_DATASET_VERSION,
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
            experiment=EXPECTED_EXPERIMENT,
            dataset_version=EXPECTED_DATASET_VERSION,
            step=0,
            tokens_seen=0,
        )

    # ========================================================
    # Resume / target safety
    # ========================================================

    if (
        starting_step
        >=
        target_step
    ):

        raise ValueError(
            f"Checkpoint is already at "
            f"step {starting_step:,}, "
            f"but target-step is "
            f"{target_step:,}."
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
    # Launch configuration
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.2 PRETRAINING CONFIGURATION"
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
        f"{EXPECTED_TOKENIZER}"
    )

    print(
        f"Dataset: "
        f"{EXPECTED_DATASET_VERSION}"
    )

    print()

    print(
        f"Training tokens: "
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
        f"Full LR schedule ends: "
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
        f"{warmup_steps:,}"
    )

    print()

    print(
        f"Log every: "
        f"{log_every:,} steps"
    )

    print(
        f"Validate every: "
        f"{eval_every:,} steps"
    )

    print(
        f"Checkpoint every: "
        f"{checkpoint_every:,} steps"
    )

    print()

    print(
        "Log:"
    )

    print(
        log_path
    )

    print()

    print(
        "Checkpoint directory:"
    )

    print(
        checkpoint_directory
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
    # We train in chunks ending at validation boundaries.
    #
    # Example:
    #   1–1000
    #   1001–2000
    #   ...
    #
    # A smoke target such as step 5 still uses the FULL
    # 50,563-step LR schedule.
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
            # Validation at every chunk boundary
            # =================================================

            validation_result = run_validation(
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
            "Resume from the most recent "
            "completed periodic checkpoint."
        )

        logger.log_event(
            "training_interrupted",
            experiment=EXPECTED_EXPERIMENT,
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
            directory=checkpoint_directory,
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
    # Integrity checks
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

    validation_finite = all(
        np.isfinite(
            result[
                "loss"
            ]
        )

        for result
        in validation_history
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
        "model":
            "MathGPT-General-v0.2",

        "experiment":
            EXPECTED_EXPERIMENT,

        "dataset_version":
            EXPECTED_DATASET_VERSION,

        "tokenizer":
            EXPECTED_TOKENIZER,

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

        "training_config":
            str(
                TRAINING_CONFIG_PATH
            ),

        "train_binary":
            str(
                train_path
            ),

        "validation_binary":
            str(
                validation_path
            ),

        "parameter_count":
            int(
                count
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
                validation_history[
                    0
                ][
                    "tokens_seen"
                ]
            ),

        "final_tokens_seen":
            int(
                tokens_seen
            ),

        "tokens_per_step":
            int(
                tokens_per_step
            ),

        "full_corpus_training_tokens":
            EXPECTED_TRAIN_TOKENS,

        "full_schedule_token_exposures":
            EXPECTED_FINAL_EXPOSURES,

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

        "recent_50_train_loss":
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

            "validation_losses_finite":
                validation_finite,

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
        experiment=EXPECTED_EXPERIMENT,
        dataset_version=EXPECTED_DATASET_VERSION,
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
    # Final acceptance
    # ========================================================

    integrity_checks = (
        summary[
            "checks"
        ]
    )

    all_integrity_passed = all(
        integrity_checks.values()
    )

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.2 PRETRAINING RUN COMPLETE"
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
        f"Final learning rate: "
        f"{optimizer.learning_rate:.10f}"
    )

    if average_throughput is not None:

        print(
            f"Average throughput: "
            f"{average_throughput:,.1f} "
            f"tokens/s"
        )

    print()

    print(
        "Integrity Checks"
    )

    print(
        "----------------"
    )

    for (
        name,
        passed,
    ) in integrity_checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    print()

    print(
        "Checkpoints created during "
        f"this invocation: "
        f"{len(all_checkpoints)}"
    )

    for path in all_checkpoints:

        print(
            f"  {path}"
        )

    print()

    print(
        "Summary:"
    )

    print(
        summary_path
    )

    if not all_integrity_passed:

        raise RuntimeError(
            "General-v0.2 training finished "
            "but failed integrity checks."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: GENERAL-v0.2 TRAINING "
        "INVOCATION PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()