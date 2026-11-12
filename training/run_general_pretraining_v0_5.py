import argparse
import hashlib
import json
import sys

from collections import deque
from datetime import datetime, timezone
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
# Frozen GENERAL-v0.5 identity
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.5.json"
)

TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.5.json"
)

EXPECTED_CONFIG_VERSION = "0.5"

EXPECTED_EXPERIMENT = (
    "general_pretraining_v0.5_16m_300m_capacity_scale"
)

EXPECTED_EXPERIMENT_TYPE = (
    "model_capacity_scaling"
)

EXPECTED_DATASET_VERSION = (
    "general_pretraining_v0.4"
)

EXPECTED_TOKENIZER = (
    "tokenizer_50m_v0.2"
)

EXPECTED_PARAMETER_COUNT = (
    16_889_856
)

EXPECTED_TRAIN_TOKENS = (
    251_490_827
)

EXPECTED_VALIDATION_TOKENS = (
    27_928_209
)

EXPECTED_BATCH_SIZE = 4
EXPECTED_CONTEXT_LENGTH = 256
EXPECTED_TOKENS_PER_STEP = 1_024

EXPECTED_MAX_STEPS = 292_969

EXPECTED_REQUESTED_EXPOSURES = (
    300_000_000
)

EXPECTED_FINAL_EXPOSURES = (
    300_000_256
)

EXPECTED_EXPOSURE_OVERAGE = 256

EXPECTED_WARMUP_STEPS = 14_294
EXPECTED_LOG_EVERY = 50
EXPECTED_EVAL_EVERY = 5_000
EXPECTED_CHECKPOINT_EVERY = 25_000

EXPECTED_MAX_LR = 0.0003
EXPECTED_MIN_LR = 0.00003

EXPECTED_BASELINE_PARAMETER_COUNT = (
    8_110_848
)

EXPECTED_BASELINE_EXPOSURE = (
    300_000_256
)


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_project_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def metadata_path_for_binary(
    binary_path,
):
    return binary_path.with_name(
        binary_path.stem
        +
        "_metadata.json"
    )


def file_sha256(path):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def parameter_count(model):
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


def parameters_finite(model):
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
    return (
        all(
            np.all(
                np.isfinite(
                    value
                )
            )
            for value
            in optimizer
            .first_moments
            .values()
        )
        and
        all(
            np.all(
                np.isfinite(
                    value
                )
            )
            for value
            in optimizer
            .second_moments
            .values()
        )
    )


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Train MathGPT-General-v0.5 "
            "from scratch: 16.89M-parameter "
            "capacity-scaling experiment at "
            "~300M token exposure using the "
            "frozen General-v0.4 corpus."
        )
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a complete "
            "General-v0.5 checkpoint."
        ),
    )

    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help=(
            "Absolute stop step for this "
            "invocation. The LR schedule always "
            "uses the full 292,969-step v0.5 "
            "schedule."
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
            "Validate frozen configuration, "
            "dataset, model, loader, and "
            "exposure schedule without training."
        ),
    )

    return parser.parse_args()


# ============================================================
# Config preflight
# ============================================================

def preflight_config(
    config,
):

    print()
    print("=" * 80)
    print(
        "GENERAL-v0.5 16.89M / 300M "
        "CONFIG PREFLIGHT"
    )
    print("=" * 80)

    data = config[
        "data"
    ]

    model_section = config[
        "model"
    ]

    training = config[
        "training"
    ]

    optimizer = config[
        "optimizer"
    ]

    scheduler = config[
        "scheduler"
    ]

    clipping = config[
        "gradient_clipping"
    ]

    logging = config[
        "logging"
    ]

    validation = config[
        "validation"
    ]

    checkpointing = config[
        "checkpointing"
    ]

    baseline = config[
        "comparison_baseline"
    ]

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

    calculated_required_steps = (
        EXPECTED_REQUESTED_EXPOSURES
        +
        calculated_tokens_per_step
        -
        1
    ) // calculated_tokens_per_step

    calculated_final_exposures = (
        int(
            training[
                "max_steps"
            ]
        )
        *
        calculated_tokens_per_step
    )

    calculated_overage = (
        calculated_final_exposures
        -
        EXPECTED_REQUESTED_EXPOSURES
    )

    checks = {
        "config_version":
            (
                str(
                    config.get(
                        "version"
                    )
                )
                ==
                EXPECTED_CONFIG_VERSION
            ),

        "experiment_identity":
            (
                config.get(
                    "experiment"
                )
                ==
                EXPECTED_EXPERIMENT
            ),

        "experiment_type_capacity_scaling":
            (
                config.get(
                    "experiment_type"
                )
                ==
                EXPECTED_EXPERIMENT_TYPE
            ),

        "model_config_path":
            (
                model_section.get(
                    "config"
                )
                ==
                "config/model_config_v0.5.json"
            ),

        "expected_parameter_count":
            (
                int(
                    model_section[
                        "expected_parameter_count"
                    ]
                )
                ==
                EXPECTED_PARAMETER_COUNT
            ),

        "baseline_parameter_count":
            (
                int(
                    baseline[
                        "parameter_count"
                    ]
                )
                ==
                EXPECTED_BASELINE_PARAMETER_COUNT
            ),

        "baseline_exposure":
            (
                int(
                    baseline[
                        "training_token_exposures"
                    ]
                )
                ==
                EXPECTED_BASELINE_EXPOSURE
            ),

        "dataset_version":
            (
                data.get(
                    "dataset_version"
                )
                ==
                EXPECTED_DATASET_VERSION
            ),

        "tokenizer_unchanged":
            (
                data.get(
                    "tokenizer"
                )
                ==
                EXPECTED_TOKENIZER
            ),

        "training_corpus_tokens":
            (
                int(
                    data[
                        "training_tokens"
                    ]
                )
                ==
                EXPECTED_TRAIN_TOKENS
            ),

        "validation_corpus_tokens":
            (
                int(
                    data[
                        "validation_tokens"
                    ]
                )
                ==
                EXPECTED_VALIDATION_TOKENS
            ),

        "batch_size_4":
            (
                int(
                    training[
                        "batch_size"
                    ]
                )
                ==
                EXPECTED_BATCH_SIZE
            ),

        "context_length_256":
            (
                int(
                    training[
                        "context_length"
                    ]
                )
                ==
                EXPECTED_CONTEXT_LENGTH
            ),

        "tokens_per_step_1024":
            (
                int(
                    training[
                        "tokens_per_step"
                    ]
                )
                ==
                EXPECTED_TOKENS_PER_STEP
            ),

        "batch_times_context_1024":
            (
                calculated_tokens_per_step
                ==
                EXPECTED_TOKENS_PER_STEP
            ),

        "seed_42":
            (
                int(
                    training[
                        "seed"
                    ]
                )
                ==
                42
            ),

        "max_steps_292969":
            (
                int(
                    training[
                        "max_steps"
                    ]
                )
                ==
                EXPECTED_MAX_STEPS
            ),

        "requested_exposure_300m":
            (
                int(
                    training[
                        "requested_token_exposures"
                    ]
                )
                ==
                EXPECTED_REQUESTED_EXPOSURES
            ),

        "target_exposure_300000256":
            (
                int(
                    training[
                        "target_token_exposures"
                    ]
                )
                ==
                EXPECTED_FINAL_EXPOSURES
            ),

        "ceiling_step_calculation":
            (
                calculated_required_steps
                ==
                EXPECTED_MAX_STEPS
            ),

        "calculated_final_exposure":
            (
                calculated_final_exposures
                ==
                EXPECTED_FINAL_EXPOSURES
            ),

        "exposure_overage_256":
            (
                calculated_overage
                ==
                EXPECTED_EXPOSURE_OVERAGE
            ),

        "optimizer_adamw":
            (
                optimizer.get(
                    "name"
                )
                ==
                "adamw"
            ),

        "max_lr_3e_4":
            (
                float(
                    optimizer[
                        "learning_rate"
                    ]
                )
                ==
                EXPECTED_MAX_LR
            ),

        "beta1_0_9":
            (
                float(
                    optimizer[
                        "beta1"
                    ]
                )
                ==
                0.9
            ),

        "beta2_0_999":
            (
                float(
                    optimizer[
                        "beta2"
                    ]
                )
                ==
                0.999
            ),

        "epsilon_1e_8":
            (
                float(
                    optimizer[
                        "epsilon"
                    ]
                )
                ==
                1e-8
            ),

        "weight_decay_0_01":
            (
                float(
                    optimizer[
                        "weight_decay"
                    ]
                )
                ==
                0.01
            ),

        "scheduler_warmup_cosine":
            (
                scheduler.get(
                    "name"
                )
                ==
                "warmup_cosine"
            ),

        "warmup_steps_14294":
            (
                int(
                    scheduler[
                        "warmup_steps"
                    ]
                )
                ==
                EXPECTED_WARMUP_STEPS
            ),

        "min_lr_3e_5":
            (
                float(
                    scheduler[
                        "min_learning_rate"
                    ]
                )
                ==
                EXPECTED_MIN_LR
            ),

        "gradient_clipping_enabled":
            (
                clipping.get(
                    "enabled"
                )
                is True
            ),

        "gradient_clip_1":
            (
                float(
                    clipping[
                        "max_norm"
                    ]
                )
                ==
                1.0
            ),

        "log_every_50":
            (
                int(
                    logging[
                        "log_every_steps"
                    ]
                )
                ==
                EXPECTED_LOG_EVERY
            ),

        "eval_every_5000":
            (
                int(
                    validation[
                        "eval_every_steps"
                    ]
                )
                ==
                EXPECTED_EVAL_EVERY
            ),

        "eval_batches_20":
            (
                int(
                    validation[
                        "eval_batches"
                    ]
                )
                ==
                20
            ),

        "validation_seed":
            (
                int(
                    validation.get(
                        "seed"
                    )
                )
                ==
                12345
            ),

        "checkpoint_every_25000":
            (
                int(
                    checkpointing[
                        "save_every_steps"
                    ]
                )
                ==
                EXPECTED_CHECKPOINT_EVERY
            ),
    }

    for name, passed in checks.items():

        print(
            f"{name:<42} {passed}"
        )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "GENERAL-v0.5 configuration "
            f"preflight failed: {failed}"
        )

    corpus_equivalents = (
        EXPECTED_FINAL_EXPOSURES
        /
        EXPECTED_TRAIN_TOKENS
    )

    warmup_fraction = (
        EXPECTED_WARMUP_STEPS
        /
        EXPECTED_MAX_STEPS
    )

    print()
    print(
        f"Corpus-equivalent exposure: "
        f"{corpus_equivalents:.6f}x"
    )

    print(
        f"Warmup fraction: "
        f"{warmup_fraction:.6%}"
    )

    print()
    print(
        "SUCCESS: GENERAL-v0.5 frozen "
        "configuration is valid."
    )

    return checks


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    *,
    binary_path,
    metadata,
    expected_tokens,
    expected_kind,
):

    actual_binary_tokens = (
        binary_path.stat().st_size
        //
        np.dtype(
            np.uint16
        ).itemsize
    )

    checks = {
        "binary_exists":
            binary_path.exists(),

        "binary_size_matches":
            (
                binary_path.stat().st_size
                ==
                expected_tokens
                *
                2
            ),

        "binary_token_count":
            (
                actual_binary_tokens
                ==
                expected_tokens
            ),

        "metadata_tokens":
            (
                int(
                    metadata.get(
                        "tokens"
                    )
                )
                ==
                expected_tokens
            ),

        "metadata_tokenizer":
            (
                metadata.get(
                    "tokenizer"
                )
                ==
                EXPECTED_TOKENIZER
            ),

        "metadata_vocab_size":
            (
                int(
                    metadata.get(
                        "vocab_size"
                    )
                )
                ==
                16_000
            ),

        "metadata_dtype":
            (
                metadata.get(
                    "dtype"
                )
                ==
                "uint16"
            ),

        "metadata_eos_token":
            (
                int(
                    metadata.get(
                        "eos_token_id"
                    )
                )
                ==
                2
            ),
    }

    # Dataset-version/split names are provenance
    # checks when the v0.4 metadata exposes them,
    # but we intentionally avoid assuming a
    # brittle internal stage label.
    if (
        "dataset_version"
        in
        metadata
    ):

        checks[
            "metadata_dataset_version"
        ] = (
            metadata[
                "dataset_version"
            ]
            ==
            EXPECTED_DATASET_VERSION
        )

    if (
        "split"
        in
        metadata
    ):

        split_value = str(
            metadata[
                "split"
            ]
        ).lower()

        checks[
            "metadata_split_kind"
        ] = (
            expected_kind
            in
            split_value
        )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            f"GENERAL-v0.5 {expected_kind} "
            f"dataset validation failed: "
            f"{failed}"
        )

    return checks


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
    validation_seed,
    step,
    tokens_seen,
    logger,
):

    print()
    print("=" * 80)
    print(
        f"VALIDATION — STEP {step:,}"
    )
    print("=" * 80)

    result = evaluate_model(
        model=model,
        binary_path=validation_path,
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=eval_batches,
        seed=validation_seed,
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
        "tokens/s"
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print()
    print("=" * 80)
    print(
        "MATHGPT — GENERAL-v0.5 "
        "16.89M CAPACITY SCALE / 300M EXPOSURE"
    )
    print("=" * 80)

    print()
    print(
        "Scientific control:"
    )
    print(
        "  Reuse the frozen General-v0.4 "
        "corpus, tokenizer, context, exposure, "
        "optimizer, and LR schedule."
    )
    print(
        "  Primary changed variable: "
        "model capacity."
    )

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    config_checks = preflight_config(
        training_config
    )

    training = training_config[
        "training"
    ]

    optimizer_config = (
        training_config[
            "optimizer"
        ]
    )

    scheduler = training_config[
        "scheduler"
    ]

    validation_config = (
        training_config[
            "validation"
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

    tokens_per_step = int(
        training[
            "tokens_per_step"
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
        training_config[
            "gradient_clipping"
        ][
            "max_norm"
        ]
    )

    log_every = int(
        training_config[
            "logging"
        ][
            "log_every_steps"
        ]
    )

    eval_every = int(
        validation_config[
            "eval_every_steps"
        ]
    )

    eval_batches = int(
        validation_config[
            "eval_batches"
        ]
    )

    validation_seed = int(
        validation_config[
            "seed"
        ]
    )

    checkpoint_every = int(
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

    train_path = resolve_project_path(
        training_config[
            "data"
        ][
            "train_binary"
        ]
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

    if not train_metadata_path.exists():
        raise FileNotFoundError(
            train_metadata_path
        )

    if not validation_metadata_path.exists():
        raise FileNotFoundError(
            validation_metadata_path
        )

    train_metadata = load_json(
        train_metadata_path
    )

    validation_metadata = load_json(
        validation_metadata_path
    )

    print()
    print("=" * 80)
    print(
        "GENERAL-v0.5 DATASET PREFLIGHT"
    )
    print("=" * 80)

    train_checks = validate_dataset(
        binary_path=train_path,
        metadata=train_metadata,
        expected_tokens=(
            EXPECTED_TRAIN_TOKENS
        ),
        expected_kind="train",
    )

    validation_checks = validate_dataset(
        binary_path=validation_path,
        metadata=validation_metadata,
        expected_tokens=(
            EXPECTED_VALIDATION_TOKENS
        ),
        expected_kind="validation",
    )

    print()
    print(
        "Training binary checks"
    )
    print(
        "----------------------"
    )

    for name, passed in (
        train_checks.items()
    ):
        print(
            f"{name:<34} {passed}"
        )

    print()
    print(
        "Validation binary checks"
    )
    print(
        "------------------------"
    )

    for name, passed in (
        validation_checks.items()
    ):
        print(
            f"{name:<34} {passed}"
        )

    # ========================================================
    # Model / loader
    # ========================================================

    print()
    print(
        "Creating GENERAL-v0.5 model..."
    )

    model = GPT.from_config(
        model_config,
        seed=seed,
    )

    count = parameter_count(
        model
    )

    require(
        count
        ==
        EXPECTED_PARAMETER_COUNT,
        (
            "Unexpected GENERAL-v0.5 "
            f"parameter count: {count:,}"
        ),
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

    require(
        train_loader.total_tokens
        ==
        EXPECTED_TRAIN_TOKENS,
        (
            "Training loader token count "
            "does not match frozen v0.4 "
            "training corpus."
        ),
    )

    print(
        f"Model parameters: "
        f"{count:,}"
    )

    print(
        f"Training corpus: "
        f"{EXPECTED_TRAIN_TOKENS:,} tokens"
    )

    print(
        f"Validation corpus: "
        f"{EXPECTED_VALIDATION_TOKENS:,} tokens"
    )

    print(
        f"Full exposure: "
        f"{EXPECTED_FINAL_EXPOSURES:,} tokens"
    )

    print(
        f"Corpus-equivalents: "
        f"{EXPECTED_FINAL_EXPOSURES / EXPECTED_TRAIN_TOKENS:.6f}x"
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
    # Preflight only
    # ========================================================

    if args.preflight_only:

        print()
        print("=" * 80)
        print(
            "PREFLIGHT-ONLY SUCCESS"
        )
        print("=" * 80)

        print()
        print(
            "Frozen v0.5 configuration, "
            "v0.4 corpus identity, 16.89M "
            "model, loader, and 300M exposure "
            "schedule all passed."
        )

        print()
        print(
            "No training steps were executed."
        )

        return

    # ========================================================
    # Run identity
    # ========================================================

    timestamp = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    if args.run_name is not None:
        run_name = args.run_name

    elif args.resume is not None:
        run_name = (
            "general_pretraining_v0_5_resume_"
            f"{timestamp}"
        )

    else:
        run_name = (
            "general_pretraining_v0_5_fresh_"
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
    # Fresh artifact protection
    # ========================================================

    if args.resume is None:

        existing = []

        if log_path.exists():
            existing.append(
                log_path
            )

        if summary_path.exists():
            existing.append(
                summary_path
            )

        if (
            checkpoint_directory.exists()
            and
            any(
                checkpoint_directory.iterdir()
            )
        ):
            existing.append(
                checkpoint_directory
            )

        if existing:
            raise FileExistsError(
                "Fresh GENERAL-v0.5 run would "
                "reuse existing artifacts. "
                "Choose a different --run-name. "
                f"Existing: {existing}"
            )

    logger = TrainingLogger(
        log_path
    )

    # ========================================================
    # Fresh / resume
    # ========================================================

    resume_path = None

    if args.resume is not None:

        resume_path = (
            resolve_project_path(
                args.resume
            )
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                resume_path
            )

        if (
            "general_pretraining_v0.5"
            not in str(
                resume_path
            )
        ):
            raise ValueError(
                "Refusing to resume v0.5 "
                "from a checkpoint outside "
                "the v0.5 checkpoint tree."
            )

        print()
        print("=" * 80)
        print(
            "RESTORING GENERAL-v0.5 CHECKPOINT"
        )
        print("=" * 80)

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

        require(
            tokens_seen
            ==
            expected_resume_tokens,
            (
                "Resume token accounting "
                "does not match v0.5."
            ),
        )

        require(
            int(
                checkpoint_metadata[
                    "parameter_count"
                ]
            )
            ==
            EXPECTED_PARAMETER_COUNT,
            (
                "Resume checkpoint parameter "
                "count is not v0.5."
            ),
        )

        logger.log_event(
            "training_resumed",
            experiment=EXPECTED_EXPERIMENT,
            dataset_version=(
                EXPECTED_DATASET_VERSION
            ),
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
            experiment_type=(
                EXPECTED_EXPERIMENT_TYPE
            ),
            dataset_version=(
                EXPECTED_DATASET_VERSION
            ),
            model_parameter_count=(
                EXPECTED_PARAMETER_COUNT
            ),
            step=0,
            tokens_seen=0,
        )

    if (
        starting_step
        >=
        target_step
    ):
        raise ValueError(
            f"Checkpoint is already at "
            f"step {starting_step:,}, but "
            f"target-step is {target_step:,}."
        )

    expected_final_tokens = (
        target_step
        *
        tokens_per_step
    )

    # ========================================================
    # Launch summary
    # ========================================================

    print()
    print("=" * 80)
    print(
        "GENERAL-v0.5 PRETRAINING "
        "CONFIGURATION"
    )
    print("=" * 80)

    print()
    print(
        f"Run name: {run_name}"
    )

    print(
        f"Mode: "
        f"{'RESUME' if resume_path else 'FRESH'}"
    )

    print()
    print(
        f"Parameters: "
        f"{count:,}"
    )

    print(
        f"Baseline parameters: "
        f"{EXPECTED_BASELINE_PARAMETER_COUNT:,}"
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
        f"Starting step: "
        f"{starting_step:,}"
    )

    print(
        f"Target step: "
        f"{target_step:,}"
    )

    print(
        f"Full schedule: "
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
        f"Tokens at target: "
        f"{expected_final_tokens:,}"
    )

    print()
    print(
        f"Requested full exposure: "
        f"{EXPECTED_REQUESTED_EXPOSURES:,}"
    )

    print(
        f"Actual full exposure: "
        f"{EXPECTED_FINAL_EXPOSURES:,}"
    )

    print(
        f"Corpus-equivalents: "
        f"{EXPECTED_FINAL_EXPOSURES / EXPECTED_TRAIN_TOKENS:.6f}x"
    )

    print()
    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Context: "
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
        f"{log_every:,}"
    )

    print(
        f"Validate every: "
        f"{eval_every:,}"
    )

    print(
        f"Checkpoint every: "
        f"{checkpoint_every:,}"
    )

    print()
    print(
        f"Log: {log_path}"
    )

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
        validation_path=(
            validation_path
        ),
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=eval_batches,
        validation_seed=(
            validation_seed
        ),
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

    recent_losses = deque(
        maxlen=50
    )

    throughput_sum = 0.0
    throughput_count = 0

    all_training_losses_finite = True
    all_gradient_norms_finite = True

    all_checkpoints = []

    # ========================================================
    # Main training loop
    # ========================================================

    try:

        while (
            optimizer.step_count
            <
            target_step
        ):

            current_step = (
                optimizer.step_count
            )

            next_eval_boundary = (
                (
                    current_step
                    //
                    eval_every
                )
                +
                1
            ) * eval_every

            chunk_end = min(
                next_eval_boundary,
                target_step,
            )

            chunk_steps = (
                chunk_end
                -
                current_step
            )

            print()
            print("=" * 80)
            print(
                f"TRAINING "
                f"{current_step + 1:,}"
                f"–"
                f"{chunk_end:,}"
            )
            print("=" * 80)

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

            all_checkpoints.extend(
                saved_checkpoints
            )

            for metrics in history:

                loss = float(
                    metrics[
                        "loss"
                    ]
                )

                grad = float(
                    metrics[
                        "gradient_norm"
                    ]
                )

                throughput = float(
                    metrics[
                        "tokens_per_second"
                    ]
                )

                recent_losses.append(
                    loss
                )

                throughput_sum += (
                    throughput
                )

                throughput_count += 1

                if not np.isfinite(
                    loss
                ):
                    all_training_losses_finite = False

                if not np.isfinite(
                    grad
                ):
                    all_gradient_norms_finite = False

            del history

            validation_result = run_validation(
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
                validation_seed=(
                    validation_seed
                ),
                step=(
                    optimizer.step_count
                ),
                tokens_seen=(
                    tokens_seen
                ),
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
        print("=" * 80)
        print(
            "GENERAL-v0.5 TRAINING INTERRUPTED"
        )
        print("=" * 80)

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
    # Final integrity
    # ========================================================

    checks = {
        "step_count_passed":
            (
                optimizer.step_count
                ==
                target_step
            ),

        "token_accounting_passed":
            (
                tokens_seen
                ==
                expected_final_tokens
            ),

        "training_losses_finite":
            all_training_losses_finite,

        "gradient_norms_finite":
            all_gradient_norms_finite,

        "validation_losses_finite":
            all(
                np.isfinite(
                    item[
                        "loss"
                    ]
                )
                for item
                in validation_history
            ),

        "model_parameters_finite":
            parameters_finite(
                model
            ),

        "optimizer_state_finite":
            optimizer_state_finite(
                optimizer
            ),
    }

    if recent_losses:

        recent_train_loss = float(
            np.mean(
                list(
                    recent_losses
                )
            )
        )

    else:
        recent_train_loss = None

    average_throughput = (
        throughput_sum
        /
        throughput_count
        if throughput_count > 0
        else None
    )

    final_validation = (
        validation_history[
            -1
        ]
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "model":
            "MathGPT-General-v0.5-16.89M",

        "experiment":
            EXPECTED_EXPERIMENT,

        "experiment_type":
            EXPECTED_EXPERIMENT_TYPE,

        "comparison_baseline": {
            "model":
                "MathGPT-General-v0.4-300M",

            "parameter_count":
                EXPECTED_BASELINE_PARAMETER_COUNT,

            "token_exposure":
                EXPECTED_BASELINE_EXPOSURE,
        },

        "dataset_version":
            EXPECTED_DATASET_VERSION,

        "tokenizer":
            EXPECTED_TOKENIZER,

        "model_config_path":
            str(
                MODEL_CONFIG_PATH
            ),

        "model_config_sha256":
            file_sha256(
                MODEL_CONFIG_PATH
            ),

        "training_config_path":
            str(
                TRAINING_CONFIG_PATH
            ),

        "training_config_sha256":
            file_sha256(
                TRAINING_CONFIG_PATH
            ),

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

        "unique_training_corpus_tokens":
            EXPECTED_TRAIN_TOKENS,

        "validation_corpus_tokens":
            EXPECTED_VALIDATION_TOKENS,

        "requested_full_exposures":
            EXPECTED_REQUESTED_EXPOSURES,

        "actual_full_exposures":
            EXPECTED_FINAL_EXPOSURES,

        "full_corpus_equivalents":
            float(
                EXPECTED_FINAL_EXPOSURES
                /
                EXPECTED_TRAIN_TOKENS
            ),

        "batch_size":
            int(
                batch_size
            ),

        "context_length":
            int(
                context_length
            ),

        "warmup_steps":
            int(
                warmup_steps
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
                str(path)
                for path
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

    logger.log_event(
        "training_completed",
        experiment=(
            EXPECTED_EXPERIMENT
        ),
        experiment_type=(
            EXPECTED_EXPERIMENT_TYPE
        ),
        dataset_version=(
            EXPECTED_DATASET_VERSION
        ),
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
    # Display
    # ========================================================

    print()
    print("=" * 80)
    print(
        "GENERAL-v0.5 CAPACITY-SCALE "
        "RUN COMPLETE"
    )
    print("=" * 80)

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

    print(
        f"Corpus-equivalents: "
        f"{tokens_seen / EXPECTED_TRAIN_TOKENS:.6f}x"
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
            "tokens/s"
        )

    print()
    print(
        "Integrity Checks"
    )
    print(
        "----------------"
    )

    for name, passed in checks.items():
        print(
            f"{name}: {passed}"
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
        f"Summary:"
    )
    print(
        summary_path
    )

    if not all(
        checks.values()
    ):
        raise RuntimeError(
            "GENERAL-v0.5 completed but "
            "failed final integrity checks."
        )

    print()
    print("=" * 80)
    print(
        "SUCCESS: GENERAL-v0.5 TRAINING "
        "INVOCATION PASSED."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
