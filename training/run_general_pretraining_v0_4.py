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
# GENERAL-v0.4 experiment identity
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)

TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.4.json"
)

ENCODING_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_v0_4_encoding_validation_report.json"
)

LEAKAGE_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "general_v0.4_sized"
    / "general_pretraining_leakage_report.json"
)


EXPECTED_CONFIG_VERSION = "0.4"

EXPECTED_EXPERIMENT = (
    "general_pretraining_v0.4_data_scale_300m"
)

EXPECTED_EXPERIMENT_TYPE = (
    "unique_data_scaling"
)

EXPECTED_DATASET_VERSION = (
    "general_pretraining_v0.4"
)

EXPECTED_DATA_STAGE = "15D"

EXPECTED_ENCODING_VALIDATION_STAGE = (
    "15E.1"
)

EXPECTED_LEAKAGE_VALIDATION_STAGE = (
    "15E.2"
)

EXPECTED_TOKENIZER = (
    "tokenizer_50m_v0.2"
)

EXPECTED_PARAMETER_COUNT = (
    8_110_848
)

EXPECTED_TRAIN_TOKENS = (
    251_490_827
)

EXPECTED_VALIDATION_TOKENS = (
    27_928_209
)

EXPECTED_TRAIN_DOCUMENTS = (
    270_366
)

EXPECTED_VALIDATION_DOCUMENTS = (
    30_289
)

EXPECTED_TRAIN_SHA256 = (
    "93560fc03217b2eeee95f9da9e5bc448"
    "2127d7e83adfdf0d0d25a6a609b2db64"
)

EXPECTED_VALIDATION_SHA256 = (
    "9a8d489330e4bbd9e747689b450f3500"
    "445180a80bb684a8d95d6136d3926069"
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

EXPECTED_WARMUP_STEPS = (
    14_294
)

# Both persistent JSON logging and terminal progress
# are intentionally every 5 steps for v0.4.
EXPECTED_LOG_EVERY = 5
EXPECTED_PROGRESS_EVERY = 5

EXPECTED_EVAL_EVERY = 5_000

EXPECTED_CHECKPOINT_EVERY = (
    25_000
)

EXPECTED_TRAIN_SPLIT = (
    "general_v0_4_train"
)

EXPECTED_VALIDATION_SPLIT = (
    "general_v0_4_validation"
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
            ) in named_parameters(model)
        )
    )


def parameters_finite(model):
    return all(
        np.all(
            np.isfinite(parameter)
        )
        for (
            _,
            parameter,
            _,
        ) in named_parameters(model)
    )


def optimizer_state_finite(optimizer):
    first_finite = all(
        np.all(
            np.isfinite(value)
        )
        for value
        in optimizer.first_moments.values()
    )

    second_finite = all(
        np.all(
            np.isfinite(value)
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
# Arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train MathGPT-General-v0.4 "
            "from scratch for approximately "
            "300M token exposures using the "
            "validated GENERAL-v0.4 corpus."
        )
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a complete "
            "GENERAL-v0.4 checkpoint."
        ),
    )

    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help=(
            "Absolute step at which this "
            "invocation stops. The learning-rate "
            "schedule always remains the full "
            "292,969-step v0.4 schedule."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Explicit run name.",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate configuration, dataset, "
            "audit reports, hashes, model, "
            "and loader without training."
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
    expected_documents,
    expected_sha256,
):
    checks = {
        "stage":
            (
                metadata.get("stage")
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
                metadata.get("split")
                ==
                expected_split
            ),

        "tokenizer":
            (
                metadata.get("tokenizer")
                ==
                EXPECTED_TOKENIZER
            ),

        "tokenizer_merges":
            (
                int(
                    metadata.get(
                        "tokenizer_merges"
                    )
                )
                ==
                15_741
            ),

        "vocab_size":
            (
                int(
                    metadata.get(
                        "vocab_size"
                    )
                )
                ==
                16_000
            ),

        "dtype":
            (
                metadata.get("dtype")
                ==
                "uint16"
            ),

        "bytes_per_token":
            (
                int(
                    metadata.get(
                        "bytes_per_token"
                    )
                )
                ==
                2
            ),

        "eos_token_id":
            (
                int(
                    metadata.get(
                        "eos_token_id"
                    )
                )
                ==
                2
            ),

        "document_count":
            (
                int(
                    metadata.get(
                        "documents"
                    )
                )
                ==
                expected_documents
            ),

        "token_count":
            (
                int(
                    metadata.get("tokens")
                )
                ==
                expected_tokens
            ),

        "binary_size":
            (
                binary_path.stat().st_size
                ==
                expected_tokens * 2
            ),
    }

    binary_tokens = np.memmap(
        binary_path,
        dtype=np.uint16,
        mode="r",
    )

    checks[
        "binary_token_count"
    ] = (
        len(binary_tokens)
        ==
        expected_tokens
    )

    checks[
        "binary_token_range"
    ] = (
        int(binary_tokens.min())
        >=
        0
        and
        int(binary_tokens.max())
        <
        16_000
    )

    del binary_tokens

    print(
        "Calculating SHA-256 for:"
    )
    print(binary_path)

    actual_sha256 = file_sha256(
        binary_path
    )

    checks[
        "binary_sha256"
    ] = (
        actual_sha256
        ==
        expected_sha256
    )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "GENERAL-v0.4 dataset "
            "preflight failed: "
            f"{failed}"
        )

    return (
        checks,
        actual_sha256,
    )


# ============================================================
# Audit validation
# ============================================================

def validate_audit_reports():
    print()
    print("=" * 76)
    print(
        "GENERAL-v0.4 AUDIT PREFLIGHT"
    )
    print("=" * 76)

    encoding_report = load_json(
        ENCODING_AUDIT_PATH
    )

    leakage_report = load_json(
        LEAKAGE_AUDIT_PATH
    )

    checks = {
        "Encoding audit stage":
            (
                encoding_report.get("stage")
                ==
                EXPECTED_ENCODING_VALIDATION_STAGE
            ),

        "Encoding dataset identity":
            (
                encoding_report.get(
                    "dataset_version"
                )
                ==
                EXPECTED_DATASET_VERSION
            ),

        "Encoding audit passed":
            bool(
                encoding_report.get(
                    "all_passed"
                )
            ),

        "Encoding train passed":
            bool(
                encoding_report[
                    "train"
                ].get(
                    "all_passed"
                )
            ),

        "Encoding validation passed":
            bool(
                encoding_report[
                    "validation"
                ].get(
                    "all_passed"
                )
            ),

        "Encoding train SHA":
            (
                encoding_report[
                    "train"
                ][
                    "binary_sha256"
                ]
                ==
                EXPECTED_TRAIN_SHA256
            ),

        "Encoding validation SHA":
            (
                encoding_report[
                    "validation"
                ][
                    "binary_sha256"
                ]
                ==
                EXPECTED_VALIDATION_SHA256
            ),

        "Leakage audit stage":
            (
                leakage_report.get("stage")
                ==
                EXPECTED_LEAKAGE_VALIDATION_STAGE
            ),

        "Leakage dataset identity":
            (
                leakage_report.get(
                    "dataset_version"
                )
                ==
                EXPECTED_DATASET_VERSION
            ),

        "Leakage audit passed":
            bool(
                leakage_report.get(
                    "all_passed"
                )
            ),

        "Exact train/validation overlap = 0":
            (
                int(
                    leakage_report.get(
                        "exact_train_validation_overlap"
                    )
                )
                ==
                0
            ),

        "Train duplicates = 0":
            (
                int(
                    leakage_report[
                        "train"
                    ][
                        "duplicate_occurrences"
                    ]
                )
                ==
                0
            ),

        "Validation duplicates = 0":
            (
                int(
                    leakage_report[
                        "validation"
                    ][
                        "duplicate_occurrences"
                    ]
                )
                ==
                0
            ),
    }

    print()

    for name, passed in checks.items():
        print(
            f"{name:<42} "
            f"{passed}"
        )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "GENERAL-v0.4 audit "
            f"preflight failed: {failed}"
        )

    print()
    print(
        "SUCCESS: GENERAL-v0.4 "
        "audit reports confirmed."
    )

    return checks


# ============================================================
# Configuration preflight
# ============================================================

def preflight_config(config):
    print()
    print("=" * 76)
    print(
        "GENERAL-v0.4 DATA-SCALE "
        "300M-EXPOSURE CONFIG PREFLIGHT"
    )
    print("=" * 76)

    data = config["data"]
    training = config["training"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]

    gradient_clipping = (
        config[
            "gradient_clipping"
        ]
    )

    logging = config["logging"]
    validation = config["validation"]

    checkpointing = (
        config[
            "checkpointing"
        ]
    )

    calculated_tokens_per_step = (
        int(training["batch_size"])
        *
        int(training["context_length"])
    )

    calculated_final_exposures = (
        int(training["max_steps"])
        *
        calculated_tokens_per_step
    )

    calculated_required_steps = (
        EXPECTED_REQUESTED_EXPOSURES
        +
        calculated_tokens_per_step
        -
        1
    ) // calculated_tokens_per_step

    calculated_overage = (
        calculated_final_exposures
        -
        EXPECTED_REQUESTED_EXPOSURES
    )

    corpus_equivalents = (
        calculated_final_exposures
        /
        EXPECTED_TRAIN_TOKENS
    )

    warmup_fraction = (
        int(scheduler["warmup_steps"])
        /
        int(training["max_steps"])
    )

    checks = {
        "Config version = 0.4":
            (
                str(config.get("version"))
                ==
                EXPECTED_CONFIG_VERSION
            ),

        "Experiment identity":
            (
                config.get("experiment")
                ==
                EXPECTED_EXPERIMENT
            ),

        "Experiment type":
            (
                config.get(
                    "experiment_type"
                )
                ==
                EXPECTED_EXPERIMENT_TYPE
            ),

        "Dataset version = v0.4":
            (
                data.get(
                    "dataset_version"
                )
                ==
                EXPECTED_DATASET_VERSION
            ),

        "Dataset stage = 15D":
            (
                data.get(
                    "dataset_stage"
                )
                ==
                EXPECTED_DATA_STAGE
            ),

        "Encoding audit stage = 15E.1":
            (
                data.get(
                    "encoding_validation_stage"
                )
                ==
                EXPECTED_ENCODING_VALIDATION_STAGE
            ),

        "Leakage audit stage = 15E.2":
            (
                data.get(
                    "leakage_validation_stage"
                )
                ==
                EXPECTED_LEAKAGE_VALIDATION_STAGE
            ),

        "Tokenizer unchanged":
            (
                data.get("tokenizer")
                ==
                EXPECTED_TOKENIZER
            ),

        "Training corpus tokens":
            (
                int(
                    data.get(
                        "training_tokens"
                    )
                )
                ==
                EXPECTED_TRAIN_TOKENS
            ),

        "Validation corpus tokens":
            (
                int(
                    data.get(
                        "validation_tokens"
                    )
                )
                ==
                EXPECTED_VALIDATION_TOKENS
            ),

        "Training documents":
            (
                int(
                    data.get(
                        "training_documents"
                    )
                )
                ==
                EXPECTED_TRAIN_DOCUMENTS
            ),

        "Validation documents":
            (
                int(
                    data.get(
                        "validation_documents"
                    )
                )
                ==
                EXPECTED_VALIDATION_DOCUMENTS
            ),

        "Train SHA frozen":
            (
                data.get(
                    "train_binary_sha256"
                )
                ==
                EXPECTED_TRAIN_SHA256
            ),

        "Validation SHA frozen":
            (
                data.get(
                    "validation_binary_sha256"
                )
                ==
                EXPECTED_VALIDATION_SHA256
            ),

        "Train duplicates = 0":
            (
                int(
                    data.get(
                        "exact_train_duplicates"
                    )
                )
                ==
                0
            ),

        "Validation duplicates = 0":
            (
                int(
                    data.get(
                        "exact_validation_duplicates"
                    )
                )
                ==
                0
            ),

        "Train/validation overlap = 0":
            (
                int(
                    data.get(
                        "exact_train_validation_overlap"
                    )
                )
                ==
                0
            ),

        "Batch size = 4":
            (
                int(training["batch_size"])
                ==
                EXPECTED_BATCH_SIZE
            ),

        "Context length = 256":
            (
                int(
                    training[
                        "context_length"
                    ]
                )
                ==
                EXPECTED_CONTEXT_LENGTH
            ),

        "Tokens per step = 1,024":
            (
                int(
                    training[
                        "tokens_per_step"
                    ]
                )
                ==
                EXPECTED_TOKENS_PER_STEP
            ),

        "Seed = 42":
            (
                int(training["seed"])
                ==
                42
            ),

        "Max steps = 292,969":
            (
                int(training["max_steps"])
                ==
                EXPECTED_MAX_STEPS
            ),

        "Requested exposures = 300M":
            (
                int(
                    training[
                        "requested_token_exposures"
                    ]
                )
                ==
                EXPECTED_REQUESTED_EXPOSURES
            ),

        "Actual exposures = 300,000,256":
            (
                int(
                    training[
                        "target_token_exposures"
                    ]
                )
                ==
                EXPECTED_FINAL_EXPOSURES
            ),

        "Warmup steps = 14,294":
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

        "AdamW beta1 = 0.9":
            (
                float(optimizer["beta1"])
                ==
                0.9
            ),

        "AdamW beta2 = 0.999":
            (
                float(optimizer["beta2"])
                ==
                0.999
            ),

        "AdamW epsilon = 1e-8":
            (
                float(optimizer["epsilon"])
                ==
                1e-8
            ),

        "Weight decay = 0.01":
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

        "Log every = 5":
            (
                int(
                    logging[
                        "log_every_steps"
                    ]
                )
                ==
                EXPECTED_LOG_EVERY
            ),

        "Validation every = 5,000":
            (
                int(
                    validation[
                        "eval_every_steps"
                    ]
                )
                ==
                EXPECTED_EVAL_EVERY
            ),

        "Checkpoint every = 25,000":
            (
                int(
                    checkpointing[
                        "save_every_steps"
                    ]
                )
                ==
                EXPECTED_CHECKPOINT_EVERY
            ),

        "Batch x context = 1,024":
            (
                calculated_tokens_per_step
                ==
                EXPECTED_TOKENS_PER_STEP
            ),

        "Ceiling(300M / 1,024) = 292,969":
            (
                calculated_required_steps
                ==
                EXPECTED_MAX_STEPS
            ),

        "Calculated exposure = 300,000,256":
            (
                calculated_final_exposures
                ==
                EXPECTED_FINAL_EXPOSURES
            ),

        "Exposure overage = 256":
            (
                calculated_overage
                ==
                EXPECTED_EXPOSURE_OVERAGE
            ),
    }

    print()

    for name, passed in checks.items():
        print(
            f"{name:<44} "
            f"{passed}"
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

    print(
        f"Persistent JSON logging: "
        f"every {EXPECTED_LOG_EVERY} steps"
    )

    print(
        f"Terminal progress: "
        f"every {EXPECTED_PROGRESS_EVERY} steps"
    )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "GENERAL-v0.4 config "
            f"preflight failed: {failed}"
        )

    print()
    print(
        "SUCCESS: GENERAL-v0.4 "
        "configuration is locked and valid."
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
    step,
    tokens_seen,
    logger,
):
    print()
    print("=" * 76)
    print(
        f"VALIDATION — STEP "
        f"{step:,}"
    )
    print("=" * 76)

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
    print("=" * 76)
    print(
        "MATHGPT — GENERAL-v0.4 "
        "UNIQUE-DATA-SCALE PRETRAINING"
    )
    print("=" * 76)

    print()
    print(
        "This launcher is dedicated to "
        "MathGPT-General-v0.4."
    )

    print(
        "The model architecture and 300M-token "
        "exposure schedule remain frozen from v0.3."
    )

    print(
        "The experimental variable is the "
        "larger unique General-v0.4 corpus."
    )

    # ========================================================
    # Load configurations
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

    validate_audit_reports()

    training = (
        training_config[
            "training"
        ]
    )

    optimizer_config = (
        training_config[
            "optimizer"
        ]
    )

    scheduler = (
        training_config[
            "scheduler"
        ]
    )

    # ========================================================
    # Core values
    # ========================================================

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
        training["seed"]
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

    beta1 = float(
        optimizer_config["beta1"]
    )

    beta2 = float(
        optimizer_config["beta2"]
    )

    epsilon = float(
        optimizer_config[
            "epsilon"
        ]
    )

    weight_decay = float(
        optimizer_config[
            "weight_decay"
        ]
    )

    warmup_steps = int(
        scheduler[
            "warmup_steps"
        ]
    )

    min_learning_rate = float(
        scheduler[
            "min_learning_rate"
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

    # User requested terminal output at the same
    # 5-step cadence as persistent JSON logging.
    progress_every = (
        EXPECTED_PROGRESS_EVERY
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
    # Dataset paths
    # ========================================================

    data_config = (
        training_config[
            "data"
        ]
    )

    train_path = resolve_project_path(
        data_config[
            "train_binary"
        ]
    )

    validation_path = (
        resolve_project_path(
            data_config[
                "validation_binary"
            ]
        )
    )

    train_metadata_path = (
        resolve_project_path(
            data_config[
                "train_metadata"
            ]
        )
    )

    validation_metadata_path = (
        resolve_project_path(
            data_config[
                "validation_metadata"
            ]
        )
    )

    for path in (
        train_path,
        validation_path,
        train_metadata_path,
        validation_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    train_metadata = load_json(
        train_metadata_path
    )

    validation_metadata = load_json(
        validation_metadata_path
    )

    # ========================================================
    # Dataset identity + fingerprints
    # ========================================================

    print()
    print("=" * 76)
    print(
        "GENERAL-v0.4 FROZEN DATASET PREFLIGHT"
    )
    print("=" * 76)
    print()

    (
        train_checks,
        train_sha256,
    ) = validate_dataset_metadata(
        metadata=train_metadata,
        binary_path=train_path,
        expected_split=(
            EXPECTED_TRAIN_SPLIT
        ),
        expected_tokens=(
            EXPECTED_TRAIN_TOKENS
        ),
        expected_documents=(
            EXPECTED_TRAIN_DOCUMENTS
        ),
        expected_sha256=(
            EXPECTED_TRAIN_SHA256
        ),
    )

    (
        validation_checks,
        validation_sha256,
    ) = validate_dataset_metadata(
        metadata=validation_metadata,
        binary_path=validation_path,
        expected_split=(
            EXPECTED_VALIDATION_SPLIT
        ),
        expected_tokens=(
            EXPECTED_VALIDATION_TOKENS
        ),
        expected_documents=(
            EXPECTED_VALIDATION_DOCUMENTS
        ),
        expected_sha256=(
            EXPECTED_VALIDATION_SHA256
        ),
    )

    print()
    print(
        f"Train tokens: "
        f"{EXPECTED_TRAIN_TOKENS:,}"
    )

    print(
        f"Validation tokens: "
        f"{EXPECTED_VALIDATION_TOKENS:,}"
    )

    print()
    print(
        "Train SHA-256:"
    )
    print(train_sha256)

    print()
    print(
        "Validation SHA-256:"
    )
    print(validation_sha256)

    print()
    print(
        "Training binary checks:"
    )

    for name, passed in (
        train_checks.items()
    ):
        print(
            f"  {name}: "
            f"{passed}"
        )

    print()
    print(
        "Validation binary checks:"
    )

    for name, passed in (
        validation_checks.items()
    ):
        print(
            f"  {name}: "
            f"{passed}"
        )

    print()
    print(
        "SUCCESS: frozen GENERAL-v0.4 "
        "dataset identity confirmed."
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
            "general_pretraining_v0_4_resume_"
            f"{timestamp}"
        )

    else:
        run_name = (
            "general_pretraining_v0_4_fresh_"
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
                "Fresh GENERAL-v0.4 run would "
                "reuse existing artifacts. "
                "Choose a different --run-name. "
                f"Existing: {existing_artifacts}"
            )

    # ========================================================
    # Model
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
    # Loader
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
            "does not match frozen "
            "GENERAL-v0.4 corpus."
        )

    print(
        f"Model parameters: "
        f"{count:,}"
    )

    print(
        f"Loader tokens: "
        f"{train_loader.total_tokens:,}"
    )

    print(
        f"Target exposure: "
        f"{EXPECTED_FINAL_EXPOSURES:,}"
    )

    print(
        f"Corpus-equivalents: "
        f"{EXPECTED_FINAL_EXPOSURES / EXPECTED_TRAIN_TOKENS:.6f}x"
    )

    # ========================================================
    # Preflight-only
    # ========================================================

    if args.preflight_only:
        print()
        print("=" * 76)
        print(
            "PREFLIGHT-ONLY SUCCESS"
        )
        print("=" * 76)

        print()
        print(
            "Model, configuration, dataset, "
            "15E audits, SHA-256 fingerprints, "
            "exposure schedule, and loader "
            "all passed."
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
                "Resume checkpoint missing: "
                f"{resume_path}"
            )

        if (
            "general_pretraining_v0.4"
            not in str(resume_path)
        ):
            raise ValueError(
                "Refusing to resume "
                "GENERAL-v0.4 from a checkpoint "
                "outside the v0.4 checkpoint tree."
            )

        print()
        print("=" * 76)
        print(
            "RESTORING GENERAL-v0.4 CHECKPOINT"
        )
        print("=" * 76)
        print()
        print(resume_path)

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
            optimizer.step_count
            !=
            starting_step
        ):
            raise RuntimeError(
                "Restored optimizer step does "
                "not match checkpoint step."
            )

        if (
            tokens_seen
            !=
            expected_resume_tokens
        ):
            raise RuntimeError(
                "Resume checkpoint token "
                "accounting does not match "
                "GENERAL-v0.4."
            )

        logger.log_event(
            "training_resumed",
            experiment=(
                EXPECTED_EXPERIMENT
            ),
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
            experiment=(
                EXPECTED_EXPERIMENT
            ),
            dataset_version=(
                EXPECTED_DATASET_VERSION
            ),
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
        target_step
        *
        tokens_per_step
    )

    # ========================================================
    # Launch summary
    # ========================================================

    print()
    print("=" * 76)
    print(
        "GENERAL-v0.4 PRETRAINING "
        "CONFIGURATION"
    )
    print("=" * 76)

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
        f"Unique training corpus: "
        f"{EXPECTED_TRAIN_TOKENS:,} tokens"
    )

    print(
        f"Validation corpus: "
        f"{EXPECTED_VALIDATION_TOKENS:,} tokens"
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
        f"Expected tokens at target: "
        f"{expected_final_tokens:,}"
    )

    print()
    print(
        f"Full requested exposure: "
        f"{EXPECTED_REQUESTED_EXPOSURES:,}"
    )

    print(
        f"Full actual exposure: "
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
        f"Persistent log every: "
        f"{log_every:,} steps"
    )

    print(
        f"Terminal progress every: "
        f"{progress_every:,} steps"
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
    print(log_path)

    print()
    print(
        "Checkpoint directory:"
    )
    print(checkpoint_directory)

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
                int(starting_step),

            "tokens_seen":
                int(tokens_seen),

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
    # Lightweight run statistics
    # ========================================================

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
            print("=" * 76)

            print(
                f"TRAINING "
                f"{current_step + 1:,}"
                f"–"
                f"{chunk_end_step:,}"
            )

            print("=" * 76)

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

                # Both are intentionally 5.
                progress_every=(
                    progress_every
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

            all_checkpoints.extend(
                saved_checkpoints
            )

            for metrics in history:
                loss = float(
                    metrics["loss"]
                )

                gradient_norm = float(
                    metrics[
                        "gradient_norm"
                    ]
                )

                tokens_per_second = float(
                    metrics[
                        "tokens_per_second"
                    ]
                )

                recent_losses.append(
                    loss
                )

                throughput_sum += (
                    tokens_per_second
                )

                throughput_count += 1

                if not np.isfinite(loss):
                    all_training_losses_finite = (
                        False
                    )

                if not np.isfinite(
                    gradient_norm
                ):
                    all_gradient_norms_finite = (
                        False
                    )

            # The runner only keeps one validation-sized
            # training chunk in memory at a time.
            del history

            validation_result = run_validation(
                model=model,
                validation_path=validation_path,
                context_length=(
                    context_length
                ),
                batch_size=batch_size,
                eval_batches=eval_batches,
                step=(
                    optimizer.step_count
                ),
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
        print("=" * 76)

        print(
            "GENERAL-v0.4 TRAINING INTERRUPTED"
        )

        print("=" * 76)
        print()

        print(
            "Resume from the most recent "
            "completed periodic or final "
            "checkpoint."
        )

        logger.log_event(
            "training_interrupted",
            experiment=(
                EXPECTED_EXPERIMENT
            ),
            dataset_version=(
                EXPECTED_DATASET_VERSION
            ),
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

    validation_losses_finite = all(
        np.isfinite(
            item["loss"]
        )
        for item
        in validation_history
    )

    # ========================================================
    # Statistics
    # ========================================================

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

    if throughput_count > 0:
        average_throughput = (
            throughput_sum
            /
            throughput_count
        )

    else:
        average_throughput = None

    final_validation = (
        validation_history[-1]
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "model":
            "MathGPT-General-v0.4",

        "experiment":
            EXPECTED_EXPERIMENT,

        "experiment_type":
            EXPECTED_EXPERIMENT_TYPE,

        "dataset_version":
            EXPECTED_DATASET_VERSION,

        "tokenizer":
            EXPECTED_TOKENIZER,

        "train_binary_sha256":
            train_sha256,

        "validation_binary_sha256":
            validation_sha256,

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
                else str(resume_path)
            ),

        "parameter_count":
            int(count),

        "starting_step":
            int(starting_step),

        "final_step":
            int(
                optimizer.step_count
            ),

        "target_step":
            int(target_step),

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
            int(tokens_seen),

        "tokens_per_step":
            int(tokens_per_step),

        "unique_training_corpus_tokens":
            EXPECTED_TRAIN_TOKENS,

        "validation_corpus_tokens":
            EXPECTED_VALIDATION_TOKENS,

        "training_documents":
            EXPECTED_TRAIN_DOCUMENTS,

        "validation_documents":
            EXPECTED_VALIDATION_DOCUMENTS,

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

        "persistent_log_every_steps":
            int(log_every),

        "terminal_progress_every_steps":
            int(progress_every),

        "warmup_steps":
            int(warmup_steps),

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

        "checks": {
            "step_count_passed":
                step_count_passed,

            "token_accounting_passed":
                token_accounting_passed,

            "training_losses_finite":
                all_training_losses_finite,

            "gradient_norms_finite":
                all_gradient_norms_finite,

            "validation_losses_finite":
                validation_losses_finite,

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
    print("=" * 76)

    print(
        "GENERAL-v0.4 DATA-SCALE "
        "300M-EXPOSURE RUN COMPLETE"
    )

    print("=" * 76)

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
    print(summary_path)

    if not all_integrity_passed:
        raise RuntimeError(
            "GENERAL-v0.4 training finished "
            "but failed integrity checks."
        )

    print()
    print("=" * 76)

    print(
        "SUCCESS: GENERAL-v0.4 "
        "TRAINING INVOCATION PASSED."
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
