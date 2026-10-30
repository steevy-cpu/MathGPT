import json
import sys
from pathlib import Path


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
# Configuration paths
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
# Expected Stage 9 configuration
# ============================================================

EXPECTED_BATCH_SIZE = 4

EXPECTED_CONTEXT_LENGTH = 256

EXPECTED_TOKENS_PER_STEP = 1_024

EXPECTED_MAX_STEPS = 8_197


EXPECTED_TOKENIZER = (
    "tokenizer_50m_v0.2"
)


EXPECTED_OPTIMIZER = (
    "adamw"
)


EXPECTED_LEARNING_RATE = (
    3e-4
)

EXPECTED_BETA1 = (
    0.9
)

EXPECTED_BETA2 = (
    0.999
)

EXPECTED_EPSILON = (
    1e-8
)

EXPECTED_WEIGHT_DECAY = (
    0.01
)


EXPECTED_SCHEDULER = (
    "warmup_cosine"
)

EXPECTED_WARMUP_STEPS = (
    400
)

EXPECTED_MIN_LEARNING_RATE = (
    3e-5
)


EXPECTED_MAX_GRAD_NORM = (
    1.0
)


EXPECTED_LOG_EVERY_STEPS = (
    10
)

EXPECTED_EVAL_EVERY_STEPS = (
    200
)

EXPECTED_EVAL_BATCHES = (
    20
)

EXPECTED_SAVE_EVERY_STEPS = (
    1_000
)


# ============================================================
# JSON loader
# ============================================================

def load_json(
    path,
):
    """
    Load one JSON configuration file.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "Training Configuration Test"
    )

    print(
        "==========================="
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

    # ========================================================
    # Training section
    # ========================================================

    training_section = (
        training_config[
            "training"
        ]
    )

    batch_size = (
        training_section[
            "batch_size"
        ]
    )

    context_length = (
        training_section[
            "context_length"
        ]
    )

    seed = (
        training_section[
            "seed"
        ]
    )

    max_steps = (
        training_section[
            "max_steps"
        ]
    )

    tokens_per_step = (
        batch_size
        *
        context_length
    )

    # ========================================================
    # Data section
    # ========================================================

    data_section = (
        training_config[
            "data"
        ]
    )

    tokenizer_name = (
        data_section[
            "tokenizer"
        ]
    )

    train_path = (
        PROJECT_ROOT
        /
        data_section[
            "train_binary"
        ]
    )

    validation_path = (
        PROJECT_ROOT
        /
        data_section[
            "validation_binary"
        ]
    )

    # ========================================================
    # Optimizer section
    # ========================================================

    optimizer_section = (
        training_config[
            "optimizer"
        ]
    )

    optimizer_name = (
        optimizer_section[
            "name"
        ]
    )

    learning_rate = (
        optimizer_section[
            "learning_rate"
        ]
    )

    beta1 = (
        optimizer_section[
            "beta1"
        ]
    )

    beta2 = (
        optimizer_section[
            "beta2"
        ]
    )

    epsilon = (
        optimizer_section[
            "epsilon"
        ]
    )

    weight_decay = (
        optimizer_section[
            "weight_decay"
        ]
    )

    # ========================================================
    # Scheduler section
    # ========================================================

    scheduler_section = (
        training_config[
            "scheduler"
        ]
    )

    scheduler_name = (
        scheduler_section[
            "name"
        ]
    )

    warmup_steps = (
        scheduler_section[
            "warmup_steps"
        ]
    )

    min_learning_rate = (
        scheduler_section[
            "min_learning_rate"
        ]
    )

    # ========================================================
    # Gradient clipping section
    # ========================================================

    clipping_section = (
        training_config[
            "gradient_clipping"
        ]
    )

    clipping_enabled = (
        clipping_section[
            "enabled"
        ]
    )

    max_grad_norm = (
        clipping_section[
            "max_norm"
        ]
    )

    # ========================================================
    # Logging section
    # ========================================================

    logging_section = (
        training_config[
            "logging"
        ]
    )

    log_every_steps = (
        logging_section[
            "log_every_steps"
        ]
    )

    # ========================================================
    # Validation section
    # ========================================================

    validation_section = (
        training_config[
            "validation"
        ]
    )

    eval_every_steps = (
        validation_section[
            "eval_every_steps"
        ]
    )

    eval_batches = (
        validation_section[
            "eval_batches"
        ]
    )

    # ========================================================
    # Checkpoint section
    # ========================================================

    checkpoint_section = (
        training_config[
            "checkpointing"
        ]
    )

    save_every_steps = (
        checkpoint_section[
            "save_every_steps"
        ]
    )

    checkpoint_directory = (
        checkpoint_section[
            "directory"
        ]
    )

    # ========================================================
    # Checks
    # ========================================================

    batch_size_passed = (
        batch_size
        ==
        EXPECTED_BATCH_SIZE
    )

    context_matches_model = (
        context_length
        ==
        model_config[
            "context_length"
        ]
        ==
        EXPECTED_CONTEXT_LENGTH
    )

    tokens_per_step_passed = (
        tokens_per_step
        ==
        EXPECTED_TOKENS_PER_STEP
    )

    seed_passed = (
        seed
        == 42
    )

    max_steps_passed = (
        max_steps
        ==
        EXPECTED_MAX_STEPS
    )

    tokenizer_passed = (
        tokenizer_name
        ==
        EXPECTED_TOKENIZER
    )

    optimizer_passed = (
        optimizer_name
        ==
        EXPECTED_OPTIMIZER
    )

    learning_rate_passed = (
        learning_rate
        ==
        EXPECTED_LEARNING_RATE
    )

    beta1_passed = (
        beta1
        ==
        EXPECTED_BETA1
    )

    beta2_passed = (
        beta2
        ==
        EXPECTED_BETA2
    )

    epsilon_passed = (
        epsilon
        ==
        EXPECTED_EPSILON
    )

    weight_decay_passed = (
        weight_decay
        ==
        EXPECTED_WEIGHT_DECAY
    )

    scheduler_name_passed = (
        scheduler_name
        ==
        EXPECTED_SCHEDULER
    )

    warmup_steps_passed = (
        warmup_steps
        ==
        EXPECTED_WARMUP_STEPS
    )

    min_lr_passed = (
        min_learning_rate
        ==
        EXPECTED_MIN_LEARNING_RATE
    )

    scheduler_range_passed = (
        warmup_steps
        <
        max_steps
    )

    scheduler_lr_range_passed = (
        0.0
        <=
        min_learning_rate
        <=
        learning_rate
    )

    gradient_clip_passed = (
        clipping_enabled
        and
        max_grad_norm
        ==
        EXPECTED_MAX_GRAD_NORM
    )

    logging_passed = (
        log_every_steps
        ==
        EXPECTED_LOG_EVERY_STEPS
    )

    validation_frequency_passed = (
        eval_every_steps
        ==
        EXPECTED_EVAL_EVERY_STEPS
    )

    validation_batches_passed = (
        eval_batches
        ==
        EXPECTED_EVAL_BATCHES
    )

    checkpoint_frequency_passed = (
        save_every_steps
        ==
        EXPECTED_SAVE_EVERY_STEPS
    )

    checkpoint_directory_passed = (
        isinstance(
            checkpoint_directory,
            str,
        )
        and
        len(
            checkpoint_directory
        )
        > 0
    )

    train_exists = (
        train_path.exists()
    )

    validation_exists = (
        validation_path.exists()
    )

    model_vocab_matches = (
        model_config[
            "vocab_size"
        ]
        ==
        16_000
    )

    # ========================================================
    # Derived training values
    # ========================================================

    total_training_token_exposures = (
        max_steps
        *
        tokens_per_step
    )

    warmup_token_exposures = (
        warmup_steps
        *
        tokens_per_step
    )

    # ========================================================
    # Display configuration
    # ========================================================

    print()

    print(
        "Training"
    )

    print(
        "--------"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Context length: "
        f"{context_length}"
    )

    print(
        f"Tokens per optimizer step: "
        f"{tokens_per_step:,}"
    )

    print(
        f"Maximum steps: "
        f"{max_steps:,}"
    )

    print(
        f"Seed: "
        f"{seed}"
    )

    print()

    print(
        "Optimizer"
    )

    print(
        "---------"
    )

    print(
        f"Optimizer: "
        f"{optimizer_name}"
    )

    print(
        f"Maximum learning rate: "
        f"{learning_rate}"
    )

    print(
        f"Beta1: "
        f"{beta1}"
    )

    print(
        f"Beta2: "
        f"{beta2}"
    )

    print(
        f"Epsilon: "
        f"{epsilon}"
    )

    print(
        f"Weight decay: "
        f"{weight_decay}"
    )

    print()

    print(
        "Scheduler"
    )

    print(
        "---------"
    )

    print(
        f"Scheduler: "
        f"{scheduler_name}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps:,}"
    )

    print(
        f"Minimum learning rate: "
        f"{min_learning_rate}"
    )

    print()

    print(
        "Gradient Clipping"
    )

    print(
        "-----------------"
    )

    print(
        f"Enabled: "
        f"{clipping_enabled}"
    )

    print(
        f"Maximum gradient norm: "
        f"{max_grad_norm}"
    )

    print()

    print(
        "Run Schedule"
    )

    print(
        "------------"
    )

    print(
        f"Log every: "
        f"{log_every_steps} steps"
    )

    print(
        f"Validate every: "
        f"{eval_every_steps} steps"
    )

    print(
        f"Validation batches: "
        f"{eval_batches}"
    )

    print(
        f"Checkpoint every: "
        f"{save_every_steps:,} steps"
    )

    print(
        f"Checkpoint directory: "
        f"{checkpoint_directory}"
    )

    print()

    print(
        "Derived Values"
    )

    print(
        "--------------"
    )

    print(
        f"Warmup token exposures: "
        f"{warmup_token_exposures:,}"
    )

    print(
        f"Total token exposures: "
        f"{total_training_token_exposures:,}"
    )

    # ========================================================
    # Final checks
    # ========================================================

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Batch size = 4: "
        f"{batch_size_passed}"
    )

    print(
        f"Context matches model: "
        f"{context_matches_model}"
    )

    print(
        f"Tokens/step = 1,024: "
        f"{tokens_per_step_passed}"
    )

    print(
        f"Seed = 42: "
        f"{seed_passed}"
    )

    print(
        f"Max steps = 8,197: "
        f"{max_steps_passed}"
    )

    print(
        f"Tokenizer v0.2: "
        f"{tokenizer_passed}"
    )

    print(
        f"Model vocabulary = 16,000: "
        f"{model_vocab_matches}"
    )

    print(
        f"AdamW configured: "
        f"{optimizer_passed}"
    )

    print(
        f"Learning rate = 3e-4: "
        f"{learning_rate_passed}"
    )

    print(
        f"Beta1 = 0.9: "
        f"{beta1_passed}"
    )

    print(
        f"Beta2 = 0.999: "
        f"{beta2_passed}"
    )

    print(
        f"Epsilon = 1e-8: "
        f"{epsilon_passed}"
    )

    print(
        f"Weight decay = 0.01: "
        f"{weight_decay_passed}"
    )

    print(
        f"Scheduler = warmup_cosine: "
        f"{scheduler_name_passed}"
    )

    print(
        f"Warmup steps = 400: "
        f"{warmup_steps_passed}"
    )

    print(
        f"Minimum LR = 3e-5: "
        f"{min_lr_passed}"
    )

    print(
        f"Warmup < max steps: "
        f"{scheduler_range_passed}"
    )

    print(
        f"Minimum LR <= maximum LR: "
        f"{scheduler_lr_range_passed}"
    )

    print(
        f"Gradient clipping = 1.0: "
        f"{gradient_clip_passed}"
    )

    print(
        f"Log every 10 steps: "
        f"{logging_passed}"
    )

    print(
        f"Validate every 200 steps: "
        f"{validation_frequency_passed}"
    )

    print(
        f"Validation batches = 20: "
        f"{validation_batches_passed}"
    )

    print(
        f"Checkpoint every 1,000 steps: "
        f"{checkpoint_frequency_passed}"
    )

    print(
        f"Checkpoint directory valid: "
        f"{checkpoint_directory_passed}"
    )

    print(
        f"Training binary exists: "
        f"{train_exists}"
    )

    print(
        f"Validation binary exists: "
        f"{validation_exists}"
    )

    # ========================================================
    # Overall result
    # ========================================================

    all_passed = (
        batch_size_passed
        and
        context_matches_model
        and
        tokens_per_step_passed
        and
        seed_passed
        and
        max_steps_passed
        and
        tokenizer_passed
        and
        model_vocab_matches
        and
        optimizer_passed
        and
        learning_rate_passed
        and
        beta1_passed
        and
        beta2_passed
        and
        epsilon_passed
        and
        weight_decay_passed
        and
        scheduler_name_passed
        and
        warmup_steps_passed
        and
        min_lr_passed
        and
        scheduler_range_passed
        and
        scheduler_lr_range_passed
        and
        gradient_clip_passed
        and
        logging_passed
        and
        validation_frequency_passed
        and
        validation_batches_passed
        and
        checkpoint_frequency_passed
        and
        checkpoint_directory_passed
        and
        train_exists
        and
        validation_exists
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: REAL PRETRAINING "
            "CONFIGURATION VALIDATED."
        )

    else:

        raise AssertionError(
            "Training configuration "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()