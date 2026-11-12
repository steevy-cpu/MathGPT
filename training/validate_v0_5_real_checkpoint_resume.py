import argparse
import hashlib
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

from training.adamw import AdamW

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


# ============================================================
# Frozen v0.5 identity
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

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "general_pretraining_v0.5"
    / "general_pretraining_v0_5_1k_smoke"
    / "mathgpt_step_001000.npz"
)

EXPECTED_PARAMETER_COUNT = 16_889_856
EXPECTED_CHECKPOINT_STEP = 1_000
EXPECTED_CHECKPOINT_TOKENS = 1_024_000

CONTINUATION_STEPS = 5
EXPECTED_FINAL_STEP = 1_005
EXPECTED_FINAL_TOKENS = 1_029_120


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_project_path(value):
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


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


def array_digest(
    digest,
    name,
    array,
):
    array = np.asarray(
        array
    )

    digest.update(
        name.encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            array.dtype
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        json.dumps(
            list(
                array.shape
            )
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        np.ascontiguousarray(
            array
        ).tobytes(
            order="C"
        )
    )


def model_state_sha256(
    model,
):
    digest = hashlib.sha256()

    for (
        name,
        parameter,
        _,
    ) in named_parameters(
        model
    ):

        array_digest(
            digest,
            name,
            parameter,
        )

    return digest.hexdigest()


def optimizer_state_sha256(
    optimizer,
):
    digest = hashlib.sha256()

    digest.update(
        f"step={optimizer.step_count}".encode(
            "utf-8"
        )
    )

    for name in sorted(
        optimizer.first_moments
    ):

        array_digest(
            digest,
            "m:" + name,
            optimizer.first_moments[
                name
            ],
        )

        array_digest(
            digest,
            "v:" + name,
            optimizer.second_moments[
                name
            ],
        )

    return digest.hexdigest()


def loader_state_json(
    loader,
):
    return json.dumps(
        loader.state_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def deterministic_metrics(
    history,
):
    keys = [
        "step",
        "loss",
        "learning_rate",
        "gradient_norm",
        "clip_scale",
        "batch_tokens",
        "tokens_seen",
    ]

    return [
        {
            key:
                record[
                    key
                ]
            for key
            in keys
        }
        for record
        in history
    ]


def model_parameters_finite(
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


def histories_finite(
    history,
):
    return all(
        np.isfinite(
            item[
                "loss"
            ]
        )
        and
        np.isfinite(
            item[
                "gradient_norm"
            ]
        )
        and
        np.isfinite(
            item[
                "learning_rate"
            ]
        )
        and
        np.isfinite(
            item[
                "clip_scale"
            ]
        )
        for item
        in history
    )


def create_runtime(
    *,
    model_config,
    training_config,
    model_seed,
    loader_seed,
):
    optimizer_config = (
        training_config[
            "optimizer"
        ]
    )

    training = (
        training_config[
            "training"
        ]
    )

    model = GPT.from_config(
        model_config,
        seed=model_seed,
    )

    require(
        model.parameter_count()
        ==
        EXPECTED_PARAMETER_COUNT,
        (
            "Unexpected GENERAL-v0.5 "
            "parameter count."
        ),
    )

    optimizer = AdamW(
        model=model,
        learning_rate=float(
            optimizer_config[
                "learning_rate"
            ]
        ),
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

    train_path = resolve_project_path(
        training_config[
            "data"
        ][
            "train_binary"
        ]
    )

    loader = TokenBatchLoader(
        binary_path=train_path,
        context_length=int(
            training[
                "context_length"
            ]
        ),
        batch_size=int(
            training[
                "batch_size"
            ]
        ),
        seed=loader_seed,
    )

    return (
        model,
        optimizer,
        loader,
    )


def continue_five_steps(
    *,
    model,
    optimizer,
    loader,
    training_config,
    starting_tokens_seen,
):
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

    clipping = (
        training_config[
            "gradient_clipping"
        ]
    )

    optimizer_config = (
        training_config[
            "optimizer"
        ]
    )

    history, tokens_seen, _ = (
        run_training_steps(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            num_steps=CONTINUATION_STEPS,
            max_learning_rate=float(
                optimizer_config[
                    "learning_rate"
                ]
            ),
            min_learning_rate=float(
                scheduler[
                    "min_learning_rate"
                ]
            ),
            warmup_steps=int(
                scheduler[
                    "warmup_steps"
                ]
            ),
            total_steps=int(
                training[
                    "max_steps"
                ]
            ),
            max_grad_norm=float(
                clipping[
                    "max_norm"
                ]
            ),
            starting_tokens_seen=(
                starting_tokens_seen
            ),
            progress_every=1,
        )
    )

    return (
        history,
        tokens_seen,
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Validate deterministic resume "
            "from the real GENERAL-v0.5 "
            "step-1000 smoke checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=str(
            DEFAULT_CHECKPOINT_PATH
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    checkpoint_path = resolve_project_path(
        args.checkpoint
    )

    require(
        checkpoint_path.exists(),
        (
            "Checkpoint missing: "
            f"{checkpoint_path}"
        ),
    )

    print()
    print("=" * 80)
    print(
        "GENERAL-v0.5 REAL CHECKPOINT "
        "RESUME VALIDATION"
    )
    print("=" * 80)

    print()
    print(
        "Checkpoint:"
    )
    print(
        f"  {checkpoint_path}"
    )

    checkpoint_sha = file_sha256(
        checkpoint_path
    )

    print()
    print(
        "Checkpoint SHA-256:"
    )
    print(
        f"  {checkpoint_sha}"
    )

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    validation_config = (
        training_config[
            "validation"
        ]
    )

    validation_path = resolve_project_path(
        training_config[
            "data"
        ][
            "validation_binary"
        ]
    )

    # ========================================================
    # Restore A
    # ========================================================

    print()
    print("=" * 80)
    print("RESTORE A")
    print("=" * 80)

    (
        model_a,
        optimizer_a,
        loader_a,
    ) = create_runtime(
        model_config=model_config,
        training_config=training_config,
        model_seed=111,
        loader_seed=111,
    )

    fresh_loader_a = loader_state_json(
        loader_a
    )

    (
        metadata_a,
        restored_step_a,
        restored_tokens_a,
    ) = restore_training_checkpoint(
        model=model_a,
        optimizer=optimizer_a,
        path=checkpoint_path,
        train_loader=loader_a,
        strict=True,
    )

    restored_loader_a = (
        loader_state_json(
            loader_a
        )
    )

    (
        history_a,
        final_tokens_a,
    ) = continue_five_steps(
        model=model_a,
        optimizer=optimizer_a,
        loader=loader_a,
        training_config=training_config,
        starting_tokens_seen=(
            restored_tokens_a
        ),
    )

    validation_a = evaluate_model(
        model=model_a,
        binary_path=validation_path,
        context_length=int(
            training_config[
                "training"
            ][
                "context_length"
            ]
        ),
        batch_size=int(
            training_config[
                "training"
            ][
                "batch_size"
            ]
        ),
        eval_batches=int(
            validation_config[
                "eval_batches"
            ]
        ),
        seed=int(
            validation_config[
                "seed"
            ]
        ),
    )

    model_sha_a = model_state_sha256(
        model_a
    )

    optimizer_sha_a = (
        optimizer_state_sha256(
            optimizer_a
        )
    )

    final_loader_a = (
        loader_state_json(
            loader_a
        )
    )

    metrics_a = deterministic_metrics(
        history_a
    )

    # ========================================================
    # Restore B from different fresh state
    # ========================================================

    print()
    print("=" * 80)
    print("RESTORE B")
    print("=" * 80)

    (
        model_b,
        optimizer_b,
        loader_b,
    ) = create_runtime(
        model_config=model_config,
        training_config=training_config,
        model_seed=999,
        loader_seed=999,
    )

    fresh_loader_b = loader_state_json(
        loader_b
    )

    (
        metadata_b,
        restored_step_b,
        restored_tokens_b,
    ) = restore_training_checkpoint(
        model=model_b,
        optimizer=optimizer_b,
        path=checkpoint_path,
        train_loader=loader_b,
        strict=True,
    )

    restored_loader_b = (
        loader_state_json(
            loader_b
        )
    )

    (
        history_b,
        final_tokens_b,
    ) = continue_five_steps(
        model=model_b,
        optimizer=optimizer_b,
        loader=loader_b,
        training_config=training_config,
        starting_tokens_seen=(
            restored_tokens_b
        ),
    )

    validation_b = evaluate_model(
        model=model_b,
        binary_path=validation_path,
        context_length=int(
            training_config[
                "training"
            ][
                "context_length"
            ]
        ),
        batch_size=int(
            training_config[
                "training"
            ][
                "batch_size"
            ]
        ),
        eval_batches=int(
            validation_config[
                "eval_batches"
            ]
        ),
        seed=int(
            validation_config[
                "seed"
            ]
        ),
    )

    model_sha_b = model_state_sha256(
        model_b
    )

    optimizer_sha_b = (
        optimizer_state_sha256(
            optimizer_b
        )
    )

    final_loader_b = (
        loader_state_json(
            loader_b
        )
    )

    metrics_b = deterministic_metrics(
        history_b
    )

    # ========================================================
    # Checks
    # ========================================================

    extra_state_a = metadata_a.get(
        "extra_state",
        {},
    )

    extra_state_b = metadata_b.get(
        "extra_state",
        {},
    )

    checks = {
        "checkpoint_parameter_count":
            (
                int(
                    metadata_a[
                        "parameter_count"
                    ]
                )
                ==
                EXPECTED_PARAMETER_COUNT
                ==
                int(
                    metadata_b[
                        "parameter_count"
                    ]
                )
            ),

        "checkpoint_step_1000":
            (
                restored_step_a
                ==
                restored_step_b
                ==
                EXPECTED_CHECKPOINT_STEP
            ),

        "checkpoint_tokens_1024000":
            (
                restored_tokens_a
                ==
                restored_tokens_b
                ==
                EXPECTED_CHECKPOINT_TOKENS
            ),

        "loader_state_present":
            (
                "train_loader_state"
                in
                extra_state_a
                and
                "train_loader_state"
                in
                extra_state_b
            ),

        "fresh_loaders_differ":
            (
                fresh_loader_a
                !=
                fresh_loader_b
            ),

        "restore_overwrites_loader_rng":
            (
                restored_loader_a
                ==
                restored_loader_b
                and
                restored_loader_a
                !=
                fresh_loader_a
                and
                restored_loader_b
                !=
                fresh_loader_b
            ),

        "steps_1001_1005_bit_exact":
            (
                metrics_a
                ==
                metrics_b
            ),

        "final_model_bit_exact":
            (
                model_sha_a
                ==
                model_sha_b
            ),

        "final_optimizer_bit_exact":
            (
                optimizer_sha_a
                ==
                optimizer_sha_b
            ),

        "final_loader_bit_exact":
            (
                final_loader_a
                ==
                final_loader_b
            ),

        "final_step_1005":
            (
                optimizer_a.step_count
                ==
                optimizer_b.step_count
                ==
                EXPECTED_FINAL_STEP
            ),

        "final_tokens_1029120":
            (
                final_tokens_a
                ==
                final_tokens_b
                ==
                EXPECTED_FINAL_TOKENS
            ),

        "history_a_finite":
            histories_finite(
                history_a
            ),

        "history_b_finite":
            histories_finite(
                history_b
            ),

        "model_a_finite":
            model_parameters_finite(
                model_a
            ),

        "model_b_finite":
            model_parameters_finite(
                model_b
            ),

        "optimizer_a_finite":
            optimizer_state_finite(
                optimizer_a
            ),

        "optimizer_b_finite":
            optimizer_state_finite(
                optimizer_b
            ),

        "validation_a_finite":
            (
                np.isfinite(
                    validation_a[
                        "loss"
                    ]
                )
                and
                np.isfinite(
                    validation_a[
                        "perplexity"
                    ]
                )
            ),

        "validation_b_finite":
            (
                np.isfinite(
                    validation_b[
                        "loss"
                    ]
                )
                and
                np.isfinite(
                    validation_b[
                        "perplexity"
                    ]
                )
            ),

        "validation_bit_exact":
            (
                validation_a[
                    "loss"
                ]
                ==
                validation_b[
                    "loss"
                ]
                and
                validation_a[
                    "perplexity"
                ]
                ==
                validation_b[
                    "perplexity"
                ]
            ),
    }

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Steps 1001–1005"
    )
    print(
        "---------------"
    )

    for item_a, item_b in zip(
        metrics_a,
        metrics_b,
    ):

        print(
            f"Step {item_a['step']}: "
            f"A loss={item_a['loss']:.9f} "
            f"B loss={item_b['loss']:.9f} "
            f"LR={item_a['learning_rate']:.10f}"
        )

    print()
    print(
        "Validation after step 1005"
    )
    print(
        "--------------------------"
    )

    print(
        f"Run A loss: "
        f"{validation_a['loss']:.9f}"
    )

    print(
        f"Run B loss: "
        f"{validation_b['loss']:.9f}"
    )

    print()
    print(
        "Final state fingerprints"
    )
    print(
        "------------------------"
    )

    print(
        f"Model A:     {model_sha_a}"
    )

    print(
        f"Model B:     {model_sha_b}"
    )

    print(
        f"Optimizer A: {optimizer_sha_a}"
    )

    print(
        f"Optimizer B: {optimizer_sha_b}"
    )

    print()
    print(
        "Checks"
    )
    print(
        "------"
    )

    for name, passed in checks.items():

        print(
            f"{name:<40} "
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
            "GENERAL-v0.5 real checkpoint "
            f"resume validation failed: {failed}"
        )

    print()
    print("=" * 80)
    print(
        "SUCCESS: GENERAL-v0.5 REAL "
        "STEP-1000 CHECKPOINT RESUME IS "
        "DETERMINISTIC."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
