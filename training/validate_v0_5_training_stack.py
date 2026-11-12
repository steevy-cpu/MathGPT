import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time

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
# Frozen smoke-test identity
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.5.json"
)

EXPECTED_PARAMETER_COUNT = 16_889_856

CONTEXT_LENGTH = 256
BATCH_SIZE = 1
TOKENS_PER_STEP = (
    CONTEXT_LENGTH
    *
    BATCH_SIZE
)

MAX_LR = 0.0003
MIN_LR = 0.00003
WARMUP_STEPS = 2
SCHEDULE_TOTAL_STEPS = 20
MAX_GRAD_NORM = 1.0

FIRST_SEGMENT_STEPS = 2
SECOND_SEGMENT_STEPS = 2

EXPECTED_CHECKPOINT_STEP = 2
EXPECTED_CHECKPOINT_TOKENS = (
    EXPECTED_CHECKPOINT_STEP
    *
    TOKENS_PER_STEP
)

EXPECTED_FINAL_STEP = 4
EXPECTED_FINAL_TOKENS = (
    EXPECTED_FINAL_STEP
    *
    TOKENS_PER_STEP
)


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


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(message)


def resolve_project_path(value):
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def discover_v0_4_train_binary():
    """
    Prefer the expected simple filename, then
    fall back to a conservative v0.4 train-bin
    search.
    """

    tokenized_dir = (
        PROJECT_ROOT
        / "data"
        / "tokenized"
    )

    preferred = (
        tokenized_dir
        / "general_v0_4_train.bin"
    )

    if preferred.exists():
        return preferred

    candidates = sorted(
        path
        for path
        in tokenized_dir.glob(
            "*.bin"
        )
        if (
            "v0_4"
            in
            path.name.lower()
            and
            "train"
            in
            path.name.lower()
            and
            "validation"
            not in
            path.name.lower()
            and
            "val"
            not in
            path.name.lower()
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover the "
            "General-v0.4 training binary. "
            "Pass --train-binary explicitly."
        )

    raise RuntimeError(
        "Multiple possible General-v0.4 "
        "training binaries were found:\n"
        +
        "\n".join(
            str(path)
            for path
            in candidates
        )
        +
        "\nPass --train-binary explicitly."
    )


def create_model(
    config,
    seed,
):
    return GPT.from_config(
        config,
        seed=seed,
    )


def create_optimizer(
    model,
):
    return AdamW(
        model=model,
        learning_rate=MAX_LR,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )


def create_loader(
    binary_path,
    seed,
):
    return TokenBatchLoader(
        binary_path=binary_path,
        context_length=(
            CONTEXT_LENGTH
        ),
        batch_size=BATCH_SIZE,
        seed=seed,
    )


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

    contiguous = np.ascontiguousarray(
        array
    )

    digest.update(
        contiguous.tobytes(
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
    """
    Ignore wall-clock throughput fields.
    """

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


def histories_finite(
    history,
):
    return all(
        np.isfinite(
            record[
                "loss"
            ]
        )
        and
        np.isfinite(
            record[
                "gradient_norm"
            ]
        )
        and
        np.isfinite(
            record[
                "learning_rate"
            ]
        )
        and
        np.isfinite(
            record[
                "clip_scale"
            ]
        )
        for record
        in history
    )


def all_model_parameters_finite(
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


def all_optimizer_state_finite(
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


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "GENERAL-v0.5 full-model training "
            "stack smoke test with exact "
            "checkpoint/resume verification."
        )
    )

    parser.add_argument(
        "--train-binary",
        type=str,
        default=None,
        help=(
            "General-v0.4 training token binary. "
            "If omitted, the script attempts "
            "safe auto-discovery."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print()
    print("=" * 78)
    print(
        "GENERAL-v0.5 TRAINING STACK "
        "SMOKE + EXACT RESUME"
    )
    print("=" * 78)

    config = load_json(
        MODEL_CONFIG_PATH
    )

    if args.train_binary is None:

        binary_path = (
            discover_v0_4_train_binary()
        )

    else:

        binary_path = (
            resolve_project_path(
                args.train_binary
            )
        )

    require(
        binary_path.exists(),
        (
            "Training binary does not exist: "
            f"{binary_path}"
        ),
    )

    binary_tokens = (
        binary_path.stat().st_size
        //
        np.dtype(
            np.uint16
        ).itemsize
    )

    minimum_tokens = (
        CONTEXT_LENGTH
        +
        2
    )

    require(
        binary_tokens
        >=
        minimum_tokens,
        (
            "Training binary is too small for "
            "the v0.5 smoke test."
        ),
    )

    print()
    print("Model config:")
    print(f"  {MODEL_CONFIG_PATH}")

    print()
    print("Training binary:")
    print(f"  {binary_path}")

    print(
        f"  tokens: {binary_tokens:,}"
    )

    print()
    print("Smoke configuration")
    print("-------------------")
    print(
        f"Batch size:       {BATCH_SIZE}"
    )
    print(
        f"Context length:   {CONTEXT_LENGTH}"
    )
    print(
        f"Tokens per step:  {TOKENS_PER_STEP}"
    )
    print(
        f"Max LR:           {MAX_LR:.7f}"
    )
    print(
        f"Min LR:           {MIN_LR:.7f}"
    )
    print(
        f"Warmup steps:     {WARMUP_STEPS}"
    )
    print(
        f"Schedule steps:   {SCHEDULE_TOTAL_STEPS}"
    )
    print(
        f"Gradient clip:    {MAX_GRAD_NORM}"
    )

    # ========================================================
    # Continuous reference run
    # ========================================================

    print()
    print("=" * 78)
    print("REFERENCE RUN A")
    print("=" * 78)

    with tempfile.TemporaryDirectory(
        prefix="mathgpt_v0_5_smoke_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        checkpoint_dir = (
            temp_dir
            / "checkpoints"
        )

        model_a = create_model(
            config,
            seed=42,
        )

        require(
            model_a.parameter_count()
            ==
            EXPECTED_PARAMETER_COUNT,
            (
                "Unexpected v0.5 parameter "
                "count in smoke test."
            ),
        )

        optimizer_a = (
            create_optimizer(
                model_a
            )
        )

        loader_a = create_loader(
            binary_path,
            seed=42,
        )

        start_a = time.perf_counter()

        (
            history_a_first,
            tokens_seen_a,
            checkpoints,
        ) = run_training_steps(
            model=model_a,
            optimizer=optimizer_a,
            train_loader=loader_a,
            num_steps=(
                FIRST_SEGMENT_STEPS
            ),
            max_learning_rate=MAX_LR,
            min_learning_rate=MIN_LR,
            warmup_steps=WARMUP_STEPS,
            total_steps=(
                SCHEDULE_TOTAL_STEPS
            ),
            max_grad_norm=(
                MAX_GRAD_NORM
            ),
            starting_tokens_seen=0,
            progress_every=1,
            checkpoint_every=2,
            checkpoint_directory=(
                checkpoint_dir
            ),
            checkpoint_prefix=(
                "v0_5_smoke"
            ),
        )

        require(
            len(
                checkpoints
            )
            ==
            1,
            (
                "Expected exactly one smoke "
                "checkpoint after step 2."
            ),
        )

        checkpoint_path = (
            checkpoints[
                0
            ]
        )

        loader_state_at_checkpoint = (
            loader_state_json(
                loader_a
            )
        )

        (
            history_a_after,
            tokens_seen_a_final,
            _,
        ) = run_training_steps(
            model=model_a,
            optimizer=optimizer_a,
            train_loader=loader_a,
            num_steps=(
                SECOND_SEGMENT_STEPS
            ),
            max_learning_rate=MAX_LR,
            min_learning_rate=MIN_LR,
            warmup_steps=WARMUP_STEPS,
            total_steps=(
                SCHEDULE_TOTAL_STEPS
            ),
            max_grad_norm=(
                MAX_GRAD_NORM
            ),
            starting_tokens_seen=(
                tokens_seen_a
            ),
            progress_every=1,
        )

        elapsed_a = (
            time.perf_counter()
            -
            start_a
        )

        reference_model_sha = (
            model_state_sha256(
                model_a
            )
        )

        reference_optimizer_sha = (
            optimizer_state_sha256(
                optimizer_a
            )
        )

        reference_loader_state = (
            loader_state_json(
                loader_a
            )
        )

        reference_after_metrics = (
            deterministic_metrics(
                history_a_after
            )
        )

        reference_final_step = (
            optimizer_a.step_count
        )

        reference_parameters_finite = (
            all_model_parameters_finite(
                model_a
            )
        )

        reference_optimizer_finite = (
            all_optimizer_state_finite(
                optimizer_a
            )
        )

        # Free the full reference state before
        # constructing the resumed model.
        del history_a_first
        del model_a
        del optimizer_a
        del loader_a

        gc.collect()

        # ====================================================
        # Resumed run B
        # ====================================================

        print()
        print("=" * 78)
        print("RESUMED RUN B")
        print("=" * 78)

        model_b = create_model(
            config,
            seed=999,
        )

        optimizer_b = (
            create_optimizer(
                model_b
            )
        )

        loader_b = create_loader(
            binary_path,
            seed=999,
        )

        fresh_loader_state = (
            loader_state_json(
                loader_b
            )
        )

        (
            metadata,
            restored_step,
            restored_tokens_seen,
        ) = restore_training_checkpoint(
            model=model_b,
            optimizer=optimizer_b,
            path=checkpoint_path,
            train_loader=loader_b,
            strict=True,
        )

        restored_loader_state = (
            loader_state_json(
                loader_b
            )
        )

        start_b = time.perf_counter()

        (
            history_b_after,
            tokens_seen_b_final,
            _,
        ) = run_training_steps(
            model=model_b,
            optimizer=optimizer_b,
            train_loader=loader_b,
            num_steps=(
                SECOND_SEGMENT_STEPS
            ),
            max_learning_rate=MAX_LR,
            min_learning_rate=MIN_LR,
            warmup_steps=WARMUP_STEPS,
            total_steps=(
                SCHEDULE_TOTAL_STEPS
            ),
            max_grad_norm=(
                MAX_GRAD_NORM
            ),
            starting_tokens_seen=(
                restored_tokens_seen
            ),
            progress_every=1,
        )

        elapsed_b = (
            time.perf_counter()
            -
            start_b
        )

        resumed_model_sha = (
            model_state_sha256(
                model_b
            )
        )

        resumed_optimizer_sha = (
            optimizer_state_sha256(
                optimizer_b
            )
        )

        resumed_loader_state = (
            loader_state_json(
                loader_b
            )
        )

        resumed_after_metrics = (
            deterministic_metrics(
                history_b_after
            )
        )

        # ====================================================
        # Final checks
        # ====================================================

        checks = {
            "parameter_count":
                (
                    model_b.parameter_count()
                    ==
                    EXPECTED_PARAMETER_COUNT
                ),

            "checkpoint_exists":
                checkpoint_path.exists(),

            "checkpoint_step_restored":
                (
                    restored_step
                    ==
                    EXPECTED_CHECKPOINT_STEP
                ),

            "checkpoint_tokens_restored":
                (
                    restored_tokens_seen
                    ==
                    EXPECTED_CHECKPOINT_TOKENS
                ),

            "loader_state_present_in_checkpoint":
                (
                    "train_loader_state"
                    in
                    metadata.get(
                        "extra_state",
                        {},
                    )
                ),

            "fresh_loader_differs":
                (
                    fresh_loader_state
                    !=
                    loader_state_at_checkpoint
                ),

            "loader_restored_exactly":
                (
                    restored_loader_state
                    ==
                    loader_state_at_checkpoint
                ),

            "steps_3_4_metrics_bit_exact":
                (
                    resumed_after_metrics
                    ==
                    reference_after_metrics
                ),

            "final_model_bit_exact":
                (
                    resumed_model_sha
                    ==
                    reference_model_sha
                ),

            "final_optimizer_bit_exact":
                (
                    resumed_optimizer_sha
                    ==
                    reference_optimizer_sha
                ),

            "final_loader_state_bit_exact":
                (
                    resumed_loader_state
                    ==
                    reference_loader_state
                ),

            "reference_final_step":
                (
                    reference_final_step
                    ==
                    EXPECTED_FINAL_STEP
                ),

            "resumed_final_step":
                (
                    optimizer_b.step_count
                    ==
                    EXPECTED_FINAL_STEP
                ),

            "reference_token_accounting":
                (
                    tokens_seen_a_final
                    ==
                    EXPECTED_FINAL_TOKENS
                ),

            "resumed_token_accounting":
                (
                    tokens_seen_b_final
                    ==
                    EXPECTED_FINAL_TOKENS
                ),

            "reference_metrics_finite":
                histories_finite(
                    history_a_after
                ),

            "resumed_metrics_finite":
                histories_finite(
                    history_b_after
                ),

            "reference_parameters_finite":
                reference_parameters_finite,

            "reference_optimizer_finite":
                reference_optimizer_finite,

            "resumed_parameters_finite":
                all_model_parameters_finite(
                    model_b
                ),

            "resumed_optimizer_finite":
                all_optimizer_state_finite(
                    optimizer_b
                ),
        }

        print()
        print("Checkpoint")
        print("----------")
        print(
            f"Path: {checkpoint_path}"
        )
        print(
            f"Restored step: "
            f"{restored_step}"
        )
        print(
            f"Restored tokens: "
            f"{restored_tokens_seen:,}"
        )

        print()
        print(
            "Reference steps 3–4"
        )
        print(
            "-------------------"
        )

        for record in (
            reference_after_metrics
        ):
            print(
                f"Step {record['step']}: "
                f"loss={record['loss']:.9f} "
                f"lr={record['learning_rate']:.10f} "
                f"grad={record['gradient_norm']:.6f}"
            )

        print()
        print(
            "Resumed steps 3–4"
        )
        print(
            "-----------------"
        )

        for record in (
            resumed_after_metrics
        ):
            print(
                f"Step {record['step']}: "
                f"loss={record['loss']:.9f} "
                f"lr={record['learning_rate']:.10f} "
                f"grad={record['gradient_norm']:.6f}"
            )

        print()
        print("Exact-state fingerprints")
        print("------------------------")
        print(
            "Reference model:    "
            f"{reference_model_sha}"
        )
        print(
            "Resumed model:      "
            f"{resumed_model_sha}"
        )
        print(
            "Reference optimizer:"
            f" {reference_optimizer_sha}"
        )
        print(
            "Resumed optimizer:  "
            f"{resumed_optimizer_sha}"
        )

        print()
        print("Checks")
        print("------")

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
                "GENERAL-v0.5 training-stack "
                "validation failed: "
                f"{failed}"
            )

        total_reference_tokens = (
            EXPECTED_FINAL_TOKENS
        )

        reference_throughput = (
            total_reference_tokens
            /
            elapsed_a
            if elapsed_a > 0.0
            else float(
                "inf"
            )
        )

        resumed_segment_tokens = (
            SECOND_SEGMENT_STEPS
            *
            TOKENS_PER_STEP
        )

        resumed_throughput = (
            resumed_segment_tokens
            /
            elapsed_b
            if elapsed_b > 0.0
            else float(
                "inf"
            )
        )

        print()
        print("Smoke throughput")
        print("----------------")
        print(
            "Reference 4-step run: "
            f"{reference_throughput:,.1f} "
            "tokens/s"
        )
        print(
            "Resumed 2-step segment: "
            f"{resumed_throughput:,.1f} "
            "tokens/s"
        )

        print()
        print("=" * 78)
        print(
            "SUCCESS: GENERAL-v0.5 TRAINING "
            "STACK + EXACT RESUME PASSED."
        )
        print("=" * 78)


if __name__ == "__main__":
    main()
