import argparse
import gc
import json
import sys
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

from training.data_loader import (
    TokenBatchLoader,
)

from training.loss import (
    cross_entropy_loss_and_backward,
)

from training.adamw import AdamW

from training.gradient_clipping import (
    clip_grad_norm,
)


# ============================================================
# Frozen v0.5 benchmark identity
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.5.json"
)

DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_v0_4_train.bin"
)

EXPECTED_PARAMETER_COUNT = 16_889_856

BATCH_SIZES = [
    1,
    2,
    4,
]

WARMUP_STEPS = 1
MEASURED_STEPS = 3

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42


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
        ) in model_parameters(
            model
        )
    )


def model_parameters(model):
    from training.parameters import (
        named_parameters,
    )

    return named_parameters(
        model
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


# ============================================================
# One real training step
# ============================================================

def run_training_step(
    *,
    model,
    optimizer,
    loader,
):
    X, Y = loader.get_batch()

    tokens_this_step = int(
        X.size
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    start = time.perf_counter()

    logits = model.forward(
        X
    )

    forward_seconds = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # Loss + dlogits
    # --------------------------------------------------------

    start = time.perf_counter()

    (
        loss,
        grad_logits,
    ) = cross_entropy_loss_and_backward(
        logits,
        Y,
    )

    loss_seconds = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    model.zero_grad()

    start = time.perf_counter()

    model.backward(
        grad_logits
    )

    backward_seconds = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    start = time.perf_counter()

    (
        gradient_norm,
        clip_scale,
    ) = clip_grad_norm(
        model=model,
        max_norm=MAX_GRAD_NORM,
    )

    clipping_seconds = (
        time.perf_counter()
        -
        start
    )

    # --------------------------------------------------------
    # AdamW update
    # --------------------------------------------------------

    start = time.perf_counter()

    optimizer.step()

    optimizer_seconds = (
        time.perf_counter()
        -
        start
    )

    total_seconds = (
        forward_seconds
        +
        loss_seconds
        +
        backward_seconds
        +
        clipping_seconds
        +
        optimizer_seconds
    )

    return {
        "loss":
            float(
                loss
            ),

        "gradient_norm":
            float(
                gradient_norm
            ),

        "clip_scale":
            float(
                clip_scale
            ),

        "tokens":
            tokens_this_step,

        "forward_seconds":
            float(
                forward_seconds
            ),

        "loss_seconds":
            float(
                loss_seconds
            ),

        "backward_seconds":
            float(
                backward_seconds
            ),

        "clipping_seconds":
            float(
                clipping_seconds
            ),

        "optimizer_seconds":
            float(
                optimizer_seconds
            ),

        "total_seconds":
            float(
                total_seconds
            ),

        "tokens_per_second":
            float(
                tokens_this_step
                /
                total_seconds
            ),

        "logits_elements":
            int(
                logits.size
            ),

        "logits_bytes":
            int(
                logits.nbytes
            ),
    }


# ============================================================
# Benchmark one batch size
# ============================================================

def benchmark_batch_size(
    *,
    batch_size,
    config,
    train_path,
):

    print()
    print(
        "=" * 76
    )

    print(
        f"GENERAL-v0.5 — BATCH SIZE "
        f"{batch_size}"
    )

    print(
        "=" * 76
    )

    loader = TokenBatchLoader(
        binary_path=train_path,
        context_length=(
            config[
                "context_length"
            ]
        ),
        batch_size=batch_size,
        seed=SEED,
    )

    model = GPT.from_config(
        config,
        seed=SEED,
    )

    require(
        model.parameter_count()
        ==
        EXPECTED_PARAMETER_COUNT,
        (
            "Unexpected v0.5 parameter "
            "count during batch benchmark."
        ),
    )

    optimizer = AdamW(
        model=model,
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    warmup_results = []
    measured_results = []

    try:

        # ----------------------------------------------------
        # One unmeasured warm-up step
        # ----------------------------------------------------

        print()
        print(
            f"Warm-up steps: "
            f"{WARMUP_STEPS}"
        )

        for step in range(
            1,
            WARMUP_STEPS + 1,
        ):

            result = run_training_step(
                model=model,
                optimizer=optimizer,
                loader=loader,
            )

            warmup_results.append(
                result
            )

            print(
                f"  warm-up {step}: "
                f"loss={result['loss']:.6f} "
                f"throughput="
                f"{result['tokens_per_second']:,.1f} "
                f"tok/s"
            )

        # ----------------------------------------------------
        # Timed measured steps
        # ----------------------------------------------------

        print()
        print(
            f"Measured steps: "
            f"{MEASURED_STEPS}"
        )

        for step in range(
            1,
            MEASURED_STEPS + 1,
        ):

            result = run_training_step(
                model=model,
                optimizer=optimizer,
                loader=loader,
            )

            measured_results.append(
                result
            )

            print()
            print(
                f"Step {step}"
            )

            print(
                f"  Loss: "
                f"{result['loss']:.6f}"
            )

            print(
                f"  Gradient norm: "
                f"{result['gradient_norm']:.4f}"
            )

            print(
                f"  Clip scale: "
                f"{result['clip_scale']:.4f}"
            )

            print(
                f"  Total time: "
                f"{result['total_seconds']:.4f}s"
            )

            print(
                f"  Throughput: "
                f"{result['tokens_per_second']:,.1f} "
                f"tokens/s"
            )

    except MemoryError:

        print()
        print(
            "MEMORY ERROR: "
            f"batch size {batch_size} "
            "could not complete."
        )

        del optimizer
        del model
        del loader

        gc.collect()

        return {
            "batch_size":
                batch_size,

            "success":
                False,

            "reason":
                "MemoryError",
        }

    # ========================================================
    # Numerical checks
    # ========================================================

    numerical_checks = {
        "losses_finite":
            all(
                np.isfinite(
                    result[
                        "loss"
                    ]
                )
                for result
                in measured_results
            ),

        "gradient_norms_finite":
            all(
                np.isfinite(
                    result[
                        "gradient_norm"
                    ]
                )
                for result
                in measured_results
            ),

        "clip_scales_finite":
            all(
                np.isfinite(
                    result[
                        "clip_scale"
                    ]
                )
                and
                0.0
                <
                result[
                    "clip_scale"
                ]
                <=
                1.0
                for result
                in measured_results
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

    require(
        all(
            numerical_checks.values()
        ),
        (
            f"Batch size {batch_size} "
            "failed numerical checks: "
            f"{numerical_checks}"
        ),
    )

    # ========================================================
    # Aggregate timings
    # ========================================================

    timing_keys = [
        "forward_seconds",
        "loss_seconds",
        "backward_seconds",
        "clipping_seconds",
        "optimizer_seconds",
        "total_seconds",
    ]

    average = {
        key:
            float(
                np.mean(
                    [
                        result[
                            key
                        ]
                        for result
                        in measured_results
                    ]
                )
            )
        for key
        in timing_keys
    }

    throughput_values = np.asarray(
        [
            result[
                "tokens_per_second"
            ]
            for result
            in measured_results
        ],
        dtype=np.float64,
    )

    loss_values = np.asarray(
        [
            result[
                "loss"
            ]
            for result
            in measured_results
        ],
        dtype=np.float64,
    )

    tokens_per_step = (
        batch_size
        *
        config[
            "context_length"
        ]
    )

    aggregate_throughput = (
        tokens_per_step
        /
        average[
            "total_seconds"
        ]
    )

    logits_mib = (
        measured_results[
            0
        ][
            "logits_bytes"
        ]
        /
        (1024 ** 2)
    )

    print()
    print(
        "Average"
    )

    print(
        "-------"
    )

    print(
        f"Tokens/step: "
        f"{tokens_per_step:,}"
    )

    print(
        f"Logits alone: "
        f"{logits_mib:,.2f} MiB"
    )

    print()

    print(
        f"Forward:  "
        f"{average['forward_seconds']:.4f}s"
    )

    print(
        f"Loss:     "
        f"{average['loss_seconds']:.4f}s"
    )

    print(
        f"Backward: "
        f"{average['backward_seconds']:.4f}s"
    )

    print(
        f"Clipping: "
        f"{average['clipping_seconds']:.4f}s"
    )

    print(
        f"AdamW:    "
        f"{average['optimizer_seconds']:.4f}s"
    )

    print()

    print(
        f"Total: "
        f"{average['total_seconds']:.4f}s"
    )

    print(
        f"Aggregate throughput: "
        f"{aggregate_throughput:,.1f} tokens/s"
    )

    print(
        f"Mean step throughput: "
        f"{np.mean(throughput_values):,.1f} tokens/s"
    )

    print(
        f"Throughput range: "
        f"{np.min(throughput_values):,.1f} – "
        f"{np.max(throughput_values):,.1f} tokens/s"
    )

    result = {
        "batch_size":
            batch_size,

        "success":
            True,

        "tokens_per_step":
            int(
                tokens_per_step
            ),

        "average_loss":
            float(
                np.mean(
                    loss_values
                )
            ),

        "aggregate_tokens_per_second":
            float(
                aggregate_throughput
            ),

        "mean_step_tokens_per_second":
            float(
                np.mean(
                    throughput_values
                )
            ),

        "minimum_step_tokens_per_second":
            float(
                np.min(
                    throughput_values
                )
            ),

        "maximum_step_tokens_per_second":
            float(
                np.max(
                    throughput_values
                )
            ),

        "logits_mib":
            float(
                logits_mib
            ),

        "numerical_checks":
            numerical_checks,

        **average,
    }

    del measured_results
    del warmup_results
    del optimizer
    del model
    del loader

    gc.collect()

    return result


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark batch sizes 1, 2, and 4 "
            "for the 16.89M-parameter "
            "MathGPT-General-v0.5 model."
        )
    )

    parser.add_argument(
        "--train-binary",
        type=str,
        default=str(
            DEFAULT_TRAIN_PATH
        ),
    )

    return parser.parse_args()


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
        "GENERAL-v0.5 REAL-DATA "
        "BATCH-SIZE BENCHMARK"
    )

    print(
        "=" * 76
    )

    config = load_json(
        MODEL_CONFIG_PATH
    )

    train_path = resolve_project_path(
        args.train_binary
    )

    require(
        train_path.exists(),
        (
            "Training binary missing: "
            f"{train_path}"
        ),
    )

    training_tokens = (
        train_path.stat().st_size
        //
        np.dtype(
            np.uint16
        ).itemsize
    )

    model = GPT.from_config(
        config,
        seed=SEED,
    )

    actual_parameter_count = (
        model.parameter_count()
    )

    del model
    gc.collect()

    require(
        actual_parameter_count
        ==
        EXPECTED_PARAMETER_COUNT,
        (
            "v0.5 model parameter count "
            "does not match frozen identity."
        ),
    )

    print()
    print(
        f"Model parameters: "
        f"{actual_parameter_count:,}"
    )

    print(
        f"Training binary: "
        f"{train_path}"
    )

    print(
        f"Training tokens: "
        f"{training_tokens:,}"
    )

    print(
        f"Context length: "
        f"{config['context_length']}"
    )

    print(
        f"Warm-up steps per batch: "
        f"{WARMUP_STEPS}"
    )

    print(
        f"Measured steps per batch: "
        f"{MEASURED_STEPS}"
    )

    benchmark_results = []

    for batch_size in BATCH_SIZES:

        result = benchmark_batch_size(
            batch_size=batch_size,
            config=config,
            train_path=train_path,
        )

        benchmark_results.append(
            result
        )

    # ========================================================
    # Final comparison
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.5 BATCH-SIZE COMPARISON"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"{'Batch':>5} | "
        f"{'Tokens/Step':>11} | "
        f"{'Sec/Step':>9} | "
        f"{'Tokens/Sec':>12} | "
        f"{'Logits MiB':>10}"
    )

    print(
        "-" * 66
    )

    successful_results = []

    for result in benchmark_results:

        if not result[
            "success"
        ]:

            print(
                f"{result['batch_size']:>5} | "
                f"{'FAILED':>11}"
            )

            continue

        successful_results.append(
            result
        )

        print(
            f"{result['batch_size']:>5} | "
            f"{result['tokens_per_step']:>11,} | "
            f"{result['total_seconds']:>9.4f} | "
            f"{result['aggregate_tokens_per_second']:>12,.1f} | "
            f"{result['logits_mib']:>10.2f}"
        )

    require(
        bool(
            successful_results
        ),
        (
            "No v0.5 batch size "
            "completed successfully."
        ),
    )

    best_result = max(
        successful_results,
        key=lambda item:
            item[
                "aggregate_tokens_per_second"
            ],
    )

    print()
    print(
        "Highest measured throughput:"
    )

    print(
        f"  batch "
        f"{best_result['batch_size']}"
    )

    print(
        f"  "
        f"{best_result['aggregate_tokens_per_second']:,.1f} "
        f"tokens/s"
    )

    print()
    print(
        "Recommended candidate for the "
        "full v0.5 schedule:"
    )

    print(
        f"  batch_size = "
        f"{best_result['batch_size']}"
    )

    print(
        f"  tokens_per_step = "
        f"{best_result['tokens_per_step']:,}"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "  This benchmark selects by measured "
        "throughput only. The full v0.5 "
        "training configuration is not frozen "
        "by this script."
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: GENERAL-v0.5 "
        "BATCH-SIZE BENCHMARK COMPLETE."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()
