import gc
import json
import sys
import time

from pathlib import Path

import numpy as np


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
# Paths
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_train.bin"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "general_train_metadata.json"
)


# ============================================================
# Benchmark settings
# ============================================================

BATCH_SIZES = [
    1,
    2,
    4,
    8,
]

MEASURED_STEPS = 3

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

SEED = 42


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


def run_training_step(
    model,
    optimizer,
    loader,
):
    """
    Run one complete real-data
    training step and time each stage.
    """

    X, Y = loader.get_batch()

    # ========================================================
    # Forward
    # ========================================================

    start = time.perf_counter()

    logits = model.forward(
        X
    )

    forward_seconds = (
        time.perf_counter()
        - start
    )

    # ========================================================
    # Loss
    # ========================================================

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
        - start
    )

    # ========================================================
    # Backward
    # ========================================================

    model.zero_grad()

    start = time.perf_counter()

    model.backward(
        grad_logits
    )

    backward_seconds = (
        time.perf_counter()
        - start
    )

    # ========================================================
    # Gradient clipping
    # ========================================================

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
        - start
    )

    # ========================================================
    # AdamW
    # ========================================================

    start = time.perf_counter()

    optimizer.step()

    optimizer_seconds = (
        time.perf_counter()
        - start
    )

    total_seconds = (
        forward_seconds
        + loss_seconds
        + backward_seconds
        + clipping_seconds
        + optimizer_seconds
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

        "forward_seconds":
            forward_seconds,

        "loss_seconds":
            loss_seconds,

        "backward_seconds":
            backward_seconds,

        "clipping_seconds":
            clipping_seconds,

        "optimizer_seconds":
            optimizer_seconds,

        "total_seconds":
            total_seconds,

        "logits_elements":
            int(
                logits.size
            ),

        "logits_bytes":
            int(
                logits.nbytes
            ),
    }


def benchmark_batch_size(
    batch_size,
    config,
):
    """
    Benchmark a fresh full-size GPT.
    """

    print()
    print("=" * 70)

    print(
        f"BATCH SIZE {batch_size}"
    )

    print("=" * 70)

    loader = TokenBatchLoader(
        binary_path=TRAIN_PATH,
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

    optimizer = AdamW(
        model=model,
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    results = []

    try:

        for step in range(
            1,
            MEASURED_STEPS + 1,
        ):

            result = run_training_step(
                model=model,
                optimizer=optimizer,
                loader=loader,
            )

            results.append(
                result
            )

            tokens_this_step = (
                batch_size
                *
                config[
                    "context_length"
                ]
            )

            step_tokens_per_second = (
                tokens_this_step
                /
                result[
                    "total_seconds"
                ]
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
                f"{step_tokens_per_second:,.1f} "
                f"tokens/s"
            )

    except MemoryError:

        print()

        print(
            "MEMORY ERROR: "
            f"batch size {batch_size} "
            "could not complete."
        )

        return {
            "batch_size":
                batch_size,

            "success":
                False,

            "reason":
                "MemoryError",
        }

    # ========================================================
    # Average timings
    # ========================================================

    average = {}

    timing_keys = [
        "forward_seconds",
        "loss_seconds",
        "backward_seconds",
        "clipping_seconds",
        "optimizer_seconds",
        "total_seconds",
    ]

    for key in timing_keys:

        average[
            key
        ] = float(
            np.mean(
                [
                    result[
                        key
                    ]
                    for result
                    in results
                ]
            )
        )

    average_loss = float(
        np.mean(
            [
                result[
                    "loss"
                ]
                for result
                in results
            ]
        )
    )

    tokens_per_step = (
        batch_size
        *
        config[
            "context_length"
        ]
    )

    tokens_per_second = (
        tokens_per_step
        /
        average[
            "total_seconds"
        ]
    )

    logits_bytes = (
        results[
            0
        ][
            "logits_bytes"
        ]
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
        f"{logits_bytes / (1024 ** 2):,.2f} MiB"
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
        f"Throughput: "
        f"{tokens_per_second:,.1f} tokens/s"
    )

    result = {
        "batch_size":
            batch_size,

        "success":
            True,

        "average_loss":
            average_loss,

        "tokens_per_step":
            tokens_per_step,

        "tokens_per_second":
            tokens_per_second,

        "logits_mib":
            (
                logits_bytes
                /
                (1024 ** 2)
            ),

        **average,
    }

    # Explicitly release large arrays/model
    # before testing the next batch size.
    del optimizer
    del model
    del loader

    gc.collect()

    return result


def main():

    print()
    print("=" * 70)

    print(
        "MATHGPT REAL-DATA BATCH-SIZE BENCHMARK"
    )

    print("=" * 70)

    config = load_json(
        CONFIG_PATH
    )

    metadata = load_json(
        METADATA_PATH
    )

    print()

    print(
        f"Model parameters: "
        f"8,110,848"
    )

    print(
        f"Training tokens: "
        f"{metadata['tokens']:,}"
    )

    print(
        f"Context length: "
        f"{config['context_length']}"
    )

    print(
        f"Steps measured per batch size: "
        f"{MEASURED_STEPS}"
    )

    benchmark_results = []

    for batch_size in BATCH_SIZES:

        result = benchmark_batch_size(
            batch_size=batch_size,
            config=config,
        )

        benchmark_results.append(
            result
        )

    # ========================================================
    # Final comparison
    # ========================================================

    print()
    print("=" * 70)

    print(
        "BATCH-SIZE COMPARISON"
    )

    print("=" * 70)

    print()

    print(
        f"{'Batch':>5} | "
        f"{'Tokens/Step':>11} | "
        f"{'Sec/Step':>9} | "
        f"{'Tokens/Sec':>12} | "
        f"{'Logits MiB':>10}"
    )

    print(
        "-" * 62
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
            f"{result['tokens_per_second']:>12,.1f} | "
            f"{result['logits_mib']:>10.2f}"
        )

    if not successful_results:

        raise RuntimeError(
            "No batch size completed successfully."
        )

    best_result = max(
        successful_results,
        key=lambda result:
            result[
                "tokens_per_second"
            ],
    )

    print()
    print(
        f"Highest measured throughput: "
        f"batch {best_result['batch_size']}"
    )

    print(
        f"Throughput: "
        f"{best_result['tokens_per_second']:,.1f} "
        f"tokens/s"
    )

    print()

    print("=" * 70)

    print(
        "SUCCESS: BATCH-SIZE BENCHMARK COMPLETE."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()