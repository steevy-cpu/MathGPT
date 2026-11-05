import subprocess
import sys
import time

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


# ============================================================
# Final Stage 9 regression tests
#
# These cover:
#
#   tokenizer / corpus
#   model architecture
#   backward propagation
#   parameter traversal
#   optimizer
#   gradient clipping
#   scheduler
#   evaluation
#   logging
#   training loop
#   checkpointing
#   deterministic resume
#   real-data training
#   real cross-machine checkpoint resume
# ============================================================

TESTS = [

    # --------------------------------------------------------
    # Data / tokenizer
    # --------------------------------------------------------

    (
        "Final v0.2 corpus encoding",
        "tokenizer/validate_general_encoding_v0_2.py",
    ),

    (
        "Real token batch loader",
        "training/test_data_loader.py",
    ),

    # --------------------------------------------------------
    # Configuration / architecture
    # --------------------------------------------------------

    (
        "Training configuration",
        "training/test_training_config.py",
    ),

    (
        "Full GPT architecture",
        "model/validate_stage6_gpt.py",
    ),

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    (
        "Cross-entropy backward",
        "training/test_loss_backward.py",
    ),

    (
        "Full GPT end-to-end backward",
        "training/test_gpt_end_to_end_backward.py",
    ),

    # --------------------------------------------------------
    # Parameter system
    # --------------------------------------------------------

    (
        "Parameter traversal",
        "training/test_parameters.py",
    ),

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    (
        "AdamW fundamentals",
        "training/test_adamw.py",
    ),

    (
        "Full-model AdamW",
        "training/test_adamw_optimizer.py",
    ),

    (
        "AdamW training behavior",
        "training/test_adamw_training.py",
    ),

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    (
        "Gradient clipping fundamentals",
        "training/test_gradient_clipping.py",
    ),

    (
        "Full GPT gradient clipping",
        "training/test_gpt_gradient_clipping.py",
    ),

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    (
        "Warmup + cosine scheduler",
        "training/test_lr_scheduler.py",
    ),

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        "Deterministic evaluation",
        "training/test_evaluation.py",
    ),

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    (
        "Persistent JSONL logger",
        "training/test_logger.py",
    ),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (
        "Single training step",
        "training/test_single_training_step.py",
    ),

    (
        "Real full-model training step",
        "training/test_real_batch_training_step.py",
    ),

    (
        "Core real training loop",
        "training/test_train_loop.py",
    ),

    (
        "Training-loop logging",
        "training/test_train_loop_logging.py",
    ),

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------

    (
        "Checkpoint save / resume",
        "training/test_checkpoint_resume.py",
    ),

    (
        "Periodic checkpointing",
        "training/test_periodic_checkpointing.py",
    ),

    (
        "Exact deterministic trainer resume",
        "training/test_exact_trainer_resume.py",
    ),

    # --------------------------------------------------------
    # Real Stage 9I checkpoint
    # --------------------------------------------------------

    (
        "Real cross-machine checkpoint resume",
        "training/test_real_checkpoint_resume.py",
    ),
]


# ============================================================
# Run one test
# ============================================================

def run_test(
    number,
    total,
    name,
    relative_path,
):

    test_path = (
        PROJECT_ROOT
        /
        relative_path
    )

    print()
    print(
        "=" * 78
    )

    print(
        f"[{number}/{total}] "
        f"{name}"
    )

    print(
        f"File: {relative_path}"
    )

    print(
        "=" * 78
    )

    if not test_path.exists():

        print()

        print(
            "FAIL: Test file does not exist."
        )

        return {
            "name":
                name,

            "path":
                relative_path,

            "passed":
                False,

            "return_code":
                None,

            "seconds":
                0.0,

            "reason":
                "file_missing",
        }

    start_time = (
        time.perf_counter()
    )

    result = subprocess.run(
        [
            sys.executable,
            str(
                test_path
            ),
        ],
        cwd=PROJECT_ROOT,
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    passed = (
        result.returncode
        ==
        0
    )

    print()

    if passed:

        print(
            f"PASS: {name}"
        )

    else:

        print(
            f"FAIL: {name}"
        )

    print(
        f"Elapsed: "
        f"{elapsed:.2f}s"
    )

    return {
        "name":
            name,

        "path":
            relative_path,

        "passed":
            passed,

        "return_code":
            result.returncode,

        "seconds":
            elapsed,

        "reason":
            None,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 78
    )

    print(
        "STAGE 9J.2 — FINAL TRAINING SYSTEM REGRESSION SUITE"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"Python:"
    )

    print(
        sys.executable
    )

    print()

    print(
        f"Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        f"Tests scheduled: "
        f"{len(TESTS)}"
    )

    suite_start = (
        time.perf_counter()
    )

    results = []

    total = len(
        TESTS
    )

    for index, (
        name,
        path,
    ) in enumerate(
        TESTS,
        start=1,
    ):

        result = run_test(
            number=index,
            total=total,
            name=name,
            relative_path=path,
        )

        results.append(
            result
        )

    suite_seconds = (
        time.perf_counter()
        -
        suite_start
    )

    # ========================================================
    # Summary
    # ========================================================

    passed_results = [
        result

        for result
        in results

        if result[
            "passed"
        ]
    ]

    failed_results = [
        result

        for result
        in results

        if not result[
            "passed"
        ]
    ]

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL REGRESSION SUMMARY"
    )

    print(
        "=" * 78
    )

    print()

    for index, result in enumerate(
        results,
        start=1,
    ):

        status = (
            "PASS"
            if result[
                "passed"
            ]
            else
            "FAIL"
        )

        print(
            f"{index:>2}. "
            f"[{status}] "
            f"{result['name']} "
            f"({result['seconds']:.2f}s)"
        )

    print()

    print(
        "-" * 78
    )

    print(
        f"Passed: "
        f"{len(passed_results)}"
        f"/"
        f"{len(results)}"
    )

    print(
        f"Failed: "
        f"{len(failed_results)}"
    )

    print(
        f"Total suite time: "
        f"{suite_seconds:.2f}s"
    )

    print(
        "-" * 78
    )

    # ========================================================
    # Failed test details
    # ========================================================

    if failed_results:

        print()
        print(
            "FAILED TESTS"
        )

        print(
            "------------"
        )

        for result in failed_results:

            print(
                f"- "
                f"{result['name']}"
            )

            print(
                f"  File: "
                f"{result['path']}"
            )

            print(
                f"  Return code: "
                f"{result['return_code']}"
            )

            if (
                result[
                    "reason"
                ]
                is not None
            ):

                print(
                    f"  Reason: "
                    f"{result['reason']}"
                )

    # ========================================================
    # Final result
    # ========================================================

    print()
    print(
        "=" * 78
    )

    if not failed_results:

        print(
            "SUCCESS: ALL STAGE 9 REGRESSION TESTS PASSED."
        )

        print()

        print(
            "MathGPT training system is ready "
            "for final Stage 9 acceptance."
        )

    else:

        print(
            "FAILURE: ONE OR MORE STAGE 9 "
            "REGRESSION TESTS FAILED."
        )

        print()

        print(
            "Do not begin full general "
            "pretraining until these are fixed."
        )

    print(
        "=" * 78
    )

    if failed_results:

        raise SystemExit(
            1
        )


if __name__ == "__main__":

    main()