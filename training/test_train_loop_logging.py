import json
import sys
import tempfile

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

from training.logger import (
    TrainingLogger,
)

from training.train_loop import (
    run_training_steps,
)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "Training Loop Logging Test"
    )

    print(
        "=========================="
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        # ====================================================
        # Synthetic token corpus
        # ====================================================

        binary_path = (
            temp_dir
            / "tiny.bin"
        )

        tokens = np.array(
            [
                1, 2, 3, 4, 5, 6,
                2, 3, 4, 5, 6, 1,
                3, 4, 5, 6, 1, 2,
                4, 5, 6, 1, 2, 3,
                5, 6, 1, 2, 3, 4,
                6, 1, 2, 3, 4, 5,
            ],
            dtype=np.uint16,
        )

        tokens.tofile(
            binary_path
        )

        # ====================================================
        # Tiny GPT
        # ====================================================

        model = GPT(
            vocab_size=7,
            context_length=3,
            d_model=4,
            num_layers=2,
            num_heads=2,
            d_ff=6,
            use_bias=True,
            tie_embeddings=True,
            seed=42,
        )

        # ====================================================
        # Data loader
        # ====================================================

        loader = TokenBatchLoader(
            binary_path=binary_path,
            context_length=3,
            batch_size=2,
            seed=42,
        )

        # ====================================================
        # Optimizer
        # ====================================================

        optimizer = AdamW(
            model=model,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
        )

        # ====================================================
        # Logger
        # ====================================================

        log_path = (
            temp_dir
            / "training.jsonl"
        )

        logger = TrainingLogger(
            log_path
        )

        # ====================================================
        # Train four steps
        #
        # log_every = 2:
        #
        # step 1 logged
        # step 2 logged
        # step 3 not logged
        # step 4 logged
        #
        # No checkpointing is enabled here.
        # ====================================================

        (
            history,
            tokens_seen,
            saved_checkpoints,
        ) = run_training_steps(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            num_steps=4,
            max_learning_rate=0.001,
            min_learning_rate=0.0001,
            warmup_steps=2,
            total_steps=10,
            max_grad_norm=1.0,
            starting_tokens_seen=0,
            progress_every=1,
            logger=logger,
            log_every=2,
        )

        # ====================================================
        # Read log
        # ====================================================

        with open(
            log_path,
            "r",
            encoding="utf-8",
        ) as file:

            records = [
                json.loads(
                    line
                )

                for line
                in file

                if line.strip()
            ]

        logged_steps = [
            record[
                "step"
            ]

            for record
            in records

            if (
                record[
                    "type"
                ]
                ==
                "train"
            )
        ]

        # ====================================================
        # Expected token accounting
        #
        # batch size = 2
        # context = 3
        #
        # 6 tokens per step
        #
        # 4 steps = 24 tokens
        # ====================================================

        expected_tokens_seen = (
            4
            *
            2
            *
            3
        )

        # ====================================================
        # Checks
        # ====================================================

        history_passed = (
            len(
                history
            )
            ==
            4
        )

        optimizer_steps_passed = (
            optimizer.step_count
            ==
            4
        )

        tokens_passed = (
            tokens_seen
            ==
            expected_tokens_seen
        )

        record_count_passed = (
            len(
                records
            )
            ==
            3
        )

        logged_steps_passed = (
            logged_steps
            ==
            [
                1,
                2,
                4,
            ]
        )

        record_types_passed = all(
            record[
                "type"
            ]
            ==
            "train"

            for record
            in records
        )

        timestamps_passed = all(
            "timestamp"
            in record

            for record
            in records
        )

        losses_finite = all(
            np.isfinite(
                record[
                    "loss"
                ]
            )

            for record
            in records
        )

        tokens_seen_logged_correctly = (
            records[
                -1
            ][
                "tokens_seen"
            ]
            ==
            expected_tokens_seen
        )

        # Since checkpointing is disabled,
        # the returned list must stay empty.
        no_checkpoints_passed = (
            saved_checkpoints
            ==
            []
        )

        # ====================================================
        # Report
        # ====================================================

        print()

        print(
            f"Training steps: "
            f"{len(history)}"
        )

        print(
            f"Optimizer step count: "
            f"{optimizer.step_count}"
        )

        print(
            f"Tokens seen: "
            f"{tokens_seen}"
        )

        print(
            f"Log records: "
            f"{len(records)}"
        )

        print(
            f"Logged steps: "
            f"{logged_steps}"
        )

        print(
            f"Checkpoints created: "
            f"{len(saved_checkpoints)}"
        )

        print()

        print(
            "Final Checks"
        )

        print(
            "------------"
        )

        print(
            f"Four training steps completed: "
            f"{history_passed}"
        )

        print(
            f"Optimizer completed four steps: "
            f"{optimizer_steps_passed}"
        )

        print(
            f"Token accounting passed: "
            f"{tokens_passed}"
        )

        print(
            f"Three records logged: "
            f"{record_count_passed}"
        )

        print(
            f"Steps 1, 2, 4 logged: "
            f"{logged_steps_passed}"
        )

        print(
            f"All records are training records: "
            f"{record_types_passed}"
        )

        print(
            f"Timestamps present: "
            f"{timestamps_passed}"
        )

        print(
            f"Logged losses finite: "
            f"{losses_finite}"
        )

        print(
            f"Logged token count correct: "
            f"{tokens_seen_logged_correctly}"
        )

        print(
            f"No checkpoints created: "
            f"{no_checkpoints_passed}"
        )

        all_passed = (
            history_passed
            and
            optimizer_steps_passed
            and
            tokens_passed
            and
            record_count_passed
            and
            logged_steps_passed
            and
            record_types_passed
            and
            timestamps_passed
            and
            losses_finite
            and
            tokens_seen_logged_correctly
            and
            no_checkpoints_passed
        )

        print()
        print(
            "=" * 65
        )

        if all_passed:

            print(
                "SUCCESS: TRAINING LOOP "
                "PERSISTENT LOGGING VALIDATED."
            )

        else:

            raise AssertionError(
                "Training-loop logging "
                "validation failed."
            )

        print(
            "=" * 65
        )


if __name__ == "__main__":

    main()