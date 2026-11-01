import json
import sys
import tempfile

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

from training.adamw import AdamW

from training.data_loader import (
    TokenBatchLoader,
)

from training.logger import (
    TrainingLogger,
)

from training.train_loop import (
    run_training_steps,
)


def read_checkpoint_metadata(
    path,
):
    """
    Read checkpoint metadata without
    modifying a model.
    """

    with np.load(
        path,
        allow_pickle=False,
    ) as checkpoint:

        return json.loads(
            str(
                checkpoint[
                    "metadata"
                ].item()
            )
        )


def main():

    print()
    print(
        "Periodic Checkpointing Test"
    )

    print(
        "==========================="
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

        optimizer = AdamW(
            model=model,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
        )

        loader = TokenBatchLoader(
            binary_path=binary_path,
            context_length=3,
            batch_size=2,
            seed=42,
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
        # Checkpoint directory
        # ====================================================

        checkpoint_directory = (
            temp_dir
            / "checkpoints"
        )

        # ====================================================
        # Train five steps.
        #
        # checkpoint_every = 2:
        #
        # step 2 saved
        # step 4 saved
        # ====================================================

        (
            history,
            tokens_seen,
            saved_checkpoints,
        ) = run_training_steps(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            num_steps=5,
            max_learning_rate=0.001,
            min_learning_rate=0.0001,
            warmup_steps=2,
            total_steps=10,
            max_grad_norm=1.0,
            starting_tokens_seen=0,
            progress_every=1,
            logger=logger,
            log_every=2,
            checkpoint_every=2,
            checkpoint_directory=(
                checkpoint_directory
            ),
            checkpoint_prefix=(
                "test_mathgpt"
            ),
        )

        # ====================================================
        # Expected checkpoint paths
        # ====================================================

        expected_step_2 = (
            checkpoint_directory
            / "test_mathgpt_step_000002.npz"
        )

        expected_step_4 = (
            checkpoint_directory
            / "test_mathgpt_step_000004.npz"
        )

        # ====================================================
        # Metadata
        # ====================================================

        metadata_2 = (
            read_checkpoint_metadata(
                expected_step_2
            )
        )

        metadata_4 = (
            read_checkpoint_metadata(
                expected_step_4
            )
        )

        # ====================================================
        # Read event log
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

        checkpoint_events = [
            record

            for record
            in records

            if (
                record[
                    "type"
                ]
                ==
                "event"
                and
                record[
                    "event"
                ]
                ==
                "checkpoint_saved"
            )
        ]

        event_steps = [
            record[
                "step"
            ]

            for record
            in checkpoint_events
        ]

        # ====================================================
        # Checks
        # ====================================================

        step_count_passed = (
            optimizer.step_count
            == 5
        )

        token_count_passed = (
            tokens_seen
            == 30
        )

        checkpoint_count_passed = (
            len(
                saved_checkpoints
            )
            == 2
        )

        step_2_exists = (
            expected_step_2.exists()
        )

        step_4_exists = (
            expected_step_4.exists()
        )

        names_passed = (
            saved_checkpoints
            ==
            [
                expected_step_2,
                expected_step_4,
            ]
        )

        metadata_step_2_passed = (
            metadata_2[
                "optimizer_step_count"
            ]
            == 2
            and
            metadata_2[
                "extra_state"
            ][
                "training_step"
            ]
            == 2
        )

        metadata_step_4_passed = (
            metadata_4[
                "optimizer_step_count"
            ]
            == 4
            and
            metadata_4[
                "extra_state"
            ][
                "training_step"
            ]
            == 4
        )

        # 6 tokens per step:
        #
        # batch 2 × context 3
        metadata_tokens_passed = (
            metadata_2[
                "extra_state"
            ][
                "tokens_seen"
            ]
            == 12
            and
            metadata_4[
                "extra_state"
            ][
                "tokens_seen"
            ]
            == 24
        )

        checkpoint_events_passed = (
            event_steps
            ==
            [
                2,
                4,
            ]
        )

        checkpoint_files_nonempty = (
            expected_step_2.stat().st_size
            > 0
            and
            expected_step_4.stat().st_size
            > 0
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
            f"Tokens seen: "
            f"{tokens_seen}"
        )

        print(
            f"Checkpoints saved: "
            f"{len(saved_checkpoints)}"
        )

        print()

        for path in saved_checkpoints:

            print(
                f"  {path.name}"
            )

        print()

        print(
            f"Checkpoint event steps: "
            f"{event_steps}"
        )

        print()

        print(
            "Step 2 checkpoint:"
        )

        print(
            f"  Optimizer step: "
            f"{metadata_2['optimizer_step_count']}"
        )

        print(
            f"  Tokens seen: "
            f"{metadata_2['extra_state']['tokens_seen']}"
        )

        print()

        print(
            "Step 4 checkpoint:"
        )

        print(
            f"  Optimizer step: "
            f"{metadata_4['optimizer_step_count']}"
        )

        print(
            f"  Tokens seen: "
            f"{metadata_4['extra_state']['tokens_seen']}"
        )

        # ====================================================
        # Final checks
        # ====================================================

        print()

        print(
            "Final Checks"
        )

        print(
            "------------"
        )

        print(
            f"Five optimizer steps completed: "
            f"{step_count_passed}"
        )

        print(
            f"Token accounting passed: "
            f"{token_count_passed}"
        )

        print(
            f"Exactly two checkpoints saved: "
            f"{checkpoint_count_passed}"
        )

        print(
            f"Step 2 checkpoint exists: "
            f"{step_2_exists}"
        )

        print(
            f"Step 4 checkpoint exists: "
            f"{step_4_exists}"
        )

        print(
            f"Checkpoint filenames passed: "
            f"{names_passed}"
        )

        print(
            f"Step 2 metadata passed: "
            f"{metadata_step_2_passed}"
        )

        print(
            f"Step 4 metadata passed: "
            f"{metadata_step_4_passed}"
        )

        print(
            f"Checkpoint token counts passed: "
            f"{metadata_tokens_passed}"
        )

        print(
            f"Checkpoint events logged: "
            f"{checkpoint_events_passed}"
        )

        print(
            f"Checkpoint files non-empty: "
            f"{checkpoint_files_nonempty}"
        )

        all_passed = (
            step_count_passed
            and
            token_count_passed
            and
            checkpoint_count_passed
            and
            step_2_exists
            and
            step_4_exists
            and
            names_passed
            and
            metadata_step_2_passed
            and
            metadata_step_4_passed
            and
            metadata_tokens_passed
            and
            checkpoint_events_passed
            and
            checkpoint_files_nonempty
        )

        print()
        print(
            "=" * 65
        )

        if all_passed:

            print(
                "SUCCESS: PERIODIC TRAINING "
                "CHECKPOINTING VALIDATED."
            )

        else:

            raise AssertionError(
                "Periodic checkpointing "
                "validation failed."
            )

        print(
            "=" * 65
        )


if __name__ == "__main__":
    main()