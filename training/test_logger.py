import json
import sys
import tempfile

from pathlib import Path


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


from training.logger import (
    TrainingLogger,
)


def main():

    print()
    print(
        "Training Logger Test"
    )

    print(
        "===================="
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        log_path = (
            Path(
                temp_dir
            )
            / "training.jsonl"
        )

        logger = TrainingLogger(
            log_path
        )

        # ====================================================
        # Fake training metrics
        # ====================================================

        training_metrics = {
            "step": 10,
            "tokens_seen": 10240,
            "loss": 9.71,
            "learning_rate": 7.5e-6,
            "gradient_norm": 1.75,
            "clip_scale": 0.57,
            "batch_tokens": 1024,
            "tokens_per_second": 1800.0,
            "total_seconds": 0.57,
        }

        logger.log_training(
            training_metrics
        )

        # ====================================================
        # Fake validation result
        # ====================================================

        validation_result = {
            "loss": 9.68,
            "perplexity": 16000.0,
            "batches": 3,
            "tokens": 3072,
            "seconds": 1.2,
            "tokens_per_second": 2560.0,
        }

        logger.log_validation(
            step=10,
            tokens_seen=10240,
            result=validation_result,
        )

        # ====================================================
        # Generic event
        # ====================================================

        logger.log_event(
            "checkpoint_saved",
            step=10,
            path="checkpoints/test.npz",
        )

        # ====================================================
        # Read JSONL back
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

        # ====================================================
        # Checks
        # ====================================================

        file_exists = (
            log_path.exists()
        )

        file_nonempty = (
            log_path.stat().st_size
            > 0
        )

        record_count_passed = (
            len(
                records
            )
            == 3
        )

        training_type_passed = (
            records[
                0
            ][
                "type"
            ]
            == "train"
        )

        validation_type_passed = (
            records[
                1
            ][
                "type"
            ]
            == "validation"
        )

        event_type_passed = (
            records[
                2
            ][
                "type"
            ]
            == "event"
        )

        training_step_passed = (
            records[
                0
            ][
                "step"
            ]
            == 10
        )

        tokens_seen_passed = (
            records[
                0
            ][
                "tokens_seen"
            ]
            == 10240
        )

        validation_loss_passed = (
            records[
                1
            ][
                "loss"
            ]
            == 9.68
        )

        event_passed = (
            records[
                2
            ][
                "event"
            ]
            ==
            "checkpoint_saved"
        )

        timestamps_passed = all(
            "timestamp"
            in record

            for record
            in records
        )

        # ====================================================
        # Report
        # ====================================================

        print()

        print(
            f"Records written: "
            f"{len(records)}"
        )

        print()

        print(
            "Training record:"
        )

        print(
            records[
                0
            ]
        )

        print()

        print(
            "Validation record:"
        )

        print(
            records[
                1
            ]
        )

        print()

        print(
            "Event record:"
        )

        print(
            records[
                2
            ]
        )

        print()

        print(
            "Final Checks"
        )

        print(
            "------------"
        )

        print(
            f"Log file exists: "
            f"{file_exists}"
        )

        print(
            f"Log file non-empty: "
            f"{file_nonempty}"
        )

        print(
            f"Three records written: "
            f"{record_count_passed}"
        )

        print(
            f"Training record type passed: "
            f"{training_type_passed}"
        )

        print(
            f"Validation record type passed: "
            f"{validation_type_passed}"
        )

        print(
            f"Event record type passed: "
            f"{event_type_passed}"
        )

        print(
            f"Training step preserved: "
            f"{training_step_passed}"
        )

        print(
            f"Tokens seen preserved: "
            f"{tokens_seen_passed}"
        )

        print(
            f"Validation loss preserved: "
            f"{validation_loss_passed}"
        )

        print(
            f"Event preserved: "
            f"{event_passed}"
        )

        print(
            f"Timestamps present: "
            f"{timestamps_passed}"
        )

        all_passed = (
            file_exists
            and
            file_nonempty
            and
            record_count_passed
            and
            training_type_passed
            and
            validation_type_passed
            and
            event_type_passed
            and
            training_step_passed
            and
            tokens_seen_passed
            and
            validation_loss_passed
            and
            event_passed
            and
            timestamps_passed
        )

        print()
        print(
            "=" * 65
        )

        if all_passed:

            print(
                "SUCCESS: TRAINING LOGGER "
                "VALIDATED."
            )

        else:

            raise AssertionError(
                "Training logger test failed."
            )

        print(
            "=" * 65
        )


if __name__ == "__main__":
    main()