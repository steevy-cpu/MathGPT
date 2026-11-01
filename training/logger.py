import json
from datetime import datetime, timezone
from pathlib import Path


class TrainingLogger:
    """
    Append-only JSONL logger for MathGPT training.

    Each line in the log file is one JSON record.

    Example:

        {"type": "train", ...}
        {"type": "validation", ...}
    """

    def __init__(
        self,
        path,
    ):
        self.path = Path(
            path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _timestamp(
        self,
    ):
        """
        Current UTC timestamp.
        """

        return (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )

    def log(
        self,
        record,
    ):
        """
        Append one JSON record.
        """

        output_record = dict(
            record
        )

        if (
            "timestamp"
            not in output_record
        ):
            output_record[
                "timestamp"
            ] = self._timestamp()

        with open(
            self.path,
            "a",
            encoding="utf-8",
        ) as file:

            json.dump(
                output_record,
                file,
                ensure_ascii=False,
            )

            file.write(
                "\n"
            )

            # Make the record visible on disk
            # immediately instead of waiting for
            # Python's file buffer.
            file.flush()

    def log_training(
        self,
        metrics,
    ):
        """
        Log one training measurement.
        """

        record = {
            "type":
                "train",

            "step":
                int(
                    metrics[
                        "step"
                    ]
                ),

            "tokens_seen":
                int(
                    metrics[
                        "tokens_seen"
                    ]
                ),

            "loss":
                float(
                    metrics[
                        "loss"
                    ]
                ),

            "learning_rate":
                float(
                    metrics[
                        "learning_rate"
                    ]
                ),

            "gradient_norm":
                float(
                    metrics[
                        "gradient_norm"
                    ]
                ),

            "clip_scale":
                float(
                    metrics[
                        "clip_scale"
                    ]
                ),

            "batch_tokens":
                int(
                    metrics[
                        "batch_tokens"
                    ]
                ),

            "tokens_per_second":
                float(
                    metrics[
                        "tokens_per_second"
                    ]
                ),

            "total_seconds":
                float(
                    metrics[
                        "total_seconds"
                    ]
                ),
        }

        self.log(
            record
        )

    def log_validation(
        self,
        *,
        step,
        tokens_seen,
        result,
    ):
        """
        Log one validation measurement.
        """

        record = {
            "type":
                "validation",

            "step":
                int(
                    step
                ),

            "tokens_seen":
                int(
                    tokens_seen
                ),

            "loss":
                float(
                    result[
                        "loss"
                    ]
                ),

            "perplexity":
                float(
                    result[
                        "perplexity"
                    ]
                ),

            "batches":
                int(
                    result[
                        "batches"
                    ]
                ),

            "tokens":
                int(
                    result[
                        "tokens"
                    ]
                ),

            "seconds":
                float(
                    result[
                        "seconds"
                    ]
                ),

            "tokens_per_second":
                float(
                    result[
                        "tokens_per_second"
                    ]
                ),
        }

        self.log(
            record
        )

    def log_event(
        self,
        event,
        **fields,
    ):
        """
        Generic event logger.

        We will use this later for things like:

            checkpoint_saved
            training_started
            training_resumed
            training_completed
        """

        record = {
            "type":
                "event",

            "event":
                str(
                    event
                ),

            **fields,
        }

        self.log(
            record
        )