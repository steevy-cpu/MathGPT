import argparse
import json

from pathlib import Path

import numpy as np


# ============================================================
# Expected v0.3 deterministic-test state
# ============================================================

EXPECTED_PARAMETER_COUNT = 8_110_848
EXPECTED_STEP = 10
EXPECTED_TOKENS_SEEN = 10_240

EXPECTED_CONTEXT_LENGTH = 256
EXPECTED_BATCH_SIZE = 4
EXPECTED_TOTAL_TOKENS = 51_776_327


# ============================================================
# Helpers
# ============================================================

def load_checkpoint(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    arrays = {}

    metadata = None

    with np.load(
        path,
        allow_pickle=False,
    ) as checkpoint:

        for key in sorted(
            checkpoint.files
        ):

            if key == "metadata":

                metadata = json.loads(
                    str(
                        checkpoint[
                            key
                        ].item()
                    )
                )

            else:

                arrays[
                    key
                ] = np.array(
                    checkpoint[
                        key
                    ],
                    copy=True,
                )

    if metadata is None:

        raise KeyError(
            f"{path} does not contain metadata."
        )

    return (
        arrays,
        metadata,
    )


def loader_state(
    metadata,
):

    return (
        metadata
        .get(
            "extra_state",
            {}
        )
        .get(
            "train_loader_state",
            {}
        )
    )


def extra_state(
    metadata,
):

    return metadata.get(
        "extra_state",
        {}
    )


# ============================================================
# Arguments
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Verify that MathGPT-General-v0.3 "
            "fresh 0→10 training is bit-exact "
            "with resumed 5→10 training."
        )
    )

    parser.add_argument(
        "--fresh",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--resumed",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    fresh_path = Path(
        args.fresh
    )

    resumed_path = Path(
        args.resumed
    )

    print()
    print(
        "=" * 76
    )

    print(
        "GENERAL-v0.3 — BIT-EXACT "
        "RESUME VERIFICATION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Fresh 0→10:"
    )

    print(
        fresh_path
    )

    print()

    print(
        "Resumed 5→10:"
    )

    print(
        resumed_path
    )

    (
        fresh_arrays,
        fresh_metadata,
    ) = load_checkpoint(
        fresh_path
    )

    (
        resumed_arrays,
        resumed_metadata,
    ) = load_checkpoint(
        resumed_path
    )

    # ========================================================
    # Array structure
    # ========================================================

    fresh_keys = set(
        fresh_arrays.keys()
    )

    resumed_keys = set(
        resumed_arrays.keys()
    )

    key_sets_identical = (
        fresh_keys
        ==
        resumed_keys
    )

    print()
    print(
        "Checkpoint structure"
    )

    print(
        "--------------------"
    )

    print(
        f"Fresh arrays: "
        f"{len(fresh_keys):,}"
    )

    print(
        f"Resumed arrays: "
        f"{len(resumed_keys):,}"
    )

    print(
        f"Array keys identical: "
        f"{key_sets_identical}"
    )

    if not key_sets_identical:

        print()

        print(
            "Only fresh:"
        )

        print(
            sorted(
                fresh_keys
                -
                resumed_keys
            )
        )

        print()

        print(
            "Only resumed:"
        )

        print(
            sorted(
                resumed_keys
                -
                fresh_keys
            )
        )

        raise AssertionError(
            "Checkpoint array structures differ."
        )

    # ========================================================
    # Bit-exact arrays
    # ========================================================

    model_arrays = 0
    first_moment_arrays = 0
    second_moment_arrays = 0

    exact_arrays = 0

    failures = []

    for key in sorted(
        fresh_keys
    ):

        first = (
            fresh_arrays[
                key
            ]
        )

        second = (
            resumed_arrays[
                key
            ]
        )

        if key.startswith(
            "model_"
        ):

            model_arrays += 1

        elif key.startswith(
            "first_moment_"
        ):

            first_moment_arrays += 1

        elif key.startswith(
            "second_moment_"
        ):

            second_moment_arrays += 1

        shape_match = (
            first.shape
            ==
            second.shape
        )

        dtype_match = (
            first.dtype
            ==
            second.dtype
        )

        value_match = (
            shape_match
            and
            dtype_match
            and
            np.array_equal(
                first,
                second,
            )
        )

        if value_match:

            exact_arrays += 1

        else:

            failures.append(
                key
            )

    arrays_bit_exact = (
        len(
            failures
        )
        ==
        0
    )

    print()
    print(
        "Array comparison"
    )

    print(
        "----------------"
    )

    print(
        f"Model arrays: "
        f"{model_arrays:,}"
    )

    print(
        f"AdamW first moments: "
        f"{first_moment_arrays:,}"
    )

    print(
        f"AdamW second moments: "
        f"{second_moment_arrays:,}"
    )

    print(
        f"Arrays compared: "
        f"{len(fresh_keys):,}"
    )

    print(
        f"Exactly identical: "
        f"{exact_arrays:,}"
    )

    print(
        f"All arrays bit-exact: "
        f"{arrays_bit_exact}"
    )

    if failures:

        print()

        print(
            "Failed arrays:"
        )

        for key in failures[
            :20
        ]:

            print(
                f"  {key}"
            )

    # ========================================================
    # Metadata
    # ========================================================

    fresh_extra = extra_state(
        fresh_metadata
    )

    resumed_extra = extra_state(
        resumed_metadata
    )

    fresh_loader = loader_state(
        fresh_metadata
    )

    resumed_loader = loader_state(
        resumed_metadata
    )

    checks = {
        "checkpoint_version":
            (
                fresh_metadata.get(
                    "checkpoint_version"
                )
                ==
                resumed_metadata.get(
                    "checkpoint_version"
                )
            ),

        "parameter_names":
            (
                fresh_metadata.get(
                    "parameter_names"
                )
                ==
                resumed_metadata.get(
                    "parameter_names"
                )
            ),

        "parameter_count":
            (
                fresh_metadata.get(
                    "parameter_count"
                )
                ==
                resumed_metadata.get(
                    "parameter_count"
                )
                ==
                EXPECTED_PARAMETER_COUNT
            ),

        "optimizer_class":
            (
                fresh_metadata.get(
                    "optimizer_class"
                )
                ==
                resumed_metadata.get(
                    "optimizer_class"
                )
            ),

        "optimizer_hyperparameters":
            (
                fresh_metadata.get(
                    "optimizer_hyperparameters"
                )
                ==
                resumed_metadata.get(
                    "optimizer_hyperparameters"
                )
            ),

        "optimizer_step_count":
            (
                fresh_metadata.get(
                    "optimizer_step_count"
                )
                ==
                resumed_metadata.get(
                    "optimizer_step_count"
                )
                ==
                EXPECTED_STEP
            ),

        "has_moment_state":
            (
                fresh_metadata.get(
                    "has_moment_state"
                )
                ==
                resumed_metadata.get(
                    "has_moment_state"
                )
            ),

        "training_step":
            (
                fresh_extra.get(
                    "training_step"
                )
                ==
                resumed_extra.get(
                    "training_step"
                )
                ==
                EXPECTED_STEP
            ),

        "tokens_seen":
            (
                fresh_extra.get(
                    "tokens_seen"
                )
                ==
                resumed_extra.get(
                    "tokens_seen"
                )
                ==
                EXPECTED_TOKENS_SEEN
            ),

        "loader_context_length":
            (
                fresh_loader.get(
                    "context_length"
                )
                ==
                resumed_loader.get(
                    "context_length"
                )
                ==
                EXPECTED_CONTEXT_LENGTH
            ),

        "loader_batch_size":
            (
                fresh_loader.get(
                    "batch_size"
                )
                ==
                resumed_loader.get(
                    "batch_size"
                )
                ==
                EXPECTED_BATCH_SIZE
            ),

        "loader_total_tokens":
            (
                fresh_loader.get(
                    "total_tokens"
                )
                ==
                resumed_loader.get(
                    "total_tokens"
                )
                ==
                EXPECTED_TOTAL_TOKENS
            ),

        "loader_maximum_start":
            (
                fresh_loader.get(
                    "maximum_start"
                )
                ==
                resumed_loader.get(
                    "maximum_start"
                )
            ),

        "loader_rng_state":
            (
                fresh_loader.get(
                    "rng_state"
                )
                ==
                resumed_loader.get(
                    "rng_state"
                )
            ),
    }

    print()
    print(
        "Metadata / trainer state"
    )

    print(
        "------------------------"
    )

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    entire_metadata_exact = (
        fresh_metadata
        ==
        resumed_metadata
    )

    print()

    print(
        f"Entire metadata identical: "
        f"{entire_metadata_exact}"
    )

    # ========================================================
    # Acceptance
    # ========================================================

    all_passed = (
        key_sets_identical
        and
        arrays_bit_exact
        and
        all(
            checks.values()
        )
        and
        entire_metadata_exact
    )

    print()
    print(
        "=" * 76
    )

    print(
        "DETERMINISTIC RESUME RESULT"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Model + optimizer arrays exact: "
        f"{arrays_bit_exact}"
    )

    print(
        f"Step exact: "
        f"{checks['training_step']}"
    )

    print(
        f"Token accounting exact: "
        f"{checks['tokens_seen']}"
    )

    print(
        f"Loader RNG exact: "
        f"{checks['loader_rng_state']}"
    )

    print(
        f"Metadata exact: "
        f"{entire_metadata_exact}"
    )

    if not all_passed:

        raise AssertionError(
            "General-v0.3 resume is "
            "NOT bit-exact."
        )

    print()
    print(
        "SUCCESS: GENERAL-v0.3 RESUME "
        "IS BIT-EXACT."
    )

    print()

    print(
        "Fresh 0→10 and resumed 5→10 "
        "produced exactly the same "
        "training state."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()