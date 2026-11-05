import argparse
import json

from pathlib import Path

import numpy as np


# ============================================================
# Helpers
# ============================================================

def load_checkpoint_payload(
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

        keys = sorted(
            checkpoint.files
        )

        if "metadata" not in keys:

            raise KeyError(
                f"{path} has no metadata."
            )

        for key in keys:

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

    return (
        arrays,
        metadata,
    )


def maximum_absolute_difference(
    first,
    second,
):

    if (
        first.shape
        !=
        second.shape
    ):

        return float(
            "inf"
        )

    if first.size == 0:

        return 0.0

    first64 = first.astype(
        np.float64,
        copy=False,
    )

    second64 = second.astype(
        np.float64,
        copy=False,
    )

    return float(
        np.max(
            np.abs(
                first64
                -
                second64
            )
        )
    )


# ============================================================
# Metadata helpers
# ============================================================

def get_extra_state(
    metadata,
):

    return metadata.get(
        "extra_state",
        {},
    )


def get_loader_state(
    metadata,
):

    return get_extra_state(
        metadata
    ).get(
        "train_loader_state",
        {},
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Compare fresh and resumed "
            "General-v0.2 checkpoints for "
            "exact deterministic equality."
        )
    )

    parser.add_argument(
        "--fresh",
        required=True,
        type=str,
        help=(
            "Fresh uninterrupted checkpoint."
        ),
    )

    parser.add_argument(
        "--resumed",
        required=True,
        type=str,
        help=(
            "Checkpoint produced after resume."
        ),
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
        "STAGE 12B.6 — GENERAL-v0.2 "
        "EXACT RESUME COMPARISON"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Fresh checkpoint:"
    )

    print(
        fresh_path
    )

    print()

    print(
        "Resumed checkpoint:"
    )

    print(
        resumed_path
    )

    (
        fresh_arrays,
        fresh_metadata,
    ) = load_checkpoint_payload(
        fresh_path
    )

    (
        resumed_arrays,
        resumed_metadata,
    ) = load_checkpoint_payload(
        resumed_path
    )

    # ========================================================
    # Key structure
    # ========================================================

    fresh_keys = set(
        fresh_arrays.keys()
    )

    resumed_keys = set(
        resumed_arrays.keys()
    )

    array_keys_match = (
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
        f"Array-key sets identical: "
        f"{array_keys_match}"
    )

    if not array_keys_match:

        only_fresh = sorted(
            fresh_keys
            -
            resumed_keys
        )

        only_resumed = sorted(
            resumed_keys
            -
            fresh_keys
        )

        print(
            f"Only in fresh: "
            f"{only_fresh}"
        )

        print(
            f"Only in resumed: "
            f"{only_resumed}"
        )

        raise AssertionError(
            "Checkpoint array structure differs."
        )

    # ========================================================
    # Exact array comparison
    # ========================================================

    model_arrays = 0

    first_moment_arrays = 0

    second_moment_arrays = 0

    exact_arrays = 0

    failed_arrays = []

    maximum_difference = 0.0

    for key in sorted(
        fresh_keys
    ):

        fresh_array = (
            fresh_arrays[
                key
            ]
        )

        resumed_array = (
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
            fresh_array.shape
            ==
            resumed_array.shape
        )

        dtype_match = (
            fresh_array.dtype
            ==
            resumed_array.dtype
        )

        exact_match = (
            shape_match
            and
            dtype_match
            and
            np.array_equal(
                fresh_array,
                resumed_array,
            )
        )

        if exact_match:

            exact_arrays += 1

        else:

            difference = (
                maximum_absolute_difference(
                    fresh_array,
                    resumed_array,
                )
            )

            maximum_difference = max(
                maximum_difference,
                difference,
            )

            failed_arrays.append(
                {
                    "key":
                        key,

                    "fresh_shape":
                        fresh_array.shape,

                    "resumed_shape":
                        resumed_array.shape,

                    "fresh_dtype":
                        str(
                            fresh_array.dtype
                        ),

                    "resumed_dtype":
                        str(
                            resumed_array.dtype
                        ),

                    "max_absolute_difference":
                        difference,
                }
            )

    all_arrays_exact = (
        len(
            failed_arrays
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
        f"AdamW first-moment arrays: "
        f"{first_moment_arrays:,}"
    )

    print(
        f"AdamW second-moment arrays: "
        f"{second_moment_arrays:,}"
    )

    print(
        f"Arrays compared: "
        f"{len(fresh_keys):,}"
    )

    print(
        f"Exactly identical arrays: "
        f"{exact_arrays:,}"
    )

    print(
        f"All arrays bit-exact: "
        f"{all_arrays_exact}"
    )

    if failed_arrays:

        print()

        print(
            "Differences:"
        )

        for failure in (
            failed_arrays[
                :10
            ]
        ):

            print(
                failure
            )

        print()

        print(
            f"Maximum absolute difference: "
            f"{maximum_difference}"
        )

    # ========================================================
    # Core checkpoint metadata
    # ========================================================

    fresh_extra = get_extra_state(
        fresh_metadata
    )

    resumed_extra = get_extra_state(
        resumed_metadata
    )

    fresh_loader = get_loader_state(
        fresh_metadata
    )

    resumed_loader = get_loader_state(
        resumed_metadata
    )

    metadata_checks = {
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
                8_110_848
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
                10
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

        "moment_state_flag":
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
                10
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
                10_240
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
                256
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
                4
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
                51_776_327
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
    ) in metadata_checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    metadata_exact = (
        fresh_metadata
        ==
        resumed_metadata
    )

    all_metadata_checks = all(
        metadata_checks.values()
    )

    print()

    print(
        f"Entire checkpoint metadata "
        f"identical: "
        f"{metadata_exact}"
    )

    # ========================================================
    # Final acceptance
    # ========================================================

    all_passed = (
        array_keys_match
        and
        all_arrays_exact
        and
        all_metadata_checks
        and
        metadata_exact
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
        f"Model parameters exact: "
        f"{all_arrays_exact}"
    )

    print(
        f"AdamW state exact: "
        f"{all_arrays_exact}"
    )

    print(
        f"Training step exact: "
        f"{metadata_checks['training_step']}"
    )

    print(
        f"Token accounting exact: "
        f"{metadata_checks['tokens_seen']}"
    )

    print(
        f"Loader RNG exact: "
        f"{metadata_checks['loader_rng_state']}"
    )

    print(
        f"Metadata exact: "
        f"{metadata_exact}"
    )

    if not all_passed:

        raise AssertionError(
            "General-v0.2 resume is NOT "
            "bit-exact."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: GENERAL-v0.2 RESUME "
        "IS BIT-EXACT."
    )

    print()

    print(
        "Fresh 0→10 and resumed 5→10 "
        "produced the same training state."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()