import json
import os
from pathlib import Path

import numpy as np


try:
    from .parameters import named_parameters

except ImportError:
    from parameters import named_parameters


CHECKPOINT_VERSION = 1


def _optimizer_hyperparameters(
    optimizer,
):
    """
    Collect optimizer settings that need
    to survive a training restart.
    """

    attributes = [
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
    ]

    values = {}

    for attribute in attributes:

        if hasattr(
            optimizer,
            attribute,
        ):

            values[
                attribute
            ] = float(
                getattr(
                    optimizer,
                    attribute,
                )
            )

    return values


def save_checkpoint(
    model,
    optimizer,
    path,
    extra_state=None,
):
    """
    Save:

        model parameters
        optimizer m
        optimizer v
        optimizer step
        optimizer hyperparameters
        extra training state

    The file is written atomically:

        temporary file
            ↓
        complete write
            ↓
        replace final checkpoint

    This reduces the chance of leaving a
    corrupted checkpoint if saving stops
    midway.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters = list(
        named_parameters(
            model
        )
    )

    parameter_names = [
        name

        for (
            name,
            _,
            _,
        ) in parameters
    ]

    # ========================================================
    # Check whether optimizer contains Adam-style state
    # ========================================================

    has_moment_state = (
        hasattr(
            optimizer,
            "first_moments",
        )
        and
        hasattr(
            optimizer,
            "second_moments",
        )
    )

    arrays = {}

    # ========================================================
    # Model weights
    # ========================================================

    for (
        index,
        (
            name,
            parameter,
            _,
        ),
    ) in enumerate(
        parameters
    ):

        arrays[
            f"model_{index:04d}"
        ] = parameter.copy()

    # ========================================================
    # Adam / AdamW state
    # ========================================================

    if has_moment_state:

        for (
            index,
            (
                name,
                parameter,
                _,
            ),
        ) in enumerate(
            parameters
        ):

            if (
                name
                not in optimizer.first_moments
            ):

                raise KeyError(
                    f"Missing first moment "
                    f"for {name}."
                )

            if (
                name
                not in optimizer.second_moments
            ):

                raise KeyError(
                    f"Missing second moment "
                    f"for {name}."
                )

            first_moment = (
                optimizer.first_moments[
                    name
                ]
            )

            second_moment = (
                optimizer.second_moments[
                    name
                ]
            )

            if (
                first_moment.shape
                != parameter.shape
            ):

                raise ValueError(
                    f"First-moment shape "
                    f"mismatch for {name}."
                )

            if (
                second_moment.shape
                != parameter.shape
            ):

                raise ValueError(
                    f"Second-moment shape "
                    f"mismatch for {name}."
                )

            arrays[
                f"first_moment_{index:04d}"
            ] = first_moment.copy()

            arrays[
                f"second_moment_{index:04d}"
            ] = second_moment.copy()

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "checkpoint_version":
            CHECKPOINT_VERSION,

        "parameter_names":
            parameter_names,

        "parameter_count":
            int(
                sum(
                    parameter.size

                    for (
                        _,
                        parameter,
                        _,
                    ) in parameters
                )
            ),

        "optimizer_class":
            optimizer.__class__.__name__,

        "optimizer_step_count":
            int(
                optimizer.step_count
            ),

        "optimizer_hyperparameters":
            _optimizer_hyperparameters(
                optimizer
            ),

        "has_moment_state":
            has_moment_state,

        "extra_state":
            (
                {}
                if extra_state is None
                else extra_state
            ),
    }

    metadata_json = json.dumps(
        metadata
    )

    arrays[
        "metadata"
    ] = np.array(
        metadata_json
    )

    # ========================================================
    # Atomic write
    # ========================================================

    temporary_path = (
        path.with_name(
            path.name
            + ".tmp"
        )
    )

    try:

        with open(
            temporary_path,
            "wb",
        ) as file:

            # Intentionally uncompressed.
            #
            # Training checkpoints favor write
            # speed over maximum compression.
            np.savez(
                file,
                **arrays,
            )

        os.replace(
            temporary_path,
            path,
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink()

    return metadata


def load_checkpoint(
    model,
    optimizer,
    path,
    strict=True,
):
    """
    Restore a checkpoint into an existing
    model and optimizer.

    Returns checkpoint metadata.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{path}"
        )

    current_parameters = list(
        named_parameters(
            model
        )
    )

    current_names = [
        name

        for (
            name,
            _,
            _,
        ) in current_parameters
    ]

    with np.load(
        path,
        allow_pickle=False,
    ) as checkpoint:

        if (
            "metadata"
            not in checkpoint
        ):

            raise ValueError(
                "Checkpoint metadata "
                "is missing."
            )

        metadata = json.loads(
            str(
                checkpoint[
                    "metadata"
                ].item()
            )
        )

        # ====================================================
        # Version check
        # ====================================================

        version = metadata.get(
            "checkpoint_version"
        )

        if (
            version
            != CHECKPOINT_VERSION
        ):

            raise ValueError(
                "Unsupported checkpoint "
                f"version: {version}"
            )

        checkpoint_names = metadata[
            "parameter_names"
        ]

        # ====================================================
        # Architecture/name check
        # ====================================================

        if (
            strict
            and
            checkpoint_names
            != current_names
        ):

            raise ValueError(
                "Checkpoint parameter names "
                "do not match the model."
            )

        checkpoint_index = {
            name: index

            for (
                index,
                name,
            ) in enumerate(
                checkpoint_names
            )
        }

        # ====================================================
        # Restore model parameters
        # ====================================================

        for (
            name,
            parameter,
            _,
        ) in current_parameters:

            if (
                name
                not in checkpoint_index
            ):

                if strict:

                    raise KeyError(
                        f"Parameter {name} "
                        "missing from checkpoint."
                    )

                continue

            index = (
                checkpoint_index[
                    name
                ]
            )

            key = (
                f"model_{index:04d}"
            )

            if (
                key
                not in checkpoint
            ):

                raise KeyError(
                    f"Missing checkpoint "
                    f"array: {key}"
                )

            saved_parameter = (
                checkpoint[
                    key
                ]
            )

            if (
                saved_parameter.shape
                != parameter.shape
            ):

                raise ValueError(
                    f"Shape mismatch for "
                    f"{name}: "
                    f"{saved_parameter.shape} "
                    f"vs "
                    f"{parameter.shape}"
                )

            np.copyto(
                parameter,
                saved_parameter,
                casting="same_kind",
            )

        # ====================================================
        # Optimizer type check
        # ====================================================

        saved_optimizer_class = (
            metadata[
                "optimizer_class"
            ]
        )

        current_optimizer_class = (
            optimizer
            .__class__
            .__name__
        )

        if (
            strict
            and
            saved_optimizer_class
            != current_optimizer_class
        ):

            raise ValueError(
                "Optimizer mismatch: "
                f"checkpoint uses "
                f"{saved_optimizer_class}, "
                f"current optimizer uses "
                f"{current_optimizer_class}."
            )

        # ====================================================
        # Restore optimizer hyperparameters
        # ====================================================

        saved_hyperparameters = (
            metadata.get(
                "optimizer_hyperparameters",
                {},
            )
        )

        for (
            name,
            value,
        ) in saved_hyperparameters.items():

            if hasattr(
                optimizer,
                name,
            ):

                setattr(
                    optimizer,
                    name,
                    float(
                        value
                    ),
                )

        # ====================================================
        # Restore optimizer step
        # ====================================================

        optimizer.step_count = int(
            metadata[
                "optimizer_step_count"
            ]
        )

        # ====================================================
        # Restore Adam/AdamW moment state
        # ====================================================

        if metadata.get(
            "has_moment_state",
            False,
        ):

            if not (
                hasattr(
                    optimizer,
                    "first_moments",
                )
                and
                hasattr(
                    optimizer,
                    "second_moments",
                )
            ):

                raise ValueError(
                    "Checkpoint contains "
                    "Adam-style state but "
                    "optimizer does not."
                )

            for (
                name,
                parameter,
                _,
            ) in current_parameters:

                if (
                    name
                    not in checkpoint_index
                ):

                    continue

                index = (
                    checkpoint_index[
                        name
                    ]
                )

                first_key = (
                    f"first_moment_"
                    f"{index:04d}"
                )

                second_key = (
                    f"second_moment_"
                    f"{index:04d}"
                )

                if (
                    first_key
                    not in checkpoint
                ):

                    raise KeyError(
                        f"Missing {first_key}."
                    )

                if (
                    second_key
                    not in checkpoint
                ):

                    raise KeyError(
                        f"Missing {second_key}."
                    )

                saved_first = (
                    checkpoint[
                        first_key
                    ]
                )

                saved_second = (
                    checkpoint[
                        second_key
                    ]
                )

                if (
                    saved_first.shape
                    != parameter.shape
                ):

                    raise ValueError(
                        "First-moment shape "
                        f"mismatch for {name}."
                    )

                if (
                    saved_second.shape
                    != parameter.shape
                ):

                    raise ValueError(
                        "Second-moment shape "
                        f"mismatch for {name}."
                    )

                np.copyto(
                    optimizer
                    .first_moments[
                        name
                    ],
                    saved_first,
                    casting="same_kind",
                )

                np.copyto(
                    optimizer
                    .second_moments[
                        name
                    ],
                    saved_second,
                    casting="same_kind",
                )

    return metadata