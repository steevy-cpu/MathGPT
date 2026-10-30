import json
from pathlib import Path

import numpy as np

from parameters import (
    named_parameters,
    parameter_count_from_traversal,
)

import sys


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


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


EXPECTED_PARAMETERS = (
    8_110_848
)


def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def main():

    print()
    print(
        "GPT Parameter Traversal Test"
    )

    print(
        "============================"
    )

    config = load_config()

    model = GPT.from_config(
        config,
        seed=42,
    )

    parameters = list(
        named_parameters(
            model
        )
    )

    # ========================================================
    # Count parameter arrays
    # ========================================================

    print()
    print(
        f"Parameter arrays found: "
        f"{len(parameters)}"
    )

    # ========================================================
    # Scalar count
    # ========================================================

    traversal_count = (
        parameter_count_from_traversal(
            model
        )
    )

    model_count = (
        model.parameter_count()
    )

    print(
        f"Traversal scalar count: "
        f"{traversal_count:,}"
    )

    print(
        f"Model parameter count: "
        f"{model_count:,}"
    )

    # ========================================================
    # Check shapes
    # ========================================================

    shape_passed = all(
        parameter.shape
        == gradient.shape

        for (
            _,
            parameter,
            gradient,
        ) in parameters
    )

    # ========================================================
    # Check dtype
    # ========================================================

    dtype_passed = all(
        parameter.dtype
        == np.float32
        and
        gradient.dtype
        == np.float32

        for (
            _,
            parameter,
            gradient,
        ) in parameters
    )

    # ========================================================
    # Check finite values
    # ========================================================

    finite_passed = all(
        np.all(
            np.isfinite(
                parameter
            )
        )
        and
        np.all(
            np.isfinite(
                gradient
            )
        )

        for (
            _,
            parameter,
            gradient,
        ) in parameters
    )

    # ========================================================
    # Check for duplicate parameter arrays
    #
    # Important because weight tying should NOT
    # cause token embedding to be counted twice.
    # ========================================================

    parameter_ids = [
        id(
            parameter
        )

        for (
            _,
            parameter,
            _,
        ) in parameters
    ]

    unique_parameters_passed = (
        len(
            parameter_ids
        )
        ==
        len(
            set(
                parameter_ids
            )
        )
    )

    # ========================================================
    # Count check
    # ========================================================

    traversal_count_passed = (
        traversal_count
        == EXPECTED_PARAMETERS
    )

    model_count_passed = (
        model_count
        == EXPECTED_PARAMETERS
    )

    # ========================================================
    # Show first and last few parameters
    # ========================================================

    print()
    print(
        "Sample parameters"
    )

    print(
        "-----------------"
    )

    for (
        name,
        parameter,
        gradient,
    ) in (
        parameters[:5]
        + parameters[-5:]
    ):

        print(
            f"{name:45} "
            f"{str(parameter.shape):18} "
            f"{parameter.size:>10,}"
        )

    # ========================================================
    # Final checks
    # ========================================================

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Traversal count = 8,110,848: "
        f"{traversal_count_passed}"
    )

    print(
        f"Model count = 8,110,848: "
        f"{model_count_passed}"
    )

    print(
        f"Parameter/gradient shapes match: "
        f"{shape_passed}"
    )

    print(
        f"All float32: "
        f"{dtype_passed}"
    )

    print(
        f"All finite: "
        f"{finite_passed}"
    )

    print(
        f"No duplicate parameters: "
        f"{unique_parameters_passed}"
    )

    all_passed = (
        traversal_count_passed
        and
        model_count_passed
        and
        shape_passed
        and
        dtype_passed
        and
        finite_passed
        and
        unique_parameters_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: GPT PARAMETER "
            "TRAVERSAL VALIDATED."
        )

    else:

        print(
            "FAILURE: Parameter traversal "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()