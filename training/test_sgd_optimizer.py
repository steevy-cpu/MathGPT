import sys
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

from training.loss import (
    cross_entropy_loss_and_backward,
)

from training.sgd import SGD

from training.parameters import (
    named_parameters,
)


def main():

    print()
    print(
        "Full-Model SGD Optimizer Test"
    )

    print(
        "============================="
    )

    # Tiny GPT
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

    optimizer = SGD(
        model=model,
        learning_rate=0.01,
    )

    token_ids = np.array(
        [
            [2, 4, 2]
        ],
        dtype=np.int64,
    )

    targets = np.array(
        [
            [4, 2, 5]
        ],
        dtype=np.int64,
    )

    # ========================================================
    # Forward + real loss
    # ========================================================

    logits = model.forward(
        token_ids
    )

    (
        loss,
        grad_logits,
    ) = cross_entropy_loss_and_backward(
        logits,
        targets,
    )

    # ========================================================
    # Backward
    # ========================================================

    model.zero_grad()

    model.backward(
        grad_logits
    )

    # ========================================================
    # Save one parameter before update
    # ========================================================

    parameter = (
        model
        .final_norm
        .gamma
    )

    gradient = (
        model
        .final_norm
        .dgamma
    )

    index = 0

    old_value = float(
        parameter[
            index
        ]
    )

    gradient_value = float(
        gradient[
            index
        ]
    )

    expected_new_value = (
        old_value
        - 0.01
        * gradient_value
    )

    # ========================================================
    # Copy ALL parameters before optimizer step
    # ========================================================

    parameters_before = {
        name: parameter.copy()

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    }

    # ========================================================
    # SGD STEP
    # ========================================================

    optimizer.step()

    new_value = float(
        parameter[
            index
        ]
    )

    # ========================================================
    # Check chosen parameter
    # ========================================================

    exact_update_passed = (
        np.isclose(
            new_value,
            expected_new_value,
            atol=1e-6,
        )
    )

    # ========================================================
    # Check that parameters with non-zero gradients moved
    # ========================================================

    changed_parameter_arrays = 0

    unchanged_parameter_arrays = 0

    for (
        name,
        current_parameter,
        gradient,
    ) in named_parameters(
        model
    ):

        old_parameter = (
            parameters_before[
                name
            ]
        )

        if np.any(
            gradient != 0
        ):

            if not np.array_equal(
                current_parameter,
                old_parameter,
            ):

                changed_parameter_arrays += 1

        else:

            if np.array_equal(
                current_parameter,
                old_parameter,
            ):

                unchanged_parameter_arrays += 1

    parameter_update_passed = (
        changed_parameter_arrays
        > 0
    )

    # ========================================================
    # Step count
    # ========================================================

    step_count_passed = (
        optimizer.step_count
        == 1
    )

    # ========================================================
    # zero_grad
    # ========================================================

    optimizer.zero_grad()

    gradients_zero = all(
        np.allclose(
            gradient,
            0.0,
            atol=0.0,
        )

        for (
            _,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    )

    # ========================================================
    # Float32 / finite parameters
    # ========================================================

    dtype_passed = all(
        parameter.dtype
        == np.float32

        for (
            _,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    )

    finite_passed = all(
        np.all(
            np.isfinite(
                parameter
            )
        )

        for (
            _,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        f"Loss before update: "
        f"{loss:.6f}"
    )

    print()

    print(
        "Selected parameter:"
    )

    print(
        "final_norm.gamma[0]"
    )

    print(
        f"Old value:      "
        f"{old_value:.8f}"
    )

    print(
        f"Gradient:       "
        f"{gradient_value:.8f}"
    )

    print(
        f"Expected value: "
        f"{expected_new_value:.8f}"
    )

    print(
        f"Actual value:   "
        f"{new_value:.8f}"
    )

    print()

    print(
        f"Parameter arrays changed: "
        f"{changed_parameter_arrays}"
    )

    print(
        f"Zero-gradient arrays unchanged: "
        f"{unchanged_parameter_arrays}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Exact SGD update passed: "
        f"{exact_update_passed}"
    )

    print(
        f"Model parameters updated: "
        f"{parameter_update_passed}"
    )

    print(
        f"Step count passed: "
        f"{step_count_passed}"
    )

    print(
        f"zero_grad passed: "
        f"{gradients_zero}"
    )

    print(
        f"Float32 preserved: "
        f"{dtype_passed}"
    )

    print(
        f"Finite parameters passed: "
        f"{finite_passed}"
    )

    all_passed = (
        exact_update_passed
        and
        parameter_update_passed
        and
        step_count_passed
        and
        gradients_zero
        and
        dtype_passed
        and
        finite_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: FULL-MODEL SGD "
            "OPTIMIZER VALIDATED."
        )

    else:

        print(
            "FAILURE: SGD optimizer "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()