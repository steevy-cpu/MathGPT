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

from training.adam import Adam

from training.parameters import (
    named_parameters,
)


def main():

    print()
    print(
        "Full-Model Adam Optimizer Test"
    )

    print(
        "=============================="
    )

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

    optimizer = Adam(
        model=model,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
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
    # Forward
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
    # Save model state before Adam step
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

    gradients_before = {
        name: gradient.copy()

        for (
            name,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    }

    # ========================================================
    # Selected parameter for exact verification
    # ========================================================

    selected_name = (
        "final_norm.gamma"
    )

    selected_parameter = (
        model.final_norm.gamma
    )

    selected_gradient = (
        model.final_norm.dgamma
    )

    selected_index = 0

    old_value = float(
        selected_parameter[
            selected_index
        ]
    )

    gradient_value = float(
        selected_gradient[
            selected_index
        ]
    )

    # ========================================================
    # Expected FIRST Adam step
    #
    # m0 = 0
    # v0 = 0
    #
    # After bias correction:
    #
    # m_hat = gradient
    # v_hat = gradient^2
    # ========================================================

    expected_first_moment = (
        0.1
        * gradient_value
    )

    expected_second_moment = (
        0.001
        * gradient_value ** 2
    )

    expected_m_hat = (
        expected_first_moment
        / (
            1.0 - 0.9
        )
    )

    expected_v_hat = (
        expected_second_moment
        / (
            1.0 - 0.999
        )
    )

    expected_update = (
        0.001
        * expected_m_hat
        / (
            np.sqrt(
                expected_v_hat
            )
            + 1e-8
        )
    )

    expected_new_value = (
        old_value
        - expected_update
    )

    # ========================================================
    # ADAM STEP
    # ========================================================

    optimizer.step()

    new_value = float(
        selected_parameter[
            selected_index
        ]
    )

    actual_first_moment = float(
        optimizer
        .first_moments[
            selected_name
        ][
            selected_index
        ]
    )

    actual_second_moment = float(
        optimizer
        .second_moments[
            selected_name
        ][
            selected_index
        ]
    )

    # ========================================================
    # Exact checks
    # ========================================================

    parameter_update_passed = (
        np.isclose(
            new_value,
            expected_new_value,
            atol=1e-6,
        )
    )

    first_moment_passed = (
        np.isclose(
            actual_first_moment,
            expected_first_moment,
            atol=1e-7,
        )
    )

    second_moment_passed = (
        np.isclose(
            actual_second_moment,
            expected_second_moment,
            atol=1e-8,
        )
    )

    # ========================================================
    # Check all nonzero-gradient arrays moved
    # ========================================================

    nonzero_gradient_arrays = 0

    changed_arrays = 0

    for (
        name,
        parameter,
        _,
    ) in named_parameters(
        model
    ):

        gradient = (
            gradients_before[
                name
            ]
        )

        if np.any(
            gradient != 0.0
        ):

            nonzero_gradient_arrays += 1

            if not np.array_equal(
                parameter,
                parameters_before[
                    name
                ],
            ):

                changed_arrays += 1

    all_nonzero_arrays_updated = (
        changed_arrays
        ==
        nonzero_gradient_arrays
    )

    # ========================================================
    # Adam state checks
    # ========================================================

    parameter_array_count = sum(
        1
        for _ in named_parameters(
            model
        )
    )

    first_state_count_passed = (
        len(
            optimizer.first_moments
        )
        ==
        parameter_array_count
    )

    second_state_count_passed = (
        len(
            optimizer.second_moments
        )
        ==
        parameter_array_count
    )

    state_shapes_passed = all(
        optimizer
        .first_moments[
            name
        ].shape
        ==
        parameter.shape
        and
        optimizer
        .second_moments[
            name
        ].shape
        ==
        parameter.shape

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    )

    state_dtype_passed = all(
        optimizer
        .first_moments[
            name
        ].dtype
        ==
        np.float32
        and
        optimizer
        .second_moments[
            name
        ].dtype
        ==
        np.float32

        for (
            name,
            _,
            _,
        ) in named_parameters(
            model
        )
    )

    state_finite_passed = all(
        np.all(
            np.isfinite(
                optimizer
                .first_moments[
                    name
                ]
            )
        )
        and
        np.all(
            np.isfinite(
                optimizer
                .second_moments[
                    name
                ]
            )
        )

        for (
            name,
            _,
            _,
        ) in named_parameters(
            model
        )
    )

    step_count_passed = (
        optimizer.step_count
        == 1
    )

    # ========================================================
    # zero_grad must NOT erase Adam state
    # ========================================================

    first_moment_copy = {
        name: state.copy()

        for (
            name,
            state,
        ) in optimizer
        .first_moments
        .items()
    }

    second_moment_copy = {
        name: state.copy()

        for (
            name,
            state,
        ) in optimizer
        .second_moments
        .items()
    }

    optimizer.zero_grad()

    gradients_zero_passed = all(
        np.all(
            gradient
            == 0.0
        )

        for (
            _,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    )

    adam_state_preserved = all(
        np.array_equal(
            optimizer
            .first_moments[
                name
            ],
            first_moment_copy[
                name
            ],
        )
        and
        np.array_equal(
            optimizer
            .second_moments[
                name
            ],
            second_moment_copy[
                name
            ],
        )

        for name
        in first_moment_copy
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
        f"{old_value:.9f}"
    )

    print(
        f"Gradient:       "
        f"{gradient_value:.9f}"
    )

    print(
        f"Expected value: "
        f"{expected_new_value:.9f}"
    )

    print(
        f"Actual value:   "
        f"{new_value:.9f}"
    )

    print()

    print(
        "Adam state:"
    )

    print(
        f"Expected m: "
        f"{expected_first_moment:.9f}"
    )

    print(
        f"Actual m:   "
        f"{actual_first_moment:.9f}"
    )

    print(
        f"Expected v: "
        f"{expected_second_moment:.12f}"
    )

    print(
        f"Actual v:   "
        f"{actual_second_moment:.12f}"
    )

    print()

    print(
        f"Parameter arrays: "
        f"{parameter_array_count}"
    )

    print(
        f"Nonzero-gradient arrays: "
        f"{nonzero_gradient_arrays}"
    )

    print(
        f"Changed arrays: "
        f"{changed_arrays}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Exact Adam parameter update: "
        f"{parameter_update_passed}"
    )

    print(
        f"First moment correct: "
        f"{first_moment_passed}"
    )

    print(
        f"Second moment correct: "
        f"{second_moment_passed}"
    )

    print(
        f"All nonzero-gradient arrays updated: "
        f"{all_nonzero_arrays_updated}"
    )

    print(
        f"First-moment state count passed: "
        f"{first_state_count_passed}"
    )

    print(
        f"Second-moment state count passed: "
        f"{second_state_count_passed}"
    )

    print(
        f"State shapes passed: "
        f"{state_shapes_passed}"
    )

    print(
        f"State float32 passed: "
        f"{state_dtype_passed}"
    )

    print(
        f"State finite passed: "
        f"{state_finite_passed}"
    )

    print(
        f"Step count passed: "
        f"{step_count_passed}"
    )

    print(
        f"zero_grad passed: "
        f"{gradients_zero_passed}"
    )

    print(
        f"Adam state survives zero_grad: "
        f"{adam_state_preserved}"
    )

    all_passed = (
        parameter_update_passed
        and
        first_moment_passed
        and
        second_moment_passed
        and
        all_nonzero_arrays_updated
        and
        first_state_count_passed
        and
        second_state_count_passed
        and
        state_shapes_passed
        and
        state_dtype_passed
        and
        state_finite_passed
        and
        step_count_passed
        and
        gradients_zero_passed
        and
        adam_state_preserved
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: FULL-MODEL ADAM "
            "OPTIMIZER VALIDATED."
        )

    else:

        print(
            "FAILURE: Adam optimizer "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()