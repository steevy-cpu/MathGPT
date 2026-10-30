import numpy as np

from adamw import adamw_update


def main():

    print()
    print(
        "AdamW Fundamentals Test"
    )

    print(
        "======================="
    )

    learning_rate = 0.001
    weight_decay = 0.1

    beta1 = 0.9
    beta2 = 0.999

    epsilon = 1e-8

    # ========================================================
    # TEST 1
    #
    # Zero loss gradient.
    #
    # Adam contribution = zero.
    #
    # But AdamW weight decay should STILL
    # shrink the parameter.
    # ========================================================

    parameter = np.array(
        [
            2.0
        ],
        dtype=np.float32,
    )

    gradient = np.array(
        [
            0.0
        ],
        dtype=np.float32,
    )

    first_moment = np.zeros_like(
        parameter
    )

    second_moment = np.zeros_like(
        parameter
    )

    adamw_update(
        parameter=parameter,
        gradient=gradient,
        first_moment=first_moment,
        second_moment=second_moment,
        step=1,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        apply_weight_decay=True,
    )

    expected_decay_only = (
        2.0
        * (
            1.0
            - learning_rate
            * weight_decay
        )
    )

    decay_only_passed = np.isclose(
        parameter[0],
        expected_decay_only,
        atol=1e-7,
    )

    moments_stay_zero = (
        np.allclose(
            first_moment,
            0.0,
        )
        and
        np.allclose(
            second_moment,
            0.0,
        )
    )

    # ========================================================
    # TEST 2
    #
    # Same parameter, but decay disabled.
    #
    # With zero gradient nothing should move.
    # ========================================================

    no_decay_parameter = np.array(
        [
            2.0
        ],
        dtype=np.float32,
    )

    no_decay_m = np.zeros_like(
        no_decay_parameter
    )

    no_decay_v = np.zeros_like(
        no_decay_parameter
    )

    adamw_update(
        parameter=no_decay_parameter,
        gradient=gradient,
        first_moment=no_decay_m,
        second_moment=no_decay_v,
        step=1,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        apply_weight_decay=False,
    )

    decay_disabled_passed = (
        np.isclose(
            no_decay_parameter[0],
            2.0,
            atol=1e-7,
        )
    )

    # ========================================================
    # TEST 3
    #
    # Gradient + weight decay.
    # ========================================================

    combined_parameter = np.array(
        [
            1.0
        ],
        dtype=np.float32,
    )

    combined_gradient = np.array(
        [
            0.5
        ],
        dtype=np.float32,
    )

    combined_m = np.zeros_like(
        combined_parameter
    )

    combined_v = np.zeros_like(
        combined_parameter
    )

    adamw_update(
        parameter=combined_parameter,
        gradient=combined_gradient,
        first_moment=combined_m,
        second_moment=combined_v,
        step=1,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        apply_weight_decay=True,
    )

    # First Adam step:
    #
    # adaptive update ~= 0.001
    #
    # decay:
    #
    # 1.0 * (1 - 0.001 * 0.1)
    # = 0.9999
    #
    # then Adam:
    #
    # ~= 0.9989

    expected_combined = (
        (
            1.0
            * (
                1.0
                - learning_rate
                * weight_decay
            )
        )
        -
        (
            learning_rate
            * 0.5
            / (
                0.5
                + epsilon
            )
        )
    )

    combined_passed = (
        np.isclose(
            combined_parameter[0],
            expected_combined,
            atol=1e-6,
        )
    )

    # Adam moments must contain the
    # LOSS gradient only.
    expected_m = 0.05

    expected_v = 0.00025

    moment_decoupling_passed = (
        np.isclose(
            combined_m[0],
            expected_m,
            atol=1e-7,
        )
        and
        np.isclose(
            combined_v[0],
            expected_v,
            atol=1e-8,
        )
    )

    dtype_passed = (
        parameter.dtype
        == np.float32
        and
        combined_parameter.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                parameter
            )
        )
        and
        np.all(
            np.isfinite(
                combined_parameter
            )
        )
    )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Decay-only test"
    )

    print(
        "---------------"
    )

    print(
        f"Original: 2.000000000"
    )

    print(
        f"Expected: "
        f"{expected_decay_only:.9f}"
    )

    print(
        f"Actual:   "
        f"{parameter[0]:.9f}"
    )

    print()

    print(
        "Gradient + decay test"
    )

    print(
        "---------------------"
    )

    print(
        f"Expected parameter: "
        f"{expected_combined:.9f}"
    )

    print(
        f"Actual parameter:   "
        f"{combined_parameter[0]:.9f}"
    )

    print()

    print(
        f"First moment:  "
        f"{combined_m[0]:.9f}"
    )

    print(
        f"Second moment: "
        f"{combined_v[0]:.9f}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Decay works with zero gradient: "
        f"{decay_only_passed}"
    )

    print(
        f"Adam moments stay zero: "
        f"{moments_stay_zero}"
    )

    print(
        f"Decay can be disabled: "
        f"{decay_disabled_passed}"
    )

    print(
        f"Gradient + decay update passed: "
        f"{combined_passed}"
    )

    print(
        f"Weight decay decoupled from moments: "
        f"{moment_decoupling_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"Finite values passed: "
        f"{finite_passed}"
    )

    all_passed = (
        decay_only_passed
        and
        moments_stay_zero
        and
        decay_disabled_passed
        and
        combined_passed
        and
        moment_decoupling_passed
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
            "SUCCESS: ADAMW FUNDAMENTALS "
            "VALIDATED."
        )

    else:

        print(
            "FAILURE: AdamW fundamentals "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()