import numpy as np

from adam import adam_update


def main():

    print()
    print(
        "Adam Fundamentals Test"
    )

    print(
        "======================"
    )

    # ========================================================
    # Settings
    # ========================================================

    learning_rate = 0.001

    beta1 = 0.9
    beta2 = 0.999

    epsilon = 1e-8

    parameter = np.array(
        [
            1.0
        ],
        dtype=np.float32,
    )

    first_moment = np.zeros_like(
        parameter
    )

    second_moment = np.zeros_like(
        parameter
    )

    # ========================================================
    # STEP 1
    #
    # gradient = 0.5
    # ========================================================

    gradient_1 = np.array(
        [
            0.5
        ],
        dtype=np.float32,
    )

    adam_update(
        parameter=parameter,
        gradient=gradient_1,
        first_moment=first_moment,
        second_moment=second_moment,
        step=1,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )

    parameter_after_step_1 = float(
        parameter[0]
    )

    m_after_step_1 = float(
        first_moment[0]
    )

    v_after_step_1 = float(
        second_moment[0]
    )

    # Expected:
    #
    # m1 = 0.05
    # v1 = 0.00025
    #
    # bias corrected:
    #
    # m_hat = 0.5
    # v_hat = 0.25
    #
    # update ~= 0.001

    expected_parameter_1 = (
        0.99900000002
    )

    expected_m_1 = (
        0.05
    )

    expected_v_1 = (
        0.00025
    )

    step_1_parameter_passed = (
        np.isclose(
            parameter_after_step_1,
            expected_parameter_1,
            atol=1e-6,
        )
    )

    step_1_m_passed = (
        np.isclose(
            m_after_step_1,
            expected_m_1,
            atol=1e-7,
        )
    )

    step_1_v_passed = (
        np.isclose(
            v_after_step_1,
            expected_v_1,
            atol=1e-8,
        )
    )

    # ========================================================
    # STEP 2
    #
    # New gradient = 0.25
    #
    # Adam must combine this gradient with
    # its history from step 1.
    # ========================================================

    gradient_2 = np.array(
        [
            0.25
        ],
        dtype=np.float32,
    )

    adam_update(
        parameter=parameter,
        gradient=gradient_2,
        first_moment=first_moment,
        second_moment=second_moment,
        step=2,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )

    parameter_after_step_2 = float(
        parameter[0]
    )

    m_after_step_2 = float(
        first_moment[0]
    )

    v_after_step_2 = float(
        second_moment[0]
    )

    expected_parameter_2 = (
        0.9980678204
    )

    expected_m_2 = (
        0.07
    )

    expected_v_2 = (
        0.00031225
    )

    step_2_parameter_passed = (
        np.isclose(
            parameter_after_step_2,
            expected_parameter_2,
            atol=1e-6,
        )
    )

    step_2_m_passed = (
        np.isclose(
            m_after_step_2,
            expected_m_2,
            atol=1e-7,
        )
    )

    step_2_v_passed = (
        np.isclose(
            v_after_step_2,
            expected_v_2,
            atol=1e-8,
        )
    )

    # ========================================================
    # Other checks
    # ========================================================

    dtype_passed = (
        parameter.dtype
        == np.float32
        and
        first_moment.dtype
        == np.float32
        and
        second_moment.dtype
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
                first_moment
            )
        )
        and
        np.all(
            np.isfinite(
                second_moment
            )
        )
    )

    # ========================================================
    # Display
    # ========================================================

    print()

    print(
        "Step 1"
    )

    print(
        "------"
    )

    print(
        f"Parameter: "
        f"{parameter_after_step_1:.9f}"
    )

    print(
        f"First moment: "
        f"{m_after_step_1:.9f}"
    )

    print(
        f"Second moment: "
        f"{v_after_step_1:.9f}"
    )

    print()

    print(
        "Step 2"
    )

    print(
        "------"
    )

    print(
        f"Parameter: "
        f"{parameter_after_step_2:.9f}"
    )

    print(
        f"First moment: "
        f"{m_after_step_2:.9f}"
    )

    print(
        f"Second moment: "
        f"{v_after_step_2:.9f}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Step 1 parameter passed: "
        f"{step_1_parameter_passed}"
    )

    print(
        f"Step 1 first moment passed: "
        f"{step_1_m_passed}"
    )

    print(
        f"Step 1 second moment passed: "
        f"{step_1_v_passed}"
    )

    print(
        f"Step 2 parameter passed: "
        f"{step_2_parameter_passed}"
    )

    print(
        f"Step 2 first moment passed: "
        f"{step_2_m_passed}"
    )

    print(
        f"Step 2 second moment passed: "
        f"{step_2_v_passed}"
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
        step_1_parameter_passed
        and
        step_1_m_passed
        and
        step_1_v_passed
        and
        step_2_parameter_passed
        and
        step_2_m_passed
        and
        step_2_v_passed
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
            "SUCCESS: ADAM FUNDAMENTALS "
            "VALIDATED."
        )

    else:

        print(
            "FAILURE: Adam fundamentals "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()