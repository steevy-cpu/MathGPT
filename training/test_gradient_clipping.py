import numpy as np

from gradient_clipping import (
    global_norm_from_gradients,
    clip_gradients,
)


def main():

    print()
    print(
        "Gradient Clipping Fundamentals Test"
    )

    print(
        "==================================="
    )

    # ========================================================
    # TEST 1
    #
    # Known global norm:
    #
    # sqrt(
    #     3^2
    #     + 4^2
    #     + 12^2
    # )
    #
    # = 13
    # ========================================================

    gradient_1 = np.array(
        [
            3.0,
            4.0,
        ],
        dtype=np.float32,
    )

    gradient_2 = np.array(
        [
            12.0
        ],
        dtype=np.float32,
    )

    gradients = [
        gradient_1,
        gradient_2,
    ]

    original_1 = (
        gradient_1.copy()
    )

    original_2 = (
        gradient_2.copy()
    )

    norm_before = (
        global_norm_from_gradients(
            gradients
        )
    )

    known_norm_passed = (
        np.isclose(
            norm_before,
            13.0,
            atol=1e-7,
        )
    )

    # ========================================================
    # Clip 13 -> maximum 5
    # ========================================================

    max_norm = 5.0

    (
        returned_norm,
        scale_factor,
    ) = clip_gradients(
        gradients=gradients,
        max_norm=max_norm,
    )

    norm_after = (
        global_norm_from_gradients(
            gradients
        )
    )

    expected_scale = (
        max_norm
        / (
            13.0
            + 1e-6
        )
    )

    # ========================================================
    # Checks
    # ========================================================

    returned_norm_passed = (
        np.isclose(
            returned_norm,
            13.0,
            atol=1e-7,
        )
    )

    scale_passed = (
        np.isclose(
            scale_factor,
            expected_scale,
            atol=1e-7,
        )
    )

    clipped_norm_passed = (
        norm_after
        <= max_norm
        + 1e-5
    )

    gradient_1_passed = (
        np.allclose(
            gradient_1,
            original_1
            * np.float32(
                expected_scale
            ),
            atol=1e-6,
        )
    )

    gradient_2_passed = (
        np.allclose(
            gradient_2,
            original_2
            * np.float32(
                expected_scale
            ),
            atol=1e-6,
        )
    )

    # ========================================================
    # Direction preservation
    #
    # Ratios between components should remain
    # unchanged because every value receives
    # the SAME multiplier.
    # ========================================================

    original_ratio = (
        original_1[0]
        / original_1[1]
    )

    clipped_ratio = (
        gradient_1[0]
        / gradient_1[1]
    )

    direction_passed = (
        np.isclose(
            original_ratio,
            clipped_ratio,
            atol=1e-7,
        )
    )

    # ========================================================
    # TEST 2
    #
    # If gradient norm is already below
    # max_norm, nothing should change.
    # ========================================================

    small_gradient = np.array(
        [
            3.0,
            4.0,
        ],
        dtype=np.float32,
    )

    small_original = (
        small_gradient.copy()
    )

    (
        small_norm,
        small_scale,
    ) = clip_gradients(
        gradients=[
            small_gradient
        ],
        max_norm=10.0,
    )

    no_clip_norm_passed = (
        np.isclose(
            small_norm,
            5.0,
            atol=1e-7,
        )
    )

    no_clip_scale_passed = (
        small_scale
        == 1.0
    )

    no_clip_values_passed = (
        np.array_equal(
            small_gradient,
            small_original,
        )
    )

    # ========================================================
    # Dtype / finite
    # ========================================================

    dtype_passed = (
        gradient_1.dtype
        == np.float32
        and
        gradient_2.dtype
        == np.float32
        and
        small_gradient.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                gradient_1
            )
        )
        and
        np.all(
            np.isfinite(
                gradient_2
            )
        )
    )

    # ========================================================
    # Display
    # ========================================================

    print()

    print(
        f"Norm before clipping: "
        f"{norm_before:.8f}"
    )

    print(
        f"Maximum norm:        "
        f"{max_norm:.8f}"
    )

    print(
        f"Scale factor:        "
        f"{scale_factor:.8f}"
    )

    print(
        f"Norm after clipping: "
        f"{norm_after:.8f}"
    )

    print()

    print(
        "Gradient 1 after clipping:"
    )

    print(
        gradient_1
    )

    print(
        "Gradient 2 after clipping:"
    )

    print(
        gradient_2
    )

    print()

    print(
        "No-clipping test:"
    )

    print(
        f"Original norm: "
        f"{small_norm:.8f}"
    )

    print(
        f"Scale factor:  "
        f"{small_scale:.8f}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Known norm = 13 passed: "
        f"{known_norm_passed}"
    )

    print(
        f"Returned original norm passed: "
        f"{returned_norm_passed}"
    )

    print(
        f"Scale factor passed: "
        f"{scale_passed}"
    )

    print(
        f"Clipped norm <= 5: "
        f"{clipped_norm_passed}"
    )

    print(
        f"Gradient 1 scaled correctly: "
        f"{gradient_1_passed}"
    )

    print(
        f"Gradient 2 scaled correctly: "
        f"{gradient_2_passed}"
    )

    print(
        f"Gradient direction preserved: "
        f"{direction_passed}"
    )

    print(
        f"No-clip norm passed: "
        f"{no_clip_norm_passed}"
    )

    print(
        f"No-clip scale = 1: "
        f"{no_clip_scale_passed}"
    )

    print(
        f"No-clip values unchanged: "
        f"{no_clip_values_passed}"
    )

    print(
        f"Float32 preserved: "
        f"{dtype_passed}"
    )

    print(
        f"Finite values passed: "
        f"{finite_passed}"
    )

    all_passed = (
        known_norm_passed
        and
        returned_norm_passed
        and
        scale_passed
        and
        clipped_norm_passed
        and
        gradient_1_passed
        and
        gradient_2_passed
        and
        direction_passed
        and
        no_clip_norm_passed
        and
        no_clip_scale_passed
        and
        no_clip_values_passed
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
            "SUCCESS: GRADIENT CLIPPING "
            "FUNDAMENTALS VALIDATED."
        )

    else:

        print(
            "FAILURE: Gradient clipping "
            "fundamentals failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()