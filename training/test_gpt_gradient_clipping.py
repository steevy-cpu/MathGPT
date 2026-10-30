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

from training.parameters import (
    named_parameters,
)

from training.gradient_clipping import (
    global_grad_norm,
    clip_grad_norm,
)

from training.adamw import AdamW


MAX_GRAD_NORM = 0.5


def main():

    print()
    print(
        "GPT Gradient Clipping Test"
    )

    print(
        "=========================="
    )

    # ========================================================
    # Tiny GPT
    # ========================================================

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

    optimizer = AdamW(
        model=model,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
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
    # Forward + backward
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

    model.zero_grad()

    model.backward(
        grad_logits
    )

    # ========================================================
    # Save gradients BEFORE clipping
    # ========================================================

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

    parameters_before_clip = {
        name: parameter.copy()

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    }

    norm_before = global_grad_norm(
        model
    )

    # ========================================================
    # Clip
    # ========================================================

    (
        returned_original_norm,
        scale_factor,
    ) = clip_grad_norm(
        model=model,
        max_norm=MAX_GRAD_NORM,
    )

    norm_after = global_grad_norm(
        model
    )

    # ========================================================
    # Expected scaling
    # ========================================================

    if (
        norm_before
        > MAX_GRAD_NORM
    ):

        expected_scale = (
            MAX_GRAD_NORM
            / (
                norm_before
                + 1e-6
            )
        )

    else:

        expected_scale = 1.0

    # ========================================================
    # Verify EVERY gradient received
    # the same scaling factor.
    # ========================================================

    every_gradient_scaled_correctly = True

    for (
        name,
        _,
        gradient,
    ) in named_parameters(
        model
    ):

        expected_gradient = (
            gradients_before[
                name
            ]
            * np.float32(
                expected_scale
            )
        )

        if not np.allclose(
            gradient,
            expected_gradient,
            atol=1e-7,
            rtol=1e-6,
        ):

            every_gradient_scaled_correctly = False

            break

    # ========================================================
    # Clipping must NOT modify parameters.
    #
    # It only modifies gradients.
    # ========================================================

    parameters_unchanged_by_clipping = all(
        np.array_equal(
            parameter,
            parameters_before_clip[
                name
            ],
        )

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    )

    # ========================================================
    # Other checks
    # ========================================================

    original_norm_returned_correctly = (
        np.isclose(
            returned_original_norm,
            norm_before,
            atol=1e-7,
        )
    )

    scale_factor_correct = (
        np.isclose(
            scale_factor,
            expected_scale,
            atol=1e-7,
        )
    )

    clipped_norm_passed = (
        norm_after
        <= MAX_GRAD_NORM
        + 1e-5
    )

    gradients_finite = all(
        np.all(
            np.isfinite(
                gradient
            )
        )

        for (
            _,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    )

    gradients_float32 = all(
        gradient.dtype
        == np.float32

        for (
            _,
            _,
            gradient,
        ) in named_parameters(
            model
        )
    )

    # ========================================================
    # Finally make sure clipped gradients are
    # accepted by AdamW.
    # ========================================================

    optimizer.step()

    parameters_finite_after_step = all(
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

    optimizer_step_passed = (
        optimizer.step_count
        == 1
    )

    # ========================================================
    # Display
    # ========================================================

    print()

    print(
        f"Loss: "
        f"{loss:.6f}"
    )

    print()

    print(
        f"Global norm BEFORE: "
        f"{norm_before:.8f}"
    )

    print(
        f"Maximum allowed norm: "
        f"{MAX_GRAD_NORM:.8f}"
    )

    print(
        f"Scale factor: "
        f"{scale_factor:.8f}"
    )

    print(
        f"Global norm AFTER: "
        f"{norm_after:.8f}"
    )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"Original norm returned correctly: "
        f"{original_norm_returned_correctly}"
    )

    print(
        f"Scale factor correct: "
        f"{scale_factor_correct}"
    )

    print(
        f"Global norm <= max norm: "
        f"{clipped_norm_passed}"
    )

    print(
        f"Every gradient scaled equally: "
        f"{every_gradient_scaled_correctly}"
    )

    print(
        f"Parameters untouched by clipping: "
        f"{parameters_unchanged_by_clipping}"
    )

    print(
        f"Gradients remain float32: "
        f"{gradients_float32}"
    )

    print(
        f"Gradients finite: "
        f"{gradients_finite}"
    )

    print(
        f"AdamW accepts clipped gradients: "
        f"{optimizer_step_passed}"
    )

    print(
        f"Parameters finite after AdamW: "
        f"{parameters_finite_after_step}"
    )

    all_passed = (
        original_norm_returned_correctly
        and
        scale_factor_correct
        and
        clipped_norm_passed
        and
        every_gradient_scaled_correctly
        and
        parameters_unchanged_by_clipping
        and
        gradients_float32
        and
        gradients_finite
        and
        optimizer_step_passed
        and
        parameters_finite_after_step
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: FULL-MODEL GLOBAL "
            "GRADIENT CLIPPING VALIDATED."
        )

    else:

        print(
            "FAILURE: GPT gradient clipping "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()