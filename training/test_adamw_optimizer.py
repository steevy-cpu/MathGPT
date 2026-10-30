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

from training.adamw import (
    AdamW,
    should_apply_weight_decay,
)

from training.parameters import (
    named_parameters,
)


def main():

    print()
    print(
        "Full-Model AdamW Optimizer Test"
    )

    print(
        "==============================="
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

    optimizer = AdamW(
        model=model,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.1,
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
    # Real GPT loss + backward
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
    # Inspect parameter groups
    # ========================================================

    decay_names = []

    no_decay_names = []

    for (
        name,
        parameter,
        _,
    ) in named_parameters(
        model
    ):

        if should_apply_weight_decay(
            parameter
        ):

            decay_names.append(
                name
            )

        else:

            no_decay_names.append(
                name
            )

    # ========================================================
    # Save state
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
    # AdamW step
    # ========================================================

    optimizer.step()

    # ========================================================
    # Verify all parameters finite
    # ========================================================

    parameters_finite = all(
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
    # Verify every parameter classification
    # ========================================================

    classification_complete = (
        len(
            decay_names
        )
        +
        len(
            no_decay_names
        )
        ==
        36
    )

    # Token embedding should decay.
    token_embedding_decays = (
        "token_embedding.weight"
        in decay_names
    )

    # Linear matrix should decay.
    linear_weight_decays = (
        "block_0.attention.query.weight"
        in decay_names
    )

    # Bias should not decay.
    bias_not_decayed = (
        "block_0.attention.query.bias"
        in no_decay_names
    )

    # LayerNorm should not decay.
    norm_not_decayed = (
        "final_norm.gamma"
        in no_decay_names
        and
        "final_norm.beta"
        in no_decay_names
    )

    # ========================================================
    # All non-zero gradients should cause movement
    # ========================================================

    changed_arrays = 0

    for (
        name,
        parameter,
        gradient,
    ) in named_parameters(
        model
    ):

        if (
            np.any(
                gradient != 0.0
            )
            and
            not np.array_equal(
                parameter,
                parameters_before[
                    name
                ],
            )
        ):

            changed_arrays += 1

    parameters_updated = (
        changed_arrays
        > 0
    )

    # ========================================================
    # Adam state
    # ========================================================

    state_count_passed = (
        len(
            optimizer.first_moments
        )
        == 36
        and
        len(
            optimizer.second_moments
        )
        == 36
    )

    state_finite = all(
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
    # Display
    # ========================================================

    print()

    print(
        f"Loss before update: "
        f"{loss:.6f}"
    )

    print()

    print(
        f"Parameter arrays: 36"
    )

    print(
        f"Decay arrays: "
        f"{len(decay_names)}"
    )

    print(
        f"No-decay arrays: "
        f"{len(no_decay_names)}"
    )

    print()

    print(
        "Example decay parameters:"
    )

    for name in decay_names[:5]:

        print(
            f"  {name}"
        )

    print()

    print(
        "Example no-decay parameters:"
    )

    for name in no_decay_names[:5]:

        print(
            f"  {name}"
        )

    print()

    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"All parameters classified: "
        f"{classification_complete}"
    )

    print(
        f"Token embedding decays: "
        f"{token_embedding_decays}"
    )

    print(
        f"Linear weights decay: "
        f"{linear_weight_decays}"
    )

    print(
        f"Biases excluded: "
        f"{bias_not_decayed}"
    )

    print(
        f"LayerNorm excluded: "
        f"{norm_not_decayed}"
    )

    print(
        f"Model parameters updated: "
        f"{parameters_updated}"
    )

    print(
        f"Parameters finite: "
        f"{parameters_finite}"
    )

    print(
        f"Adam state count passed: "
        f"{state_count_passed}"
    )

    print(
        f"Adam state finite: "
        f"{state_finite}"
    )

    print(
        f"Step count passed: "
        f"{step_count_passed}"
    )

    all_passed = (
        classification_complete
        and
        token_embedding_decays
        and
        linear_weight_decays
        and
        bias_not_decayed
        and
        norm_not_decayed
        and
        parameters_updated
        and
        parameters_finite
        and
        state_count_passed
        and
        state_finite
        and
        step_count_passed
    )

    print()
    print(
        "=" * 65
    )

    if all_passed:

        print(
            "SUCCESS: FULL-MODEL ADAMW "
            "OPTIMIZER VALIDATED."
        )

    else:

        print(
            "FAILURE: AdamW optimizer "
            "validation failed."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()