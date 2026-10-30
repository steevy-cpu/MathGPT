import sys
import tempfile

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
    cross_entropy_loss,
    cross_entropy_loss_and_backward,
)

from training.adamw import AdamW

from training.gradient_clipping import (
    clip_grad_norm,
)

from training.parameters import (
    named_parameters,
)

from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


MAX_GRAD_NORM = 1.0


def create_model(
    seed,
):
    return GPT(
        vocab_size=7,
        context_length=3,
        d_model=4,
        num_layers=2,
        num_heads=2,
        d_ff=6,
        use_bias=True,
        tie_embeddings=True,
        seed=seed,
    )


def create_optimizer(
    model,
    learning_rate=0.001,
    weight_decay=0.01,
):
    return AdamW(
        model=model,
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=weight_decay,
    )


def training_step(
    model,
    optimizer,
    token_ids,
    targets,
):
    """
    One realistic training step:

        forward
        loss
        backward
        gradient clipping
        AdamW
    """

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

    (
        original_grad_norm,
        clip_scale,
    ) = clip_grad_norm(
        model=model,
        max_norm=MAX_GRAD_NORM,
    )

    optimizer.step()

    return (
        float(
            loss
        ),
        float(
            original_grad_norm
        ),
        float(
            clip_scale
        ),
    )


def models_match(
    model_a,
    model_b,
):
    """
    Compare every model parameter.
    """

    parameters_a = {
        name: parameter

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model_a
        )
    }

    parameters_b = {
        name: parameter

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            model_b
        )
    }

    if (
        parameters_a.keys()
        != parameters_b.keys()
    ):

        return False

    return all(
        np.array_equal(
            parameters_a[
                name
            ],
            parameters_b[
                name
            ],
        )

        for name
        in parameters_a
    )


def optimizer_states_match(
    optimizer_a,
    optimizer_b,
):
    """
    Compare AdamW m and v state.
    """

    if (
        optimizer_a.step_count
        != optimizer_b.step_count
    ):

        return False

    if (
        optimizer_a.first_moments.keys()
        != optimizer_b.first_moments.keys()
    ):

        return False

    for name in (
        optimizer_a.first_moments
    ):

        if not np.array_equal(
            optimizer_a
            .first_moments[
                name
            ],
            optimizer_b
            .first_moments[
                name
            ],
        ):

            return False

        if not np.array_equal(
            optimizer_a
            .second_moments[
                name
            ],
            optimizer_b
            .second_moments[
                name
            ],
        ):

            return False

    return True


def main():

    print()
    print(
        "Checkpoint Resume Test"
    )

    print(
        "======================"
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
    # ORIGINAL TRAINING RUN
    # ========================================================

    model_a = create_model(
        seed=42
    )

    optimizer_a = create_optimizer(
        model_a
    )

    # Step 1
    (
        step_1_loss,
        step_1_grad_norm,
        step_1_clip_scale,
    ) = training_step(
        model_a,
        optimizer_a,
        token_ids,
        targets,
    )

    # ========================================================
    # Save after step 1
    # ========================================================

    with tempfile.TemporaryDirectory() as temp_dir:

        checkpoint_path = (
            Path(
                temp_dir
            )
            / "test_checkpoint.npz"
        )

        save_metadata = (
            save_checkpoint(
                model=model_a,
                optimizer=optimizer_a,
                path=checkpoint_path,
                extra_state={
                    "training_step": 1,
                    "tokens_seen": 3,
                    "purpose":
                        "checkpoint resume test",
                },
            )
        )

        checkpoint_exists = (
            checkpoint_path.exists()
        )

        checkpoint_nonempty = (
            checkpoint_path.stat().st_size
            > 0
        )

        # ====================================================
        # Create COMPLETELY NEW model.
        #
        # Different seed proves weights really
        # come from the checkpoint.
        # ====================================================

        model_b = create_model(
            seed=999
        )

        # Deliberately use different optimizer
        # hyperparameters.
        #
        # load_checkpoint() should restore them.
        optimizer_b = create_optimizer(
            model_b,
            learning_rate=0.5,
            weight_decay=0.5,
        )

        models_different_before_load = (
            not models_match(
                model_a,
                model_b,
            )
        )

        # ====================================================
        # LOAD
        # ====================================================

        load_metadata = (
            load_checkpoint(
                model=model_b,
                optimizer=optimizer_b,
                path=checkpoint_path,
            )
        )

        # ====================================================
        # Check state immediately after load
        # ====================================================

        models_equal_after_load = (
            models_match(
                model_a,
                model_b,
            )
        )

        optimizer_equal_after_load = (
            optimizer_states_match(
                optimizer_a,
                optimizer_b,
            )
        )

        step_restored = (
            optimizer_b.step_count
            ==
            optimizer_a.step_count
            ==
            1
        )

        learning_rate_restored = (
            np.isclose(
                optimizer_b.learning_rate,
                optimizer_a.learning_rate,
            )
        )

        weight_decay_restored = (
            np.isclose(
                optimizer_b.weight_decay,
                optimizer_a.weight_decay,
            )
        )

        extra_state_restored = (
            load_metadata[
                "extra_state"
            ][
                "training_step"
            ]
            == 1
            and
            load_metadata[
                "extra_state"
            ][
                "tokens_seen"
            ]
            == 3
        )

        parameter_count_restored = (
            load_metadata[
                "parameter_count"
            ]
            ==
            model_a.parameter_count()
        )

        # ====================================================
        # Evaluate both BEFORE step 2
        # ====================================================

        loss_a_before_step_2 = (
            cross_entropy_loss(
                model_a.forward(
                    token_ids
                ),
                targets,
            )
        )

        loss_b_before_step_2 = (
            cross_entropy_loss(
                model_b.forward(
                    token_ids
                ),
                targets,
            )
        )

        pre_resume_loss_match = (
            np.isclose(
                loss_a_before_step_2,
                loss_b_before_step_2,
                atol=1e-8,
            )
        )

        # ====================================================
        # STEP 2
        #
        # A continues normally.
        #
        # B continues from checkpoint.
        # ====================================================

        (
            loss_a_step_2,
            norm_a_step_2,
            scale_a_step_2,
        ) = training_step(
            model_a,
            optimizer_a,
            token_ids,
            targets,
        )

        (
            loss_b_step_2,
            norm_b_step_2,
            scale_b_step_2,
        ) = training_step(
            model_b,
            optimizer_b,
            token_ids,
            targets,
        )

        # ====================================================
        # Compare post-step-2 state
        # ====================================================

        step_2_loss_match = (
            np.isclose(
                loss_a_step_2,
                loss_b_step_2,
                atol=1e-8,
            )
        )

        step_2_norm_match = (
            np.isclose(
                norm_a_step_2,
                norm_b_step_2,
                atol=1e-8,
            )
        )

        step_2_scale_match = (
            np.isclose(
                scale_a_step_2,
                scale_b_step_2,
                atol=1e-8,
            )
        )

        models_equal_after_step_2 = (
            models_match(
                model_a,
                model_b,
            )
        )

        optimizer_equal_after_step_2 = (
            optimizer_states_match(
                optimizer_a,
                optimizer_b,
            )
        )

        both_at_step_2 = (
            optimizer_a.step_count
            ==
            optimizer_b.step_count
            ==
            2
        )

        final_loss_a = (
            cross_entropy_loss(
                model_a.forward(
                    token_ids
                ),
                targets,
            )
        )

        final_loss_b = (
            cross_entropy_loss(
                model_b.forward(
                    token_ids
                ),
                targets,
            )
        )

        final_loss_match = (
            np.isclose(
                final_loss_a,
                final_loss_b,
                atol=1e-8,
            )
        )

        # ====================================================
        # Display
        # ====================================================

        print()

        print(
            "Step 1 before checkpoint"
        )

        print(
            "------------------------"
        )

        print(
            f"Loss: "
            f"{step_1_loss:.8f}"
        )

        print(
            f"Gradient norm: "
            f"{step_1_grad_norm:.8f}"
        )

        print(
            f"Clip scale: "
            f"{step_1_clip_scale:.8f}"
        )

        print(
            f"Optimizer step: "
            f"{save_metadata['optimizer_step_count']}"
        )

        print()

        print(
            "Checkpoint"
        )

        print(
            "----------"
        )

        print(
            f"Exists: "
            f"{checkpoint_exists}"
        )

        print(
            f"File size: "
            f"{checkpoint_path.stat().st_size:,} bytes"
        )

        print(
            f"Parameter count: "
            f"{load_metadata['parameter_count']:,}"
        )

        print(
            f"Saved optimizer: "
            f"{load_metadata['optimizer_class']}"
        )

        print()

        print(
            "Resume comparison"
        )

        print(
            "-----------------"
        )

        print(
            f"Loss A before step 2: "
            f"{loss_a_before_step_2:.8f}"
        )

        print(
            f"Loss B before step 2: "
            f"{loss_b_before_step_2:.8f}"
        )

        print()

        print(
            f"Step 2 loss A: "
            f"{loss_a_step_2:.8f}"
        )

        print(
            f"Step 2 loss B: "
            f"{loss_b_step_2:.8f}"
        )

        print()

        print(
            f"Final loss A: "
            f"{final_loss_a:.8f}"
        )

        print(
            f"Final loss B: "
            f"{final_loss_b:.8f}"
        )

        # ====================================================
        # Final checks
        # ====================================================

        print()

        print(
            "Final Checks"
        )

        print(
            "------------"
        )

        print(
            f"Checkpoint exists: "
            f"{checkpoint_exists}"
        )

        print(
            f"Checkpoint non-empty: "
            f"{checkpoint_nonempty}"
        )

        print(
            f"Fresh model differed before load: "
            f"{models_different_before_load}"
        )

        print(
            f"Model restored exactly: "
            f"{models_equal_after_load}"
        )

        print(
            f"AdamW state restored exactly: "
            f"{optimizer_equal_after_load}"
        )

        print(
            f"Step count restored: "
            f"{step_restored}"
        )

        print(
            f"Learning rate restored: "
            f"{learning_rate_restored}"
        )

        print(
            f"Weight decay restored: "
            f"{weight_decay_restored}"
        )

        print(
            f"Extra training state restored: "
            f"{extra_state_restored}"
        )

        print(
            f"Parameter count restored: "
            f"{parameter_count_restored}"
        )

        print(
            f"Pre-step-2 losses match: "
            f"{pre_resume_loss_match}"
        )

        print(
            f"Step-2 losses match: "
            f"{step_2_loss_match}"
        )

        print(
            f"Step-2 gradient norms match: "
            f"{step_2_norm_match}"
        )

        print(
            f"Step-2 clip scales match: "
            f"{step_2_scale_match}"
        )

        print(
            f"Models match after resumed step: "
            f"{models_equal_after_step_2}"
        )

        print(
            f"AdamW states match after resumed step: "
            f"{optimizer_equal_after_step_2}"
        )

        print(
            f"Both optimizers reached step 2: "
            f"{both_at_step_2}"
        )

        print(
            f"Final losses match: "
            f"{final_loss_match}"
        )

        all_passed = (
            checkpoint_exists
            and
            checkpoint_nonempty
            and
            models_different_before_load
            and
            models_equal_after_load
            and
            optimizer_equal_after_load
            and
            step_restored
            and
            learning_rate_restored
            and
            weight_decay_restored
            and
            extra_state_restored
            and
            parameter_count_restored
            and
            pre_resume_loss_match
            and
            step_2_loss_match
            and
            step_2_norm_match
            and
            step_2_scale_match
            and
            models_equal_after_step_2
            and
            optimizer_equal_after_step_2
            and
            both_at_step_2
            and
            final_loss_match
        )

        print()
        print(
            "=" * 65
        )

        if all_passed:

            print(
                "SUCCESS: CHECKPOINT RESUME "
                "IS EXACT AND CONTINUOUS."
            )

        else:

            print(
                "FAILURE: Checkpoint resume "
                "validation failed."
            )

        print(
            "=" * 65
        )


if __name__ == "__main__":
    main()