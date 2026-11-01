import json
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

from training.adamw import AdamW

from training.data_loader import (
    TokenBatchLoader,
)

from training.parameters import (
    named_parameters,
)

from training.train_loop import (
    run_training_steps,
)

from training.checkpoint_manager import (
    restore_training_checkpoint,
)


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
):
    return AdamW(
        model=model,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )


def create_loader(
    binary_path,
    seed,
):
    return TokenBatchLoader(
        binary_path=binary_path,
        context_length=3,
        batch_size=2,
        seed=seed,
    )


def model_parameters_equal(
    model_a,
    model_b,
):
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
        !=
        parameters_b.keys()
    ):
        return False

    return all(
        np.array_equal(
            parameters_a[name],
            parameters_b[name],
        )

        for name
        in parameters_a
    )


def optimizer_states_equal(
    optimizer_a,
    optimizer_b,
):
    if (
        optimizer_a.step_count
        !=
        optimizer_b.step_count
    ):
        return False

    if (
        optimizer_a.first_moments.keys()
        !=
        optimizer_b.first_moments.keys()
    ):
        return False

    for name in (
        optimizer_a.first_moments
    ):

        if not np.array_equal(
            optimizer_a.first_moments[name],
            optimizer_b.first_moments[name],
        ):
            return False

        if not np.array_equal(
            optimizer_a.second_moments[name],
            optimizer_b.second_moments[name],
        ):
            return False

    return True


def loader_states_equal(
    loader_a,
    loader_b,
):
    state_a = loader_a.state_dict()
    state_b = loader_b.state_dict()

    return (
        json.dumps(
            state_a,
            sort_keys=True,
        )
        ==
        json.dumps(
            state_b,
            sort_keys=True,
        )
    )


def metric_histories_equal(
    history_a,
    history_b,
):
    """
    Compare deterministic mathematical metrics.

    Timing / tokens-per-second are intentionally
    excluded because wall-clock performance is
    naturally nondeterministic.
    """

    if (
        len(history_a)
        !=
        len(history_b)
    ):
        return False

    metric_keys = [
        "step",
        "loss",
        "learning_rate",
        "gradient_norm",
        "clip_scale",
        "batch_tokens",
        "tokens_seen",
    ]

    for (
        metrics_a,
        metrics_b,
    ) in zip(
        history_a,
        history_b,
    ):

        for key in metric_keys:

            if (
                metrics_a[key]
                !=
                metrics_b[key]
            ):
                return False

    return True


def main():

    print()
    print(
        "Exact Full-Trainer Resume Test"
    )

    print(
        "=============================="
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        # ====================================================
        # Synthetic corpus
        # ====================================================

        binary_path = (
            temp_dir
            / "tiny.bin"
        )

        tokens = np.array(
            [
                1, 2, 3, 4, 5, 6,
                2, 3, 4, 5, 6, 1,
                3, 4, 5, 6, 1, 2,
                4, 5, 6, 1, 2, 3,
                5, 6, 1, 2, 3, 4,
                6, 1, 2, 3, 4, 5,
            ],
            dtype=np.uint16,
        )

        tokens.tofile(
            binary_path
        )

        checkpoint_directory = (
            temp_dir
            / "checkpoints"
        )

        # ====================================================
        # CONTINUOUS RUN A
        # ====================================================

        model_a = create_model(
            seed=42
        )

        optimizer_a = create_optimizer(
            model_a
        )

        loader_a = create_loader(
            binary_path,
            seed=42,
        )

        # ----------------------------------------------------
        # Steps 1 and 2.
        #
        # Step 2 creates checkpoint.
        # ----------------------------------------------------

        (
            history_a_first,
            tokens_seen_a,
            checkpoints,
        ) = run_training_steps(
            model=model_a,
            optimizer=optimizer_a,
            train_loader=loader_a,
            num_steps=2,
            max_learning_rate=0.001,
            min_learning_rate=0.0001,
            warmup_steps=2,
            total_steps=10,
            max_grad_norm=1.0,
            starting_tokens_seen=0,
            progress_every=1,
            checkpoint_every=2,
            checkpoint_directory=(
                checkpoint_directory
            ),
            checkpoint_prefix=(
                "resume_test"
            ),
        )

        checkpoint_path = (
            checkpoints[
                0
            ]
        )

        # State immediately after checkpoint.
        loader_state_at_checkpoint = (
            loader_a.state_dict()
        )

        # ----------------------------------------------------
        # Continue A through steps 3,4,5.
        # ----------------------------------------------------

        (
            history_a_after,
            tokens_seen_a_final,
            _,
        ) = run_training_steps(
            model=model_a,
            optimizer=optimizer_a,
            train_loader=loader_a,
            num_steps=3,
            max_learning_rate=0.001,
            min_learning_rate=0.0001,
            warmup_steps=2,
            total_steps=10,
            max_grad_norm=1.0,
            starting_tokens_seen=(
                tokens_seen_a
            ),
            progress_every=1,
        )

        # ====================================================
        # RESUMED RUN B
        #
        # Deliberately initialize EVERYTHING differently.
        # The checkpoint must overwrite the relevant state.
        # ====================================================

        model_b = create_model(
            seed=999
        )

        optimizer_b = create_optimizer(
            model_b
        )

        loader_b = create_loader(
            binary_path,
            seed=999,
        )

        # Confirm loaders differ before restore.
        loaders_different_before = (
            not loader_states_equal(
                loader_a,
                loader_b,
            )
        )

        # ====================================================
        # Restore checkpoint
        # ====================================================

        (
            metadata,
            restored_step,
            restored_tokens_seen,
        ) = restore_training_checkpoint(
            model=model_b,
            optimizer=optimizer_b,
            path=checkpoint_path,
            train_loader=loader_b,
            strict=True,
        )

        # ====================================================
        # Compare restored loader with the loader state
        # captured at step 2.
        # ====================================================

        restored_loader_state = (
            loader_b.state_dict()
        )

        loader_state_restored = (
            json.dumps(
                restored_loader_state,
                sort_keys=True,
            )
            ==
            json.dumps(
                loader_state_at_checkpoint,
                sort_keys=True,
            )
        )

        # ====================================================
        # Resume B through steps 3,4,5
        # ====================================================

        (
            history_b_after,
            tokens_seen_b_final,
            _,
        ) = run_training_steps(
            model=model_b,
            optimizer=optimizer_b,
            train_loader=loader_b,
            num_steps=3,
            max_learning_rate=0.001,
            min_learning_rate=0.0001,
            warmup_steps=2,
            total_steps=10,
            max_grad_norm=1.0,
            starting_tokens_seen=(
                restored_tokens_seen
            ),
            progress_every=1,
        )

        # ====================================================
        # Final comparisons
        # ====================================================

        restored_step_passed = (
            restored_step
            == 2
        )

        restored_tokens_passed = (
            restored_tokens_seen
            == 12
        )

        checkpoint_loader_state_present = (
            "train_loader_state"
            in metadata[
                "extra_state"
            ]
        )

        histories_equal = (
            metric_histories_equal(
                history_a_after,
                history_b_after,
            )
        )

        models_equal = (
            model_parameters_equal(
                model_a,
                model_b,
            )
        )

        optimizers_equal = (
            optimizer_states_equal(
                optimizer_a,
                optimizer_b,
            )
        )

        loaders_equal_final = (
            loader_states_equal(
                loader_a,
                loader_b,
            )
        )

        tokens_equal_final = (
            tokens_seen_a_final
            ==
            tokens_seen_b_final
            ==
            30
        )

        final_steps_equal = (
            optimizer_a.step_count
            ==
            optimizer_b.step_count
            ==
            5
        )

        # ====================================================
        # Report
        # ====================================================

        print()

        print(
            f"Checkpoint: "
            f"{checkpoint_path.name}"
        )

        print(
            f"Restored step: "
            f"{restored_step}"
        )

        print(
            f"Restored tokens seen: "
            f"{restored_tokens_seen}"
        )

        print()

        print(
            "Continuous steps 3-5 losses:"
        )

        for metrics in history_a_after:

            print(
                f"  Step "
                f"{metrics['step']}: "
                f"{metrics['loss']:.9f}"
            )

        print()

        print(
            "Resumed steps 3-5 losses:"
        )

        for metrics in history_b_after:

            print(
                f"  Step "
                f"{metrics['step']}: "
                f"{metrics['loss']:.9f}"
            )

        print()

        print(
            "Final Checks"
        )

        print(
            "------------"
        )

        print(
            f"Fresh loader differed before restore: "
            f"{loaders_different_before}"
        )

        print(
            f"Loader RNG stored in checkpoint: "
            f"{checkpoint_loader_state_present}"
        )

        print(
            f"Loader RNG restored exactly: "
            f"{loader_state_restored}"
        )

        print(
            f"Training step restored: "
            f"{restored_step_passed}"
        )

        print(
            f"Tokens seen restored: "
            f"{restored_tokens_passed}"
        )

        print(
            f"Steps 3-5 metrics identical: "
            f"{histories_equal}"
        )

        print(
            f"Final model parameters identical: "
            f"{models_equal}"
        )

        print(
            f"Final AdamW state identical: "
            f"{optimizers_equal}"
        )

        print(
            f"Final loader RNG state identical: "
            f"{loaders_equal_final}"
        )

        print(
            f"Final token accounting identical: "
            f"{tokens_equal_final}"
        )

        print(
            f"Both runs reached step 5: "
            f"{final_steps_equal}"
        )

        all_passed = (
            loaders_different_before
            and
            checkpoint_loader_state_present
            and
            loader_state_restored
            and
            restored_step_passed
            and
            restored_tokens_passed
            and
            histories_equal
            and
            models_equal
            and
            optimizers_equal
            and
            loaders_equal_final
            and
            tokens_equal_final
            and
            final_steps_equal
        )

        print()
        print(
            "=" * 65
        )

        if all_passed:

            print(
                "SUCCESS: FULL TRAINER RESUME "
                "IS EXACT AND DETERMINISTIC."
            )

        else:

            raise AssertionError(
                "Exact trainer resume "
                "validation failed."
            )

        print(
            "=" * 65
        )


if __name__ == "__main__":
    main()