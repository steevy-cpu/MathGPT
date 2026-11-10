import argparse
import json
import shlex
import statistics
import subprocess
import sys

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

SCORER_SCRIPT = (
    PROJECT_ROOT
    /
    "laboratory/tools/continuation_scorer.py"
)

ABLATION_SCRIPT = (
    PROJECT_ROOT
    /
    "laboratory/tools/attention_head_ablation.py"
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "EXP-0002 systematic L0H1 "
            "generalization experiment."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "laboratory/experiments/"
            "exp_0002_probes.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "laboratory/artifacts/"
            "exp_0002_systematic"
        ),
    )

    parser.add_argument(
        "--summary",
        type=str,
        default=(
            "laboratory/artifacts/"
            "exp_0002_systematic_summary.json"
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def project_path(
    value,
):

    path = Path(
        value
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    return path


def load_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def run_command(
    command,
):

    print()
    print(
        "$ "
        +
        shlex.join(
            [
                str(
                    value
                )

                for value
                in command
            ]
        )
    )

    subprocess.run(
        [
            str(
                value
            )

            for value
            in command
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def contains_four(
    continuation,
):

    text = (
        continuation
        .strip()
        .lower()
    )

    return text.startswith(
        "four"
    )


def standardized_four_effect(
    *,
    preferred,
    comparison,
    ablation_margin_delta,
):
    """
    Return a standardized effect where:

        positive
        =
        intact head pushes toward FOUR

        negative
        =
        intact head pushes away from FOUR

    If neither candidate is four, return None.
    """

    preferred_is_four = (
        contains_four(
            preferred
        )
    )

    comparison_is_four = (
        contains_four(
            comparison
        )
    )

    if (
        preferred_is_four
        and
        not comparison_is_four
    ):

        # Margin:
        # four - comparison
        #
        # Negative ablation delta means
        # removing the head hurts four.
        #
        # Therefore intact head supported four.
        return float(
            -ablation_margin_delta
        )

    if (
        comparison_is_four
        and
        not preferred_is_four
    ):

        # Margin:
        # preferred - four
        #
        # Positive ablation delta means
        # removing the head helps preferred.
        #
        # Therefore intact head had supported four.
        return float(
            ablation_margin_delta
        )

    return None


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    config_path = project_path(
        args.config
    )

    output_dir = project_path(
        args.output_dir
    )

    summary_path = project_path(
        args.summary
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_json(
        config_path
    )

    layer = int(
        config[
            "target_layer"
        ]
    )

    head = int(
        config[
            "target_head"
        ]
    )

    print()
    print(
        "=" * 80
    )

    print(
        "EXP-0002 — SYSTEMATIC L0H1 "
        "GENERALIZATION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Target component: "
        f"L{layer}H{head}"
    )

    print(
        f"Probe count: "
        f"{len(config['probes'])}"
    )

    results = []

    # ========================================================
    # Probe loop
    # ========================================================

    for (
        probe_number,
        probe,
    ) in enumerate(
        config[
            "probes"
        ],
        start=1,
    ):

        probe_id = (
            probe[
                "id"
            ]
        )

        print()
        print(
            "=" * 80
        )

        print(
            f"[{probe_number}/"
            f"{len(config['probes'])}] "
            f"{probe_id}"
        )

        print(
            "=" * 80
        )

        print(
            f"Category: "
            f"{probe['category']}"
        )

        print(
            f"Prompt: "
            f"{probe['prompt']}"
        )

        print(
            f"Preferred: "
            f"{repr(probe['preferred'])}"
        )

        print(
            f"Comparison: "
            f"{repr(probe['comparison'])}"
        )

        # ====================================================
        # Artifact paths
        # ====================================================

        scorer_path = (
            output_dir
            /
            f"{probe_id.lower()}_score.json"
        )

        ablation_path = (
            output_dir
            /
            f"{probe_id.lower()}_l0h1_ablation.json"
        )

        # ====================================================
        # L6.2 scorer
        # ====================================================

        scorer_command = [
            sys.executable,
            SCORER_SCRIPT,

            "--checkpoint",
            args.checkpoint,

            "--prompt",
            probe[
                "prompt"
            ],

            "--continuation",
            probe[
                "preferred"
            ],

            "--continuation",
            probe[
                "comparison"
            ],

            "--preferred-index",
            "0",

            "--output",
            scorer_path,
        ]

        if (
            args.expected_sha256
            is not None
        ):

            scorer_command.extend(
                [
                    "--expected-sha256",
                    args.expected_sha256,
                ]
            )

        run_command(
            scorer_command
        )

        scorer = load_json(
            scorer_path
        )

        shared_prefix_ids = [
            int(
                token_id
            )

            for token_id
            in scorer[
                "discriminative_analysis"
            ][
                "shared_prefix_token_ids"
            ]
        ]

        if not shared_prefix_ids:

            print(
                "Shared prefix: none"
            )

        else:

            print(
                f"Shared prefix IDs: "
                f"{shared_prefix_ids}"
            )

        # ====================================================
        # L6.8 head ablation
        # ====================================================

        ablation_command = [
            sys.executable,
            ABLATION_SCRIPT,

            "--checkpoint",
            args.checkpoint,

            "--prompt",
            probe[
                "prompt"
            ],

            "--scorer-artifact",
            scorer_path,

            "--preferred-target-index",
            "0",

            "--comparison-target-index",
            "1",

            "--layer",
            str(
                layer
            ),

            "--head",
            str(
                head
            ),

            "--output",
            ablation_path,
        ]

        if (
            args.expected_sha256
            is not None
        ):

            ablation_command.extend(
                [
                    "--expected-sha256",
                    args.expected_sha256,
                ]
            )

        for token_id in shared_prefix_ids:

            ablation_command.extend(
                [
                    "--append-token-id",
                    str(
                        token_id
                    ),
                ]
            )

        run_command(
            ablation_command
        )

        ablation = load_json(
            ablation_path
        )

        head_result = (
            ablation[
                "results"
            ][
                0
            ]
        )

        # ====================================================
        # Extract measurements
        # ====================================================

        full_continuation_margin = None

        if scorer[
            "comparisons"
        ]:

            full_continuation_margin = float(
                scorer[
                    "comparisons"
                ][
                    0
                ][
                    "margin"
                ]
            )

        baseline_discriminative_margin = float(
            ablation[
                "baseline"
            ][
                "logit_margin"
            ]
        )

        ablated_discriminative_margin = float(
            head_result[
                "ablated"
            ][
                "logit_margin"
            ]
        )

        margin_delta = float(
            head_result[
                "effect"
            ][
                "logit_margin_delta"
            ]
        )

        four_effect = (
            standardized_four_effect(
                preferred=(
                    probe[
                        "preferred"
                    ]
                ),

                comparison=(
                    probe[
                        "comparison"
                    ]
                ),

                ablation_margin_delta=(
                    margin_delta
                ),
            )
        )

        result = {
            "id":
                probe_id,

            "category":
                probe[
                    "category"
                ],

            "prompt":
                probe[
                    "prompt"
                ],

            "preferred":
                probe[
                    "preferred"
                ],

            "comparison":
                probe[
                    "comparison"
                ],

            "contains_two":
                bool(
                    probe[
                        "contains_two"
                    ]
                ),

            "shared_prefix_token_ids":
                shared_prefix_ids,

            "full_continuation_margin":
                full_continuation_margin,

            "baseline_discriminative_margin":
                baseline_discriminative_margin,

            "ablated_discriminative_margin":
                ablated_discriminative_margin,

            "ablation_margin_delta":
                margin_delta,

            "intact_l0h1_four_support_effect":
                four_effect,

            "preferred_rank_baseline":
                int(
                    ablation[
                        "baseline"
                    ][
                        "preferred"
                    ][
                        "rank"
                    ]
                ),

            "preferred_rank_ablated":
                int(
                    head_result[
                        "ablated"
                    ][
                        "preferred"
                    ][
                        "rank"
                    ]
                ),

            "kl_baseline_to_ablation":
                float(
                    head_result[
                        "effect"
                    ][
                        "distribution_change"
                    ][
                        "kl_baseline_to_intervention"
                    ]
                ),

            "scorer_artifact":
                str(
                    scorer_path
                ),

            "ablation_artifact":
                str(
                    ablation_path
                ),
        }

        results.append(
            result
        )

    # ========================================================
    # Group summaries
    # ========================================================

    four_results = [
        result

        for result
        in results

        if (
            result[
                "intact_l0h1_four_support_effect"
            ]
            is not None
        )
    ]

    with_two = [
        result[
            "intact_l0h1_four_support_effect"
        ]

        for result
        in four_results

        if result[
            "contains_two"
        ]
    ]

    without_two = [
        result[
            "intact_l0h1_four_support_effect"
        ]

        for result
        in four_results

        if not result[
            "contains_two"
        ]
    ]

    group_summary = {
        "four_comparison_probe_count":
            len(
                four_results
            ),

        "contains_two_four_probe_count":
            len(
                with_two
            ),

        "no_two_four_probe_count":
            len(
                without_two
            ),

        "mean_four_support_effect_with_two":
            (
                None
                if not with_two
                else float(
                    statistics.mean(
                        with_two
                    )
                )
            ),

        "mean_four_support_effect_without_two":
            (
                None
                if not without_two
                else float(
                    statistics.mean(
                        without_two
                    )
                )
            ),
    }

    # ========================================================
    # Final report
    # ========================================================

    report = {
        "experiment_id":
            config[
                "experiment_id"
            ],

        "title":
            config[
                "title"
            ],

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "checkpoint":
            args.checkpoint,

        "expected_checkpoint_sha256":
            args.expected_sha256,

        "target_component":
            {
                "layer":
                    layer,

                "head":
                    head,

                "head_id":
                    f"L{layer}H{head}",
            },

        "probe_count":
            len(
                results
            ),

        "results":
            results,

        "group_summary":
            group_summary,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # Display final table
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        "EXP-0002 SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        (
            f"{'ID':<6}"
            f"{'Category':<24}"
            f"{'Contains Two':>14}"
            f"{'Base Margin':>14}"
            f"{'Δ Margin':>14}"
            f"{'Four Effect':>14}"
            f"{'KL':>12}"
        )
    )

    print(
        "-" * 100
    )

    for result in results:

        four_effect = (
            result[
                "intact_l0h1_four_support_effect"
            ]
        )

        four_text = (
            "n/a"
            if four_effect is None
            else
            f"{four_effect:+.6f}"
        )

        print(
            (
                f"{result['id']:<6}"
                f"{result['category']:<24}"
                f"{str(result['contains_two']):>14}"
                f"{result['baseline_discriminative_margin']:>+14.6f}"
                f"{result['ablation_margin_delta']:>+14.6f}"
                f"{four_text:>14}"
                f"{result['kl_baseline_to_ablation']:>12.6f}"
            )
        )

    print()
    print(
        "Standardized Four Effect"
    )

    print(
        "------------------------"
    )

    print(
        "Positive = intact L0H1 pushes toward FOUR."
    )

    print(
        "Negative = intact L0H1 pushes away from FOUR."
    )

    print()

    print(
        (
            "Mean four-support effect, "
            "probes containing Two: "
            f"{group_summary['mean_four_support_effect_with_two']}"
        )
    )

    print(
        (
            "Mean four-support effect, "
            "probes without Two: "
            f"{group_summary['mean_four_support_effect_without_two']}"
        )
    )

    print()
    print(
        "Summary artifact:"
    )

    print(
        summary_path
    )

    print()
    print(
        "=" * 80
    )

    print(
        "SUCCESS: EXP-0002 SYSTEMATIC "
        "PROBE RUN COMPLETE."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()