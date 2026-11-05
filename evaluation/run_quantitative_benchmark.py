import argparse
import hashlib
import json
import math
import sys

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

import numpy as np


# ============================================================
# Project root
# ============================================================

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


# ============================================================
# Imports
# ============================================================

from model.gpt import GPT

from tokenizer.bpe_tokenizer import (
    BPETokenizer,
)

from training.adamw import (
    AdamW,
)

from training.checkpoint import (
    load_checkpoint,
)


# ============================================================
# Paths
# ============================================================

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "model_config_v0.1.json"
)


TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "training_config_v0.1.json"
)


TOKENIZER_DIR = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer_50m_v0.2"
)


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "general_pretraining"
    / "general_pretraining_full_v0.1"
    / "mathgpt_step_008197.npz"
)


DEFAULT_BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "benchmarks"
    / "quantitative_benchmark_v0.1.json"
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


# ============================================================
# Helpers
# ============================================================

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


def resolve_path(
    path,
):

    path = Path(
        path
    )

    if path.is_absolute():

        return path

    return (
        PROJECT_ROOT
        /
        path
    )


def file_sha256(
    path,
):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024
                *
                1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# Benchmark validation
# ============================================================

def validate_benchmark(
    benchmark,
):

    required = {
        "version",
        "name",
        "target_continuations",
        "preference_tests",
    }

    missing = (
        required
        -
        set(
            benchmark.keys()
        )
    )

    if missing:

        raise KeyError(
            "Benchmark missing keys: "
            f"{sorted(missing)}"
        )

    target_ids = []

    for test in (
        benchmark[
            "target_continuations"
        ]
    ):

        for key in (
            "id",
            "category",
            "prompt",
            "target",
        ):

            if key not in test:

                raise KeyError(
                    f"Target test missing "
                    f"{key}."
                )

        target_ids.append(
            test[
                "id"
            ]
        )

    if (
        len(
            target_ids
        )
        !=
        len(
            set(
                target_ids
            )
        )
    ):

        raise ValueError(
            "Duplicate target test IDs."
        )

    preference_ids = []

    for test in (
        benchmark[
            "preference_tests"
        ]
    ):

        for key in (
            "id",
            "category",
            "prompt",
            "preferred",
            "rejected",
        ):

            if key not in test:

                raise KeyError(
                    f"Preference test missing "
                    f"{key}."
                )

        preference_ids.append(
            test[
                "id"
            ]
        )

    if (
        len(
            preference_ids
        )
        !=
        len(
            set(
                preference_ids
            )
        )
    ):

        raise ValueError(
            "Duplicate preference test IDs."
        )


# ============================================================
# Token probability / rank
# ============================================================

def token_statistics(
    logits,
    target_token_id,
):

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    target_token_id = int(
        target_token_id
    )

    if not (
        0
        <=
        target_token_id
        <
        logits.shape[
            0
        ]
    ):

        raise ValueError(
            "Target token outside vocabulary."
        )

    maximum = np.max(
        logits
    )

    shifted = (
        logits
        -
        maximum
    )

    exponentials = np.exp(
        shifted
    )

    denominator = float(
        np.sum(
            exponentials
        )
    )

    if (
        not np.isfinite(
            denominator
        )
        or
        denominator <= 0.0
    ):

        raise FloatingPointError(
            "Invalid softmax denominator."
        )

    log_denominator = math.log(
        denominator
    )

    log_probability = (
        float(
            shifted[
                target_token_id
            ]
        )
        -
        log_denominator
    )

    probability = math.exp(
        log_probability
    )

    target_logit = (
        logits[
            target_token_id
        ]
    )

    rank = (
        1
        +
        int(
            np.sum(
                logits
                >
                target_logit
            )
        )
    )

    return {
        "probability":
            float(
                probability
            ),

        "log_probability":
            float(
                log_probability
            ),

        "rank":
            int(
                rank
            ),
    }


# ============================================================
# Score one continuation
# ============================================================

def score_continuation(
    *,
    model,
    tokenizer,
    prompt,
    continuation,
):

    prompt_ids = tokenizer.encode(
        prompt,
        add_bos=False,
        add_eos=False,
    )

    continuation_ids = (
        tokenizer.encode(
            continuation,
            add_bos=False,
            add_eos=False,
        )
    )

    if not prompt_ids:

        prompt_ids = [
            tokenizer.bos_token_id
        ]

    if not continuation_ids:

        raise ValueError(
            "Continuation tokenized "
            "to an empty sequence."
        )

    context_ids = list(
        prompt_ids
    )

    token_results = []

    total_log_probability = 0.0

    for position, target_id in enumerate(
        continuation_ids
    ):

        model_context = (
            context_ids[
                -model.context_length:
            ]
        )

        input_ids = np.asarray(
            [
                model_context
            ],
            dtype=np.int64,
        )

        logits = model.forward(
            input_ids
        )

        next_token_logits = (
            logits[
                0,
                -1,
                :
            ]
        )

        statistics = token_statistics(
            next_token_logits,
            target_id,
        )

        total_log_probability += (
            statistics[
                "log_probability"
            ]
        )

        token_results.append(
            {
                "position":
                    int(
                        position
                    ),

                "token_id":
                    int(
                        target_id
                    ),

                "probability":
                    statistics[
                        "probability"
                    ],

                "log_probability":
                    statistics[
                        "log_probability"
                    ],

                "rank":
                    statistics[
                        "rank"
                    ],
            }
        )

        context_ids.append(
            int(
                target_id
            )
        )

    token_count = len(
        continuation_ids
    )

    mean_log_probability = (
        total_log_probability
        /
        token_count
    )

    mean_nll = (
        -mean_log_probability
    )

    try:

        perplexity = math.exp(
            mean_nll
        )

    except OverflowError:

        perplexity = float(
            "inf"
        )

    ranks = [
        result[
            "rank"
        ]

        for result
        in token_results
    ]

    return {
        "prompt":
            prompt,

        "continuation":
            continuation,

        "prompt_token_ids":
            [
                int(
                    token_id
                )

                for token_id
                in prompt_ids
            ],

        "continuation_token_ids":
            [
                int(
                    token_id
                )

                for token_id
                in continuation_ids
            ],

        "continuation_token_count":
            int(
                token_count
            ),

        "sum_log_probability":
            float(
                total_log_probability
            ),

        "mean_log_probability":
            float(
                mean_log_probability
            ),

        "mean_nll":
            float(
                mean_nll
            ),

        "perplexity":
            float(
                perplexity
            ),

        "first_token_probability":
            float(
                token_results[
                    0
                ][
                    "probability"
                ]
            ),

        "first_token_rank":
            int(
                token_results[
                    0
                ][
                    "rank"
                ]
            ),

        "mean_token_rank":
            float(
                np.mean(
                    ranks
                )
            ),

        "median_token_rank":
            float(
                np.median(
                    ranks
                )
            ),

        "top1_fraction":
            float(
                np.mean(
                    [
                        rank <= 1
                        for rank
                        in ranks
                    ]
                )
            ),

        "top10_fraction":
            float(
                np.mean(
                    [
                        rank <= 10
                        for rank
                        in ranks
                    ]
                )
            ),

        "top100_fraction":
            float(
                np.mean(
                    [
                        rank <= 100
                        for rank
                        in ranks
                    ]
                )
            ),

        "token_results":
            token_results,
    }


# ============================================================
# Markdown report
# ============================================================

def write_markdown_report(
    path,
    report,
):

    aggregate = (
        report[
            "aggregate"
        ]
    )

    lines = [
        "# MathGPT Quantitative Benchmark",
        "",
        (
            f"- Model: "
            f"`{report['model']['label']}`"
        ),
        (
            f"- Checkpoint: "
            f"`{report['model']['checkpoint_name']}`"
        ),
        (
            f"- Training step: "
            f"`{report['model']['training_step']:,}`"
        ),
        (
            f"- Parameters: "
            f"`{report['model']['parameter_count']:,}`"
        ),
        (
            f"- Benchmark version: "
            f"`{report['benchmark']['version']}`"
        ),
        "",
        "## Aggregate Results",
        "",
        (
            "- First-target Top-1 accuracy: "
            f"{aggregate['first_token_top1_accuracy']:.2%}"
        ),
        (
            "- First-target Top-10 accuracy: "
            f"{aggregate['first_token_top10_accuracy']:.2%}"
        ),
        (
            "- First-target Top-100 accuracy: "
            f"{aggregate['first_token_top100_accuracy']:.2%}"
        ),
        (
            "- Median first-target rank: "
            f"{aggregate['median_first_token_rank']:.1f}"
        ),
        (
            "- Preference accuracy: "
            f"{aggregate['preference_accuracy']:.2%}"
        ),
        (
            "- Mean preference margin: "
            f"{aggregate['mean_preference_margin']:+.6f}"
        ),
        "",
        "## Target Continuations",
        "",
        (
            "| Test | Category | Target | "
            "First Rank | Mean NLL | PPL |"
        ),
        (
            "|---|---|---|---:|---:|---:|"
        ),
    ]

    for result in (
        report[
            "target_results"
        ]
    ):

        target = (
            result[
                "target"
            ]
            .replace(
                "|",
                "\\|",
            )
        )

        score = (
            result[
                "score"
            ]
        )

        lines.append(
            (
                f"| {result['id']} "
                f"| {result['category']} "
                f"| `{target}` "
                f"| {score['first_token_rank']} "
                f"| {score['mean_nll']:.6f} "
                f"| {score['perplexity']:.2f} |"
            )
        )

    lines.extend(
        [
            "",
            "## Contrastive Preferences",
            "",
            (
                "| Test | Category | "
                "Preferred NLL | Rejected NLL | "
                "Margin | Correct |"
            ),
            (
                "|---|---|---:|---:|---:|---|"
            ),
        ]
    )

    for result in (
        report[
            "preference_results"
        ]
    ):

        lines.append(
            (
                f"| {result['id']} "
                f"| {result['category']} "
                f"| "
                f"{result['preferred_score']['mean_nll']:.6f} "
                f"| "
                f"{result['rejected_score']['mean_nll']:.6f} "
                f"| {result['margin']:+.6f} "
                f"| {result['correct']} |"
            )
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


# ============================================================
# Arguments
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed MathGPT "
            "quantitative benchmark."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            DEFAULT_CHECKPOINT_PATH
        ),
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(
            DEFAULT_BENCHMARK_PATH
        ),
    )

    parser.add_argument(
        "--model-label",
        type=str,
        default=(
            "MathGPT-General-v0.1"
        ),
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    benchmark_path = resolve_path(
        args.benchmark
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint missing: "
            f"{checkpoint_path}"
        )

    if not benchmark_path.exists():

        raise FileNotFoundError(
            f"Benchmark missing: "
            f"{benchmark_path}"
        )

    benchmark = load_json(
        benchmark_path
    )

    validate_benchmark(
        benchmark
    )

    model_config = load_json(
        MODEL_CONFIG_PATH
    )

    training_config = load_json(
        TRAINING_CONFIG_PATH
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        BPETokenizer.from_files(
            config_path=(
                TOKENIZER_DIR
                /
                "tokenizer_config.json"
            ),

            merges_path=(
                TOKENIZER_DIR
                /
                "merges.json"
            ),

            vocab_path=(
                TOKENIZER_DIR
                /
                "vocab.json"
            ),
        )
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "Loading MathGPT..."
    )

    model = GPT.from_config(
        model_config,
        seed=42,
    )

    optimizer = AdamW(
        model=model,

        learning_rate=(
            training_config[
                "optimizer"
            ][
                "learning_rate"
            ]
        ),

        beta1=(
            training_config[
                "optimizer"
            ][
                "beta1"
            ]
        ),

        beta2=(
            training_config[
                "optimizer"
            ][
                "beta2"
            ]
        ),

        epsilon=(
            training_config[
                "optimizer"
            ][
                "epsilon"
            ]
        ),

        weight_decay=(
            training_config[
                "optimizer"
            ][
                "weight_decay"
            ]
        ),
    )

    metadata = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        strict=True,
    )

    checkpoint_hash = file_sha256(
        checkpoint_path
    )

    benchmark_hash = file_sha256(
        benchmark_path
    )

    target_tests = (
        benchmark[
            "target_continuations"
        ]
    )

    preference_tests = (
        benchmark[
            "preference_tests"
        ]
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE 11D — QUANTITATIVE CAPABILITY BENCHMARK"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Model: "
        f"{args.model_label}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path.name}"
    )

    print(
        f"Training step: "
        f"{metadata['optimizer_step_count']:,}"
    )

    print(
        f"Parameters: "
        f"{metadata['parameter_count']:,}"
    )

    print()

    print(
        f"Target continuation tests: "
        f"{len(target_tests)}"
    )

    print(
        f"Preference tests: "
        f"{len(preference_tests)}"
    )

    # ========================================================
    # Target tests
    # ========================================================

    target_results = []

    print()
    print(
        "=" * 76
    )

    print(
        "TARGET CONTINUATION TESTS"
    )

    print(
        "=" * 76
    )

    for index, test in enumerate(
        target_tests,
        start=1,
    ):

        score = score_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=(
                test[
                    "prompt"
                ]
            ),
            continuation=(
                test[
                    "target"
                ]
            ),
        )

        result = {
            "id":
                test[
                    "id"
                ],

            "category":
                test[
                    "category"
                ],

            "prompt":
                test[
                    "prompt"
                ],

            "target":
                test[
                    "target"
                ],

            "score":
                score,
        }

        target_results.append(
            result
        )

        print()
        print(
            f"[{index}/"
            f"{len(target_tests)}] "
            f"{test['id']}"
        )

        print(
            f"  Prompt: "
            f"{test['prompt']}"
        )

        print(
            f"  Target: "
            f"{test['target']}"
        )

        print(
            f"  Target tokens: "
            f"{score['continuation_token_count']}"
        )

        print(
            f"  First-token rank: "
            f"{score['first_token_rank']:,}"
        )

        print(
            f"  First-token probability: "
            f"{score['first_token_probability']:.8f}"
        )

        print(
            f"  Mean NLL: "
            f"{score['mean_nll']:.6f}"
        )

        print(
            f"  Target PPL: "
            f"{score['perplexity']:.2f}"
        )

    # ========================================================
    # Preference tests
    # ========================================================

    preference_results = []

    print()
    print(
        "=" * 76
    )

    print(
        "CONTRASTIVE PREFERENCE TESTS"
    )

    print(
        "=" * 76
    )

    for index, test in enumerate(
        preference_tests,
        start=1,
    ):

        preferred_score = (
            score_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=(
                    test[
                        "prompt"
                    ]
                ),
                continuation=(
                    test[
                        "preferred"
                    ]
                ),
            )
        )

        rejected_score = (
            score_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=(
                    test[
                        "prompt"
                    ]
                ),
                continuation=(
                    test[
                        "rejected"
                    ]
                ),
            )
        )

        # Positive margin means preferred
        # continuation has lower NLL.
        margin = (
            rejected_score[
                "mean_nll"
            ]
            -
            preferred_score[
                "mean_nll"
            ]
        )

        correct = (
            margin
            >
            0.0
        )

        result = {
            "id":
                test[
                    "id"
                ],

            "category":
                test[
                    "category"
                ],

            "prompt":
                test[
                    "prompt"
                ],

            "preferred":
                test[
                    "preferred"
                ],

            "rejected":
                test[
                    "rejected"
                ],

            "preferred_score":
                preferred_score,

            "rejected_score":
                rejected_score,

            "margin":
                float(
                    margin
                ),

            "correct":
                bool(
                    correct
                ),
        }

        preference_results.append(
            result
        )

        print()
        print(
            f"[{index}/"
            f"{len(preference_tests)}] "
            f"{test['id']}"
        )

        print(
            f"  Preferred: "
            f"{test['preferred']}"
        )

        print(
            f"  Rejected:  "
            f"{test['rejected']}"
        )

        print(
            f"  Preferred NLL: "
            f"{preferred_score['mean_nll']:.6f}"
        )

        print(
            f"  Rejected NLL:  "
            f"{rejected_score['mean_nll']:.6f}"
        )

        print(
            f"  Margin: "
            f"{margin:+.6f}"
        )

        print(
            f"  Correct: "
            f"{correct}"
        )

    # ========================================================
    # Aggregate metrics
    # ========================================================

    first_ranks = [
        result[
            "score"
        ][
            "first_token_rank"
        ]

        for result
        in target_results
    ]

    all_target_token_results = [
        token_result

        for result
        in target_results

        for token_result
        in result[
            "score"
        ][
            "token_results"
        ]
    ]

    preference_margins = [
        result[
            "margin"
        ]

        for result
        in preference_results
    ]

    aggregate = {
        "target_test_count":
            len(
                target_results
            ),

        "target_token_count":
            len(
                all_target_token_results
            ),

        "first_token_top1_accuracy":
            float(
                np.mean(
                    [
                        rank <= 1
                        for rank
                        in first_ranks
                    ]
                )
            ),

        "first_token_top10_accuracy":
            float(
                np.mean(
                    [
                        rank <= 10
                        for rank
                        in first_ranks
                    ]
                )
            ),

        "first_token_top100_accuracy":
            float(
                np.mean(
                    [
                        rank <= 100
                        for rank
                        in first_ranks
                    ]
                )
            ),

        "median_first_token_rank":
            float(
                np.median(
                    first_ranks
                )
            ),

        "mean_first_token_rank":
            float(
                np.mean(
                    first_ranks
                )
            ),

        "all_target_tokens_top1":
            float(
                np.mean(
                    [
                        result[
                            "rank"
                        ]
                        <=
                        1

                        for result
                        in all_target_token_results
                    ]
                )
            ),

        "all_target_tokens_top10":
            float(
                np.mean(
                    [
                        result[
                            "rank"
                        ]
                        <=
                        10

                        for result
                        in all_target_token_results
                    ]
                )
            ),

        "all_target_tokens_top100":
            float(
                np.mean(
                    [
                        result[
                            "rank"
                        ]
                        <=
                        100

                        for result
                        in all_target_token_results
                    ]
                )
            ),

        "mean_target_nll":
            float(
                np.mean(
                    [
                        result[
                            "score"
                        ][
                            "mean_nll"
                        ]

                        for result
                        in target_results
                    ]
                )
            ),

        "preference_test_count":
            len(
                preference_results
            ),

        "preference_correct":
            int(
                sum(
                    result[
                        "correct"
                    ]

                    for result
                    in preference_results
                )
            ),

        "preference_accuracy":
            float(
                np.mean(
                    [
                        result[
                            "correct"
                        ]

                        for result
                        in preference_results
                    ]
                )
            ),

        "mean_preference_margin":
            float(
                np.mean(
                    preference_margins
                )
            ),

        "median_preference_margin":
            float(
                np.median(
                    preference_margins
                )
            ),
    }

    # ========================================================
    # Report
    # ========================================================

    report = {
        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model": {
            "label":
                args.model_label,

            "checkpoint":
                str(
                    checkpoint_path
                ),

            "checkpoint_name":
                checkpoint_path.name,

            "checkpoint_sha256":
                checkpoint_hash,

            "training_step":
                int(
                    metadata[
                        "optimizer_step_count"
                    ]
                ),

            "parameter_count":
                int(
                    metadata[
                        "parameter_count"
                    ]
                ),

            "vocab_size":
                int(
                    model.vocab_size
                ),

            "context_length":
                int(
                    model.context_length
                ),
        },

        "benchmark": {
            "name":
                benchmark[
                    "name"
                ],

            "version":
                benchmark[
                    "version"
                ],

            "definition_path":
                str(
                    benchmark_path
                ),

            "definition_sha256":
                benchmark_hash,

            "target_test_count":
                len(
                    target_results
                ),

            "preference_test_count":
                len(
                    preference_results
                ),
        },

        "aggregate":
            aggregate,

        "target_results":
            target_results,

        "preference_results":
            preference_results,
    }

    # ========================================================
    # Save
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output_prefix is None:

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        output_prefix = (
            f"{args.model_label}"
            f"_quantitative_"
            f"v{benchmark['version']}_"
            f"{timestamp}"
        )

    else:

        output_prefix = (
            args.output_prefix
        )

    json_path = (
        RESULTS_DIR
        /
        (
            output_prefix
            +
            ".json"
        )
    )

    markdown_path = (
        RESULTS_DIR
        /
        (
            output_prefix
            +
            ".md"
        )
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    write_markdown_report(
        markdown_path,
        report,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE 11D QUANTITATIVE RESULTS"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Target tests: "
        f"{aggregate['target_test_count']}"
    )

    print(
        f"Target tokens scored: "
        f"{aggregate['target_token_count']}"
    )

    print()

    print(
        f"First-token Top-1: "
        f"{aggregate['first_token_top1_accuracy']:.2%}"
    )

    print(
        f"First-token Top-10: "
        f"{aggregate['first_token_top10_accuracy']:.2%}"
    )

    print(
        f"First-token Top-100: "
        f"{aggregate['first_token_top100_accuracy']:.2%}"
    )

    print(
        f"Median first-token rank: "
        f"{aggregate['median_first_token_rank']:.1f}"
    )

    print()

    print(
        f"All target tokens Top-1: "
        f"{aggregate['all_target_tokens_top1']:.2%}"
    )

    print(
        f"All target tokens Top-10: "
        f"{aggregate['all_target_tokens_top10']:.2%}"
    )

    print(
        f"All target tokens Top-100: "
        f"{aggregate['all_target_tokens_top100']:.2%}"
    )

    print()

    print(
        f"Preference accuracy: "
        f"{aggregate['preference_correct']}"
        f"/"
        f"{aggregate['preference_test_count']} "
        f"("
        f"{aggregate['preference_accuracy']:.2%}"
        f")"
    )

    print(
        f"Mean preference margin: "
        f"{aggregate['mean_preference_margin']:+.6f}"
    )

    print()

    print(
        "JSON:"
    )

    print(
        json_path
    )

    print()

    print(
        "Markdown:"
    )

    print(
        markdown_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: QUANTITATIVE "
        "CAPABILITY BENCHMARK COMPLETED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()
