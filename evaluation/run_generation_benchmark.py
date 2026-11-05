import argparse
import hashlib
import json
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

from evaluation.generation import (
    generate_text,
)


# ============================================================
# Default paths
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


DEFAULT_BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "benchmarks"
    / "generation_benchmark_v0.1.json"
)


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "general_pretraining"
    / "general_pretraining_full_v0.1"
    / "mathgpt_step_008197.npz"
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


# ============================================================
# File helpers
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


def file_sha256(
    path,
):
    """
    SHA-256 fingerprint.

    This allows us to prove exactly which
    checkpoint and benchmark definition
    produced a result.
    """

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


# ============================================================
# Benchmark validation
# ============================================================

def validate_benchmark(
    benchmark,
):

    required_top_level = {
        "version",
        "name",
        "decoding_profiles",
        "prompts",
    }

    missing = (
        required_top_level
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

    profiles = (
        benchmark[
            "decoding_profiles"
        ]
    )

    prompts = (
        benchmark[
            "prompts"
        ]
    )

    if not profiles:

        raise ValueError(
            "Benchmark has no "
            "decoding profiles."
        )

    if not prompts:

        raise ValueError(
            "Benchmark has no prompts."
        )

    profile_names = []

    for profile in profiles:

        if "name" not in profile:

            raise KeyError(
                "Decoding profile missing name."
            )

        if "strategy" not in profile:

            raise KeyError(
                "Decoding profile missing strategy."
            )

        if (
            profile[
                "strategy"
            ]
            not in {
                "greedy",
                "sample",
            }
        ):

            raise ValueError(
                "Unsupported decoding strategy."
            )

        profile_names.append(
            profile[
                "name"
            ]
        )

    if (
        len(
            profile_names
        )
        !=
        len(
            set(
                profile_names
            )
        )
    ):

        raise ValueError(
            "Duplicate decoding profile names."
        )

    prompt_ids = []

    for prompt in prompts:

        required_prompt_keys = {
            "id",
            "category",
            "prompt",
        }

        missing_prompt_keys = (
            required_prompt_keys
            -
            set(
                prompt.keys()
            )
        )

        if missing_prompt_keys:

            raise KeyError(
                "Prompt missing keys: "
                f"{sorted(missing_prompt_keys)}"
            )

        if not isinstance(
            prompt[
                "prompt"
            ],
            str,
        ):

            raise TypeError(
                "Prompt text must be string."
            )

        prompt_ids.append(
            prompt[
                "id"
            ]
        )

    if (
        len(
            prompt_ids
        )
        !=
        len(
            set(
                prompt_ids
            )
        )
    ):

        raise ValueError(
            "Duplicate prompt IDs."
        )


# ============================================================
# Generation metrics
# ============================================================

def longest_consecutive_token_run(
    token_ids,
):

    if not token_ids:

        return 0

    longest = 1

    current = 1

    previous = (
        token_ids[
            0
        ]
    )

    for token_id in (
        token_ids[
            1:
        ]
    ):

        if (
            token_id
            ==
            previous
        ):

            current += 1

            longest = max(
                longest,
                current,
            )

        else:

            current = 1

        previous = (
            token_id
        )

    return int(
        longest
    )


def generation_metrics(
    result,
):

    token_ids = [
        int(
            token_id
        )

        for token_id
        in result[
            "generated_token_ids"
        ]
    ]

    generated_count = len(
        token_ids
    )

    if generated_count > 0:

        unique_ratio = (
            len(
                set(
                    token_ids
                )
            )
            /
            generated_count
        )

    else:

        unique_ratio = 0.0

    text = (
        result[
            "generated_text"
        ]
    )

    return {
        "generated_tokens":
            int(
                generated_count
            ),

        "generated_characters":
            int(
                len(
                    text
                )
            ),

        "unique_token_ratio":
            float(
                unique_ratio
            ),

        "longest_consecutive_token_run":
            longest_consecutive_token_run(
                token_ids
            ),

        "replacement_character_count":
            int(
                text.count(
                    "\ufffd"
                )
            ),

        "stopped_on_eos":
            bool(
                result[
                    "stopped_on_eos"
                ]
            ),
    }


# ============================================================
# Markdown helper
# ============================================================

def safe_markdown_code(
    text,
):

    return text.replace(
        "```",
        "``\u200b`",
    )


def write_markdown_report(
    *,
    output_path,
    report,
):

    lines = []

    lines.append(
        "# MathGPT Generation Benchmark"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- Model: `{report['model']['label']}`"
    )

    lines.append(
        f"- Checkpoint: `{report['model']['checkpoint_name']}`"
    )

    lines.append(
        f"- Training step: `{report['model']['training_step']:,}`"
    )

    lines.append(
        f"- Parameters: `{report['model']['parameter_count']:,}`"
    )

    lines.append(
        f"- Benchmark: `{report['benchmark']['name']}`"
    )

    lines.append(
        f"- Benchmark version: `{report['benchmark']['version']}`"
    )

    lines.append(
        f"- Generations: `{len(report['results'])}`"
    )

    lines.append(
        ""
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    for result in report[
        "results"
    ]:

        lines.append(
            (
                f"## {result['prompt_id']} "
                f"— {result['decoding_profile']}"
            )
        )

        lines.append(
            ""
        )

        lines.append(
            f"Category: `{result['category']}`"
        )

        lines.append(
            ""
        )

        lines.append(
            "**Prompt**"
        )

        lines.append(
            ""
        )

        lines.append(
            "```text"
        )

        lines.append(
            safe_markdown_code(
                result[
                    "prompt"
                ]
            )
        )

        lines.append(
            "```"
        )

        lines.append(
            ""
        )

        lines.append(
            "**Generated continuation**"
        )

        lines.append(
            ""
        )

        lines.append(
            "```text"
        )

        lines.append(
            safe_markdown_code(
                result[
                    "generated_text"
                ]
            )
        )

        lines.append(
            "```"
        )

        lines.append(
            ""
        )

        metrics = (
            result[
                "metrics"
            ]
        )

        lines.append(
            (
                f"- Generated tokens: "
                f"{metrics['generated_tokens']}"
            )
        )

        lines.append(
            (
                f"- EOS: "
                f"{metrics['stopped_on_eos']}"
            )
        )

        lines.append(
            (
                f"- Unique token ratio: "
                f"{metrics['unique_token_ratio']:.4f}"
            )
        )

        lines.append(
            (
                f"- Longest repeated-token run: "
                f"{metrics['longest_consecutive_token_run']}"
            )
        )

        lines.append(
            ""
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed MathGPT "
            "generation benchmark."
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

    checkpoint_path = (
        resolve_path(
            args.checkpoint
        )
    )

    benchmark_path = (
        resolve_path(
            args.benchmark
        )
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

    # ========================================================
    # Benchmark
    # ========================================================

    benchmark = load_json(
        benchmark_path
    )

    validate_benchmark(
        benchmark
    )

    # ========================================================
    # Config
    # ========================================================

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

    training_step = int(
        metadata[
            "optimizer_step_count"
        ]
    )

    parameter_count = int(
        metadata[
            "parameter_count"
        ]
    )

    # ========================================================
    # Fingerprints
    # ========================================================

    print()
    print(
        "Calculating checkpoint SHA-256..."
    )

    checkpoint_hash = (
        file_sha256(
            checkpoint_path
        )
    )

    benchmark_hash = (
        file_sha256(
            benchmark_path
        )
    )

    # ========================================================
    # Header
    # ========================================================

    profiles = (
        benchmark[
            "decoding_profiles"
        ]
    )

    prompts = (
        benchmark[
            "prompts"
        ]
    )

    total_generations = (
        len(
            profiles
        )
        *
        len(
            prompts
        )
    )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE 11C — FIXED GENERATION BENCHMARK"
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
        f"{training_step:,}"
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    print()

    print(
        f"Benchmark: "
        f"{benchmark['name']}"
    )

    print(
        f"Version: "
        f"{benchmark['version']}"
    )

    print(
        f"Prompts: "
        f"{len(prompts)}"
    )

    print(
        f"Decoding profiles: "
        f"{len(profiles)}"
    )

    print(
        f"Total generations: "
        f"{total_generations}"
    )

    # ========================================================
    # Run benchmark
    # ========================================================

    results = []

    generation_index = 0

    for prompt_spec in prompts:

        for profile in profiles:

            generation_index += 1

            print()
            print(
                "=" * 76
            )

            print(
                f"[{generation_index}/"
                f"{total_generations}] "
                f"{prompt_spec['id']} "
                f"| "
                f"{profile['name']}"
            )

            print(
                "=" * 76
            )

            print()

            print(
                "Prompt:"
            )

            print(
                prompt_spec[
                    "prompt"
                ]
            )

            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=(
                    prompt_spec[
                        "prompt"
                    ]
                ),
                max_new_tokens=(
                    profile.get(
                        "max_new_tokens",
                        50,
                    )
                ),
                strategy=(
                    profile[
                        "strategy"
                    ]
                ),
                temperature=(
                    profile.get(
                        "temperature",
                        1.0,
                    )
                ),
                top_k=(
                    profile.get(
                        "top_k"
                    )
                ),
                top_p=(
                    profile.get(
                        "top_p"
                    )
                ),
                seed=(
                    profile.get(
                        "seed",
                        42,
                    )
                ),
                stop_on_eos=True,
            )

            metrics = (
                generation_metrics(
                    result
                )
            )

            record = {
                "prompt_id":
                    prompt_spec[
                        "id"
                    ],

                "category":
                    prompt_spec[
                        "category"
                    ],

                "prompt":
                    prompt_spec[
                        "prompt"
                    ],

                "decoding_profile":
                    profile[
                        "name"
                    ],

                "settings":
                    {
                        "strategy":
                            profile[
                                "strategy"
                            ],

                        "temperature":
                            profile.get(
                                "temperature"
                            ),

                        "top_k":
                            profile.get(
                                "top_k"
                            ),

                        "top_p":
                            profile.get(
                                "top_p"
                            ),

                        "seed":
                            profile.get(
                                "seed"
                            ),

                        "max_new_tokens":
                            profile.get(
                                "max_new_tokens",
                                50,
                            ),
                    },

                "generated_text":
                    result[
                        "generated_text"
                    ],

                "full_text":
                    result[
                        "text"
                    ],

                "generated_token_ids":
                    [
                        int(
                            token_id
                        )

                        for token_id
                        in result[
                            "generated_token_ids"
                        ]
                    ],

                "metrics":
                    metrics,
            }

            results.append(
                record
            )

            print()

            print(
                "Generated:"
            )

            print(
                result[
                    "generated_text"
                ]
            )

            print()

            print(
                "Metrics:"
            )

            print(
                f"  tokens: "
                f"{metrics['generated_tokens']}"
            )

            print(
                f"  EOS: "
                f"{metrics['stopped_on_eos']}"
            )

            print(
                f"  unique-token ratio: "
                f"{metrics['unique_token_ratio']:.4f}"
            )

            print(
                f"  longest repeat run: "
                f"{metrics['longest_consecutive_token_run']}"
            )

    # ========================================================
    # Final report
    # ========================================================

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output_prefix is None:

        output_prefix = (
            f"{args.model_label}"
            f"_generation_benchmark_"
            f"v{benchmark['version']}_"
            f"{timestamp}"
        )

    else:

        output_prefix = (
            args.output_prefix
        )

    json_output_path = (
        RESULTS_DIR
        /
        (
            output_prefix
            +
            ".json"
        )
    )

    markdown_output_path = (
        RESULTS_DIR
        /
        (
            output_prefix
            +
            ".md"
        )
    )

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
                training_step,

            "parameter_count":
                parameter_count,

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

            "prompt_count":
                len(
                    prompts
                ),

            "decoding_profile_count":
                len(
                    profiles
                ),

            "generation_count":
                len(
                    results
                ),
        },

        "results":
            results,
    }

    with open(
        json_output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    write_markdown_report(
        output_path=(
            markdown_output_path
        ),
        report=report,
    )

    # ========================================================
    # Final summary
    # ========================================================

    eos_count = sum(
        1

        for result
        in results

        if result[
            "metrics"
        ][
            "stopped_on_eos"
        ]
    )

    average_unique_ratio = (
        float(
            np.mean(
                [
                    result[
                        "metrics"
                    ][
                        "unique_token_ratio"
                    ]

                    for result
                    in results
                ]
            )
        )
    )

    maximum_repeat_run = max(
        result[
            "metrics"
        ][
            "longest_consecutive_token_run"
        ]

        for result
        in results
    )

    print()
    print(
        "=" * 76
    )

    print(
        "STAGE 11C BENCHMARK COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        f"Generations completed: "
        f"{len(results)}"
    )

    print(
        f"EOS completions: "
        f"{eos_count}"
    )

    print(
        f"Average unique-token ratio: "
        f"{average_unique_ratio:.4f}"
    )

    print(
        f"Maximum consecutive-token run: "
        f"{maximum_repeat_run}"
    )

    print()

    print(
        "JSON result:"
    )

    print(
        json_output_path
    )

    print()

    print(
        "Markdown report:"
    )

    print(
        markdown_output_path
    )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: FIXED GENERATION "
        "BENCHMARK COMPLETED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()