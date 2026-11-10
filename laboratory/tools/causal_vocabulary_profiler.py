import argparse
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
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Laboratory imports
# ============================================================

from laboratory.core.runtime import (
    LAB_TOOL_VERSION,
    MARS_PROTOCOL_VERSION,
    LaboratoryRuntime,
    sha256_file,
    token_statistics,
    token_text,
)

from laboratory.tools.attention_head_ablation import (
    distribution_metrics,
    forward_with_head_scale,
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_TRACKED_TEXTS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "plus",
    "minus",
    "equal",
    "equals",
    "France",
    "London",
]


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L7.2 causal_vocabulary_profiler — "
            "measure the next-token vocabulary-wide "
            "causal logit effect of one attention head."
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
        "--prompt",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--append-token-id",
        action="append",
        type=int,
        default=[],
    )

    parser.add_argument(
        "--layer",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--head",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--tracked-text",
        action="append",
        default=[],
        help=(
            "Text whose first tokenizer token "
            "will be tracked. Repeat as needed. "
            "If omitted, a default numeric/control "
            "set is used."
        ),
    )

    parser.add_argument(
        "--tracked-token-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Exact vocabulary token ID to track. "
            "Repeat as needed."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=str,
    )

    return parser.parse_args()


# ============================================================
# Softmax
# ============================================================

def softmax(
    logits,
):

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    maximum = np.max(
        logits
    )

    exp_values = np.exp(
        logits
        -
        maximum
    )

    return (
        exp_values
        /
        np.sum(
            exp_values
        )
    )


# ============================================================
# Stable vocabulary ranks
# ============================================================

def vocabulary_ranks(
    logits,
):

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    order = np.argsort(
        -logits,
        kind="stable",
    )

    ranks = np.empty(
        logits.shape[
            0
        ],
        dtype=np.int64,
    )

    ranks[
        order
    ] = np.arange(
        1,
        logits.shape[
            0
        ]
        +
        1,
        dtype=np.int64,
    )

    return ranks


# ============================================================
# One vocabulary record
# ============================================================

def vocabulary_record(
    *,
    runtime,
    token_id,
    baseline_logits,
    ablated_logits,
    causal_effect,
    baseline_probabilities,
    ablated_probabilities,
    baseline_ranks,
    ablated_ranks,
    effect_mean,
    effect_std,
):

    token_id = int(
        token_id
    )

    effect = float(
        causal_effect[
            token_id
        ]
    )

    if effect_std > 0.0:

        effect_z = float(
            (
                effect
                -
                effect_mean
            )
            /
            effect_std
        )

    else:

        effect_z = 0.0

    return {
        "token_id":
            token_id,

        "token_text":
            token_text(
                runtime.tokenizer,
                token_id,
            ),

        "causal_logit_effect":
            effect,

        "causal_effect_z":
            effect_z,

        "baseline_logit":
            float(
                baseline_logits[
                    token_id
                ]
            ),

        "ablated_logit":
            float(
                ablated_logits[
                    token_id
                ]
            ),

        "baseline_probability":
            float(
                baseline_probabilities[
                    token_id
                ]
            ),

        "ablated_probability":
            float(
                ablated_probabilities[
                    token_id
                ]
            ),

        "probability_delta":
            float(
                baseline_probabilities[
                    token_id
                ]
                -
                ablated_probabilities[
                    token_id
                ]
            ),

        "baseline_rank":
            int(
                baseline_ranks[
                    token_id
                ]
            ),

        "ablated_rank":
            int(
                ablated_ranks[
                    token_id
                ]
            ),

        "rank_delta_ablation_minus_baseline":
            int(
                ablated_ranks[
                    token_id
                ]
                -
                baseline_ranks[
                    token_id
                ]
            ),
    }


# ============================================================
# Track user-facing text
# ============================================================

def tracked_text_record(
    *,
    runtime,
    text,
    baseline_logits,
    ablated_logits,
    causal_effect,
    baseline_probabilities,
    ablated_probabilities,
    baseline_ranks,
    ablated_ranks,
    effect_mean,
    effect_std,
):

    token_ids = (
        runtime
        .tokenizer
        .encode(
            text,
            add_bos=False,
            add_eos=False,
        )
    )

    token_ids = [
        int(
            token_id
        )

        for token_id
        in token_ids
    ]

    if not token_ids:

        return {
            "text":
                text,

            "token_ids":
                [],

            "token_count":
                0,

            "first_token":
                None,
        }

    first_token_id = (
        token_ids[
            0
        ]
    )

    first_record = vocabulary_record(
        runtime=runtime,

        token_id=(
            first_token_id
        ),

        baseline_logits=(
            baseline_logits
        ),

        ablated_logits=(
            ablated_logits
        ),

        causal_effect=(
            causal_effect
        ),

        baseline_probabilities=(
            baseline_probabilities
        ),

        ablated_probabilities=(
            ablated_probabilities
        ),

        baseline_ranks=(
            baseline_ranks
        ),

        ablated_ranks=(
            ablated_ranks
        ),

        effect_mean=(
            effect_mean
        ),

        effect_std=(
            effect_std
        ),
    )

    return {
        "text":
            text,

        "token_ids":
            token_ids,

        "token_count":
            int(
                len(
                    token_ids
                )
            ),

        "token_texts":
            [
                token_text(
                    runtime.tokenizer,
                    token_id,
                )

                for token_id
                in token_ids
            ],

        "first_token":
            first_record,

        "warning":
            (
                None
                if len(
                    token_ids
                )
                ==
                1
                else
                (
                    "Tracked text uses multiple BPE "
                    "tokens. The reported causal "
                    "measurement refers only to its "
                    "first token at this next-token "
                    "decision point."
                )
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print()
    print(
        "=" * 80
    )

    print(
        "L7.2 — CAUSAL VOCABULARY PROFILER"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Permission level: P3"
    )

    print(
        "Evidence level: E3 "
        "(vocabulary-wide causal intervention)"
    )

    print()

    # ========================================================
    # Runtime
    # ========================================================

    runtime = LaboratoryRuntime(
        checkpoint_path=(
            args.checkpoint
        ),

        expected_checkpoint_sha256=(
            args.expected_sha256
        ),
    )

    # ========================================================
    # Validate target head
    # ========================================================

    if not (
        0
        <=
        args.layer
        <
        runtime.model.num_layers
    ):

        raise ValueError(
            f"Layer must be between 0 and "
            f"{runtime.model.num_layers - 1}."
        )

    if not (
        0
        <=
        args.head
        <
        runtime.model.num_heads
    ):

        raise ValueError(
            f"Head must be between 0 and "
            f"{runtime.model.num_heads - 1}."
        )

    head_id = (
        f"L{args.layer}H{args.head}"
    )

    if args.top_k <= 0:

        raise ValueError(
            "--top-k must be positive."
        )

    # ========================================================
    # Exact context
    # ========================================================

    prompt_token_ids = (
        runtime.encode_prompt(
            args.prompt
        )
    )

    appended_token_ids = [
        int(
            token_id
        )

        for token_id
        in args.append_token_id
    ]

    for token_id in appended_token_ids:

        if not (
            0
            <=
            token_id
            <
            runtime.model.vocab_size
        ):

            raise ValueError(
                f"Appended token ID "
                f"{token_id} is invalid."
            )

    context_token_ids = (
        list(
            prompt_token_ids
        )
        +
        appended_token_ids
    )

    if not context_token_ids:

        raise ValueError(
            "Context cannot be empty."
        )

    if (
        len(
            context_token_ids
        )
        >
        runtime.model.context_length
    ):

        raise ValueError(
            "Context exceeds model "
            "context length."
        )

    input_ids = np.asarray(
        [
            context_token_ids
        ],
        dtype=np.int64,
    )

    # ========================================================
    # True baseline
    # ========================================================

    baseline_logits_full = (
        runtime.model.forward(
            input_ids
        )
    )

    baseline_logits = (
        baseline_logits_full[
            0,
            -1,
            :
        ].astype(
            np.float64,
            copy=False,
        )
    )

    # ========================================================
    # Scale-1 control
    # ========================================================

    (
        control_logits_full,
        control_record,
    ) = forward_with_head_scale(
        runtime=runtime,

        token_ids=input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            args.head
        ),

        head_scale=1.0,
    )

    control_exact = bool(
        np.array_equal(
            baseline_logits_full,
            control_logits_full,
        )
    )

    control_max_difference = float(
        np.max(
            np.abs(
                baseline_logits_full.astype(
                    np.float64
                )
                -
                control_logits_full.astype(
                    np.float64
                )
            )
        )
    )

    # ========================================================
    # Full head ablation
    # ========================================================

    (
        ablated_logits_full,
        ablation_record,
    ) = forward_with_head_scale(
        runtime=runtime,

        token_ids=input_ids,

        target_layer=(
            args.layer
        ),

        target_head=(
            args.head
        ),

        head_scale=0.0,
    )

    ablated_logits = (
        ablated_logits_full[
            0,
            -1,
            :
        ].astype(
            np.float64,
            copy=False,
        )
    )

    # ========================================================
    # Vocabulary-wide causal effect
    #
    # Positive:
    # intact head raises this token's logit.
    #
    # Negative:
    # intact head lowers this token's logit.
    # ========================================================

    causal_effect = (
        baseline_logits
        -
        ablated_logits
    )

    effect_mean = float(
        np.mean(
            causal_effect
        )
    )

    effect_std = float(
        np.std(
            causal_effect
        )
    )

    effect_l2 = float(
        np.linalg.norm(
            causal_effect
        )
    )

    effect_mean_abs = float(
        np.mean(
            np.abs(
                causal_effect
            )
        )
    )

    effect_max = float(
        np.max(
            causal_effect
        )
    )

    effect_min = float(
        np.min(
            causal_effect
        )
    )

    # ========================================================
    # Probabilities and ranks
    # ========================================================

    baseline_probabilities = (
        softmax(
            baseline_logits
        )
    )

    ablated_probabilities = (
        softmax(
            ablated_logits
        )
    )

    baseline_ranks = (
        vocabulary_ranks(
            baseline_logits
        )
    )

    ablated_ranks = (
        vocabulary_ranks(
            ablated_logits
        )
    )

    # ========================================================
    # Top supported / suppressed tokens
    # ========================================================

    top_k = min(
        args.top_k,
        runtime.model.vocab_size,
    )

    supported_order = np.argsort(
        -causal_effect,
        kind="stable",
    )[
        :top_k
    ]

    suppressed_order = np.argsort(
        causal_effect,
        kind="stable",
    )[
        :top_k
    ]

    top_supported = [
        vocabulary_record(
            runtime=runtime,

            token_id=(
                token_id
            ),

            baseline_logits=(
                baseline_logits
            ),

            ablated_logits=(
                ablated_logits
            ),

            causal_effect=(
                causal_effect
            ),

            baseline_probabilities=(
                baseline_probabilities
            ),

            ablated_probabilities=(
                ablated_probabilities
            ),

            baseline_ranks=(
                baseline_ranks
            ),

            ablated_ranks=(
                ablated_ranks
            ),

            effect_mean=(
                effect_mean
            ),

            effect_std=(
                effect_std
            ),
        )

        for token_id
        in supported_order
    ]

    top_suppressed = [
        vocabulary_record(
            runtime=runtime,

            token_id=(
                token_id
            ),

            baseline_logits=(
                baseline_logits
            ),

            ablated_logits=(
                ablated_logits
            ),

            causal_effect=(
                causal_effect
            ),

            baseline_probabilities=(
                baseline_probabilities
            ),

            ablated_probabilities=(
                ablated_probabilities
            ),

            baseline_ranks=(
                baseline_ranks
            ),

            ablated_ranks=(
                ablated_ranks
            ),

            effect_mean=(
                effect_mean
            ),

            effect_std=(
                effect_std
            ),
        )

        for token_id
        in suppressed_order
    ]

    # ========================================================
    # Tracked texts
    # ========================================================

    if args.tracked_text:

        tracked_texts = list(
            args.tracked_text
        )

    else:

        tracked_texts = list(
            DEFAULT_TRACKED_TEXTS
        )

    tracked_text_records = [
        tracked_text_record(
            runtime=runtime,

            text=text,

            baseline_logits=(
                baseline_logits
            ),

            ablated_logits=(
                ablated_logits
            ),

            causal_effect=(
                causal_effect
            ),

            baseline_probabilities=(
                baseline_probabilities
            ),

            ablated_probabilities=(
                ablated_probabilities
            ),

            baseline_ranks=(
                baseline_ranks
            ),

            ablated_ranks=(
                ablated_ranks
            ),

            effect_mean=(
                effect_mean
            ),

            effect_std=(
                effect_std
            ),
        )

        for text
        in tracked_texts
    ]

    # ========================================================
    # Exact tracked token IDs
    # ========================================================

    tracked_token_records = []

    for token_id in args.tracked_token_id:

        token_id = int(
            token_id
        )

        if not (
            0
            <=
            token_id
            <
            runtime.model.vocab_size
        ):

            raise ValueError(
                f"Tracked token ID "
                f"{token_id} is invalid."
            )

        tracked_token_records.append(
            vocabulary_record(
                runtime=runtime,

                token_id=(
                    token_id
                ),

                baseline_logits=(
                    baseline_logits
                ),

                ablated_logits=(
                    ablated_logits
                ),

                causal_effect=(
                    causal_effect
                ),

                baseline_probabilities=(
                    baseline_probabilities
                ),

                ablated_probabilities=(
                    ablated_probabilities
                ),

                baseline_ranks=(
                    baseline_ranks
                ),

                ablated_ranks=(
                    ablated_ranks
                ),

                effect_mean=(
                    effect_mean
                ),

                effect_std=(
                    effect_std
                ),
            )
        )

    # ========================================================
    # Global distribution change
    # ========================================================

    distribution_change = (
        distribution_metrics(
            baseline_logits,
            ablated_logits,
        )
    )

    # ========================================================
    # Output paths
    # ========================================================

    output_prefix = Path(
        args.output_prefix
    )

    if not output_prefix.is_absolute():

        output_prefix = (
            PROJECT_ROOT
            /
            output_prefix
        )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_prefix.with_suffix(
            ".json"
        )
    )

    npz_path = (
        output_prefix.with_suffix(
            ".npz"
        )
    )

    # ========================================================
    # Raw artifact
    # ========================================================

    np.savez(
        npz_path,

        context_token_ids=np.asarray(
            context_token_ids,
            dtype=np.int64,
        ),

        baseline_next_token_logits=(
            baseline_logits.astype(
                np.float32
            )
        ),

        ablated_next_token_logits=(
            ablated_logits.astype(
                np.float32
            )
        ),

        causal_logit_effect=(
            causal_effect.astype(
                np.float32
            )
        ),

        baseline_probabilities=(
            baseline_probabilities.astype(
                np.float32
            )
        ),

        ablated_probabilities=(
            ablated_probabilities.astype(
                np.float32
            )
        ),
    )

    npz_hash = (
        sha256_file(
            npz_path
        )
    )

    # ========================================================
    # Integrity
    # ========================================================

    all_finite = bool(
        np.all(
            np.isfinite(
                baseline_logits
            )
        )
        and
        np.all(
            np.isfinite(
                ablated_logits
            )
        )
        and
        np.all(
            np.isfinite(
                causal_effect
            )
        )
    )

    checks = {
        "checkpoint_identity_valid":
            all(
                runtime
                .identity_checks
                .values()
            ),

        "scale_one_control_exact":
            control_exact,

        "scale_zero_fully_ablates_head":
            (
                ablation_record[
                    "intervened_last_token_l2"
                ]
                ==
                0.0
            ),

        "vocabulary_size_matches":
            (
                causal_effect.shape[
                    0
                ]
                ==
                runtime.model.vocab_size
            ),

        "all_measurements_finite":
            all_finite,
    }

    all_checks_passed = all(
        checks.values()
    )

    # ========================================================
    # Context records
    # ========================================================

    context_records = [
        {
            "position":
                int(
                    position
                ),

            "token_id":
                int(
                    token_id
                ),

            "token_text":
                token_text(
                    runtime.tokenizer,
                    token_id,
                ),
        }

        for (
            position,
            token_id,
        ) in enumerate(
            context_token_ids
        )
    ]

    # ========================================================
    # JSON report
    # ========================================================

    report = {
        "mars_protocol":
            MARS_PROTOCOL_VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "tool": {
            "name":
                "causal_vocabulary_profiler",

            "version":
                LAB_TOOL_VERSION,

            "permission_level":
                "P3",

            "evidence_level":
                "E3",
        },

        "status":
            (
                "success"
                if all_checks_passed
                else "failed"
            ),

        "model":
            runtime.model_record(),

        "input": {
            "prompt":
                args.prompt,

            "prompt_token_ids":
                [
                    int(
                        token_id
                    )

                    for token_id
                    in prompt_token_ids
                ],

            "appended_token_ids":
                appended_token_ids,

            "context":
                context_records,
        },

        "intervention": {
            "type":
                "attention_head_ablation",

            "layer":
                int(
                    args.layer
                ),

            "head":
                int(
                    args.head
                ),

            "head_id":
                head_id,

            "scale":
                0.0,

            "record":
                ablation_record,
        },

        "control_verification": {
            "scale":
                1.0,

            "exact":
                control_exact,

            "max_absolute_difference":
                control_max_difference,

            "record":
                control_record,
        },

        "effect_definition": {
            "formula":
                (
                    "baseline_next_token_logit "
                    "- ablated_next_token_logit"
                ),

            "positive":
                (
                    "The intact head increases "
                    "this vocabulary token's logit."
                ),

            "negative":
                (
                    "The intact head suppresses "
                    "this vocabulary token's logit."
                ),
        },

        "effect_statistics": {
            "vocabulary_size":
                int(
                    causal_effect.shape[
                        0
                    ]
                ),

            "mean":
                effect_mean,

            "std":
                effect_std,

            "l2_norm":
                effect_l2,

            "mean_absolute_effect":
                effect_mean_abs,

            "maximum_support_effect":
                effect_max,

            "maximum_suppression_effect":
                effect_min,
        },

        "top_supported_tokens":
            top_supported,

        "top_suppressed_tokens":
            top_suppressed,

        "tracked_texts":
            tracked_text_records,

        "tracked_token_ids":
            tracked_token_records,

        "distribution_change":
            distribution_change,

        "artifacts": {
            "json":
                str(
                    json_path
                ),

            "raw_npz":
                str(
                    npz_path
                ),

            "raw_npz_sha256":
                npz_hash,
        },

        "checks":
            checks,
    }

    # ========================================================
    # Save report
    # ========================================================

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

    # ========================================================
    # Display
    # ========================================================

    print(
        "Experiment"
    )

    print(
        "----------"
    )

    print(
        f"Prompt: "
        f"{repr(args.prompt)}"
    )

    print(
        f"Head: {head_id}"
    )

    print()

    print(
        "Context:"
    )

    for record in context_records:

        print(
            (
                f"  [{record['position']}] "
                f"id={record['token_id']:<5} "
                f"text={repr(record['token_text'])}"
            )
        )

    print()
    print(
        "Control verification"
    )

    print(
        "--------------------"
    )

    print(
        f"Scale-1 exact: "
        f"{control_exact}"
    )

    print(
        f"Max absolute difference: "
        f"{control_max_difference}"
    )

    # ========================================================
    # Global effect
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "VOCABULARY-WIDE CAUSAL EFFECT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Effect = baseline logit - ablated logit"
    )

    print(
        "Positive = intact head supports token"
    )

    print(
        "Negative = intact head suppresses token"
    )

    print()

    print(
        f"Vocabulary size: "
        f"{runtime.model.vocab_size:,}"
    )

    print(
        f"Effect mean: "
        f"{effect_mean:+.8f}"
    )

    print(
        f"Effect std: "
        f"{effect_std:.8f}"
    )

    print(
        f"Effect L2: "
        f"{effect_l2:.8f}"
    )

    print(
        f"Mean |effect|: "
        f"{effect_mean_abs:.8f}"
    )

    # ========================================================
    # Top supported
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        f"TOP {top_k} TOKENS SUPPORTED BY {head_id}"
    )

    print(
        "=" * 80
    )

    print()

    for (
        rank,
        record,
    ) in enumerate(
        top_supported,
        start=1,
    ):

        print(
            (
                f"#{rank:<3} "
                f"id={record['token_id']:<5} "
                f"text={repr(record['token_text']):<18} "
                f"effect="
                f"{record['causal_logit_effect']:+.6f} "
                f"z="
                f"{record['causal_effect_z']:+.3f} "
                f"rank "
                f"{record['baseline_rank']:,}"
                f"→"
                f"{record['ablated_rank']:,}"
            )
        )

    # ========================================================
    # Top suppressed
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        f"TOP {top_k} TOKENS SUPPRESSED BY {head_id}"
    )

    print(
        "=" * 80
    )

    print()

    for (
        rank,
        record,
    ) in enumerate(
        top_suppressed,
        start=1,
    ):

        print(
            (
                f"#{rank:<3} "
                f"id={record['token_id']:<5} "
                f"text={repr(record['token_text']):<18} "
                f"effect="
                f"{record['causal_logit_effect']:+.6f} "
                f"z="
                f"{record['causal_effect_z']:+.3f} "
                f"rank "
                f"{record['baseline_rank']:,}"
                f"→"
                f"{record['ablated_rank']:,}"
            )
        )

    # ========================================================
    # Tracked vocabulary
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "TRACKED TOKENS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        (
            f"{'Text':<12}"
            f"{'Token':<14}"
            f"{'ID':>8}"
            f"{'Effect':>14}"
            f"{'Z':>10}"
            f"{'Base Rank':>14}"
            f"{'Abl Rank':>14}"
        )
    )

    print(
        "-" * 86
    )

    for record in tracked_text_records:

        first = (
            record[
                "first_token"
            ]
        )

        if first is None:

            print(
                (
                    f"{record['text']:<12}"
                    f"{'<empty>':<14}"
                )
            )

            continue

        print(
            (
                f"{record['text']:<12}"
                f"{repr(first['token_text']):<14}"
                f"{first['token_id']:>8}"
                f"{first['causal_logit_effect']:>+14.6f}"
                f"{first['causal_effect_z']:>+10.3f}"
                f"{first['baseline_rank']:>14,}"
                f"{first['ablated_rank']:>14,}"
            )
        )

        if (
            record[
                "warning"
            ]
            is not None
        ):

            print(
                (
                    f"  note: "
                    f"{record['text']} tokenizes as "
                    f"{record['token_texts']}; "
                    f"table reports its first token."
                )
            )

    # ========================================================
    # Distribution change
    # ========================================================

    print()
    print(
        "Distribution change"
    )

    print(
        "-------------------"
    )

    print(
        (
            f"KL(base || ablated): "
            f"{distribution_change['kl_baseline_to_intervention']:.8f}"
        )
    )

    print(
        (
            f"Probability L1: "
            f"{distribution_change['probability_l1_distance']:.8f}"
        )
    )

    print(
        (
            f"Logit L2: "
            f"{distribution_change['logit_l2_difference']:.8f}"
        )
    )

    print(
        (
            f"Top-1 changed: "
            f"{distribution_change['top1_changed']}"
        )
    )

    # ========================================================
    # Checks
    # ========================================================

    print()
    print(
        "Checks"
    )

    print(
        "------"
    )

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    print()
    print(
        "Artifacts"
    )

    print(
        "---------"
    )

    print(
        f"JSON: "
        f"{json_path}"
    )

    print(
        f"NPZ:  "
        f"{npz_path}"
    )

    print(
        f"NPZ SHA-256:"
    )

    print(
        npz_hash
    )

    if not all_checks_passed:

        raise RuntimeError(
            "L7.2 causal vocabulary "
            "profiler failed."
        )

    print()
    print(
        "=" * 80
    )

    print(
        "SUCCESS: L7.2 CAUSAL VOCABULARY "
        "PROFILER PASSED."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()