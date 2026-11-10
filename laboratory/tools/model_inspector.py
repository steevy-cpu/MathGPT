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
)


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "L6.1 model_inspector — "
            "read-only MathGPT laboratory instrument."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="MathGPT checkpoint to inspect.",
    )

    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help=(
            "Optional expected checkpoint "
            "SHA-256 fingerprint."
        ),
    )

    parser.add_argument(
        "--verify-prompt",
        type=str,
        default=(
            "Two plus two equals"
        ),
        help=(
            "Prompt used to verify that the "
            "laboratory traced forward pass "
            "matches GPT.forward()."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional JSON output path."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print()
    print(
        "=" * 76
    )

    print(
        "L6.1 — MODEL INSPECTOR"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Permission level: P0"
    )

    print(
        "Evidence level: E1 "
        "(direct model measurement)"
    )

    print()

    # ========================================================
    # Load runtime
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
    # Parameter measurements
    # ========================================================

    parameter_records = (
        runtime.parameter_records()
    )

    all_parameters_finite = all(
        record[
            "finite"
        ]

        for record
        in parameter_records
    )

    total_parameter_scalars = int(
        sum(
            record[
                "count"
            ]

            for record
            in parameter_records
        )
    )

    parameter_array_count = (
        len(
            parameter_records
        )
    )

    dtype_counts = {}

    for record in parameter_records:

        dtype = (
            record[
                "dtype"
            ]
        )

        dtype_counts[
            dtype
        ] = (
            dtype_counts.get(
                dtype,
                0,
            )
            +
            1
        )

    # ========================================================
    # Verify traced lab forward
    # ========================================================

    forward_verification = (
        runtime.verify_trace_forward(
            args.verify_prompt
        )
    )

    # ========================================================
    # Identity checks
    # ========================================================

    identity_checks = {
        name:
            bool(
                passed
            )

        for (
            name,
            passed,
        ) in runtime.identity_checks.items()
    }

    identity_checks[
        "all_parameters_finite"
    ] = (
        all_parameters_finite
    )

    identity_checks[
        "parameter_traversal_count"
    ] = (
        total_parameter_scalars
        ==
        8_110_848
    )

    identity_checks[
        "lab_trace_matches_gpt_forward"
    ] = (
        forward_verification[
            "exact"
        ]
    )

    all_checks_passed = all(
        identity_checks.values()
    )

    # ========================================================
    # MARS-style result
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
                "model_inspector",

            "version":
                LAB_TOOL_VERSION,

            "permission_level":
                "P0",

            "evidence_level":
                "E1",
        },

        "status":
            (
                "success"
                if all_checks_passed
                else "failed"
            ),

        "model":
            runtime.model_record(),

        "architecture": {
            "architecture":
                runtime.model_config[
                    "architecture"
                ],

            "normalization":
                runtime.model_config[
                    "normalization"
                ],

            "activation":
                runtime.model_config[
                    "activation"
                ],

            "position_embedding":
                runtime.model_config[
                    "position_embedding"
                ],

            "tie_embeddings":
                runtime.model_config[
                    "tie_embeddings"
                ],

            "use_bias":
                runtime.model_config[
                    "use_bias"
                ],

            "dropout":
                runtime.model_config[
                    "dropout"
                ],

            "dtype":
                runtime.model_config[
                    "dtype"
                ],
        },

        "parameters": {
            "array_count":
                int(
                    parameter_array_count
                ),

            "scalar_count":
                int(
                    total_parameter_scalars
                ),

            "all_finite":
                bool(
                    all_parameters_finite
                ),

            "dtype_array_counts":
                dtype_counts,
        },

        "forward_verification": {
            "prompt":
                args.verify_prompt,

            "exact":
                bool(
                    forward_verification[
                        "exact"
                    ]
                ),

            "max_absolute_difference":
                float(
                    forward_verification[
                        "max_absolute_difference"
                    ]
                ),

            "logit_shape":
                forward_verification[
                    "shape"
                ],
        },

        "checks":
            identity_checks,
    }

    # ========================================================
    # Display
    # ========================================================

    model_record = (
        report[
            "model"
        ]
    )

    print(
        "Checkpoint"
    )

    print(
        "----------"
    )

    print(
        model_record[
            "checkpoint_path"
        ]
    )

    print()

    print(
        f"SHA-256:"
    )

    print(
        model_record[
            "checkpoint_sha256"
        ]
    )

    print()

    print(
        f"Training step: "
        f"{model_record['training_step']:,}"
    )

    tokens_seen = (
        model_record.get(
            "tokens_seen"
        )
    )

    if tokens_seen is not None:

        print(
            f"Tokens seen: "
            f"{int(tokens_seen):,}"
        )

    print()

    print(
        "Architecture"
    )

    print(
        "------------"
    )

    print(
        f"Parameters: "
        f"{model_record['parameter_count']:,}"
    )

    print(
        f"Layers: "
        f"{model_record['num_layers']}"
    )

    print(
        f"Heads: "
        f"{model_record['num_heads']}"
    )

    print(
        f"Head dimension: "
        f"{model_record['head_dim']}"
    )

    print(
        f"d_model: "
        f"{model_record['d_model']}"
    )

    print(
        f"d_ff: "
        f"{model_record['d_ff']:,}"
    )

    print(
        f"Vocabulary: "
        f"{model_record['vocab_size']:,}"
    )

    print(
        f"Context length: "
        f"{model_record['context_length']}"
    )

    print()

    print(
        "Parameter integrity"
    )

    print(
        "-------------------"
    )

    print(
        f"Parameter arrays: "
        f"{parameter_array_count}"
    )

    print(
        f"Parameter scalars: "
        f"{total_parameter_scalars:,}"
    )

    print(
        f"All finite: "
        f"{all_parameters_finite}"
    )

    print()

    print(
        "Laboratory forward verification"
    )

    print(
        "-------------------------------"
    )

    print(
        f"Prompt: "
        f"{args.verify_prompt}"
    )

    print(
        f"Exact logits: "
        f"{forward_verification['exact']}"
    )

    print(
        f"Max absolute difference: "
        f"{forward_verification['max_absolute_difference']}"
    )

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
    ) in identity_checks.items():

        print(
            f"{name}: "
            f"{passed}"
        )

    # ========================================================
    # Optional output
    # ========================================================

    if args.output is not None:

        output_path = Path(
            args.output
        )

        if not output_path.is_absolute():

            output_path = (
                PROJECT_ROOT
                /
                output_path
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print()

        print(
            "JSON artifact:"
        )

        print(
            output_path
        )

    # ========================================================
    # Acceptance
    # ========================================================

    if not all_checks_passed:

        raise RuntimeError(
            "L6.1 model inspection failed."
        )

    print()
    print(
        "=" * 76
    )

    print(
        "SUCCESS: L6.1 MODEL INSPECTOR PASSED."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()