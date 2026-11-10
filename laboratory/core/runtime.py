import hashlib
import json
import sys

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
# MathGPT imports
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

from training.parameters import (
    named_parameters,
)


# ============================================================
# Frozen architecture / tokenizer locations
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


TOKENIZER_CONFIG_PATH = (
    TOKENIZER_DIR
    / "tokenizer_config.json"
)


TOKENIZER_MERGES_PATH = (
    TOKENIZER_DIR
    / "merges.json"
)


TOKENIZER_VOCAB_PATH = (
    TOKENIZER_DIR
    / "vocab.json"
)


# ============================================================
# Laboratory constants
# ============================================================

MARS_PROTOCOL_VERSION = (
    "MARS-0.1"
)


LAB_TOOL_VERSION = (
    "0.1"
)


EXPECTED_PARAMETER_COUNT = (
    8_110_848
)


EXPECTED_VOCAB_SIZE = (
    16_000
)


EXPECTED_CONTEXT_LENGTH = (
    256
)


EXPECTED_D_MODEL = (
    256
)


EXPECTED_NUM_LAYERS = (
    5
)


EXPECTED_NUM_HEADS = (
    4
)


EXPECTED_D_FF = (
    1_024
)


# ============================================================
# General helpers
# ============================================================

def load_json(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def resolve_project_path(
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


def sha256_file(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

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


def all_finite(
    array,
):

    return bool(
        np.all(
            np.isfinite(
                array
            )
        )
    )


# ============================================================
# Safe token display
# ============================================================

def token_text(
    tokenizer,
    token_id,
):

    token_id = int(
        token_id
    )

    if (
        token_id
        ==
        tokenizer.pad_token_id
    ):

        return "<PAD>"

    if (
        token_id
        ==
        tokenizer.bos_token_id
    ):

        return "<BOS>"

    if (
        token_id
        ==
        tokenizer.eos_token_id
    ):

        return "<EOS>"

    token_bytes = (
        tokenizer
        .token_bytes
        .get(
            token_id
        )
    )

    if token_bytes is None:

        return (
            f"<UNKNOWN:{token_id}>"
        )

    return token_bytes.decode(
        "utf-8",
        errors="replace",
    )


# ============================================================
# Read-only final LayerNorm
# ============================================================

def readonly_layer_norm(
    layer_norm,
    x,
):
    """
    Apply an existing LayerNorm without modifying
    the LayerNorm forward cache.

    This is useful for interpretability tools where
    we want to project intermediate residual-stream
    representations through the trained final norm.

    No parameters are modified.
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    mean = np.mean(
        x,
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )

    centered = (
        x
        -
        mean
    )

    variance = np.mean(
        centered
        *
        centered,
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )

    inv_std = (
        np.float32(
            1.0
        )
        /
        np.sqrt(
            variance
            +
            layer_norm.epsilon
        )
    )

    normalized = (
        centered
        *
        inv_std
    )

    output = (
        layer_norm.gamma
        *
        normalized
        +
        layer_norm.beta
    )

    return output.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Token statistics
# ============================================================

def token_statistics(
    logits,
    target_token_id,
):
    """
    Compute deterministic next-token statistics.

    Returns:
        probability
        log_probability
        rank

    Rank 1 means highest logit.
    """

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    if logits.ndim != 1:

        raise ValueError(
            "logits must be a 1D vocabulary vector."
        )

    target_token_id = int(
        target_token_id
    )

    if not (
        0
        <=
        target_token_id
        <
        logits.shape[0]
    ):

        raise ValueError(
            "target token ID is outside "
            "the vocabulary."
        )

    maximum_logit = np.max(
        logits
    )

    shifted = (
        logits
        -
        maximum_logit
    )

    exp_logits = np.exp(
        shifted
    )

    denominator = np.sum(
        exp_logits
    )

    probability = (
        exp_logits[
            target_token_id
        ]
        /
        denominator
    )

    log_probability = (
        shifted[
            target_token_id
        ]
        -
        np.log(
            denominator
        )
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

        "logit":
            float(
                target_logit
            ),
    }


# ============================================================
# Laboratory runtime
# ============================================================

class LaboratoryRuntime:
    """
    Shared read-only MathGPT runtime for laboratory tools.

    Responsibilities:
        - load exact model architecture
        - load tokenizer
        - restore checkpoint
        - fingerprint artifacts
        - expose safe model metadata
        - provide deterministic forward tracing

    This runtime DOES NOT:
        - optimize parameters
        - save checkpoints
        - mutate canonical checkpoint files
    """

    def __init__(
        self,
        checkpoint_path,
        expected_checkpoint_sha256=None,
        seed=42,
    ):

        self.checkpoint_path = (
            resolve_project_path(
                checkpoint_path
            )
        )

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint missing: "
                f"{self.checkpoint_path}"
            )

        self.seed = int(
            seed
        )

        # ====================================================
        # Load frozen configs
        # ====================================================

        self.model_config = load_json(
            MODEL_CONFIG_PATH
        )

        self.training_config = load_json(
            TRAINING_CONFIG_PATH
        )

        # ====================================================
        # Fingerprint checkpoint BEFORE loading it
        # ====================================================

        self.checkpoint_sha256 = (
            sha256_file(
                self.checkpoint_path
            )
        )

        if (
            expected_checkpoint_sha256
            is not None
            and
            self.checkpoint_sha256
            !=
            expected_checkpoint_sha256
        ):

            raise RuntimeError(
                "Checkpoint SHA-256 mismatch.\n"
                f"Expected: "
                f"{expected_checkpoint_sha256}\n"
                f"Actual:   "
                f"{self.checkpoint_sha256}"
            )

        # ====================================================
        # Load tokenizer
        # ====================================================

        self.tokenizer = (
            BPETokenizer.from_files(
                config_path=(
                    TOKENIZER_CONFIG_PATH
                ),

                merges_path=(
                    TOKENIZER_MERGES_PATH
                ),

                vocab_path=(
                    TOKENIZER_VOCAB_PATH
                ),
            )
        )

        # ====================================================
        # Build fresh architecture
        # ====================================================

        self.model = GPT.from_config(
            self.model_config,
            seed=self.seed,
        )

        # ====================================================
        # Temporary optimizer
        #
        # Current checkpoint loading restores model and
        # optimizer together.
        #
        # Laboratory inference never calls optimizer.step().
        # ====================================================

        optimizer_config = (
            self.training_config[
                "optimizer"
            ]
        )

        self.optimizer = AdamW(
            model=self.model,

            learning_rate=float(
                optimizer_config[
                    "learning_rate"
                ]
            ),

            beta1=float(
                optimizer_config[
                    "beta1"
                ]
            ),

            beta2=float(
                optimizer_config[
                    "beta2"
                ]
            ),

            epsilon=float(
                optimizer_config[
                    "epsilon"
                ]
            ),

            weight_decay=float(
                optimizer_config[
                    "weight_decay"
                ]
            ),
        )

        # ====================================================
        # Restore checkpoint
        # ====================================================

        self.checkpoint_metadata = (
            load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                path=self.checkpoint_path,
                strict=True,
            )
        )

        # ====================================================
        # Validate architecture
        # ====================================================

        self._validate_model_identity()

    # ========================================================
    # Identity validation
    # ========================================================

    def _validate_model_identity(
        self,
    ):

        checks = {
            "parameter_count":
                (
                    self.model.parameter_count()
                    ==
                    EXPECTED_PARAMETER_COUNT
                ),

            "checkpoint_parameter_count":
                (
                    int(
                        self.checkpoint_metadata[
                            "parameter_count"
                        ]
                    )
                    ==
                    EXPECTED_PARAMETER_COUNT
                ),

            "vocab_size":
                (
                    self.model.vocab_size
                    ==
                    EXPECTED_VOCAB_SIZE
                ),

            "context_length":
                (
                    self.model.context_length
                    ==
                    EXPECTED_CONTEXT_LENGTH
                ),

            "d_model":
                (
                    self.model.d_model
                    ==
                    EXPECTED_D_MODEL
                ),

            "num_layers":
                (
                    self.model.num_layers
                    ==
                    EXPECTED_NUM_LAYERS
                ),

            "num_heads":
                (
                    self.model.num_heads
                    ==
                    EXPECTED_NUM_HEADS
                ),

            "d_ff":
                (
                    self.model.d_ff
                    ==
                    EXPECTED_D_FF
                ),
        }

        failed = [
            name

            for (
                name,
                passed,
            ) in checks.items()

            if not passed
        ]

        if failed:

            raise RuntimeError(
                "Laboratory model identity "
                f"check failed: {failed}"
            )

        self.identity_checks = (
            checks
        )

    # ========================================================
    # Parameter inspection
    # ========================================================

    def parameter_records(
        self,
    ):

        records = []

        for (
            name,
            parameter,
            _,
        ) in named_parameters(
            self.model
        ):

            records.append(
                {
                    "name":
                        name,

                    "shape":
                        [
                            int(
                                dimension
                            )

                            for dimension
                            in parameter.shape
                        ],

                    "dtype":
                        str(
                            parameter.dtype
                        ),

                    "count":
                        int(
                            parameter.size
                        ),

                    "finite":
                        all_finite(
                            parameter
                        ),

                    "mean":
                        float(
                            np.mean(
                                parameter,
                                dtype=np.float64,
                            )
                        ),

                    "std":
                        float(
                            np.std(
                                parameter,
                                dtype=np.float64,
                            )
                        ),

                    "min":
                        float(
                            np.min(
                                parameter
                            )
                        ),

                    "max":
                        float(
                            np.max(
                                parameter
                            )
                        ),
                }
            )

        return records

    # ========================================================
    # Artifact fingerprints
    # ========================================================

    def tokenizer_fingerprints(
        self,
    ):

        return {
            "tokenizer_config_sha256":
                sha256_file(
                    TOKENIZER_CONFIG_PATH
                ),

            "merges_sha256":
                sha256_file(
                    TOKENIZER_MERGES_PATH
                ),

            "vocab_sha256":
                sha256_file(
                    TOKENIZER_VOCAB_PATH
                ),
        }

    # ========================================================
    # Model record
    # ========================================================

    def model_record(
        self,
    ):

        extra_state = (
            self.checkpoint_metadata.get(
                "extra_state",
                {},
            )
        )

        return {
            "checkpoint_path":
                str(
                    self.checkpoint_path
                ),

            "checkpoint_name":
                self.checkpoint_path.name,

            "checkpoint_sha256":
                self.checkpoint_sha256,

            "training_step":
                int(
                    self.checkpoint_metadata[
                        "optimizer_step_count"
                    ]
                ),

            "tokens_seen":
                extra_state.get(
                    "tokens_seen"
                ),

            "parameter_count":
                int(
                    self.model.parameter_count()
                ),

            "vocab_size":
                int(
                    self.model.vocab_size
                ),

            "context_length":
                int(
                    self.model.context_length
                ),

            "d_model":
                int(
                    self.model.d_model
                ),

            "num_layers":
                int(
                    self.model.num_layers
                ),

            "num_heads":
                int(
                    self.model.num_heads
                ),

            "head_dim":
                int(
                    self.model.d_model
                    //
                    self.model.num_heads
                ),

            "d_ff":
                int(
                    self.model.d_ff
                ),

            "tokenizer":
                "tokenizer_50m_v0.2",

            "tokenizer_fingerprints":
                self.tokenizer_fingerprints(),
        }

    # ========================================================
    # Prompt encoding
    # ========================================================

    def encode_prompt(
        self,
        prompt,
    ):

        token_ids = (
            self.tokenizer.encode(
                prompt,
                add_bos=False,
                add_eos=False,
            )
        )

        if not token_ids:

            token_ids = [
                self.tokenizer.bos_token_id
            ]

        if (
            len(
                token_ids
            )
            >
            self.model.context_length
        ):

            token_ids = (
                token_ids[
                    -self.model.context_length:
                ]
            )

        return [
            int(
                token_id
            )

            for token_id
            in token_ids
        ]

    # ========================================================
    # Read-only traced forward pass
    # ========================================================

    def forward_with_trace(
        self,
        token_ids,
    ):
        """
        Forward pass with explicit intermediate captures.

        Important:
            This reproduces the existing MathGPT forward
            computation. It does not modify trainable
            parameters.

        Returns:
            logits
            trace dictionary
        """

        token_ids = np.asarray(
            token_ids,
            dtype=np.int64,
        )

        if token_ids.ndim == 1:

            token_ids = token_ids[
                None,
                :
            ]

        if token_ids.ndim != 2:

            raise ValueError(
                "token_ids must have shape "
                "(batch, sequence)."
            )

        if (
            token_ids.shape[1]
            >
            self.model.context_length
        ):

            raise ValueError(
                "Sequence exceeds model "
                "context length."
            )

        trace = {}

        # ====================================================
        # Embedding output
        # ====================================================

        x = (
            self.model
            .input_embedding
            .forward(
                token_ids
            )
        )

        trace[
            "embedding_output"
        ] = np.array(
            x,
            copy=True,
        )

        # ====================================================
        # Transformer blocks
        # ====================================================

        for (
            block_index,
            block,
        ) in enumerate(
            self.model.blocks
        ):

            block_prefix = (
                f"block_{block_index}"
            )

            block_input = (
                x
            )

            normalized_x = (
                block.norm1.forward(
                    block_input
                )
            )

            (
                attention_output,
                attention_weights,
            ) = (
                block
                .attention
                .forward(
                    normalized_x
                )
            )

            attention_residual = (
                block_input
                +
                attention_output
            )

            normalized_attention = (
                block.norm2.forward(
                    attention_residual
                )
            )

            mlp_output = (
                block.mlp.forward(
                    normalized_attention
                )
            )

            x = (
                attention_residual
                +
                mlp_output
            )

            trace[
                f"{block_prefix}.input"
            ] = np.array(
                block_input,
                copy=True,
            )

            trace[
                f"{block_prefix}.norm1"
            ] = np.array(
                normalized_x,
                copy=True,
            )

            trace[
                f"{block_prefix}.attention_output"
            ] = np.array(
                attention_output,
                copy=True,
            )

            trace[
                f"{block_prefix}.attention_weights"
            ] = np.array(
                attention_weights,
                copy=True,
            )

            trace[
                f"{block_prefix}.attention_residual"
            ] = np.array(
                attention_residual,
                copy=True,
            )

            trace[
                f"{block_prefix}.norm2"
            ] = np.array(
                normalized_attention,
                copy=True,
            )

            trace[
                f"{block_prefix}.mlp_output"
            ] = np.array(
                mlp_output,
                copy=True,
            )

            trace[
                f"{block_prefix}.output"
            ] = np.array(
                x,
                copy=True,
            )

        # ====================================================
        # Final norm
        # ====================================================

        final_hidden = (
            self.model
            .final_norm
            .forward(
                x
            )
        )

        trace[
            "final_layernorm"
        ] = np.array(
            final_hidden,
            copy=True,
        )

        # ====================================================
        # Tied vocabulary projection
        # ====================================================

        embedding_weight = (
            self.model
            .input_embedding
            .token_embedding
            .weight
        )

        logits = (
            final_hidden
            @
            embedding_weight.T
        )

        logits = logits.astype(
            np.float32,
            copy=False,
        )

        return (
            logits,
            trace,
        )

    # ========================================================
    # Verify lab forward path
    # ========================================================

    def verify_trace_forward(
        self,
        prompt,
    ):
        """
        Prove that the laboratory traced forward pass
        reproduces GPT.forward() exactly.
        """

        prompt_ids = (
            self.encode_prompt(
                prompt
            )
        )

        input_ids = np.asarray(
            [
                prompt_ids
            ],
            dtype=np.int64,
        )

        normal_logits = (
            self.model.forward(
                input_ids
            )
        )

        traced_logits, _ = (
            self.forward_with_trace(
                input_ids
            )
        )

        exact = bool(
            np.array_equal(
                normal_logits,
                traced_logits,
            )
        )

        max_absolute_difference = float(
            np.max(
                np.abs(
                    normal_logits.astype(
                        np.float64
                    )
                    -
                    traced_logits.astype(
                        np.float64
                    )
                )
            )
        )

        return {
            "exact":
                exact,

            "max_absolute_difference":
                max_absolute_difference,

            "shape":
                [
                    int(
                        value
                    )

                    for value
                    in normal_logits.shape
                ],
        }