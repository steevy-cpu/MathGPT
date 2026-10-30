def named_parameters(
    model,
):
    """
    Yield every UNIQUE trainable parameter
    together with its matching gradient.

    Yields:
        name,
        parameter,
        gradient

    Important:
        The token embedding matrix is tied to
        the output projection, so it appears
        only ONCE here.
    """

    # ========================================================
    # Input embeddings
    # ========================================================

    yield (
        "token_embedding.weight",
        model
        .input_embedding
        .token_embedding
        .weight,
        model
        .input_embedding
        .token_embedding
        .dweight,
    )

    yield (
        "position_embedding.weight",
        model
        .input_embedding
        .position_embedding
        .weight,
        model
        .input_embedding
        .position_embedding
        .dweight,
    )

    # ========================================================
    # Transformer blocks
    # ========================================================

    for (
        block_index,
        block,
    ) in enumerate(
        model.blocks
    ):

        prefix = (
            f"block_{block_index}"
        )

        # ----------------------------------------------------
        # LayerNorm 1
        # ----------------------------------------------------

        yield (
            f"{prefix}.norm1.gamma",
            block.norm1.gamma,
            block.norm1.dgamma,
        )

        yield (
            f"{prefix}.norm1.beta",
            block.norm1.beta,
            block.norm1.dbeta,
        )

        # ----------------------------------------------------
        # Attention projections
        # ----------------------------------------------------

        projections = [
            (
                "query",
                block
                .attention
                .query_projection,
            ),
            (
                "key",
                block
                .attention
                .key_projection,
            ),
            (
                "value",
                block
                .attention
                .value_projection,
            ),
            (
                "output",
                block
                .attention
                .output_projection,
            ),
        ]

        for (
            projection_name,
            projection,
        ) in projections:

            yield (
                (
                    f"{prefix}.attention."
                    f"{projection_name}.weight"
                ),
                projection.weight,
                projection.dweight,
            )

            if (
                projection.bias
                is not None
            ):

                yield (
                    (
                        f"{prefix}.attention."
                        f"{projection_name}.bias"
                    ),
                    projection.bias,
                    projection.dbias,
                )

        # ----------------------------------------------------
        # LayerNorm 2
        # ----------------------------------------------------

        yield (
            f"{prefix}.norm2.gamma",
            block.norm2.gamma,
            block.norm2.dgamma,
        )

        yield (
            f"{prefix}.norm2.beta",
            block.norm2.beta,
            block.norm2.dbeta,
        )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        yield (
            f"{prefix}.mlp.fc_in.weight",
            block.mlp.fc_in.weight,
            block.mlp.fc_in.dweight,
        )

        if (
            block.mlp.fc_in.bias
            is not None
        ):

            yield (
                f"{prefix}.mlp.fc_in.bias",
                block.mlp.fc_in.bias,
                block.mlp.fc_in.dbias,
            )

        yield (
            f"{prefix}.mlp.fc_out.weight",
            block.mlp.fc_out.weight,
            block.mlp.fc_out.dweight,
        )

        if (
            block.mlp.fc_out.bias
            is not None
        ):

            yield (
                f"{prefix}.mlp.fc_out.bias",
                block.mlp.fc_out.bias,
                block.mlp.fc_out.dbias,
            )

    # ========================================================
    # Final LayerNorm
    # ========================================================

    yield (
        "final_norm.gamma",
        model.final_norm.gamma,
        model.final_norm.dgamma,
    )

    yield (
        "final_norm.beta",
        model.final_norm.beta,
        model.final_norm.dbeta,
    )


def parameter_count_from_traversal(
    model,
):
    """
    Count parameter scalars using the traversal.
    """

    return sum(
        parameter.size
        for (
            _,
            parameter,
            _,
        ) in named_parameters(
            model
        )
    )