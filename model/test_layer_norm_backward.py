import numpy as np

from layer_norm import LayerNorm


BATCH_SIZE = 2
SEQUENCE_LENGTH = 2
D_MODEL = 4

EPSILON = 1e-3


def scalar_loss(
    layer,
    x,
    grad_output,
):
    """
    Artificial scalar loss:

        L = sum(
            LayerNorm(x)
            * grad_output
        )
    """

    output = layer.forward(
        x
    )

    return float(
        np.sum(
            output
            * grad_output
        )
    )


def main():

    print()
    print(
        "LayerNorm Backward Test"
    )

    print(
        "======================="
    )

    rng = np.random.default_rng(
        123
    )

    layer = LayerNorm(
        d_model=D_MODEL,
        epsilon=1e-5,
    )

    # Non-trivial learned parameters
    layer.gamma[:] = np.array(
        [
            0.8,
            1.1,
            0.9,
            1.3,
        ],
        dtype=np.float32,
    )

    layer.beta[:] = np.array(
        [
            0.1,
            -0.2,
            0.05,
            0.3,
        ],
        dtype=np.float32,
    )

    x = rng.normal(
        loc=1.0,
        scale=2.0,
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            D_MODEL,
        ),
    ).astype(
        np.float32
    )

    grad_output = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            D_MODEL,
        ),
    ).astype(
        np.float32
    )

    # Analytical gradients
    output = layer.forward(
        x
    )

    grad_input = layer.backward(
        grad_output
    )

    analytical_dinput = (
        grad_input.copy()
    )

    analytical_dgamma = (
        layer.dgamma.copy()
    )

    analytical_dbeta = (
        layer.dbeta.copy()
    )

    print()
    print(
        f"Input shape: "
        f"{x.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"dInput shape: "
        f"{analytical_dinput.shape}"
    )

    print(
        f"dGamma shape: "
        f"{analytical_dgamma.shape}"
    )

    print(
        f"dBeta shape: "
        f"{analytical_dbeta.shape}"
    )

    # Numerical dX
    input_index = (
        0,
        1,
        2,
    )

    original_input = float(
        x[
            input_index
        ]
    )

    x[
        input_index
    ] = (
        original_input
        + EPSILON
    )

    loss_plus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    x[
        input_index
    ] = (
        original_input
        - EPSILON
    )

    loss_minus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    x[
        input_index
    ] = original_input

    numerical_dinput = (
        loss_plus
        - loss_minus
    ) / (
        2.0
        * EPSILON
    )

    expected_dinput = (
        analytical_dinput[
            input_index
        ]
    )

    # Numerical dGamma
    gamma_index = 1

    original_gamma = float(
        layer.gamma[
            gamma_index
        ]
    )

    layer.gamma[
        gamma_index
    ] = (
        original_gamma
        + EPSILON
    )

    loss_plus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    layer.gamma[
        gamma_index
    ] = (
        original_gamma
        - EPSILON
    )

    loss_minus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    layer.gamma[
        gamma_index
    ] = original_gamma

    numerical_dgamma = (
        loss_plus
        - loss_minus
    ) / (
        2.0
        * EPSILON
    )

    expected_dgamma = (
        analytical_dgamma[
            gamma_index
        ]
    )

    # Numerical dBeta
    beta_index = 3

    original_beta = float(
        layer.beta[
            beta_index
        ]
    )

    layer.beta[
        beta_index
    ] = (
        original_beta
        + EPSILON
    )

    loss_plus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    layer.beta[
        beta_index
    ] = (
        original_beta
        - EPSILON
    )

    loss_minus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    layer.beta[
        beta_index
    ] = original_beta

    numerical_dbeta = (
        loss_plus
        - loss_minus
    ) / (
        2.0
        * EPSILON
    )

    expected_dbeta = (
        analytical_dbeta[
            beta_index
        ]
    )

    # Comparisons
    dinput_passed = np.isclose(
        numerical_dinput,
        expected_dinput,
        atol=2e-3,
        rtol=2e-3,
    )

    dgamma_passed = np.isclose(
        numerical_dgamma,
        expected_dgamma,
        atol=2e-3,
        rtol=2e-3,
    )

    dbeta_passed = np.isclose(
        numerical_dbeta,
        expected_dbeta,
        atol=2e-3,
        rtol=2e-3,
    )

    shape_passed = (
        analytical_dinput.shape
        == x.shape
        and
        analytical_dgamma.shape
        == layer.gamma.shape
        and
        analytical_dbeta.shape
        == layer.beta.shape
    )

    dtype_passed = (
        analytical_dinput.dtype
        == np.float32
        and
        analytical_dgamma.dtype
        == np.float32
        and
        analytical_dbeta.dtype
        == np.float32
    )

    finite_passed = (
        np.all(
            np.isfinite(
                analytical_dinput
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dgamma
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dbeta
            )
        )
    )

    # Display
    print()
    print(
        "Numerical Gradient Checks"
    )

    print(
        "-------------------------"
    )

    print()

    print(
        f"dX numerical:  "
        f"{numerical_dinput:.6f}"
    )

    print(
        f"dX analytical: "
        f"{expected_dinput:.6f}"
    )

    print()

    print(
        f"dGamma numerical:  "
        f"{numerical_dgamma:.6f}"
    )

    print(
        f"dGamma analytical: "
        f"{expected_dgamma:.6f}"
    )

    print()

    print(
        f"dBeta numerical:  "
        f"{numerical_dbeta:.6f}"
    )

    print(
        f"dBeta analytical: "
        f"{expected_dbeta:.6f}"
    )

    print()
    print(
        "Final Checks"
    )

    print(
        "------------"
    )

    print(
        f"dInput gradient passed: "
        f"{dinput_passed}"
    )

    print(
        f"dGamma gradient passed: "
        f"{dgamma_passed}"
    )

    print(
        f"dBeta gradient passed: "
        f"{dbeta_passed}"
    )

    print(
        f"Shapes passed: "
        f"{shape_passed}"
    )

    print(
        f"Float32 passed: "
        f"{dtype_passed}"
    )

    print(
        f"Finite gradients passed: "
        f"{finite_passed}"
    )

    if (
        dinput_passed
        and dgamma_passed
        and dbeta_passed
        and shape_passed
        and dtype_passed
        and finite_passed
    ):

        print()

        print(
            "SUCCESS: LayerNorm backward "
            "tests passed."
        )

    else:

        raise AssertionError(
            "LayerNorm backward test failed."
        )


if __name__ == "__main__":
    main()