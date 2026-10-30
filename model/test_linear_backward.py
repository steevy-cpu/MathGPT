import numpy as np

from linear import Linear


INPUT_DIM = 3
OUTPUT_DIM = 2

BATCH_SIZE = 2
SEQUENCE_LENGTH = 2

EPSILON = 1e-3


def scalar_loss(
    layer,
    x,
    grad_output,
):
    """
    Artificial scalar loss:

        L = sum(
            Linear(x)
            * grad_output
        )

    This lets grad_output act as
    the upstream gradient.
    """

    output = layer.forward(
        x
    )

    loss = np.sum(
        output
        * grad_output
    )

    return float(
        loss
    )


def main():

    print()
    print("Linear Backward Test")
    print("====================")

    layer = Linear(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        use_bias=True,
        seed=42,
    )

    rng = np.random.default_rng(
        123
    )

    x = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            INPUT_DIM,
        ),
    ).astype(
        np.float32
    )

    grad_output = rng.normal(
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            OUTPUT_DIM,
        ),
    ).astype(
        np.float32
    )

    # ========================================================
    # Forward + analytical backward
    # ========================================================

    output = layer.forward(
        x
    )

    grad_input = layer.backward(
        grad_output
    )

    analytical_dweight = (
        layer.dweight.copy()
    )

    analytical_dbias = (
        layer.dbias.copy()
    )

    analytical_dinput = (
        grad_input.copy()
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
        f"grad_output shape: "
        f"{grad_output.shape}"
    )

    print(
        f"dInput shape: "
        f"{grad_input.shape}"
    )

    print(
        f"dWeight shape: "
        f"{layer.dweight.shape}"
    )

    print(
        f"dBias shape: "
        f"{layer.dbias.shape}"
    )

    # ========================================================
    # Shape tests
    # ========================================================

    input_shape_passed = (
        analytical_dinput.shape
        == x.shape
    )

    weight_shape_passed = (
        analytical_dweight.shape
        == layer.weight.shape
    )

    bias_shape_passed = (
        analytical_dbias.shape
        == layer.bias.shape
    )

    # ========================================================
    # Numerical gradient check: one weight
    # ========================================================

    weight_index = (
        1,
        0,
    )

    original_weight = float(
        layer.weight[
            weight_index
        ]
    )

    layer.weight[
        weight_index
    ] = (
        original_weight
        + EPSILON
    )

    loss_plus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    layer.weight[
        weight_index
    ] = (
        original_weight
        - EPSILON
    )

    loss_minus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    # Restore original value.
    layer.weight[
        weight_index
    ] = original_weight

    numerical_dweight = (
        loss_plus
        - loss_minus
    ) / (
        2.0
        * EPSILON
    )

    expected_dweight = (
        analytical_dweight[
            weight_index
        ]
    )

    weight_gradient_passed = (
        np.isclose(
            numerical_dweight,
            expected_dweight,
            atol=1e-3,
            rtol=1e-3,
        )
    )

    # ========================================================
    # Numerical gradient check: one bias
    # ========================================================

    bias_index = 1

    original_bias = float(
        layer.bias[
            bias_index
        ]
    )

    layer.bias[
        bias_index
    ] = (
        original_bias
        + EPSILON
    )

    loss_plus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    layer.bias[
        bias_index
    ] = (
        original_bias
        - EPSILON
    )

    loss_minus = scalar_loss(
        layer,
        x,
        grad_output,
    )

    layer.bias[
        bias_index
    ] = original_bias

    numerical_dbias = (
        loss_plus
        - loss_minus
    ) / (
        2.0
        * EPSILON
    )

    expected_dbias = (
        analytical_dbias[
            bias_index
        ]
    )

    bias_gradient_passed = (
        np.isclose(
            numerical_dbias,
            expected_dbias,
            atol=1e-3,
            rtol=1e-3,
        )
    )

    # ========================================================
    # Numerical gradient check: one input
    # ========================================================

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

    input_gradient_passed = (
        np.isclose(
            numerical_dinput,
            expected_dinput,
            atol=1e-3,
            rtol=1e-3,
        )
    )

    # ========================================================
    # Other checks
    # ========================================================

    dtype_passed = (
        analytical_dinput.dtype
        == np.float32
        and
        analytical_dweight.dtype
        == np.float32
        and
        analytical_dbias.dtype
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
                analytical_dweight
            )
        )
        and
        np.all(
            np.isfinite(
                analytical_dbias
            )
        )
    )

    # ========================================================
    # Print gradient comparison
    # ========================================================

    print()
    print("Numerical Gradient Checks")
    print("-------------------------")

    print()

    print(
        f"dW numerical:  "
        f"{numerical_dweight:.6f}"
    )

    print(
        f"dW analytical: "
        f"{expected_dweight:.6f}"
    )

    print()

    print(
        f"db numerical:  "
        f"{numerical_dbias:.6f}"
    )

    print(
        f"db analytical: "
        f"{expected_dbias:.6f}"
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
    print("Final Checks")
    print("------------")

    print(
        f"dInput shape passed: "
        f"{input_shape_passed}"
    )

    print(
        f"dWeight shape passed: "
        f"{weight_shape_passed}"
    )

    print(
        f"dBias shape passed: "
        f"{bias_shape_passed}"
    )

    print(
        f"dWeight gradient passed: "
        f"{weight_gradient_passed}"
    )

    print(
        f"dBias gradient passed: "
        f"{bias_gradient_passed}"
    )

    print(
        f"dInput gradient passed: "
        f"{input_gradient_passed}"
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
        input_shape_passed
        and weight_shape_passed
        and bias_shape_passed
        and weight_gradient_passed
        and bias_gradient_passed
        and input_gradient_passed
        and dtype_passed
        and finite_passed
    ):

        print()

        print(
            "SUCCESS: Linear backward "
            "tests passed."
        )

    else:

        raise AssertionError(
            "Linear backward test failed."
        )


if __name__ == "__main__":
    main()