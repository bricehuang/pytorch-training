import torch

from tasks.task01_tensor_workout import creation_workout, tensor_workout


def test_creation_workout_has_expected_constructors() -> None:
    tensors = creation_workout()
    expected = {
        "tensor",
        "zeros",
        "ones",
        "empty",
        "arange",
        "linspace",
        "randn",
        "full",
        "eye",
    }
    assert set(tensors) == expected
    assert all(isinstance(value, torch.Tensor) for value in tensors.values())
    assert tensors["eye"].ndim == 2
    assert tensors["eye"].shape[0] == tensors["eye"].shape[1]


def test_tensor_workout_values_and_shapes() -> None:
    x = torch.tensor(
        [
            [[-1.0, 2.0, 3.0, -4.0], [5.0, -6.0, 7.0, 8.0]],
            [[9.0, 10.0, -11.0, 12.0], [-13.0, 14.0, 15.0, -16.0]],
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        ]
    )
    out = tensor_workout(x)

    assert out["last_token"].shape == (3, 4)
    torch.testing.assert_close(out["last_token"], x[:, -1])
    torch.testing.assert_close(out["even_features"], x[..., ::2])
    torch.testing.assert_close(out["first_two_batches"], x[:2])
    torch.testing.assert_close(out["positive_mask"], x > 0)
    torch.testing.assert_close(out["positive_values"], x[x > 0])
    torch.testing.assert_close(out["sequence_mean"], x.mean(dim=1, keepdim=True))
    torch.testing.assert_close(out["feature_max"], x.amax(dim=-1))
    assert out["normalized"].shape == x.shape
    torch.testing.assert_close(
        out["normalized"].mean(dim=1),
        torch.zeros_like(x[:, 0]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(out["doubled_sequence"], torch.cat((x, x), dim=1))
    torch.testing.assert_close(out["flattened_tokens"], x.reshape(-1, x.shape[-1]))


def test_tensor_workout_preserves_dtype_and_device() -> None:
    x = torch.randn(1, 3, 7, dtype=torch.float64)
    out = tensor_workout(x)
    for key, value in out.items():
        if key == "positive_mask":
            assert value.dtype == torch.bool
        else:
            assert value.device == x.device
