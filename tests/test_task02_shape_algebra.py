import torch
import torch.nn.functional as F

from tasks.task02_shape_algebra import merge_heads, pairwise_metrics, split_heads


def test_pairwise_metrics_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    y = torch.randn(2, 4, 5, dtype=torch.float64)

    dot, cosine, distance = pairwise_metrics(x, y)

    expected_dot = x @ y.transpose(-1, -2)
    expected_cosine = F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).transpose(-1, -2)
    expected_distance = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(dim=-1)

    torch.testing.assert_close(dot, expected_dot)
    torch.testing.assert_close(cosine, expected_cosine)
    torch.testing.assert_close(distance, expected_distance)
    assert dot.shape == cosine.shape == distance.shape == (2, 3, 4)


def test_split_and_merge_heads_round_trip() -> None:
    x = torch.randn(2, 5, 12)
    split = split_heads(x, num_heads=3)
    assert split.shape == (2, 3, 5, 4)
    torch.testing.assert_close(merge_heads(split), x)


def test_split_heads_accepts_noncontiguous_input() -> None:
    x = torch.randn(2, 12, 5).transpose(1, 2)
    assert not x.is_contiguous()
    reconstructed = merge_heads(split_heads(x, num_heads=3))
    torch.testing.assert_close(reconstructed, x)
