"""H61 tests for the coordinate-focused supervision.

These tests prove the supervision actually changes the gradient density: a
coordinate token must carry more weight than a structural token, and the
weighted loss must reduce to the plain token mean when every weight is one.
"""
import unittest

from h61.supervision import (
    DIGIT_WEIGHT,
    completion_weights,
    coordinate_token_flags,
    weighted_lm_loss,
)

try:  # The CPU qualification environment has no torch; the remote session does.
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


def decode_map(mapping):
    def decode(ids):
        return mapping[ids[0]]

    return decode


class SupervisionTests(unittest.TestCase):
    def test_digit_tokens_are_marked_and_structural_tokens_are_not(self):
        mapping = {1: "{\"type\"", 2: "\":[720", 3: ",176", 4: "]}"}
        flags = coordinate_token_flags([1, 2, 3, 4], decode_map(mapping))
        self.assertEqual(flags, [False, True, True, False])

    def test_completion_weights_up_weight_only_the_coordinate_tokens(self):
        mapping = {1: "{\"type\"", 2: "\":[720", 3: ",176", 4: "]}"}
        weights = completion_weights([1, 2, 3, 4], decode_map(mapping), 8.0)
        self.assertEqual(weights, [1.0, 8.0, 8.0, 1.0])
        self.assertEqual(weights[1], DIGIT_WEIGHT)

    def test_a_completion_without_a_coordinate_token_is_refused(self):
        mapping = {1: "{\"type\"", 2: "\":\"drag\"}"}
        with self.assertRaises(ValueError):
            completion_weights([1, 2], decode_map(mapping), 8.0)

    def test_bounded_weight_is_enforced(self):
        mapping = {1: "1"}
        with self.assertRaises(ValueError):
            completion_weights([1], decode_map(mapping), 0.5)
        with self.assertRaises(ValueError):
            completion_weights([1], decode_map(mapping), 65.0)


@unittest.skipUnless(torch is not None, "torch is required for the weighted loss proof")
class WeightedLossTests(unittest.TestCase):
    def test_weighted_loss_equals_the_plain_mean_when_every_weight_is_one(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 5, 11)
        labels = torch.tensor([[-100, 3, 4, 5, 6]])
        weights = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
        weighted = weighted_lm_loss(logits, labels, weights)
        plain = torch.nn.functional.cross_entropy(
            logits[0, :-1, :].float(), labels[0, 1:]
        )
        self.assertTrue(torch.allclose(weighted, plain, atol=1e-5))

    def test_a_heavier_coordinate_weight_moves_the_loss_toward_the_coordinate(self):
        torch.manual_seed(11)
        logits = torch.randn(1, 5, 11)
        labels = torch.tensor([[-100, 3, 4, 5, 6]])
        structural = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
        heavy = torch.tensor([[0.0, 8.0, 8.0, 1.0, 1.0]])
        first = weighted_lm_loss(logits, labels, structural)
        second = weighted_lm_loss(logits, labels, heavy)
        self.assertFalse(torch.allclose(first, second, atol=1e-6))

    def test_unsupervised_positions_must_carry_zero_weight(self):
        logits = torch.randn(1, 4, 7)
        # The second position is unsupervised and it is inside the loss
        # window, so a weight on it would silently supervise the prompt.
        labels = torch.tensor([[2, -100, 3, 4]])
        with self.assertRaises(ValueError):
            weighted_lm_loss(logits, labels, torch.ones(1, 4))

    def test_a_supervised_position_must_carry_a_positive_weight(self):
        logits = torch.randn(1, 4, 7)
        labels = torch.tensor([[-100, 2, 3, 4]])
        with self.assertRaises(ValueError):
            weighted_lm_loss(logits, labels, torch.zeros(1, 4))

    def test_a_span_without_any_supervised_token_is_refused(self):
        logits = torch.randn(1, 4, 7)
        labels = torch.tensor([[-100, -100, -100, -100]])
        weights = torch.zeros(1, 4)
        with self.assertRaises(ValueError):
            weighted_lm_loss(logits, labels, weights)


if __name__ == "__main__":
    unittest.main()
