"""Numerical and gradient checks for causal masking and sequence PPO."""

import math
import unittest

import torch

from rl_emo_per.losses import (
    classification_cross_entropy,
    clipped_policy_loss,
    masked_token_nll,
    normalize_rewards,
    sequence_cross_entropy,
    sequence_log_probs,
)


class CausalLossTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = torch.tensor(
            [[[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.5, 0.5]],
             [[0.4, 0.6], [0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]], dtype=torch.float64,
        )
        self.logits = self.probabilities.log().requires_grad_()
        self.ids = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        self.mask = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 0]])

    def test_sequence_probability_is_product_of_shifted_response_tokens(self):
        actual = sequence_log_probs(self.logits, self.ids, self.mask)
        expected = torch.tensor([math.log(0.3 * 0.4), math.log(0.4)], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_token_mean_weights_response_lengths_and_masks_gradients(self):
        loss = masked_token_nll(self.logits, self.ids, self.mask)
        self.assertAlmostEqual(loss.item(), -math.log(0.3 * 0.4 * 0.4) / 3)
        loss.backward()
        expected = torch.zeros_like(self.logits)
        expected[0, 1] = torch.tensor([-0.7, 0.7]) / 3
        expected[0, 2] = torch.tensor([0.6, -0.6]) / 3
        expected[1, 0] = torch.tensor([-0.6, 0.6]) / 3
        torch.testing.assert_close(self.logits.grad, expected, atol=1e-8, rtol=1e-7)

    def test_label_smoothing_mixes_with_uniform_distribution(self):
        smoothing = 0.2
        selected = [(0.3, 0.7), (0.4, 0.6), (0.4, 0.6)]
        expected = sum(-0.9 * math.log(target) - 0.1 * math.log(other) for target, other in selected) / 3
        actual = masked_token_nll(self.logits, self.ids, self.mask, label_smoothing=smoothing)
        self.assertAlmostEqual(actual.item(), expected)

    def test_sequence_probability_gradient_matches_finite_difference(self):
        self.assertTrue(torch.autograd.gradcheck(
            lambda logits: sequence_log_probs(logits, self.ids, self.mask), (self.logits,),
        ))

    def test_context_logits_do_not_change_loss(self):
        alternate = self.logits.detach().clone()
        alternate[0, 0] = torch.tensor([1000.0, -1000.0])
        alternate[1, 1:] = torch.tensor([-1000.0, 1000.0])
        torch.testing.assert_close(
            masked_token_nll(alternate, self.ids, self.mask),
            masked_token_nll(self.logits, self.ids, self.mask),
        )

    def test_padding_cannot_be_selected_as_response(self):
        attention = torch.ones_like(self.mask)
        attention[0, 3] = 0
        with self.assertRaisesRegex(ValueError, "padding"):
            sequence_cross_entropy(self.logits, self.ids, attention, self.mask)

    def test_invalid_mask_and_empty_response_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero and one"):
            sequence_log_probs(self.logits, self.ids, self.mask.float() * 0.5)
        mask = self.mask.clone()
        mask[1] = 0
        with self.assertRaisesRegex(ValueError, "at least one"):
            sequence_log_probs(self.logits, self.ids, mask)

    def test_invalid_tokens_and_nonfinite_logits_are_rejected(self):
        invalid = self.ids.clone()
        invalid[0, 2] = 9
        with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
            masked_token_nll(self.logits, invalid, self.mask)
        invalid_logits = self.logits.detach().clone()
        invalid_logits[0, 1, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            masked_token_nll(invalid_logits, self.ids, self.mask)

    def test_classifier_loss_value_and_gradient(self):
        logits = torch.tensor([[math.log(0.2), math.log(0.8)]], dtype=torch.float64, requires_grad=True)
        loss = classification_cross_entropy(logits, torch.tensor([1]))
        self.assertAlmostEqual(loss.item(), -math.log(0.8))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[0.2, -0.2]], dtype=torch.float64))


class PolicyLossTests(unittest.TestCase):
    def test_clipping_handles_both_advantage_signs_and_detaches_rollout(self):
        ratios = torch.tensor([1.5, 0.5, 1.5, 0.5], dtype=torch.float64)
        new = (ratios.log() - 4).requires_grad_()
        old = torch.full((4,), -4.0, dtype=torch.float64, requires_grad=True)
        advantages = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float64, requires_grad=True)
        loss = clipped_policy_loss(new, old, advantages, clip_epsilon=0.2)
        # Positive: min(3,2.4), min(1,1.6). Negative: min(-3,-2.4), min(-1,-1.6).
        self.assertAlmostEqual(loss.item(), -(2.4 + 1.0 - 3.0 - 1.6) / 4)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([0, -0.25, 0.75, 0], dtype=torch.float64))
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)

    def test_ppo_gradient_matches_finite_difference_away_from_clip_boundaries(self):
        new = torch.tensor([-2.1, -1.9, -2.5, -1.5], dtype=torch.float64, requires_grad=True)
        old = torch.full((4,), -2.0, dtype=torch.float64)
        advantages = torch.tensor([0.7, -0.2, -0.8, 0.3], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda logp: clipped_policy_loss(logp, old, advantages), (new,)))

    def test_normalization_is_population_based_and_detached(self):
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        actual = normalize_rewards(rewards)
        expected = torch.tensor([-math.sqrt(1.5), 0, math.sqrt(1.5)], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        self.assertAlmostEqual(actual.std(unbiased=False).item(), 1)

    def test_constant_and_singleton_rewards_have_zero_advantage(self):
        for rewards in (torch.tensor([4.0]), torch.full((4,), -2.0)):
            torch.testing.assert_close(normalize_rewards(rewards), torch.zeros_like(rewards))

    def test_invalid_rollout_values_do_not_silently_broadcast_or_overflow(self):
        with self.assertRaisesRegex(ValueError, "matching vectors"):
            clipped_policy_loss(torch.zeros(2), torch.zeros(1), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "overflowed"):
            clipped_policy_loss(torch.tensor([1000.0]), torch.tensor([0.0]), torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_rewards(torch.tensor([float("nan")]))
        with self.assertRaisesRegex(ValueError, "clip_epsilon"):
            clipped_policy_loss(torch.zeros(2), torch.zeros(2), torch.ones(2), clip_epsilon=-0.1)


if __name__ == "__main__":
    unittest.main()
