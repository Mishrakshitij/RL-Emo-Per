"""Reward equation checks, including METEOR resource error behavior."""

import builtins
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rl_emo_per.rewards import (
    classifier_reward,
    jaccard_similarity,
    meteor_consistency,
    repetition_reward,
    weighted_reward,
)


class RepetitionTests(unittest.TestCase):
    def test_set_overlap_ignores_case_punctuation_and_token_frequency(self):
        self.assertAlmostEqual(jaccard_similarity("HELP help, today!", "Help me today."), 2 / 3)
        self.assertAlmostEqual(repetition_reward("help today", "help me today"), -2 / 3)
        self.assertAlmostEqual(repetition_reward("help today", "help me today", positive_overlap=True), 2 / 3)

    def test_disjoint_and_empty_utterances(self):
        self.assertEqual(repetition_reward("", ""), 0)
        self.assertEqual(repetition_reward("", "hello"), 0)
        self.assertEqual(repetition_reward("hello", "goodbye"), 0)

    def test_supplied_spacy_lemmas_are_used(self):
        def nlp(text):
            lemma = {"running": "run", "ran": "run"}
            return [SimpleNamespace(text=token, lemma_=lemma.get(token, token), is_punct=token == "!", is_space=False)
                    for token in text.split()]
        self.assertEqual(jaccard_similarity("running !", "ran", nlp=nlp), 1)
        self.assertEqual(jaccard_similarity("running", "ran"), 0)


class ClassifierRewardTests(unittest.TestCase):
    def test_target_probability_penalty_and_limits(self):
        probabilities = torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.0, 1.0]], dtype=torch.float64)
        reward = classifier_reward(probabilities, torch.tensor([0, 1, 0]))
        torch.testing.assert_close(reward, torch.tensor([0.4, -1.4, -2.0], dtype=torch.float64))
        torch.testing.assert_close(
            classifier_reward(probabilities[-1:], torch.tensor([1])), torch.tensor([1.0], dtype=torch.float64),
        )

    def test_reward_gradient_through_softmax(self):
        logits = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(-1)
        classifier_reward(probabilities, torch.tensor([0])).sum().backward()
        positive = 3 * probabilities[0, 0] * probabilities[0, 1]
        torch.testing.assert_close(logits.grad, torch.stack((positive, -positive)).reshape(1, 2))

    def test_invalid_probabilities_and_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            classifier_reward(torch.tensor([[0.2, 0.3]]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            classifier_reward(torch.tensor([[0.2, 0.8]]), torch.tensor([2]))
        with self.assertRaisesRegex(ValueError, "at least one"):
            classifier_reward(torch.tensor([[0.2, 0.8]]), torch.tensor([0]), beta=0.5)

    def test_default_weighted_reward_and_gradient(self):
        components = [torch.tensor([value], dtype=torch.float64, requires_grad=True) for value in (-0.5, 0.8, 0.4, -1.4)]
        total = weighted_reward(*components)
        self.assertAlmostEqual(total.item(), -0.1)
        total.sum().backward()
        for component, expected in zip(components, (0.1, 0.1, 0.55, 0.25)):
            self.assertAlmostEqual(component.grad.item(), expected)

    def test_component_shapes_cannot_broadcast(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            weighted_reward(torch.zeros(2), torch.zeros(1), torch.zeros(2), torch.zeros(2))


class MeteorTests(unittest.TestCase):
    def test_missing_nltk_has_actionable_error(self):
        original_import = builtins.__import__

        def without_nltk(name, *args, **kwargs):
            if name == "nltk" or name.startswith("nltk."):
                raise ImportError("absent in fixture")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_nltk):
            with self.assertRaisesRegex(RuntimeError, "pip install nltk"):
                meteor_consistency("a dog runs", "a dog runs")

    def _nltk_scorer(self):
        try:
            from nltk.translate.meteor_score import single_meteor_score
        except ImportError:
            self.skipTest("NLTK is not installed")
        return single_meteor_score

    def test_missing_wordnet_has_actionable_error(self):
        self._nltk_scorer()
        missing = SimpleNamespace(ensure_loaded=lambda: (_ for _ in ()).throw(LookupError("missing")))
        with patch("nltk.corpus.wordnet", missing):
            with self.assertRaisesRegex(RuntimeError, "nltk.downloader wordnet"):
                meteor_consistency("a dog runs", "a dog runs")

    def test_actual_meteor_exact_and_synonym_scores_with_fixed_corpus(self):
        scorer = self._nltk_scorer()
        # A tiny corpus fixture makes synonym matching deterministic and avoids
        # requiring a network download during the numerical metric test.
        def synsets(word):
            if word != "hound":
                return []
            return [SimpleNamespace(lemmas=lambda: [SimpleNamespace(name=lambda: "dog")])]

        wordnet = SimpleNamespace(synsets=synsets)
        with patch("rl_emo_per.rewards._meteor_backend", return_value=(scorer, wordnet)):
            # Three aligned tokens, one chunk: 1 - 0.5 * (1 / 3)^3.
            self.assertAlmostEqual(meteor_consistency("a dog runs", "a dog runs"), 53 / 54)
            self.assertAlmostEqual(meteor_consistency("a dog runs", "a hound runs"), 53 / 54)
            self.assertEqual(meteor_consistency("a dog runs", "unrelated"), 0)
            self.assertEqual(meteor_consistency("", ""), 0)


if __name__ == "__main__":
    unittest.main()
