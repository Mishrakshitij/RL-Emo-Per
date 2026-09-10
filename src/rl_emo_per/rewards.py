"""Text and classifier rewards used by the RL-Emo-Per policy objective."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import torch
from torch import Tensor


DEFAULT_REWARD_WEIGHTS = (0.1, 0.1, 0.55, 0.25)


def _tokens(text: str, nlp: Any = None) -> set[str]:
    if not isinstance(text, str):
        raise ValueError("reward text must be a string")
    if nlp is None:
        return set(re.findall(r"\b\w+\b", text.casefold()))
    return {
        (token.lemma_ or token.text).casefold().strip()
        for token in nlp(text)
        if not token.is_punct and not token.is_space and (token.lemma_ or token.text).strip()
    }


def jaccard_similarity(previous: str, response: str, *, nlp: Any = None) -> float:
    """Compute unigram set overlap; an empty union has similarity zero.

    Supply a spaCy pipeline with a lemmatizer for the paper's normalization.
    Without ``nlp``, normalization uses case-folded word tokens without stemming.
    """
    before, current = _tokens(previous, nlp), _tokens(response, nlp)
    union = before | current
    return len(before & current) / len(union) if union else 0.0


def repetition_reward(previous: str, response: str, *, positive_overlap: bool = False, nlp: Any = None) -> float:
    """Penalize Jaccard overlap, following the released reward implementation.

    ``positive_overlap=True`` instead uses the manuscript's displayed positive
    similarity equation. An absent/empty previous response gives zero reward.
    """
    similarity = jaccard_similarity(previous, response, nlp=nlp)
    return similarity if positive_overlap else -similarity


def _meteor_backend():
    try:
        from nltk.corpus import wordnet
        from nltk.translate.meteor_score import single_meteor_score
    except ImportError as error:
        raise RuntimeError("METEOR requires NLTK; install it with `python -m pip install nltk`.") from error
    try:
        wordnet.ensure_loaded()
    except LookupError as error:
        raise RuntimeError(
            "METEOR requires the NLTK WordNet corpus; run `python -m nltk.downloader wordnet`."
        ) from error
    return single_meteor_score, wordnet


def meteor_consistency(reference: str, candidate: str) -> float:
    """Return NLTK METEOR with exact, stem, and WordNet synonym matching.

    Inputs use whitespace tokenization, as in the released implementation.
    Dependencies and the WordNet corpus must be installed explicitly; this
    function never downloads resources or substitutes another metric.
    """
    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise ValueError("METEOR reference and candidate must be strings")
    scorer, wordnet = _meteor_backend()
    try:
        return float(scorer(reference.split(), candidate.split(), wordnet=wordnet))
    except LookupError as error:
        raise RuntimeError(
            "METEOR could not load its NLTK resources; run `python -m nltk.downloader wordnet`."
        ) from error


def classifier_reward(probabilities: Tensor, target_labels: Tensor, *, beta: float = 2.0) -> Tensor:
    """Compute ``p(target) - beta * (1 - p(target))`` for each candidate.

    ``probabilities`` has shape ``[batch, classes]`` and must already be a
    normalized distribution. ``target_labels`` supplies one gold class per row.
    The function preserves gradients; rollout collection should use no-grad.
    """
    if not math.isfinite(beta) or beta < 1:
        raise ValueError("beta must be finite and at least one")
    if not isinstance(probabilities, Tensor) or not probabilities.is_floating_point():
        raise ValueError("probabilities must be a floating-point tensor")
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, classes >= 2]")
    if not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("probabilities must be finite and in [0, 1]")
    stable_probs = probabilities.float() if probabilities.dtype in (torch.float16, torch.bfloat16) else probabilities
    tolerance = 2e-3 if probabilities.dtype in (torch.float16, torch.bfloat16) else 1e-5
    if not torch.allclose(stable_probs.sum(-1), torch.ones_like(stable_probs[:, 0]), atol=tolerance, rtol=tolerance):
        raise ValueError("each classifier probability row must sum to one")
    if not isinstance(target_labels, Tensor) or target_labels.shape != probabilities.shape[:1]:
        raise ValueError("target_labels must supply one class ID per probability row")
    if target_labels.device != probabilities.device or target_labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_labels must be integers on the probabilities device")
    if not ((target_labels >= 0) & (target_labels < probabilities.shape[-1])).all():
        raise ValueError("target_labels contain an out-of-range class ID")
    selected = stable_probs.gather(-1, target_labels.long().unsqueeze(-1)).squeeze(-1)
    return selected - beta * (1 - selected)


def weighted_reward(
    repetition: Tensor,
    consistency: Tensor,
    emotion: Tensor,
    persuasion: Tensor,
    *,
    weights: Sequence[float] = DEFAULT_REWARD_WEIGHTS,
) -> Tensor:
    """Sum the four rewards with weights ordered as R1, R2, R3, and R4.

    Each component must be a finite vector with one value per candidate, on
    the same device. Weights are finite nonnegative coefficients; they need
    not sum to one, so individual reward ablations can set weights to zero.
    """
    if len(weights) != 4 or not all(math.isfinite(weight) and weight >= 0 for weight in weights):
        raise ValueError("weights must contain four finite nonnegative coefficients")
    components = (repetition, consistency, emotion, persuasion)
    if not isinstance(repetition, Tensor):
        raise ValueError("reward components must be tensors")
    for component in components:
        if not isinstance(component, Tensor) or not component.is_floating_point():
            raise ValueError("reward components must be floating-point tensors")
        if component.ndim != 1 or component.numel() == 0 or component.shape != repetition.shape:
            raise ValueError("reward components must be matching nonempty vectors")
        if component.device != repetition.device or not torch.isfinite(component).all():
            raise ValueError("reward components must be finite and on the same device")
    result = sum(weight * component for weight, component in zip(weights, components))
    if not torch.isfinite(result).all():
        raise ValueError("weighted reward overflowed")
    return result
