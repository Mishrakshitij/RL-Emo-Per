"""Causal language-model and sequence PPO objectives implemented in PyTorch."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _finite_float(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")


def _binary_mask(mask: Tensor, shape: torch.Size, device: torch.device, name: str) -> Tensor:
    if not isinstance(mask, Tensor) or mask.shape != shape or mask.device != device:
        raise ValueError(f"{name} must match the token shape and device")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f"{name} must contain only zero and one")
    return mask.bool()


def _causal_inputs(logits: Tensor, input_ids: Tensor, response_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    _finite_float(logits, "logits")
    if logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[2] < 2:
        raise ValueError("logits must have shape [batch, sequence >= 2, vocabulary >= 2]")
    if not isinstance(input_ids, Tensor) or input_ids.shape != logits.shape[:2]:
        raise ValueError("input_ids must match logits' batch and sequence dimensions")
    if input_ids.device != logits.device or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be an integer tensor on the logits device")
    if not ((input_ids >= 0) & (input_ids < logits.shape[-1])).all():
        raise ValueError("input_ids contain an out-of-vocabulary token")
    mask = _binary_mask(response_mask, input_ids.shape, logits.device, "response_mask")[:, 1:]
    if not mask.any(dim=1).all():
        raise ValueError("every sequence needs at least one response token after position zero")
    # Keep double precision for numerical checks; avoid half-precision reductions.
    shifted_logits = logits[:, :-1]
    if shifted_logits.dtype in (torch.float16, torch.bfloat16):
        shifted_logits = shifted_logits.float()
    return shifted_logits, input_ids[:, 1:].long(), mask


def sequence_log_probs(logits: Tensor, input_ids: Tensor, response_mask: Tensor) -> Tensor:
    """Sum response-token log probabilities for each complete input sequence.

    All inputs include context and response positions. Logits at position ``t``
    predict the ID at ``t + 1``; the mask selects target positions, includes EOS,
    and excludes context and padding. Position zero cannot be predicted and its
    mask value is ignored. Each row must contain a selected target after it.
    """
    shifted_logits, targets, mask = _causal_inputs(logits, input_ids, response_mask)
    token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.masked_fill(~mask, 0.0).sum(dim=-1)


def _smoothing(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("label_smoothing must be finite and in [0, 1)")


def masked_token_nll(
    logits: Tensor, input_ids: Tensor, response_mask: Tensor, *, label_smoothing: float = 0.0
) -> Tensor:
    """Return token-mean causal cross entropy over response targets only.

    Optional smoothing mixes each target distribution with the uniform
    vocabulary distribution. Longer responses contribute more selected tokens;
    this objective does not average the per-sequence averages.
    """
    _smoothing(label_smoothing)
    shifted_logits, targets, mask = _causal_inputs(logits, input_ids, response_mask)
    token_losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        targets.reshape(-1), reduction="none", label_smoothing=label_smoothing,
    ).reshape_as(targets)
    return token_losses.masked_fill(~mask, 0.0).sum() / mask.sum()


def sequence_cross_entropy(
    logits: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute response-only causal cross entropy, validating padding masks."""
    if not isinstance(input_ids, Tensor):
        raise ValueError("input_ids must be a tensor")
    attention = _binary_mask(attention_mask, input_ids.shape, input_ids.device, "attention_mask")
    response = _binary_mask(response_mask, input_ids.shape, input_ids.device, "response_mask")
    if (response & ~attention).any():
        raise ValueError("response_mask must not select padding tokens")
    return masked_token_nll(logits, input_ids, response, label_smoothing=label_smoothing)


def classification_cross_entropy(logits: Tensor, targets: Tensor, *, label_smoothing: float = 0.0) -> Tensor:
    """Return mean cross entropy for integer class IDs and unnormalized logits."""
    _smoothing(label_smoothing)
    _finite_float(logits, "logits")
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError("classifier logits must have shape [batch, classes >= 2]")
    if not isinstance(targets, Tensor) or targets.shape != logits.shape[:1]:
        raise ValueError("targets must contain one class ID per batch item")
    if targets.device != logits.device or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must be integers on the logits device")
    if not ((targets >= 0) & (targets < logits.shape[-1])).all():
        raise ValueError("targets contain an out-of-range class ID")
    stable_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    return F.cross_entropy(stable_logits, targets.long(), label_smoothing=label_smoothing)


def normalize_rewards(rewards: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Detach and standardize a rollout's rewards using population deviation.

    Normalization belongs to the complete rollout, before PPO minibatching.
    A constant or singleton rollout yields zero advantages.
    """
    _finite_float(rewards, "rewards")
    if rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional rollout tensor")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    values = rewards.detach()
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    centered = values - values.mean()
    return centered / values.std(unbiased=False).clamp_min(eps)


def normalize_advantages(rewards: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Alias for reward normalization used by the sequence PPO trainer."""
    return normalize_rewards(rewards, eps=eps)


def clipped_policy_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> Tensor:
    """Negative mean sequence PPO clipped surrogate, for gradient descent.

    Inputs are vectors with one value per sampled response. Old-policy log
    probabilities and advantages are detached inside this function. Clipping
    applies to the probability ratio, not to token log probabilities. An
    overflowing ratio raises an error instead of silently changing the objective.
    """
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be finite and in (0, 1)")
    for value, name in ((new_log_probs, "new_log_probs"), (old_log_probs, "old_log_probs"), (advantages, "advantages")):
        _finite_float(value, name)
        if value.ndim != 1 or value.shape != new_log_probs.shape or value.device != new_log_probs.device:
            raise ValueError("PPO inputs must be matching vectors on the same device")
    new = new_log_probs.float() if new_log_probs.dtype in (torch.float16, torch.bfloat16) else new_log_probs
    old = old_log_probs.detach().to(dtype=new.dtype)
    advantage = advantages.detach().to(dtype=new.dtype)
    ratios = (new - old).exp()
    if not torch.isfinite(ratios).all():
        raise ValueError("PPO probability ratio overflowed; inspect the rollout and learning rate")
    surrogate = torch.minimum(ratios * advantage, ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    if not torch.isfinite(surrogate).all():
        raise ValueError("PPO surrogate overflowed; inspect the reward scale")
    return -surrogate.mean()
