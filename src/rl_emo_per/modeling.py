"""Model loading and response masks shared by training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import END_OF_TURN


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_model(model_name: str | Path, device=None):
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(str(model_name))
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model.to(resolve_device(device)), tokenizer


def context_length(model, requested: int) -> int:
    if requested < 2:
        raise ValueError("max_length must be at least 2")
    for name in ("max_position_embeddings", "n_positions"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and value > 0:
            return min(requested, value)
    return requested


def classifier_context_length(model, tokenizer, requested: int) -> int:
    """Respect tokenizer limits and RoBERTa's reserved position embeddings."""
    limit = context_length(model, requested)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 1 < tokenizer_limit < 10**9:
        limit = min(limit, tokenizer_limit)
    if getattr(model.config, "model_type", None) in {"roberta", "xlm-roberta"}:
        # RoBERTa starts non-padding positions at padding_idx + 1.
        limit = min(limit, model.config.max_position_embeddings - model.config.pad_token_id - 1)
    if limit < 2:
        raise ValueError("The classifier has no usable token context")
    return limit


def prompt_token_ids(tokenizer, prompt: str, max_length: int) -> list[int]:
    if max_length < 1:
        raise ValueError("At least one prompt token must fit in the context")
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens:
        start = tokenizer.bos_token_id
        if start is None:
            start = tokenizer.eos_token_id
        if start is None:
            raise ValueError("An empty prompt requires a BOS or EOS token")
        tokens = [start]
    return tokens[-max_length:]


def response_token_ids(tokenizer, response: str, max_length: int) -> list[int]:
    """Include a turn delimiter and EOS; truncate targets before padding."""
    if max_length < 1:
        raise ValueError("At least one response token must fit in the context")
    if not response.endswith(END_OF_TURN):
        response += END_OF_TURN
    tokens = tokenizer.encode(response, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        tokens = tokens[: max_length - 1] + [eos]
    else:
        tokens = tokens[:max_length]
    if not tokens:
        raise ValueError("The response tokenized to an empty sequence")
    return tokens


def pack_token_sequences(
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    pad_token_id: int,
    device=None,
) -> dict[str, torch.Tensor]:
    """Right-pad explicit actions without changing their sampled token IDs.

    Padding can share the EOS ID: attention and response masks are derived from
    sequence lengths, never from token values. The first response token is
    predicted by the final prompt token and therefore belongs to the loss.
    """
    if not prompts or len(prompts) != len(responses):
        raise ValueError("A batch needs equally many prompts and responses")
    if any(not prompt or not response for prompt, response in zip(prompts, responses)):
        raise ValueError("Every example needs a prompt and a response")
    width = max(len(prompt) + len(response) for prompt, response in zip(prompts, responses))
    input_ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, (prompt, response) in enumerate(zip(prompts, responses)):
        end = len(prompt) + len(response)
        input_ids[index, :end] = torch.tensor(list(prompt) + list(response), dtype=torch.long)
        attention_mask[index, :end] = 1
        response_mask[index, len(prompt) : end] = True
    return {
        key: value.to(resolve_device(device))
        for key, value in {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }.items()
    }


def pack_responses(tokenizer, examples: Sequence[dict], max_length: int, device=None):
    """Keep response tokens and the most recent history within the context."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    prompts, responses = [], []
    for example in examples:
        response = response_token_ids(tokenizer, example["response"], max_length - 1)
        prompt = prompt_token_ids(tokenizer, example["prompt"], max_length - len(response))
        prompts.append(prompt)
        responses.append(response)
    return pack_token_sequences(prompts, responses, tokenizer.pad_token_id, device)


def model_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits
