"""Bounded nucleus sampling and response-only language-model evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .data import END_OF_TURN, PREFIXES, dialogue_examples, load_records
from .losses import masked_token_nll
from .modeling import (
    context_length,
    load_causal_model,
    model_logits,
    pack_responses,
    prompt_token_ids,
    seed_everything,
)


class TurnEndingCriteria(StoppingCriteria):
    """Stop each candidate when its generated text reaches three newlines."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(
            input_ids[:, self.prompt_length :], skip_special_tokens=True
        )
        return torch.tensor(
            [END_OF_TURN in text for text in texts], dtype=torch.bool, device=input_ids.device
        )


def _trim_action(tokens: list[int], tokenizer) -> list[int]:
    eos = tokenizer.eos_token_id
    if eos is not None and eos in tokens:
        tokens = tokens[: tokens.index(eos) + 1]
    # The generated candidate can have padding after an earlier turn stop while
    # another candidate is still running. Retain exactly the terminated action.
    for index in range(1, len(tokens) + 1):
        if END_OF_TURN in tokenizer.decode(tokens[:index], skip_special_tokens=True):
            return tokens[:index]
    return tokens


@torch.no_grad()
def sample_responses(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    num_candidates: int = 1,
    max_new_tokens: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.8,
) -> list[dict]:
    if num_candidates < 1 or max_new_tokens < 1:
        raise ValueError("num_candidates and max_new_tokens must be positive")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    if not prompt_ids:
        raise ValueError("A prompt must contain at least one token")
    device = next(model.parameters()).device
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_candidates,
        do_sample=True,
        top_p=top_p,
        top_k=0,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([TurnEndingCriteria(tokenizer, len(prompt_ids))]),
        use_cache=True,
    )
    result = []
    for sequence in generated:
        action = _trim_action(sequence[len(prompt_ids) :].tolist(), tokenizer)
        text = tokenizer.decode(action, skip_special_tokens=True).split(END_OF_TURN, 1)[0].strip()
        result.append({"text": text, "token_ids": action})
    return result


def generate_response(
    model_path: str | Path,
    history: str | list[dict],
    *,
    speaker: str = "persuader",
    max_length: int = 1024,
    max_new_tokens: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.8,
    num_candidates: int = 1,
    seed: int = 42,
    device=None,
) -> list[str]:
    """Generate from a complete prompt, or chronological speaker/text records.

    String histories are treated as complete prompts. For a list of records,
    the requested next speaker's prefix is appended automatically.
    """
    if speaker not in PREFIXES:
        raise ValueError(f"Unknown speaker: {speaker}")
    seed_everything(seed)
    model, tokenizer = load_causal_model(model_path, device)
    limit = context_length(model, max_length)
    if not 1 <= max_new_tokens < limit:
        raise ValueError("max_new_tokens must leave at least one prompt token")
    if isinstance(history, str):
        prompt = history
    else:
        prompt = "".join(PREFIXES[row["speaker"]] + row["text"] + END_OF_TURN for row in history)
        prompt += PREFIXES[speaker]
    prompt_ids = prompt_token_ids(tokenizer, prompt, limit - max_new_tokens)
    return [
        result["text"]
        for result in sample_responses(
            model, tokenizer, prompt_ids, num_candidates=num_candidates,
            max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature,
        )
    ]


@torch.no_grad()
def evaluate_examples(model, tokenizer, examples, *, batch_size=1, max_length=1024) -> dict:
    if not examples or batch_size < 1:
        raise ValueError("Evaluation requires examples and a positive batch size")
    model.eval()
    max_length = context_length(model, max_length)
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    for start in range(0, len(examples), batch_size):
        batch = pack_responses(tokenizer, examples[start : start + batch_size], max_length, device)
        loss = masked_token_nll(model_logits(model, batch), batch["input_ids"], batch["response_mask"])
        count = int(batch["response_mask"][:, 1:].sum())
        total_loss += float(loss) * count
        total_tokens += count
    nll = total_loss / total_tokens
    return {
        "nll": nll,
        "perplexity": math.exp(nll) if nll < 709 else float("inf"),
        "response_tokens": total_tokens,
        "examples": len(examples),
    }


def evaluate_language_model(
    data_path, model_path, *, speaker="persuader", batch_size=1, max_length=1024, device=None
) -> dict:
    model, tokenizer = load_causal_model(model_path, device)
    examples = dialogue_examples(load_records(data_path), speaker=speaker)
    return evaluate_examples(model, tokenizer, examples, batch_size=batch_size, max_length=max_length)
