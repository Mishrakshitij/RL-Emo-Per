"""Classifier, speaker language-model, and sequence PPO training in PyTorch.

PPO uses the paper's sequence likelihoods, normalized scalar rewards and clipped
surrogate. A frozen copy of the current policy collects each batch; all old
likelihoods are recorded before any update. Classifier parameters stay frozen.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from .data import dialogue_examples, load_records
from .generation import evaluate_examples, sample_responses
from .labels import labels_for_task
from .losses import (
    classification_cross_entropy,
    clipped_policy_loss,
    masked_token_nll,
    normalize_rewards,
    sequence_log_probs,
)
from .modeling import (
    classifier_context_length,
    context_length,
    load_causal_model,
    load_tokenizer,
    model_logits,
    pack_responses,
    pack_token_sequences,
    prompt_token_ids,
    resolve_device,
    response_token_ids,
    seed_everything,
)
from .rewards import classifier_reward, meteor_consistency, repetition_reward, weighted_reward


def _check_training_args(epochs, batch_size, learning_rate, max_steps):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when supplied")


def _load_splits(train_path, validation_path):
    train = load_records(train_path)
    validation = load_records(validation_path)
    overlap = {row["dialogue_id"] for row in train} & {row["dialogue_id"] for row in validation}
    if overlap:
        raise ValueError("Training and validation contain the same dialogue IDs")
    if any(row["split"] != "train" for row in train) or any(row["split"] != "validation" for row in validation):
        raise ValueError("Training and validation files must contain their respective split rows")
    return train, validation


def _batches(examples, batch_size, rng=None):
    indices = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def _optimizer(model, learning_rate):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = no_decay if parameter.ndim < 2 or "bias" in name else decay
            group.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
        eps=1e-6,
    )


def _save_checkpoint(model, tokenizer, output_dir, configuration, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _classification_examples(records, task):
    labels_for_task(task)  # Fail early for an unknown task.
    examples = []
    for row in records:
        field = "emotion_id" if task == "emotion" else "strategy_id"
        target = row[field]
        if target is not None:
            examples.append({"text": row["text"], "target": int(target != 0) if task == "binary" else target})
    if not examples:
        raise ValueError(f"No labeled examples for task {task}")
    return examples


def _classification_batch(tokenizer, examples, max_length, device):
    inputs = tokenizer(
        [row["text"] for row in examples], padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)
    targets = torch.tensor([row["target"] for row in examples], dtype=torch.long, device=device)
    return inputs, targets


@torch.no_grad()
def _evaluate_classifier(model, tokenizer, examples, *, batch_size, max_length, device):
    model.eval()
    confusion = torch.zeros((model.config.num_labels, model.config.num_labels), dtype=torch.long)
    total_loss = 0.0
    for rows in _batches(examples, batch_size):
        inputs, targets = _classification_batch(tokenizer, rows, max_length, device)
        logits = model(**inputs).logits
        total_loss += float(classification_cross_entropy(logits, targets)) * len(rows)
        predictions = logits.argmax(-1)
        for target, prediction in zip(targets.cpu(), predictions.cpu()):
            confusion[target, prediction] += 1
    correct = confusion.diag().float()
    denominators = confusion.sum(0) + confusion.sum(1)
    # Score the complete class vocabulary, including classes absent in this
    # split. This keeps macro F1 comparable between evaluation runs.
    f1 = torch.where(denominators > 0, 2 * correct / denominators.clamp_min(1), 0)
    return {
        "loss": total_loss / len(examples),
        "accuracy": float(correct.sum() / len(examples)),
        "macro_f1": float(f1.mean()),
        "examples": len(examples),
    }


def train_classifier(
    train_path,
    validation_path,
    output_dir,
    *,
    task="emotion",
    model_name="roberta-large",
    epochs=10,
    batch_size=32,
    learning_rate=2e-5,
    max_length=256,
    seed=42,
    device=None,
    max_steps=None,
    label_smoothing=0.0,
):
    """Train a complete named classifier head and select by validation loss."""
    _check_training_args(epochs, batch_size, learning_rate, max_steps)
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    seed_everything(seed)
    device = resolve_device(device)
    train, validation = _load_splits(train_path, validation_path)
    train = _classification_examples(train, task)
    validation = _classification_examples(validation, task)
    labels = labels_for_task(task)
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_name), num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={label: index for index, label in enumerate(labels)},
        ignore_mismatched_sizes=True,
    ).to(device)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    max_length = classifier_context_length(model, tokenizer, max_length)
    model.config.problem_type = "single_label_classification"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.rl_emo_per_task = task
    optimizer = _optimizer(model, learning_rate)
    configuration = {
        "stage": "classifier", "task": task, "model_name": str(model_name), "epochs": epochs,
        "batch_size": batch_size, "learning_rate": learning_rate, "max_length": max_length,
        "seed": seed, "max_steps": max_steps, "label_smoothing": label_smoothing,
    }
    rng, steps, history, best_loss = random.Random(seed), 0, [], math.inf
    best_metrics = None
    for epoch in range(epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for rows in _batches(train, batch_size, rng):
            inputs, targets = _classification_batch(tokenizer, rows, max_length, device)
            optimizer.zero_grad(set_to_none=True)
            loss = classification_cross_entropy(model(**inputs).logits, targets, label_smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(rows)
            count += len(rows)
            if max_steps is not None and steps >= max_steps:
                break
        scores = _evaluate_classifier(
            model, tokenizer, validation, batch_size=batch_size, max_length=max_length, device=device
        )
        metrics = {"epoch": epoch + 1, "steps": steps, "training_loss": loss_sum / count, "validation": scores}
        history.append(metrics)
        if scores["loss"] < best_loss:
            best_loss, best_metrics = scores["loss"], metrics
            _save_checkpoint(model, tokenizer, output_dir, configuration, metrics)
        if max_steps is not None and steps >= max_steps:
            break
    result = {"steps": steps, "best": best_metrics, "history": history}
    (Path(output_dir) / "training_history.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def train_mle(
    train_path,
    validation_path,
    output_dir,
    *,
    speaker="persuader",
    model_name="gpt2-medium",
    epochs=1,
    batch_size=1,
    learning_rate=3e-5,
    max_length=1024,
    warmup_steps=100,
    seed=42,
    device=None,
    max_steps=None,
    label_smoothing=0.02,
):
    """Fit one speaker's conditional LM; run once for each ARDM speaker."""
    _check_training_args(epochs, batch_size, learning_rate, max_steps)
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    seed_everything(seed)
    train, validation = _load_splits(train_path, validation_path)
    train = dialogue_examples(train, speaker=speaker)
    validation = dialogue_examples(validation, speaker=speaker)
    if not train or not validation:
        raise ValueError("Both splits must contain examples for the requested speaker")
    model, tokenizer = load_causal_model(model_name, device)
    device = next(model.parameters()).device
    max_length = context_length(model, max_length)
    model.config.rl_emo_per_speaker = speaker
    optimizer = _optimizer(model, learning_rate)
    total_steps = epochs * math.ceil(len(train) / batch_size)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    def learning_rate_factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    configuration = {
        "stage": "mle", "speaker": speaker, "model_name": str(model_name), "epochs": epochs,
        "batch_size": batch_size, "learning_rate": learning_rate, "max_length": max_length,
        "warmup_steps": warmup_steps, "seed": seed, "max_steps": max_steps,
        "label_smoothing": label_smoothing,
    }
    rng, steps, history, best_loss = random.Random(seed), 0, [], math.inf
    best_metrics = None
    for epoch in range(epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        for rows in _batches(train, batch_size, rng):
            batch = pack_responses(tokenizer, rows, max_length, device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_token_nll(
                model_logits(model, batch), batch["input_ids"], batch["response_mask"],
                label_smoothing=label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            steps += 1
            count = int(batch["response_mask"][:, 1:].sum())
            loss_sum += float(loss.detach()) * count
            token_count += count
            if max_steps is not None and steps >= max_steps:
                break
        scores = evaluate_examples(model, tokenizer, validation, batch_size=batch_size, max_length=max_length)
        metrics = {"epoch": epoch + 1, "steps": steps, "training_loss": loss_sum / token_count, "validation": scores}
        history.append(metrics)
        if scores["nll"] < best_loss:
            best_loss, best_metrics = scores["nll"], metrics
            _save_checkpoint(model, tokenizer, output_dir, configuration, metrics)
        if max_steps is not None and steps >= max_steps:
            break
    result = {"steps": steps, "best": best_metrics, "history": history}
    (Path(output_dir) / "training_history.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


class RewardScorer:
    """Frozen, named reward classifiers and the paper's four reward terms."""

    def __init__(
        self, emotion_model, strategy_model, *, device=None,
        weights=(0.1, 0.1, 0.55, 0.25), beta=2.0, spacy_model=None,
    ):
        self.device = resolve_device(device)
        if len(weights) != 4 or any(value < 0 or not math.isfinite(value) for value in weights):
            raise ValueError("Four finite, non-negative reward weights are required")
        if beta < 1 or not math.isfinite(beta):
            raise ValueError("beta must be finite and at least 1")
        self.weights, self.beta = tuple(weights), beta
        self.classifiers, self.tokenizers = {}, {}
        for task, path in (("emotion", emotion_model), ("strategy", strategy_model)):
            model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(self.device).eval()
            labels = labels_for_task(task)
            expected = dict(enumerate(labels))
            if model.config.num_labels != len(labels) or model.config.id2label != expected:
                raise ValueError(f"The {task} checkpoint must use the dataset's named class order")
            model.requires_grad_(False)
            self.classifiers[task] = model
            self.tokenizers[task] = load_tokenizer(path)
        self.nlp = None
        if spacy_model:
            import spacy

            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    @torch.no_grad()
    def score(self, candidates: list[str], example: dict) -> torch.Tensor:
        scores = {}
        for task in ("emotion", "strategy"):
            tokenizer = self.tokenizers[task]
            model = self.classifiers[task]
            inputs = tokenizer(
                candidates, padding=True, truncation=True,
                max_length=classifier_context_length(model, tokenizer, 256), return_tensors="pt",
            ).to(self.device)
            probabilities = model(**inputs).logits.softmax(-1)
            targets = torch.full(
                (len(candidates),), example[f"{task}_id"], dtype=torch.long, device=self.device
            )
            scores[task] = classifier_reward(probabilities, targets, beta=self.beta)
        repetition = torch.tensor(
            [repetition_reward(example["previous_response"], text, nlp=self.nlp) for text in candidates],
            device=self.device,
        )
        consistency = torch.tensor(
            [meteor_consistency(example["text"], text) for text in candidates], device=self.device
        )
        return weighted_reward(repetition, consistency, scores["emotion"], scores["strategy"], weights=self.weights)


@torch.no_grad()
def collect_rollout(
    reference,
    tokenizer,
    examples,
    scorer,
    *,
    max_length,
    max_new_tokens,
    num_candidates,
    human_reward,
    top_p,
    temperature,
):
    """Record exact sampled actions and gold actions under one frozen policy.

    Nucleus sampling proposes candidates, as in the paper. PPO's likelihood
    ratio uses the underlying language-model sequence probabilities, before
    sampling temperature and top-p truncation, for both old and active policies.
    """
    reference.eval()
    prompts, responses, rewards = [], [], []
    device = next(reference.parameters()).device
    for example in examples:
        prompt = prompt_token_ids(tokenizer, example["prompt"], max_length - max_new_tokens)
        candidates = sample_responses(
            reference, tokenizer, prompt, num_candidates=num_candidates,
            max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature,
        )
        candidate_rewards = scorer.score([candidate["text"] for candidate in candidates], example)
        prompts.append(prompt)
        responses.append(response_token_ids(tokenizer, example["response"], max_length - len(prompt)))
        rewards.append(float(human_reward))
        for candidate, reward in zip(candidates, candidate_rewards):
            prompts.append(prompt)
            responses.append(candidate["token_ids"])
            rewards.append(float(reward))
    batch = pack_token_sequences(prompts, responses, tokenizer.pad_token_id, device)
    old_log_probs = sequence_log_probs(model_logits(reference, batch), batch["input_ids"], batch["response_mask"])
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    return batch, old_log_probs.detach(), normalize_rewards(reward_tensor), reward_tensor


def train_rl(
    train_path,
    validation_path,
    output_dir,
    *,
    model_name,
    emotion_model,
    strategy_model,
    epochs=1,
    batch_size=1,
    learning_rate=2e-5,
    max_length=1024,
    max_new_tokens=50,
    num_candidates=2,
    clip_epsilon=0.2,
    ppo_epochs=1,
    human_reward=10.0,
    top_p=0.9,
    temperature=0.8,
    reward_weights=(0.1, 0.1, 0.55, 0.25),
    beta=2.0,
    seed=42,
    device=None,
    max_steps=None,
    spacy_model=None,
):
    """Fine-tune the persuader; max_steps counts collected rollout batches."""
    _check_training_args(epochs, batch_size, learning_rate, max_steps)
    if ppo_epochs < 1 or num_candidates < 1 or not 0 < clip_epsilon < 1:
        raise ValueError("ppo_epochs/candidates must be positive and clip_epsilon in (0, 1)")
    if not math.isfinite(human_reward):
        raise ValueError("human_reward must be finite")
    seed_everything(seed)
    train, validation = _load_splits(train_path, validation_path)
    train, validation = dialogue_examples(train), dialogue_examples(validation)
    if not train or not validation:
        raise ValueError("Both splits must contain persuader examples")
    model, tokenizer = load_causal_model(model_name, device)
    device = next(model.parameters()).device
    max_length = context_length(model, max_length)
    if not 1 <= max_new_tokens < max_length:
        raise ValueError("max_new_tokens must leave at least one prompt token")
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    scorer = RewardScorer(
        emotion_model, strategy_model, device=device, weights=reward_weights, beta=beta,
        spacy_model=spacy_model,
    )
    model.config.rl_emo_per_speaker = "persuader"
    optimizer = _optimizer(model, learning_rate)
    configuration = {
        "stage": "rl", "model_name": str(model_name), "emotion_model": str(emotion_model),
        "strategy_model": str(strategy_model), "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "max_length": max_length, "max_new_tokens": max_new_tokens,
        "num_candidates": num_candidates, "clip_epsilon": clip_epsilon, "ppo_epochs": ppo_epochs,
        "human_reward": human_reward, "top_p": top_p, "temperature": temperature,
        "reward_weights": list(reward_weights), "beta": beta, "seed": seed,
        "max_steps": max_steps, "spacy_model": spacy_model,
    }
    rng, steps, optimizer_steps, history, best_loss = random.Random(seed), 0, 0, [], math.inf
    best_metrics = None
    for epoch in range(epochs):
        rollout_metrics = []
        for rows in _batches(train, batch_size, rng):
            reference.load_state_dict(model.state_dict())
            batch, old_log_probs, advantages, rewards = collect_rollout(
                reference, tokenizer, rows, scorer, max_length=max_length,
                max_new_tokens=max_new_tokens, num_candidates=num_candidates,
                human_reward=human_reward, top_p=top_p, temperature=temperature,
            )
            # eval() disables dropout while preserving autograd. With unchanged
            # weights, old/new likelihoods then coincide at the first PPO pass.
            model.eval()
            for _ in range(ppo_epochs):
                optimizer.zero_grad(set_to_none=True)
                new_log_probs = sequence_log_probs(
                    model_logits(model, batch), batch["input_ids"], batch["response_mask"]
                )
                loss = clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=clip_epsilon)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
            with torch.no_grad():
                new_log_probs = sequence_log_probs(
                    model_logits(model, batch), batch["input_ids"], batch["response_mask"]
                )
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                rollout_metrics.append({
                    "policy_loss": float(loss.detach()), "reward": float(rewards.mean()),
                    "clip_fraction": float(((ratio - 1).abs() > clip_epsilon).float().mean()),
                    "approx_kl": float((ratio - 1 - log_ratio).mean()),
                    "actions": len(rewards),
                })
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        scores = evaluate_examples(model, tokenizer, validation, batch_size=batch_size, max_length=max_length)
        metrics = {
            "epoch": epoch + 1, "steps": steps, "optimizer_steps": optimizer_steps,
            "validation": scores,
            "rollout": {key: sum(row[key] for row in rollout_metrics) / len(rollout_metrics) for key in rollout_metrics[0]},
        }
        history.append(metrics)
        if scores["nll"] < best_loss:
            best_loss, best_metrics = scores["nll"], metrics
            _save_checkpoint(model, tokenizer, output_dir, configuration, metrics)
        if max_steps is not None and steps >= max_steps:
            break
    result = {"steps": steps, "optimizer_steps": optimizer_steps, "best": best_metrics, "history": history}
    (Path(output_dir) / "training_history.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
