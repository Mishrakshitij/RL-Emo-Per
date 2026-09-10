"""Offline integration checks using tiny, locally initialized HF models."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from rl_emo_per.data import FIELDS, dialogue_examples, load_records
from rl_emo_per.generation import evaluate_language_model, generate_response
from rl_emo_per.labels import EMOTIONS, STRATEGIES, labels_for_task
from rl_emo_per.losses import sequence_log_probs
from rl_emo_per.modeling import load_causal_model, model_logits, pack_responses
from rl_emo_per.training import collect_rollout, train_classifier, train_mle, train_rl


def tiny_assets(tmp_path):
    torch.manual_seed(3)
    words = ["<unk>", "<eos>", "A", "B", ":", "hello", "please", "help", "children", "thank", "you", "yes", "today", "."]
    tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>", bos_token="<eos>"
    )
    causal = tmp_path / "initial_lm"
    GPT2LMHeadModel(GPT2Config(
        vocab_size=len(words), n_layer=1, n_head=2, n_embd=16, n_positions=64,
        bos_token_id=1, eos_token_id=1, pad_token_id=1,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )).save_pretrained(causal)
    tokenizer.save_pretrained(causal)
    classifiers = {}
    for task in ("emotion", "strategy", "binary"):
        labels = labels_for_task(task)
        path = tmp_path / f"initial_{task}"
        BertForSequenceClassification(BertConfig(
            vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=64, num_labels=len(labels),
            pad_token_id=1, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            id2label=dict(enumerate(labels)), label2id={label: index for index, label in enumerate(labels)},
        )).save_pretrained(path)
        tokenizer.save_pretrained(path)
        classifiers[task] = path
    paths = {}
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for turn in range(2):
                for speaker in ("persuader", "persuadee"):
                    target = speaker == "persuader"
                    writer.writerow({
                        "dialogue_id": split + "_dialogue", "turn_id": turn, "speaker": speaker,
                        "text": ("please help children today" if turn else "hello children") if target else "yes thank you",
                        "emotion": EMOTIONS[turn] if target else "", "emotion_id": turn if target else "",
                        "strategy": STRATEGIES[turn] if target else "", "strategy_id": turn if target else "",
                        "split": split,
                    })
        paths[split] = path
    return {"lm": causal, "classifiers": classifiers, "tokenizer": tokenizer, **paths}


def test_response_mask_keeps_eos_but_excludes_prompt_and_padding(tiny_assets):
    tokenizer = tiny_assets["tokenizer"]
    examples = [{"prompt": "A:", "response": "hello"}, {"prompt": "A: hello B: yes A:", "response": "please help children"}]
    batch = pack_responses(tokenizer, examples, 32, "cpu")
    eos = tokenizer.eos_token_id
    assert batch["input_ids"][0, 3] == eos
    assert batch["response_mask"][0, :4].tolist() == [False, False, True, True]
    assert not batch["response_mask"][0, 4:].any()
    assert not batch["attention_mask"][0, 4:].any()
    assert batch["input_ids"][0, 4:].eq(eos).all()


def test_classifier_training_uses_validation_and_saves_named_head(tiny_assets, tmp_path, task):
    output = tmp_path / "trained_classifier"
    source = tiny_assets["classifiers"][task]
    before = AutoModelForSequenceClassification.from_pretrained(source).classifier.weight.detach().clone()
    result = train_classifier(
        tiny_assets["train"], tiny_assets["validation"], output,
        task=task, model_name=source, max_steps=1, batch_size=2, max_length=32,
        learning_rate=1e-3, device="cpu",
    )
    saved = AutoModelForSequenceClassification.from_pretrained(output)
    assert result["steps"] == 1
    assert result["best"]["validation"]["examples"] == 2
    assert math.isfinite(result["best"]["validation"]["loss"])
    assert saved.config.id2label == dict(enumerate(labels_for_task(task)))
    assert not torch.equal(before, saved.classifier.weight)
    assert json.loads((output / "metrics.json").read_text())["validation"]["examples"] == 2


def test_mle_training_checkpoint_evaluation_and_generation(tiny_assets, tmp_path, speaker):
    output = tmp_path / "trained_lm"
    before = AutoModelForCausalLM.from_pretrained(tiny_assets["lm"]).transformer.wte.weight.detach().clone()
    result = train_mle(
        tiny_assets["train"], tiny_assets["validation"], output,
        model_name=tiny_assets["lm"], speaker=speaker, max_steps=1, batch_size=2,
        learning_rate=1e-3, warmup_steps=0, max_length=32, device="cpu",
    )
    saved = AutoModelForCausalLM.from_pretrained(output)
    assert not torch.equal(before, saved.transformer.wte.weight)
    assert saved.config.rl_emo_per_speaker == speaker
    assert result["best"]["validation"]["examples"] == 2
    evaluation = evaluate_language_model(
        tiny_assets["validation"], output, speaker=speaker, max_length=32, device="cpu"
    )
    assert math.isclose(evaluation["nll"], result["best"]["validation"]["nll"], rel_tol=1e-6)
    responses = generate_response(output, [], speaker=speaker, max_length=32, max_new_tokens=4, num_candidates=2, device="cpu")
    assert len(responses) == 2
    assert all(isinstance(response, str) for response in responses)


class DeterministicScorer:
    def __init__(self, *args, **kwargs):
        pass

    def score(self, candidates, example):
        return torch.tensor([-0.5 + index * 0.1 for index in range(len(candidates))])


def test_rollout_keeps_sampling_context_and_detaches_old_policy(tiny_assets):
    model, tokenizer = load_causal_model(tiny_assets["lm"], "cpu")
    model.eval()
    rows = dialogue_examples(load_records(tiny_assets["train"]))
    batch, old, advantages, rewards = collect_rollout(
        model, tokenizer, rows[-1:], DeterministicScorer(), max_length=20, max_new_tokens=4,
        num_candidates=2, human_reward=10, top_p=0.9, temperature=0.8,
    )
    assert torch.allclose(rewards, torch.tensor([10, -0.5, -0.4]))
    assert not old.requires_grad and not advantages.requires_grad
    assert math.isclose(float(advantages.mean()), 0, abs_tol=1e-7)
    assert math.isclose(float(advantages.std(correction=0)), 1, rel_tol=1e-6)
    fresh = sequence_log_probs(model_logits(model, batch), batch["input_ids"], batch["response_mask"])
    assert torch.allclose(old, fresh, atol=1e-6)
    for index in range(1, 3):
        prompts = [batch["input_ids"][i][~batch["response_mask"][i] & batch["attention_mask"][i].bool()] for i in (0, index)]
        assert torch.equal(*prompts)


def test_sequence_ppo_updates_parameters_and_validates(tiny_assets, tmp_path):
    output = tmp_path / "trained_rl"
    before = AutoModelForCausalLM.from_pretrained(tiny_assets["lm"]).transformer.wte.weight.detach().clone()
    result = train_rl(
        tiny_assets["train"], tiny_assets["validation"], output,
        model_name=tiny_assets["lm"], emotion_model=tiny_assets["classifiers"]["emotion"],
        strategy_model=tiny_assets["classifiers"]["strategy"], max_steps=2,
        ppo_epochs=2, max_length=32, max_new_tokens=4, learning_rate=1e-3, device="cpu",
    )
    saved = AutoModelForCausalLM.from_pretrained(output)
    assert result["steps"] == 2 and result["optimizer_steps"] == 4
    assert result["best"]["validation"]["examples"] == 2
    assert result["best"]["rollout"]["actions"] == 3
    assert math.isfinite(result["best"]["validation"]["nll"])
    assert not torch.equal(before, saved.transformer.wte.weight)


def test_training_rejects_dialogue_leakage(tiny_assets, tmp_path):
    with unittest.TestCase().assertRaisesRegex(ValueError, "same dialogue IDs"):
        train_mle(
            tiny_assets["train"], tiny_assets["train"], tmp_path / "bad",
            model_name=tiny_assets["lm"], max_steps=1, device="cpu",
        )


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.path = Path(self.scratch.name)
        self.assets = tiny_assets(self.path)

    def tearDown(self):
        self.scratch.cleanup()

    def test_response_mask(self):
        test_response_mask_keeps_eos_but_excludes_prompt_and_padding(self.assets)

    def test_emotion_classifier(self):
        test_classifier_training_uses_validation_and_saves_named_head(self.assets, self.path, "emotion")

    def test_strategy_classifier(self):
        test_classifier_training_uses_validation_and_saves_named_head(self.assets, self.path, "strategy")

    def test_binary_classifier(self):
        test_classifier_training_uses_validation_and_saves_named_head(self.assets, self.path, "binary")

    def test_persuader_mle(self):
        test_mle_training_checkpoint_evaluation_and_generation(self.assets, self.path, "persuader")

    def test_persuadee_mle(self):
        test_mle_training_checkpoint_evaluation_and_generation(self.assets, self.path, "persuadee")

    def test_rollout(self):
        test_rollout_keeps_sampling_context_and_detaches_old_policy(self.assets)

    def test_ppo(self):
        with mock.patch("rl_emo_per.training.RewardScorer", DeterministicScorer):
            test_sequence_ppo_updates_parameters_and_validates(self.assets, self.path)

    def test_split_leakage(self):
        test_training_rejects_dialogue_leakage(self.assets, self.path)


if __name__ == "__main__":
    unittest.main()
