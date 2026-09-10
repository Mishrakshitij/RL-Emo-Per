"""Command-line interface; dataset inspection does not load ML dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _training_arguments(parser, *, learning_rate, batch_size, max_length, epochs):
    parser.add_argument("--train-path", default="datasets/train.csv")
    parser.add_argument("--validation-path", default="datasets/validation.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--max-length", type=int, default=max_length)
    parser.add_argument("--max-steps", type=int,
                        help="Limit training steps (RL: rollout batches; classifier/MLE: optimizer updates)")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda:0; auto-detected by default")
    parser.add_argument("--seed", type=int, default=42)


def build_parser():
    parser = argparse.ArgumentParser(description="RL-Emo-Per: empathetic persuasive dialogue models")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-data", help="Check dataset hashes, labels, statistics and splits")
    verify.add_argument("--data-dir", default="datasets")
    stats = commands.add_parser("stats", help="Print measured dataset statistics")
    stats.add_argument("--data-path", default="datasets/dialogues.csv")

    classifier = commands.add_parser("train-classifier", help="Train an emotion, strategy or binary classifier")
    _training_arguments(classifier, learning_rate=2e-5, batch_size=32, max_length=256, epochs=10)
    classifier.add_argument("--task", choices=["emotion", "strategy", "binary"], required=True)
    classifier.add_argument("--model-name", default="roberta-large")

    mle = commands.add_parser("train-mle", help="Train one speaker's conditional language model")
    _training_arguments(mle, learning_rate=3e-5, batch_size=1, max_length=1024, epochs=1)
    mle.add_argument("--speaker", choices=["persuader", "persuadee"], default="persuader")
    mle.add_argument("--model-name", default="gpt2-medium")
    mle.add_argument("--warmup-steps", type=int, default=100)

    rl = commands.add_parser("train-rl", help="Fine-tune the persuader with custom sequence PPO")
    _training_arguments(rl, learning_rate=2e-5, batch_size=1, max_length=1024, epochs=1)
    rl.add_argument("--model-name", required=True, help="Persuader MLE checkpoint directory")
    rl.add_argument("--emotion-model", required=True, help="23-class emotion checkpoint directory")
    rl.add_argument("--strategy-model", required=True, help="11-class strategy checkpoint directory")
    rl.add_argument("--max-new-tokens", type=int, default=50)
    rl.add_argument("--num-candidates", type=int, default=2)
    rl.add_argument("--clip-epsilon", type=float, default=0.2)
    rl.add_argument("--ppo-epochs", type=int, default=1)
    rl.add_argument("--human-reward", type=float, default=10.0)
    rl.add_argument("--top-p", type=float, default=0.9)
    rl.add_argument("--temperature", type=float, default=0.8)
    rl.add_argument("--reward-weights", type=float, nargs=4, default=(0.1, 0.1, 0.55, 0.25),
                    metavar=("REPETITION", "CONSISTENCY", "EMOTION", "STRATEGY"))
    rl.add_argument("--beta", type=float, default=2.0)
    rl.add_argument("--spacy-model", help="Optional installed spaCy model for unigram normalization")

    generate = commands.add_parser("generate", help="Generate responses to a JSON dialogue history")
    generate.add_argument("--model-path", required=True)
    generate.add_argument("--history", required=True, type=Path,
                          help="JSON list of {speaker, text} turns")
    generate.add_argument("--speaker", choices=["persuader", "persuadee"], default="persuader")
    generate.add_argument("--max-length", type=int, default=1024)
    generate.add_argument("--max-new-tokens", type=int, default=50)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--num-candidates", type=int, default=1)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--device")

    evaluate = commands.add_parser("evaluate", help="Measure held-out response NLL and perplexity")
    evaluate.add_argument("--data-path", default="datasets/validation.csv")
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--speaker", choices=["persuader", "persuadee"], default="persuader")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--max-length", type=int, default=1024)
    evaluate.add_argument("--device")
    return parser


def main(argv=None):
    parser = build_parser()
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    try:
        if command == "verify-data":
            from .data import verify_dataset
            result = verify_dataset(options["data_dir"])
        elif command == "stats":
            from .data import dataset_summary, load_records
            result = dataset_summary(load_records(options["data_path"]))
        elif command in {"train-classifier", "train-mle", "train-rl"}:
            from .training import train_classifier, train_mle, train_rl
            result = {"train-classifier": train_classifier, "train-mle": train_mle, "train-rl": train_rl}[command](**options)
        elif command == "generate":
            from .generation import generate_response
            options["history"] = json.loads(options["history"].read_text(encoding="utf-8"))
            result = {"responses": generate_response(**options)}
        else:
            from .generation import evaluate_language_model
            result = evaluate_language_model(**options)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
