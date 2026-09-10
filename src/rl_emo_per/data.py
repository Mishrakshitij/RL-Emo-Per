"""CSV loading, integrity checks, and chronological dialogue examples.

This module uses only the Python standard library. Each row is one speaker turn;
emotion and strategy targets belong to the persuader. Persuadee turns provide
dialogue context and have empty target fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .labels import EMOTIONS, STRATEGIES

FIELDS = (
    "dialogue_id", "turn_id", "speaker", "text", "emotion", "emotion_id",
    "strategy", "strategy_id", "split",
)
PREFIXES = {"persuader": "A:", "persuadee": "B:"}
END_OF_TURN = "\n\n\n"


def load_records(path: str | Path) -> list[dict]:
    """Load and validate one CSV; preserve the original text byte-for-byte."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"{path}: expected CSV columns {list(FIELDS)}")
        rows = []
        try:
            for row in reader:
                if set(row) != set(FIELDS) or any(row[name] is None for name in FIELDS):
                    raise ValueError(f"{path}: malformed CSV row at line {reader.line_num}")
                for name in ("turn_id", "emotion_id", "strategy_id"):
                    value = row[name]
                    try:
                        row[name] = int(value) if value != "" else None
                    except ValueError as error:
                        raise ValueError(f"{path}: invalid {name} at line {reader.line_num}: {value!r}") from error
                for name in ("emotion", "strategy"):
                    row[name] = row[name] or None
                rows.append(row)
        except csv.Error as error:
            raise ValueError(f"{path}: malformed CSV at line {reader.line_num}: {error}") from error
    validate_records(rows)
    return rows


def validate_records(rows: list[dict]) -> None:
    """Reject missing targets, duplicate turns, unknown classes and split leakage."""
    if not rows:
        raise ValueError("Dataset is empty")
    seen = set()
    splits = {}
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("Unexpected or missing record fields")
        dialogue_id = row["dialogue_id"]
        if not isinstance(dialogue_id, str) or not dialogue_id:
            raise ValueError("Empty dialogue ID")
        if type(row["turn_id"]) is not int or row["turn_id"] < 0:
            raise ValueError("Turn IDs must be non-negative integers")
        if row["speaker"] not in PREFIXES:
            raise ValueError(f"Unknown speaker: {row['speaker']}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Empty utterance")
        key = (dialogue_id, row["turn_id"], row["speaker"])
        if key in seen:
            raise ValueError(f"Duplicate turn: {key}")
        seen.add(key)
        split = row["split"]
        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown split: {split}")
        if splits.setdefault(dialogue_id, split) != split:
            raise ValueError(f"Dialogue crosses splits: {dialogue_id}")
        for field, labels in (("emotion", EMOTIONS), ("strategy", STRATEGIES)):
            index = row[f"{field}_id"]
            label = row[field]
            if row["speaker"] == "persuadee":
                if index is not None or label is not None:
                    raise ValueError("Persuadee context must not carry persuader targets")
            elif type(index) is not int or not 0 <= index < len(labels) or label != labels[index]:
                raise ValueError(f"Invalid {field} label for {key}: {index}, {label}")


def dialogue_examples(records: list[dict], speaker: str = "persuader") -> list[dict]:
    """Build response targets using only earlier turns in the same dialogue.

    Both speakers share a round index. The persuader speaks first in each round.
    Labels for a target utterance never appear in the language-model prompt.
    """
    if speaker not in PREFIXES:
        raise ValueError(f"Unknown speaker: {speaker}")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["dialogue_id"]].append(row)
    examples = []
    for dialogue_id in sorted(grouped):
        history = []
        previous = {role: "" for role in PREFIXES}
        for row in sorted(grouped[dialogue_id], key=lambda r: (r["turn_id"], r["speaker"] == "persuadee")):
            role = row["speaker"]
            if role == speaker:
                examples.append({
                    **row,
                    "prompt": "".join(history) + PREFIXES[role],
                    "response": row["text"] + END_OF_TURN,
                    "previous_response": previous[role],
                })
            history.append(PREFIXES[role] + row["text"] + END_OF_TURN)
            previous[role] = row["text"]
    return examples


def dataset_summary(records: list[dict]) -> dict:
    """Return counts measured from the records, including zero-count classes."""
    targets = [r for r in records if r["speaker"] == "persuader"]
    return {
        "dialogues": len({r["dialogue_id"] for r in records}),
        "utterances": len(records),
        "persuader_utterances": len(targets),
        "persuadee_utterances": len(records) - len(targets),
        "emotion_counts": {label: sum(r["emotion"] == label for r in targets) for label in EMOTIONS},
        "strategy_counts": {label: sum(r["strategy"] == label for r in targets) for label in STRATEGIES},
        "splits": {
            name: {
                "dialogues": len({r["dialogue_id"] for r in records if r["split"] == name}),
                "utterances": sum(r["split"] == name for r in records),
                "persuader_utterances": sum(r["split"] == name for r in targets),
            }
            for name in ("train", "validation")
        },
    }


def verify_dataset(directory: str | Path) -> dict:
    """Verify checksums, class counts, and mutually exclusive dialogue splits."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    checksums = manifest.get("sha256") if isinstance(manifest, dict) else None
    required_files = {"dialogues.csv", "train.csv", "validation.csv", "paper_emotion_counts.json"}
    if not isinstance(checksums, dict) or not required_files.issubset(checksums):
        raise ValueError("Manifest sha256 checksums must include every required dataset artifact")
    for name, digest in checksums.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("Manifest sha256 entries must have filenames and lowercase SHA-256 digests")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Manifest path must stay inside dataset directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    rows = load_records(directory / "dialogues.csv")
    stats = dataset_summary(rows)
    if stats != manifest["statistics"]:
        raise ValueError("Dataset statistics differ from manifest")
    for name in ("train", "validation"):
        actual = load_records(directory / f"{name}.csv")
        expected = [row for row in rows if row["split"] == name]
        if actual != expected:
            raise ValueError(f"{name}.csv differs from its rows in dialogues.csv")
    expected_emotions = json.loads((directory / "paper_emotion_counts.json").read_text())
    if stats["emotion_counts"] != expected_emotions:
        raise ValueError("Emotion distribution differs from the paper")
    return stats
