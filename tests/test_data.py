"""CSV integrity, dialogue chronology, and release verification checks."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rl_emo_per.data import (
    END_OF_TURN,
    FIELDS,
    dataset_summary,
    dialogue_examples,
    load_records,
    validate_records,
    verify_dataset,
)
from rl_emo_per.labels import BINARY_LABELS, EMOTIONS, STRATEGIES, labels_for_task


def record(dialogue="train-dialogue", turn=0, speaker="persuader", text="Opening offer", split="train", label=0):
    targeted = speaker == "persuader"
    return {
        "dialogue_id": dialogue, "turn_id": turn, "speaker": speaker, "text": text,
        "emotion": EMOTIONS[label] if targeted else None,
        "emotion_id": label if targeted else None,
        "strategy": STRATEGIES[label] if targeted else None,
        "strategy_id": label if targeted else None, "split": split,
    }


def write_csv(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


class DialogueExampleTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            record(turn=0, text="Earlier system text"),
            record(turn=0, speaker="persuadee", text="Earlier user text"),
            record(turn=1, text="Current system target", label=2),
            record(turn=1, speaker="persuadee", text="Future user answer"),
        ]

    def test_round_order_uses_only_preceding_context(self):
        examples = dialogue_examples(list(reversed(self.rows)))
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["prompt"], "A:")
        expected = "A:Earlier system text" + END_OF_TURN + "B:Earlier user text" + END_OF_TURN + "A:"
        self.assertEqual(examples[1]["prompt"], expected)
        self.assertEqual(examples[1]["response"], "Current system target" + END_OF_TURN)
        self.assertEqual(examples[1]["previous_response"], "Earlier system text")
        self.assertNotIn("Current system target", examples[1]["prompt"])
        self.assertNotIn("Future user answer", examples[1]["prompt"])

    def test_targets_do_not_leak_into_prompts_and_input_is_unchanged(self):
        before = copy.deepcopy(self.rows)
        example = dialogue_examples(self.rows)[1]
        for value in (EMOTIONS[2], STRATEGIES[2], "emotion_id", "strategy_id"):
            self.assertNotIn(value, example["prompt"])
        self.assertEqual(example["emotion_id"], 2)
        self.assertEqual(example["strategy_id"], 2)
        self.assertEqual(self.rows, before)

    def test_persuadee_target_sees_same_round_persuader(self):
        examples = dialogue_examples(self.rows, speaker="persuadee")
        self.assertEqual(examples[0]["prompt"], "A:Earlier system text" + END_OF_TURN + "B:")
        self.assertIn("A:Current system target" + END_OF_TURN, examples[1]["prompt"])
        self.assertNotIn("Future user answer", examples[1]["prompt"])
        self.assertEqual(examples[1]["previous_response"], "Earlier user text")
        self.assertIsNone(examples[1]["emotion_id"])

    def test_history_and_previous_response_reset_between_dialogues(self):
        rows = self.rows + [record(dialogue="other-dialogue", text="Other conversation", split="validation")]
        examples = dialogue_examples(rows)
        other = next(row for row in examples if row["dialogue_id"] == "other-dialogue")
        self.assertEqual(other["prompt"], "A:")
        self.assertEqual(other["previous_response"], "")
        self.assertNotIn("Other conversation", examples[-1]["prompt"])

    def test_summary_counts_targets_and_keeps_absent_classes(self):
        stats = dataset_summary(self.rows)
        self.assertEqual(stats["utterances"], 4)
        self.assertEqual(stats["dialogues"], 1)
        self.assertEqual(stats["persuader_utterances"], 2)
        self.assertEqual(stats["persuadee_utterances"], 2)
        self.assertEqual(stats["emotion_counts"][EMOTIONS[0]], 1)
        self.assertEqual(stats["emotion_counts"][EMOTIONS[2]], 1)
        self.assertEqual(stats["emotion_counts"][EMOTIONS[-1]], 0)
        self.assertEqual(len(stats["emotion_counts"]), 23)
        self.assertEqual(len(stats["strategy_counts"]), 11)
        self.assertEqual(stats["splits"]["validation"]["utterances"], 0)


class ValidationTests(unittest.TestCase):
    def test_duplicate_speaker_turns_are_rejected_but_both_speakers_can_share_round(self):
        first = record()
        validate_records([first, record(speaker="persuadee")])
        with self.assertRaisesRegex(ValueError, "Duplicate turn"):
            validate_records([first, dict(first)])

    def test_dialogue_split_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_records([record(), record(turn=1, split="validation")])

    def test_malformed_rows_and_target_mismatches_are_rejected(self):
        cases = (
            {"turn_id": -1}, {"turn_id": True}, {"text": " \t\r\n "},
            {"speaker": "unknown"}, {"split": "test"}, {"dialogue_id": ""},
            {"emotion_id": None}, {"emotion_id": len(EMOTIONS)}, {"emotion": EMOTIONS[1]},
            {"strategy_id": -1}, {"strategy_id": True}, {"strategy": "unknown"},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_records([{**record(), **changes}])
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_records([{**record(), "extra": "value"}])
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_records([])

    def test_persuadee_cannot_inherit_persuader_labels(self):
        row = record(speaker="persuadee")
        row.update(emotion_id=0, emotion=EMOTIONS[0])
        with self.assertRaisesRegex(ValueError, "context"):
            validate_records([row])


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "records.csv"

    def test_unicode_quotes_commas_whitespace_and_newlines_survive_roundtrip(self):
        text = '  “Please,” she said: café, 東京, مرحبًا 💛\r\nSecond line\n  '
        rows = [record(text=text), record(speaker="persuadee", text=" A comma, and a \"quote\". ")]
        write_csv(self.path, rows)
        loaded = load_records(self.path)
        self.assertEqual(loaded, rows)
        self.assertEqual(loaded[0]["text"].encode("utf-8"), text.encode("utf-8"))
        self.assertIsNone(loaded[1]["emotion_id"])
        self.assertIsNone(loaded[1]["strategy"])

    def test_header_must_have_exact_columns_and_order(self):
        self.path.write_text(",".join(reversed(FIELDS)) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "columns"):
            load_records(self.path)

    def test_short_and_long_csv_rows_raise_value_error(self):
        for values in (
            ["conversation", "0", "persuader", "hello"],
            ["conversation", "0", "persuader", "hello", EMOTIONS[0], "0", STRATEGIES[0], "0", "train", "extra"],
        ):
            with self.subTest(values=values):
                with self.path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(FIELDS)
                    writer.writerow(values)
                with self.assertRaises(ValueError):
                    load_records(self.path)

    def test_noninteger_csv_targets_are_rejected(self):
        write_csv(self.path, [{**record(), "emotion_id": "zero"}])
        with self.assertRaises(ValueError):
            load_records(self.path)

    def test_unterminated_quoted_text_has_a_csv_error(self):
        self.path.write_text(",".join(FIELDS) + '\nconversation,0,persuader,"unterminated text', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "malformed CSV"):
            load_records(self.path)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.rows = [
            record(), record(speaker="persuadee"),
            record(dialogue="validation-dialogue", split="validation", label=1),
            record(dialogue="validation-dialogue", split="validation", speaker="persuadee"),
        ]
        write_csv(self.directory / "dialogues.csv", self.rows)
        for split in ("train", "validation"):
            write_csv(self.directory / f"{split}.csv", [row for row in self.rows if row["split"] == split])
        (self.directory / "paper_emotion_counts.json").write_text(
            json.dumps(dataset_summary(self.rows)["emotion_counts"]), encoding="utf-8",
        )
        self.manifest = {
            "sha256": {name: self.digest(name) for name in (
                "dialogues.csv", "train.csv", "validation.csv", "paper_emotion_counts.json",
            )},
            "statistics": dataset_summary(self.rows),
        }
        self.save_manifest()

    def digest(self, name):
        return hashlib.sha256((self.directory / name).read_bytes()).hexdigest()

    def save_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_complete_consistent_release_verifies(self):
        stats = verify_dataset(self.directory)
        self.assertEqual(stats["dialogues"], 2)
        self.assertEqual(stats["utterances"], 4)
        self.assertEqual(stats["persuader_utterances"], 2)
        self.assertEqual(stats["splits"]["train"]["dialogues"], 1)
        self.assertEqual(stats["splits"]["validation"]["dialogues"], 1)

    def test_changed_file_is_detected_before_statistics(self):
        with (self.directory / "dialogues.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            verify_dataset(self.directory)

    def test_wrong_split_contents_are_detected_even_with_matching_hash(self):
        changed = [{**row, "text": "Changed split text"} for row in self.rows if row["split"] == "train"]
        write_csv(self.directory / "train.csv", changed)
        self.manifest["sha256"]["train.csv"] = self.digest("train.csv")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "differs from its rows"):
            verify_dataset(self.directory)

    def test_emotion_counts_are_checked_against_paper_counts(self):
        counts = dict(self.manifest["statistics"]["emotion_counts"])
        counts[EMOTIONS[0]] += 1
        (self.directory / "paper_emotion_counts.json").write_text(json.dumps(counts), encoding="utf-8")
        self.manifest["sha256"]["paper_emotion_counts.json"] = self.digest("paper_emotion_counts.json")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "distribution differs"):
            verify_dataset(self.directory)

    def test_manifest_cannot_reference_a_parent_directory(self):
        self.manifest["sha256"]["../outside.csv"] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "inside dataset directory"):
            verify_dataset(self.directory)

    def test_manifest_must_cover_required_artifacts(self):
        del self.manifest["sha256"]["dialogues.csv"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "checksum|Checksum|sha256"):
            verify_dataset(self.directory)


class LabelTests(unittest.TestCase):
    def test_label_vocabularies_are_unique_complete_and_stable(self):
        for task, labels, count in (("emotion", EMOTIONS, 23), ("strategy", STRATEGIES, 11), ("binary", BINARY_LABELS, 2)):
            self.assertEqual(labels_for_task(task), labels)
            self.assertEqual(len(labels), count)
            self.assertEqual(len(set(labels)), count)
        self.assertEqual(EMOTIONS[0], "sentimental")
        self.assertEqual(STRATEGIES[0], "no_strategy")
        self.assertEqual(BINARY_LABELS, ("not_persuasive", "persuasive"))


class ShippedDatasetTests(unittest.TestCase):
    def test_shipped_dataset_matches_manifest_and_paper(self):
        directory = Path(__file__).resolve().parents[1] / "datasets"
        if not (directory / "manifest.json").exists():
            self.skipTest("Dataset release artifacts are not present")
        stats = verify_dataset(directory)
        self.assertGreater(stats["dialogues"], 0)
        self.assertEqual(sum(stats["emotion_counts"].values()), stats["persuader_utterances"])
        self.assertEqual(sum(stats["strategy_counts"].values()), stats["persuader_utterances"])


if __name__ == "__main__":
    unittest.main()
