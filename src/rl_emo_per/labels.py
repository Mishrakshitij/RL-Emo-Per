"""Stable class order used by dataset IDs and classifier checkpoints."""

EMOTIONS = (
    "sentimental", "afraid", "proud", "trusting", "joyful", "angry", "sad",
    "jealous", "grateful", "prepared", "embarrassed", "surprised", "annoyed",
    "lonely", "guilty", "confident", "disappointed", "caring", "apprehensive",
    "anxious", "hopeful", "content", "impressed",
)
STRATEGIES = (
    "no_strategy", "task-related-inquiry", "credibility-appeal", "logical-appeal",
    "personal-related-inquiry", "source-related-inquiry", "donation-information",
    "foot-in-the-door", "emotion-appeal", "self-modeling", "personal-story",
)
BINARY_LABELS = ("not_persuasive", "persuasive")


def labels_for_task(task: str) -> tuple[str, ...]:
    """Return the complete class vocabulary for a classifier task."""
    return {"emotion": EMOTIONS, "strategy": STRATEGIES, "binary": BINARY_LABELS}[task]
