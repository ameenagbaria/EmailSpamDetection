from __future__ import annotations

from typing import Tuple

from datasets import Dataset, load_dataset


DATASET_NAME = "SetFit/enron_spam"


def load_enron_splits() -> Tuple[Dataset, Dataset]:
    """Load train/test splits from Hugging Face."""
    ds = load_dataset(DATASET_NAME)
    return ds["train"], ds["test"]


def get_text_and_labels(split: Dataset) -> tuple[list[str], list[int]]:
    """
    Return text and binary labels.

    The dataset includes email text and labels.
    Prefer the 'text' column when present, otherwise combine subject/body-like
    fields if needed.
    """
    columns = set(split.column_names)

    if "text" in columns:
        texts = [str(x) for x in split["text"]]
    elif {"subject", "message"}.issubset(columns):
        texts = [f"{s} {m}".strip() for s, m in zip(split["subject"], split["message"])]
    elif "message" in columns:
        texts = [str(x) for x in split["message"]]
    else:
        raise ValueError(f"Could not find a usable text column. Found columns: {sorted(columns)}")

    # Most HF binary datasets expose either 'label' or 'label_text'.
    if "label" in columns:
        labels = [int(x) for x in split["label"]]
    elif "label_text" in columns:
        label_map = {"ham": 0, "spam": 1}
        labels = [label_map[str(x).lower()] for x in split["label_text"]]
    else:
        raise ValueError(f"Could not find a usable label column. Found columns: {sorted(columns)}")

    return texts, labels