from __future__ import annotations

import re
from typing import Tuple

from datasets import Dataset, load_dataset


#DATASET_NAME = "SetFit/enron_spam"


def load_data_splits(dataset_name) -> Tuple[Dataset, Dataset]:
    """Load train/test splits from Hugging Face."""
    ds = load_dataset(dataset_name)
    return ds["train"], ds["test"]


def get_text_and_labels(split: Dataset) -> tuple[list[str], list[int]]:
    """
    Return text and binary labels.

    The dataset includes email text and labels.
    Prefer the 'text' column when present, otherwise combine subject/text-like
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


def preprocess(text):

    # Replace URLs
    text = re.sub(r'http[s]?://\S+', 'URL', text)
    text = re.sub(r'www\.\S+', 'URL', text)

    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', text)

    # Replace numbers
    text = re.sub(r'\b\d+\b', 'NUMBER', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())

    # Convert to lowercase
    text = text.lower()
    
    return text


def extract_metadata(text):
    """Extract metadata features from email text."""

    num_words = len(text.split())
    num_characters = len(text)

    # URL features
    num_urls = len(re.findall(r'http[s]?://\S+', text)) + len(re.findall(r'www\.\S+', text)) + len(re.findall(r'[^\s]+\.com/[^\s]', text)) + len(re.findall(r'[^\s]+\.site/[^\s]', text)) + len(re.findall(r'[^\s]+bit\.ly/[^\s]', text))
    hasurl = int(num_urls > 0)

    # Punctuation features
    num_exclamations = text.count('!')
    num_dollar_signs = text.count('$')

    # Capitalization features
    num_all_caps_words = sum(1 for word in text.split() if word.isupper())
    uppercase_ratio = num_all_caps_words / num_words
    
    # Digits features
    num_digits = sum(c.isdigit() for c in text)
    has_digits = int(num_digits > 0)

    # Keyword features
    keywords_verify = ['verify', 'verification', 'confirm', 'confirmation', 'account', 'password', 'bank', 'social security', 'click', 'ssn', 'credit card', 'debit card']
    contains_verify_keywords = int(any(keyword in text.lower() for keyword in keywords_verify))
    keywords_offer = ['free', 'win', 'winner', 'prize', 'money', 'offer', 'click', 'buy', 'cheap', 'limited time', 'exclusive', 'deal', 'discount']
    contains_offer_keywords = int(any(keyword in text.lower() for keyword in keywords_offer))
    keywords_urgency = ['urgent', 'immediately', 'asap', 'important', 'attention', 'act now']
    contains_urgency_keywords = int(any(keyword in text.lower() for keyword in keywords_urgency))

    return [
        num_words,
        num_characters,
        num_urls,
        hasurl,
        num_exclamations,
        num_dollar_signs,
        num_all_caps_words,
        uppercase_ratio,
        num_digits,
        has_digits,
        contains_verify_keywords,
        contains_offer_keywords,
        contains_urgency_keywords
    ]

