from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

STAGES = ["stage_1", "stage_2", "stage_3"]
MODEL_CHOICES = ["naive_bayes", "logistic_regression"]
VECTORIZER_CHOICES = ["bow", "tfidf"]


def build_pipeline(vectorizer_name: str, model_name: str, max_features: int) -> Pipeline:
    if vectorizer_name == "bow":
        vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
    elif vectorizer_name == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    else:
        raise ValueError(f"Unsupported vectorizer: {vectorizer_name}")

    if model_name == "naive_bayes":
        model = MultinomialNB()
    elif model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([
        ("vectorizer", vectorizer),
        ("model", model),
    ])


def get_score_values(pipeline: Pipeline, X_test: list[str]) -> np.ndarray:
    """Return a continuous spam score for ROC/AUC."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X_test)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X_test)
    raise ValueError("Model does not support predict_proba or decision_function.")


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path, label_names) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curve_plot(y_true: list[int], y_scores: np.ndarray, output_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return auc


