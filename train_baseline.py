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

from data import get_text_and_labels, load_enron_splits


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


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline spam/ham classifier on SetFit/enron_spam.")
    parser.add_argument("--vectorizer", choices=VECTORIZER_CHOICES, default="tfidf")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="logistic_regression")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split, test_split = load_enron_splits()
    X_train, y_train = get_text_and_labels(train_split)
    X_test, y_test = get_text_and_labels(test_split)

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Vectorizer: {args.vectorizer}")
    print(f"Model: {args.model}")

    pipeline = build_pipeline(args.vectorizer, args.model, args.max_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_scores = get_score_values(pipeline, X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_test, y_scores)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            "ham": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "spam": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"],
            output_dict=True,
            zero_division=0,
        ),
        "config": {
            "dataset": "SetFit/enron_spam",
            "vectorizer": args.vectorizer,
            "model": args.model,
            "max_features": args.max_features,
        },
    }

    metrics_path = output_dir / f"metrics_{args.vectorizer}_{args.model}.json"
    model_path = output_dir / f"model_{args.vectorizer}_{args.model}.joblib"
    cm_plot_path = output_dir / f"confusion_matrix_{args.vectorizer}_{args.model}.png"
    roc_plot_path = output_dir / f"roc_curve_{args.vectorizer}_{args.model}.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipeline, model_path)
    save_confusion_matrix_plot(cm, cm_plot_path)
    save_roc_curve_plot(y_test, y_scores, roc_plot_path)

    print("\nResults")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion matrix [ [ham->ham, ham->spam], [spam->ham, spam->spam] ]:")
    print(cm.tolist())
    print("\nPer-class metrics:")
    print(f"Ham  - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    print(f"Spam - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")
    print(f"Saved ROC curve plot to: {roc_plot_path}")


if __name__ == "__main__":
    main()
