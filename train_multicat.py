from train_utils import *
from data import *

def main():
    parser = argparse.ArgumentParser(description="Train a multi-category classifier on the jason23322/high-accuracy-email-classifier dataset.")
    parser.add_argument("--vectorizer", choices=VECTORIZER_CHOICES, default="tfidf")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="logistic_regression")
    parser.add_argument("--stage", choices=STAGES, default="Undefined Stage")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    multiclass_dataset = "jason23322/high-accuracy-email-classifier"
    train_split, test_split = load_data_splits(multiclass_dataset)

    X_train = train_split["text"]
    print(X_train)
    X_test = test_split["text"]

    # Preprocess text
    X_train = [preprocess(x) for x in X_train]
    X_test = [preprocess(x) for x in X_test]

    y_train = train_split["category_id"]
    y_test = test_split["category_id"]

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Vectorizer: {args.vectorizer}")
    print(f"Model: {args.model}")

    pipeline = build_pipeline(args.vectorizer, args.model, args.max_features)

    # Use unigrams and bigrams for better performance on multi-category classification
    pipeline.named_steps["vectorizer"].set_params(ngram_range=(1, 2))

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Create category id and category name mapping for interpretability in metrics output
    label_to_name = {}

    for category_id, category in zip(train_split["category_id"], train_split["category"]):
        label_to_name[category_id] = category

    label_to_name = dict(sorted(label_to_name.items()))

    # Category IDs
    labels = list(label_to_name.keys())

    # Category names
    label_names = list(label_to_name.values())
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class_metrics = {}

    for i, label in enumerate(labels):
        category_name = label_to_name[label]
        per_class_metrics[category_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "config": {
            "dataset": multiclass_dataset,
            "vectorizer": args.vectorizer,
            "model": args.model,
            "max_features": args.max_features,
        },
    }

    metrics_path = output_dir / f"metrics_{args.stage}_{args.vectorizer}_{args.model}.json"
    model_path = output_dir / f"model_{args.stage}_{args.vectorizer}_{args.model}.joblib"
    cm_plot_path = output_dir / f"confusion_matrix_{args.stage}_{args.vectorizer}_{args.model}.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipeline, model_path)

    save_confusion_matrix_plot(cm, cm_plot_path, label_names)

    print("\nResults")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix:")
    print(cm.tolist())
    print("\nPer-class metrics:")
    for category_name, class_metrics in per_class_metrics.items():
        print(f"{category_name}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1: {class_metrics['f1']:.4f}")
        print(f"  Support: {class_metrics['support']}")
    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

if __name__ == "__main__":
    main()