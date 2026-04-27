from email import parser
from data import *
from train_utils import *
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

def main():
    parser = argparse.ArgumentParser(description="Train a multi-category transformer-based classifier on the jason23322/high-accuracy-email-classifier dataset.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="logistic_regression")
    parser.add_argument("--stage", choices=STAGES, default="Undefined Stage")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    multiclass_dataset_name = "jason23322/high-accuracy-email-classifier"
    train_split, test_split = load_data_splits(multiclass_dataset_name)

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Create category id and category name mapping for interpretability in metrics output
    label_to_name = {}

    for category_id, category in zip(train_split["category_id"], train_split["category"]):
        label_to_name[category_id] = category

    label_to_name = dict(sorted(label_to_name.items()))

    # Category IDs
    labels = list(label_to_name.keys())

    # Category names
    label_names = list(label_to_name.values())

    # Tokenize the text
    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    tokenized_train = train_split.map(tokenize_function, batched=True)
    tokenized_test = test_split.map(tokenize_function, batched=True)

    # Rename category_id to labels because the model expects the columns to be named this way
    tokenized_train = tokenized_train.rename_column("category_id", "labels")
    tokenized_test = tokenized_test.rename_column("category_id", "labels")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Instantiate the model and define training arguments
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_name))
    training_args = TrainingArguments(
        output_dir="artifacts", 
        eval_strategy="epoch",
        logging_dir="artifacts/logs",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3
    )

    # Define metric computation function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
        }

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_test = np.array(predictions.label_ids)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

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
            "dataset": multiclass_dataset_name,
            "model": args.model,
            "max_features": args.max_features,
        },
    }

    metrics_path = output_dir / f"metrics_{args.stage}_{args.model}.json"
    cm_plot_path = output_dir / f"confusion_matrix_{args.stage}_{args.model}.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model("artifacts")
    tokenizer.save_pretrained("artifacts")

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
    print(f"\nSaved model to: artifacts")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

if __name__ == "__main__":
    main()