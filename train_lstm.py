from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data import load_data_splits, get_text_and_labels
from train_utils import save_confusion_matrix_plot, save_roc_curve_plot, STAGES


class TextDataset(Dataset):
    """PyTorch Dataset for tokenized email sequences."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SpamLSTM(nn.Module):
    """LSTM model for binary spam classification."""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.5):
        super(SpamLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        dropped = self.dropout(hidden_concat)
        out = self.fc(dropped)
        return self.sigmoid(out).squeeze()


def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    """Build vocabulary from training texts."""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(max_vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def texts_to_sequences(texts, vocab, max_length=200):
    """Convert texts to padded integer sequences."""
    sequences = []
    for text in texts:
        words = text.lower().split()
        sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
        
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))
        
        sequences.append(sequence)
    
    return np.array(sequences)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy().astype(int))
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM spam classifier on SetFit/enron_spam.")
    parser.add_argument("--stage", choices=STAGES, default="stage_2")
    parser.add_argument("--max-vocab", type=int, default=20000, help="Maximum vocabulary size")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    enron_dataset = "SetFit/enron_spam"
    train_split, test_split = load_data_splits(enron_dataset)
    X_train, y_train = get_text_and_labels(train_split)
    X_test, y_test = get_text_and_labels(test_split)
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    print("Building vocabulary...")
    vocab = build_vocab(X_train, max_vocab_size=args.max_vocab)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    print("Converting texts to sequences...")
    X_train_seq = texts_to_sequences(X_train, vocab, max_length=args.max_length)
    X_test_seq = texts_to_sequences(X_test, vocab, max_length=args.max_length)
    
    train_dataset = TextDataset(X_train_seq, y_train)
    test_dataset = TextDataset(X_test_seq, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Initializing model...")
    model = SpamLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 40)
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, y_probs = evaluate(model, test_loader, device)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
    
    model.load_state_dict(best_model_state)
    
    print("\nFinal evaluation on test set...")
    y_true, y_pred, y_probs = evaluate(model, test_loader, device)
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    roc_auc = roc_auc_score(y_true, y_probs)
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
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
            y_true, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0
        ),
        "config": {
            "dataset": "SetFit/enron_spam",
            "model": "lstm",
            "max_vocab": args.max_vocab,
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
        },
    }
    
    metrics_path = output_dir / f"metrics_{args.stage}_lstm.json"
    model_path = output_dir / f"model_{args.stage}_lstm.pth"
    vocab_path = output_dir / f"vocab_{args.stage}_lstm.joblib"
    cm_plot_path = output_dir / f"confusion_matrix_{args.stage}_lstm.png"
    roc_plot_path = output_dir / f"roc_curve_{args.stage}_lstm.png"
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    torch.save({
        'model_state_dict': best_model_state,
        'config': metrics['config']
    }, model_path)
    
    joblib.dump(vocab, vocab_path)
    
    label_names = ["ham", "spam"]
    save_confusion_matrix_plot(cm, cm_plot_path, label_names)
    save_roc_curve_plot(y_true, y_probs, roc_plot_path)
    
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
    print(f"Saved vocabulary to: {vocab_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")
    print(f"Saved ROC curve plot to: {roc_plot_path}")


if __name__ == "__main__":
    main()