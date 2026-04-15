# Enron Spam Baseline

A simple baseline binary classifier for the Hugging Face dataset `SetFit/enron_spam`.

## Files

- `data.py`: loads the dataset from Hugging Face and extracts text + labels
- `train_baseline.py`: trains and evaluates a bag-of-words or TF-IDF model with Naive Bayes or Logistic Regression
- `predict.py`: runs inference with a saved model
- `requirements.txt`: Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Train a baseline

### TF-IDF + Logistic Regression

```bash
python train_baseline.py --vectorizer tfidf --model logistic_regression
```

### Bag of Words + Naive Bayes

```bash
python train_baseline.py --vectorizer bow --model naive_bayes
```

## Optional arguments

```bash
python train_baseline.py --vectorizer tfidf --model logistic_regression --max-features 20000 --output-dir artifacts
```

## Run inference

```bash
python predict.py --model-path artifacts/model_tfidf_logistic_regression.joblib --text "Congratulations, you've won a free vacation! Click here now."
```

## Notes

- Labels are treated as `0 = ham`, `1 = spam`.
- The script uses the dataset's provided `train` and `test` splits.
- `CountVectorizer` gives bag-of-words features.
- `TfidfVectorizer` gives TF-IDF features.
- `MultinomialNB` and `LogisticRegression` are both strong baseline models for text classification.

