from __future__ import annotations

import argparse

import joblib


LABEL_MAP = {0: "ham", 1: "spam"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a saved spam classifier.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    pred = int(model.predict([args.text])[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([args.text])[0]
        print({"prediction": LABEL_MAP[pred], "ham_prob": float(proba[0]), "spam_prob": float(proba[1])})
    else:
        print({"prediction": LABEL_MAP[pred]})


if __name__ == "__main__":
    main()