from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import features, make_sequences


def _scores(y, probability):
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


def evaluate(seed: int = 27):
    train_seq, _, train_y = make_sequences(1800, seed=seed)
    test_seq, group, test_y = make_sequences(700, seed=seed + 1)
    report = {}
    for name, trend_aware in {"last_observation": False, "trend_aware": True}.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(features(train_seq, trend_aware), train_y)
        probability = model.predict_proba(features(test_seq, trend_aware))[:, 1]
        report[name] = {"overall": _scores(test_y, probability)}
        report[name]["younger_group"] = _scores(test_y[group == 0], probability[group == 0])
        report[name]["older_group"] = _scores(test_y[group == 1], probability[group == 1])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = evaluate(args.seed)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
