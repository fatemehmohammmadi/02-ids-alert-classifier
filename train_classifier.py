"""Train and evaluate IDS alert classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent / "data" / "ids_alerts.csv"
OUT_DIR = Path(__file__).parent / "outputs"

CAT = ["signature", "proto"]
NUM = ["src_bytes", "dst_bytes", "duration_s", "failed_logins", "same_srv_rate", "dst_host_count", "severity"]


def build_pipeline(model) -> Pipeline:
    # keep preprocessing inside the pipeline so train/serve stay aligned
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler(), NUM),
        ]
    )
    return Pipeline([("pre", pre), ("clf", model)])


def plot_cm(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=["benign", "attack"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["benign", "attack"], yticklabels=["benign", "attack"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run generate_alerts.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df[CAT + NUM]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    from sklearn.metrics import f1_score, precision_recall_fscore_support

    best_name, best_f1, best_pipe = None, -1.0, None
    metrics: dict = {}
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        report = classification_report(y_test, pred, digits=3)
        print(f"\n===== {name} =====\n{report}")
        plot_cm(y_test, pred, f"Confusion — {name}", OUT_DIR / f"cm_{name}.png")

        f1 = f1_score(y_test, pred, average="macro")
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        metrics[name] = {
            "precision_macro": round(float(p), 4),
            "recall_macro": round(float(r), 4),
            "f1_macro": round(float(f), 4),
            "classification_report": report,
        }
        if f1 > best_f1:
            best_name, best_f1, best_pipe = name, f1, pipe

    joblib.dump(best_pipe, OUT_DIR / "model.joblib")
    metrics["best"] = {"name": best_name, "macro_f1": round(float(best_f1), 4)}
    import json

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "best_model.txt").write_text(f"{best_name} macro_f1={best_f1:.4f}\n", encoding="utf-8")
    print(f"\nBest model: {best_name} (macro F1={best_f1:.4f})")
    print(f"Saved -> {OUT_DIR / 'model.joblib'}")
    print(f"Saved -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
