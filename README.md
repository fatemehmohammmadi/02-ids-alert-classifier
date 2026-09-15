# IDS Alert Classifier

Train a simple supervised model on Suricata/Snort-style alert fields and separate `benign` vs `attack`.

This is the kind of baseline I spin up before trying deep models — categorical signatures + a handful of numeric counters, then Logistic Regression / Random Forest.

## Layout

```
generate_alerts.py      # synthetic alert table
train_classifier.py     # train + metrics + confusion matrices
data/ids_alerts.csv
outputs/
  metrics.json
  cm_logreg.png
  cm_rf.png
  best_model.txt
```

## Setup

```bash
pip install -r requirements.txt
python generate_alerts.py
python train_classifier.py
```

Python 3.10+ recommended.

## Features used

- Categorical: `signature`, `proto`
- Numeric: bytes, duration, failed logins, same-srv rate, dst host count, severity

One-hot + scaling goes through a sklearn `Pipeline`, so the same object can be dumped with joblib for later inference (local only; `.joblib` is ignored in git because it's bulky).

## Sample run

On the included synthetic set both models hit very high scores — the classes are intentionally separable so the pipeline is easy to demo. Check `outputs/metrics.json` and the confusion matrix PNGs.

> Real IDS data is messier. Expect class imbalance, concept drift, and signature churn.

## Why bother

- Good interview / portfolio piece for Security Data Science
- Easy to extend to multi-class (DoS / Probe / R2L / …)
- Clear metrics that SOC folks actually care about (precision/recall on `attack`)

## Possible next steps

- Pull CIC-IDS or UNSW-NB15 and redo the same pipeline
- Calibrate probabilities for alert thresholds
- Tiny FastAPI wrapper for scoring a single JSON alert

## License

MIT
