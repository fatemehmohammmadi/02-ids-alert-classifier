# 02 — IDS Alert Classifier

Supervised classification of IDS-style alerts into **benign** vs **attack** using classical sklearn models.

## Learning goals

- Build Suricata/Snort-like alert tables
- Encode mixed categorical/numeric features
- Compare Logistic Regression vs Random Forest
- Report Precision / Recall / F1 / confusion matrix (SOC-critical)

## Layout

```
02-ids-alert-classifier/
├── README.md
├── requirements.txt
├── generate_alerts.py
├── train_classifier.py
├── data/ids_alerts.csv
└── outputs/
    ├── cm_logreg.png
    ├── cm_rf.png
    ├── metrics.json
    └── best_model.txt
```

## Setup & run

```bash
cd 02-ids-alert-classifier
pip install -r requirements.txt
python generate_alerts.py
python train_classifier.py
```

## Sample run (committed)

| Artifact | Description |
|----------|-------------|
| `outputs/metrics.json` | Per-model precision/recall/F1 |
| `outputs/cm_*.png` | Confusion matrices |
| `outputs/best_model.txt` | Winning model name + macro F1 |

On synthetic data, both models typically reach near-perfect scores (easy separable features by design).
