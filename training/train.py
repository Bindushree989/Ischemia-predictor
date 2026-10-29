"""Offline training pipeline (SAD: Dataset -> Training pipeline -> exported models).

Usage:
  python -m training.train --data data/heart.csv --models models [--force-validate]
"""
import argparse
import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

from common.features import FEATURES, load_csv
from common.preprocess import PIPELINE_VERSION, build_pipeline, pipeline_path
from common.qml_model import QMLClassifier


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--force-validate", action="store_true",
                    help="mark models validated even if < 85%% (report real numbers in the test report)")
    ap.add_argument("--qml-samples", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.models, exist_ok=True)

    df = load_csv(a.data)
    print("records:", len(df), "positives:", int(df.label.sum()), "negatives:", int((df.label == 0).sum()))
    X, y = df[FEATURES].values, df.label.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_pipeline().fit(Xtr)
    joblib.dump(pipe, pipeline_path(a.models))
    Ztr, Zte = pipe.transform(Xtr), pipe.transform(Xte)
    pd.DataFrame(np.c_[Xte, yte], columns=FEATURES + ["label"]).to_csv(
        os.path.join(a.models, "heldout.csv"), index=False)  # for TC-22

    ml = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42).fit(Ztr, ytr)
    # DL: small MLP, regularisation chosen by 5-fold CV on the training split only
    grid = GridSearchCV(MLPClassifier(max_iter=2000, random_state=42),
                        {"hidden_layer_sizes": [(16,), (32,)], "alpha": [0.1, 1.0, 3.0]},
                        cv=5, scoring="accuracy").fit(Ztr, ytr)
    dl = grid.best_estimator_
    print("DL best params:", grid.best_params_)
    top4 = np.argsort(ml.feature_importances_)[::-1][:4]
    rng = np.random.default_rng(42)
    n = min(a.qml_samples, len(Ztr))
    idx = rng.choice(len(Ztr), n, replace=False)
    print("Training QML on", n, "samples, qubits:", [FEATURES[i] for i in top4])
    qml_m = QMLClassifier(top4).fit(Ztr[idx], ytr[idx])

    registry = []
    for mtype, model in (("ML", ml), ("DL", dl), ("QML", qml_m)):
        pred = (model.predict_proba(Zte)[:, 1] >= 0.5).astype(int)
        acc, rec = accuracy_score(yte, pred), recall_score(yte, pred)
        ok = acc >= 0.85 and rec >= 0.85
        version = f"{mtype.lower()}-1.0"
        fn = f"{version}.joblib"
        joblib.dump(model, os.path.join(a.models, fn))
        status = "validated" if (ok or a.force_validate) else "unvalidated"
        print(f"{mtype}: accuracy={acc:.3f} recall={rec:.3f} -> {status}")
        registry.append({"type": mtype, "version": version, "file": fn,
                         "sha256": sha256(os.path.join(a.models, fn)),
                         "pipeline_version": PIPELINE_VERSION, "status": status,
                         "is_active": status == "validated",
                         "metrics": {"accuracy": round(acc, 4), "recall": round(rec, 4)}})
    with open(os.path.join(a.models, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    print("Done. Registry written to", os.path.join(a.models, "registry.json"))


if __name__ == "__main__":
    main()
