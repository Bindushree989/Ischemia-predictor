"""Offline training pipeline (SAD: Dataset -> Training pipeline -> exported models).

Usage:
  python -m training.train --data data/heart.csv --models models [--qubits 4] [--force-validate]

Outputs in --models: pipeline-1.0.joblib, ml/dl/qml-1.0.joblib, registry.json,
heldout.csv (for TC-22) and training_report.json (metrics, CV scores, provenance).
"""
import argparse
import datetime
import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from common.features import FEATURES, load_csv
from common.preprocess import PIPELINE_VERSION, build_pipeline, pipeline_path
from common.qml_model import QMLClassifier


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, Z, y):
    prob = model.predict_proba(Z)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred), 4),
            "roc_auc": round(roc_auc_score(y, prob), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def cv_scores(estimator, X, y, seed):
    """5-fold CV with preprocessing refitted inside every fold (no leakage)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = {}
    for name in ("accuracy", "recall"):
        s = cross_val_score(make_pipeline(build_pipeline(), estimator), X, y, cv=cv, scoring=name)
        out[name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qubits", type=int, default=4, help="number of features fed to the QML circuit")
    ap.add_argument("--qml-epochs", type=int, default=15)
    ap.add_argument("--force-validate", action="store_true",
                    help="mark models validated even if < 85%% (report the real numbers)")
    a = ap.parse_args()
    os.makedirs(a.models, exist_ok=True)

    df = load_csv(a.data)
    print("records:", len(df), "positives:", int(df.label.sum()), "negatives:", int((df.label == 0).sum()))
    X, y = df[FEATURES].values, df.label.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)

    pipe = build_pipeline().fit(Xtr)
    joblib.dump(pipe, pipeline_path(a.models))
    Ztr, Zte = pipe.transform(Xtr), pipe.transform(Xte)
    pd.DataFrame(np.c_[Xte, yte], columns=FEATURES + ["label"]).to_csv(
        os.path.join(a.models, "heldout.csv"), index=False)

    ml = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=a.seed)
    grid = GridSearchCV(MLPClassifier(max_iter=2000, random_state=a.seed),
                        {"hidden_layer_sizes": [(16,), (32,)], "alpha": [0.1, 1.0, 3.0]},
                        cv=5, scoring="accuracy").fit(Ztr, ytr)
    print("DL best params:", grid.best_params_)
    dl = grid.best_estimator_

    print("Cross-validating ML and DL on the training split ...")
    cv = {"ML": cv_scores(ml, Xtr, ytr, a.seed),
          "DL": cv_scores(MLPClassifier(max_iter=2000, random_state=a.seed, **grid.best_params_),
                          Xtr, ytr, a.seed)}
    ml.fit(Ztr, ytr)

    top = np.argsort(ml.feature_importances_)[::-1][:a.qubits]
    print("Training QML on", len(Ztr), "samples, qubits:", [FEATURES[i] for i in top])
    qml_m = QMLClassifier(top, epochs=a.qml_epochs, seed=a.seed).fit(Ztr, ytr)

    registry, report = [], {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": sha256(a.data), "rows": int(len(df)), "seed": a.seed,
        "train_rows": int(len(Ztr)), "test_rows": int(len(Zte)),
        "dl_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in grid.best_params_.items()},
        "qml": {"features": [FEATURES[i] for i in top], "loss_history": [round(l, 4) for l in qml_m.loss_history_]},
        "models": {}}
    for mtype, model in (("ML", ml), ("DL", dl), ("QML", qml_m)):
        m = evaluate(model, Zte, yte)
        ok = m["accuracy"] >= 0.85 and m["recall"] >= 0.85
        version = f"{mtype.lower()}-1.0"
        fn = f"{version}.joblib"
        joblib.dump(model, os.path.join(a.models, fn))
        status = "validated" if (ok or a.force_validate) else "unvalidated"
        extra = f"  CV acc {cv[mtype]['accuracy']['mean']:.3f}±{cv[mtype]['accuracy']['std']:.3f}" if mtype in cv else ""
        print(f"{mtype}: accuracy={m['accuracy']:.3f} recall={m['recall']:.3f} "
              f"auc={m['roc_auc']:.3f}{extra} -> {status}")
        registry.append({"type": mtype, "version": version, "file": fn,
                         "sha256": sha256(os.path.join(a.models, fn)),
                         "pipeline_version": PIPELINE_VERSION, "status": status,
                         "is_active": status == "validated", "metrics": m})
        report["models"][mtype] = {"heldout": m, "cv": cv.get(mtype), "status": status}
    with open(os.path.join(a.models, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    with open(os.path.join(a.models, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Done. Wrote registry.json and training_report.json to", a.models)


if __name__ == "__main__":
    main()
