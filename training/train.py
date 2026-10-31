"""Offline training pipeline (SAD: Dataset -> Training pipeline -> exported models).

Usage:
  python -m training.train --data data/heart.csv --models models [--force-validate]
"""
import argparse
import hashlib
import json
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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
    ap.add_argument("--experiment", default=os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "ischemia-predictor"))
    a = ap.parse_args()
    os.makedirs(a.models, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(a.experiment)
    dataset_sha256 = sha256(a.data)

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
        probabilities = model.predict_proba(Zte)[:, 1]
        pred = (probabilities >= 0.5).astype(int)
        acc = accuracy_score(yte, pred)
        rec = recall_score(yte, pred)
        precision = precision_score(yte, pred, zero_division=0)
        f1 = f1_score(yte, pred, zero_division=0)
        roc_auc = roc_auc_score(yte, probabilities)
        ok = acc >= 0.85 and rec >= 0.85
        version = f"{mtype.lower()}-1.0"
        fn = f"{version}.joblib"
        joblib.dump(model, os.path.join(a.models, fn))
        status = "validated" if (ok or a.force_validate) else "unvalidated"
        print(f"{mtype}: accuracy={acc:.3f} recall={rec:.3f} -> {status}")
        model_path = os.path.join(a.models, fn)
        metrics = {
            "accuracy": round(float(acc), 4),
            "recall": round(float(rec), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
        }
        entry = {"type": mtype, "version": version, "file": fn,
                 "sha256": sha256(model_path),
                 "pipeline_version": PIPELINE_VERSION, "status": status,
                 "is_active": status == "validated",
                 "metrics": metrics}
        registry.append(entry)
        if mtype == "ML":
            model_params = {"n_estimators": 400, "min_samples_leaf": 2}
        elif mtype == "DL":
            model_params = {
                "hidden_layer_sizes": str(grid.best_params_["hidden_layer_sizes"]),
                "alpha": grid.best_params_["alpha"],
                "max_iter": 2000,
                "cross_validation_folds": 5,
            }
        else:
            model_params = {
                "qml_samples": n,
                "qubits": model.n,
                "layers": model.layers,
                "steps": model.steps,
                "learning_rate": model.lr,
                "features": ",".join(FEATURES[i] for i in model.feature_idx),
            }
        with mlflow.start_run(run_name=f"{mtype.lower()}-{version}"):
            mlflow.set_tags({
                "model_type": mtype,
                "model_version": version,
                "pipeline_version": PIPELINE_VERSION,
                "validation_status": status,
            })
            mlflow.log_params({
                "dataset": os.path.basename(a.data),
                "dataset_sha256": dataset_sha256,
                "random_state": 42,
                "force_validate": a.force_validate,
                "decision_threshold": 0.5,
                **model_params,
            })
            mlflow.log_metrics({**metrics, "validated": float(status == "validated")})
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(pipeline_path(a.models), artifact_path="preprocessing")
            mlflow.log_dict(entry, "registry_entry.json")

    with open(os.path.join(a.models, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    print("Done. Registry written to", os.path.join(a.models, "registry.json"))


if __name__ == "__main__":
    main()
