"""Offline training pipeline (SAD: Dataset -> Training pipeline -> exported models).

Usage:
  python -m training.train --data data/heart.csv --models models [--qubits 4] [--force-validate]

Outputs in --models: pipeline-1.0.joblib, six model artifacts, registry.json,
heldout.csv (for TC-22), training_report.json, and six MLflow runs.
"""
import argparse
import datetime
import hashlib
import json
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

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
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "recall": round(float(recall_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, prob)), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def cv_scores(estimator, X, y, seed):
    """5-fold CV with preprocessing refitted inside every fold (no leakage)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = {}
    for name in ("accuracy", "recall"):
        s = cross_val_score(make_pipeline(build_pipeline(), estimator), X, y, cv=cv, scoring=name)
        out[name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
    return out


def model_parameters(model, model_type, qml_samples=None):
    """Return small, serializable parameter dictionaries for reports and MLflow."""
    if isinstance(model, QMLClassifier):
        return {"qml_samples": qml_samples, "qubits": model.n, "layers": model.layers,
                "epochs": model.epochs, "learning_rate": model.lr,
                "features": ",".join(FEATURES[i] for i in model.feature_idx)}
    parameters = model.get_params(deep=False)
    if model_type == "ML" and isinstance(model, RandomForestClassifier):
        return {"n_estimators": parameters["n_estimators"],
                "min_samples_leaf": parameters["min_samples_leaf"]}
    if model_type == "ML":
        return {"n_estimators": parameters["n_estimators"], "max_depth": parameters["max_depth"],
                "learning_rate": parameters["learning_rate"], "subsample": parameters["subsample"]}
    if model_type == "DL":
        return {"hidden_layer_sizes": str(parameters["hidden_layer_sizes"]),
                "alpha": parameters["alpha"], "max_iter": parameters["max_iter"],
                "cross_validation_folds": 5}
    raise ValueError(f"Unsupported model type for parameters: {model_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qubits", type=int, default=4, help="number of features fed to the QML circuit")
    ap.add_argument("--qml-epochs", type=int, default=15)
    ap.add_argument("--qml-samples", type=int, default=300)
    ap.add_argument("--experiment", default=os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "ischemia-predictor"))
    ap.add_argument("--force-validate", action="store_true",
                    help="mark models validated even if < 85%% (report the real numbers)")
    a = ap.parse_args()
    os.makedirs(a.models, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(a.experiment)
    dataset_sha256 = sha256(a.data)

    df = load_csv(a.data)
    print("records:", len(df), "positives:", int(df.label.sum()), "negatives:", int((df.label == 0).sum()))
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)

    pipe = build_pipeline().fit(Xtr)
    joblib.dump(pipe, pipeline_path(a.models))
    Ztr, Zte = pipe.transform(Xtr), pipe.transform(Xte)
    pd.DataFrame(np.c_[Xte, yte], columns=FEATURES + ["label"]).to_csv(
        os.path.join(a.models, "heldout.csv"), index=False)

    ml = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=a.seed)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
        random_state=a.seed, n_jobs=1,
    )
    grid = GridSearchCV(MLPClassifier(max_iter=2000, random_state=a.seed),
                        {"hidden_layer_sizes": [(16,), (32,)], "alpha": [0.1, 1.0, 3.0]},
                        cv=5, scoring="accuracy").fit(Ztr, ytr)
    print("DL best params:", grid.best_params_)
    dl = grid.best_estimator_
    deep_dl = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), alpha=0.1, max_iter=2000,
        early_stopping=True, random_state=a.seed,
    )

    print("Cross-validating classical ML and DL models on the training split ...")
    cv = {
        "ml-1.0": cv_scores(ml, Xtr, ytr, a.seed),
        "ml-2.0": cv_scores(xgb, Xtr, ytr, a.seed),
        "dl-1.0": cv_scores(
            MLPClassifier(max_iter=2000, random_state=a.seed, **grid.best_params_),
            Xtr, ytr, a.seed,
        ),
        "dl-2.0": cv_scores(deep_dl, Xtr, ytr, a.seed),
    }
    ml.fit(Ztr, ytr)
    xgb.fit(Ztr, ytr)
    deep_dl.fit(Ztr, ytr)

    top = np.argsort(ml.feature_importances_)[::-1][:a.qubits]
    n = min(a.qml_samples, len(Ztr))
    qml_rng = np.random.default_rng(a.seed)
    qml_indices = qml_rng.choice(len(Ztr), n, replace=False)
    print("Training QML on", n, "samples, qubits:", [FEATURES[i] for i in top])
    qml_m = QMLClassifier(top, epochs=a.qml_epochs, seed=a.seed).fit(
        Ztr[qml_indices], ytr[qml_indices])
    qml_deep = QMLClassifier(top, layers=5, epochs=a.qml_epochs, seed=a.seed + 1).fit(
        Ztr[qml_indices], ytr[qml_indices])

    registry, report = [], {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": sha256(a.data), "rows": int(len(df)), "seed": a.seed,
        "train_rows": int(len(Ztr)), "test_rows": int(len(Zte)),
        "dl_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in grid.best_params_.items()},
        "qml": {
            "features": [FEATURES[i] for i in top],
            "loss_history": {
                "qml-1.0": [round(l, 4) for l in qml_m.loss_history_],
                "qml-2.0": [round(l, 4) for l in qml_deep.loss_history_],
            },
        },
        "models": {}}
    candidates = (
        ("ML", "ml-1.0", "scikit-learn", "Random Forest", ml, None),
        ("ML", "ml-2.0", "XGBoost", "XGBoost Classifier", xgb, None),
        ("DL", "dl-1.0", "scikit-learn", "Tuned MLP", dl, None),
        ("DL", "dl-2.0", "scikit-learn", "Deep MLP (64-32-16)", deep_dl, None),
        ("QML", "qml-1.0", "PennyLane", "Variational Classifier (3 layers)", qml_m, n),
        ("QML", "qml-2.0", "PennyLane", "Variational Classifier (5 layers)", qml_deep, n),
    )
    for mtype, version, framework, algorithm, model, qml_samples in candidates:
        m = evaluate(model, Zte, yte)
        ok = m["accuracy"] >= 0.85 and m["recall"] >= 0.85
        fn = f"{version}.joblib"
        model_path = os.path.join(a.models, fn)
        joblib.dump(model, model_path)
        status = "validated" if (ok or a.force_validate) else "unvalidated"
        model_cv = cv.get(version)
        extra = (f"  CV acc {model_cv['accuracy']['mean']:.3f}±"
                 f"{model_cv['accuracy']['std']:.3f}") if model_cv else ""
        print(f"{mtype}: accuracy={m['accuracy']:.3f} recall={m['recall']:.3f} "
              f"auc={m['roc_auc']:.3f}{extra} -> {status}")
        entry = {"type": mtype, "version": version, "framework": framework,
                 "algorithm": algorithm, "file": fn,
                 "sha256": sha256(model_path),
                 "pipeline_version": PIPELINE_VERSION, "status": status,
                 "is_active": status == "validated" and version.endswith("-1.0"),
                 "metrics": m}
        registry.append(entry)
        report["models"][version] = {
            "type": mtype, "framework": framework, "algorithm": algorithm,
            "heldout": m, "cv": model_cv, "status": status, "is_active": False,
        }

        report["models"][version]["is_active"] = entry["is_active"]

    for entry, candidate in zip(registry, candidates):
        mtype, version, framework, algorithm, model, qml_samples = candidate
        model_path = os.path.join(a.models, entry["file"])
        model_params = model_parameters(model, mtype, qml_samples)
        metrics = {key: float(value) for key, value in entry["metrics"].items()
                   if isinstance(value, (int, float))}
        metrics.update({key: float(value)
                        for key, value in entry["metrics"]["confusion"].items()})
        with mlflow.start_run(run_name=f"{mtype.lower()}-{version}"):
            mlflow.set_tags({
                "model_type": mtype,
                "model_version": version,
                "framework": framework,
                "algorithm": algorithm,
                "pipeline_version": PIPELINE_VERSION,
                "validation_status": entry["status"],
            })
            mlflow.log_params({
                "dataset": os.path.basename(a.data),
                "dataset_sha256": dataset_sha256,
                "seed": a.seed,
                "experiment": a.experiment,
                "force_validate": a.force_validate,
                "decision_threshold": 0.5,
                **model_params,
            })
            mlflow.log_metrics(metrics)
            mlflow.log_metric("validated", float(entry["status"] == "validated"))
            mlflow.log_metric("active", float(entry["is_active"]))
            if version in cv:
                mlflow.log_metrics({
                    f"cv_{metric}_{stat}": value
                    for metric, scores in cv[version].items()
                    for stat, value in scores.items()
                })
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(pipeline_path(a.models), artifact_path="preprocessing")
            mlflow.log_dict(entry, "registry_entry.json")
    with open(os.path.join(a.models, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    with open(os.path.join(a.models, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Done. Wrote registry.json and training_report.json to", a.models)


if __name__ == "__main__":
    main()
