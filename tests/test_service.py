"""Tests for the prediction microservice. Run: pytest -q"""
import json
import numpy as np
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from common.features import FEATURES
from common.preprocess import PIPELINE_VERSION
from tests.util import sha


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, len(FEATURES)))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    joblib.dump(model, tmp_path / "ml-1.0.joblib")
    (tmp_path / "registry.json").write_text(json.dumps([{
        "type": "ML", "version": "ml-1.0", "file": "ml-1.0.joblib", "sha256": sha(tmp_path / "ml-1.0.joblib"),
        "pipeline_version": PIPELINE_VERSION, "status": "validated", "metrics": {"accuracy": 0.9}}]))
    from services import model_service as ms
    monkeypatch.setattr(ms, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(ms, "_cache", {})
    return ms.create_app().test_client(), tmp_path


def body(**over):
    d = {"features": [0.1] * len(FEATURES), "pipeline_version": PIPELINE_VERSION, "model_version": "ml-1.0"}
    d.update(over)
    return d


def test_predict_ok(svc):
    r = svc[0].post("/predict", json=body())
    j = r.get_json()
    assert r.status_code == 200 and 0 <= j["probability"] <= 100 and len(j["importances"]) == 5
    assert j["model_type"] == "ML" and j["latency_ms"] >= 0


def test_rejects_nan_and_inf(svc):
    for bad in ("NaN", "Infinity", "-Infinity"):
        raw = '{"features":[%s%s],"pipeline_version":"%s","model_version":"ml-1.0"}' % (
            bad, ",0" * (len(FEATURES) - 1), PIPELINE_VERSION)
        r = svc[0].post("/predict", data=raw, content_type="application/json")
        assert r.status_code == 400


def test_rejects_wrong_length_and_types(svc):
    assert svc[0].post("/predict", json=body(features=[0.1])).status_code == 400
    assert svc[0].post("/predict", json=body(features=["a"] * len(FEATURES))).status_code == 400
    assert svc[0].post("/predict", json=body(features=[True] * len(FEATURES))).status_code == 400


def test_pipeline_mismatch(svc):
    assert svc[0].post("/predict", json=body(pipeline_version="pipeline-9")).status_code == 409


def test_oversized_request(svc):
    assert svc[0].post("/predict", data=b"x" * 50000, content_type="application/json").status_code == 413


def test_tampered_model_blocked(svc):
    c, path = svc
    (path / "ml-1.0.joblib").write_bytes(b"tampered")
    assert c.post("/predict", json=body()).status_code == 503


def test_info(svc):
    j = svc[0].get("/info").get_json()
    assert j["type"] == "ML" and j["models"][0]["metrics"]["accuracy"] == 0.9
