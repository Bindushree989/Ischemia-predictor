"""Prediction microservice (ML, DL or QML). Choose with MODEL_TYPE env var.
Internal API: POST /predict, GET /health. Stateless, read-only (SAD 2.8)."""
import hashlib
import json
import os
import joblib
import numpy as np
from flask import Flask, jsonify, request

from common.features import FEATURES
from common.preprocess import PIPELINE_VERSION

MODEL_TYPE = os.environ.get("MODEL_TYPE", "ML").upper()
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
_cache = {}


class IntegrityError(Exception):
    pass


def load_model(version):
    if version in _cache:
        return _cache[version]
    with open(os.path.join(MODELS_DIR, "registry.json")) as f:
        reg = json.load(f)
    entry = next((e for e in reg if e["type"] == MODEL_TYPE and e["version"] == version
                  and e["status"] == "validated"), None)
    if entry is None:
        raise IntegrityError("model not registered/validated")
    path = os.path.join(MODELS_DIR, os.path.basename(entry["file"]))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    if h.hexdigest() != entry["sha256"]:          # SEC-07
        raise IntegrityError("checksum mismatch")
    _cache[version] = joblib.load(path)
    return _cache[version]


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok", type=MODEL_TYPE)

    @app.post("/predict")
    def predict():
        d = request.get_json(silent=True)
        ok = (isinstance(d, dict) and isinstance(d.get("features"), list)
              and len(d["features"]) == len(FEATURES)
              and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in d["features"])
              and isinstance(d.get("pipeline_version"), str)
              and isinstance(d.get("model_version"), str))
        if not ok:                                  # SEC-06 schema validation
            return jsonify(error="invalid schema"), 400
        if d["pipeline_version"] != PIPELINE_VERSION:
            return jsonify(error="pipeline mismatch"), 409
        try:
            model = load_model(d["model_version"])
        except (IntegrityError, FileNotFoundError, OSError) as e:
            return jsonify(error=str(e)), 503
        x = np.array([d["features"]], dtype=float)
        p = float(model.predict_proba(x)[0, 1])
        imps = []
        for i, name in enumerate(FEATURES):         # leave-one-out (set to mean) attribution
            x2 = x.copy()
            x2[0, i] = 0.0
            delta = p - float(model.predict_proba(x2)[0, 1])
            imps.append((abs(delta), {"name": name,
                                      "direction": "increases" if delta >= 0 else "decreases"}))
        imps.sort(key=lambda t: -t[0])
        return jsonify(label="Ischemia Likely" if p >= 0.5 else "Ischemia Unlikely",
                       probability=round(p * 100, 1), importances=[i[1] for i in imps[:5]],
                       model_version=d["model_version"])
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
