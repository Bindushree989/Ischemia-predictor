"""Prediction microservice (ML, DL or QML). Choose with MODEL_TYPE env var.
Internal API: POST /predict, GET /health, GET /info. Stateless, read-only (SAD 2.8)."""
import hashlib
import json
import logging
import math
import os
import time
import uuid
import joblib
import numpy as np
from flask import Flask, jsonify, request

from common.features import FEATURES
from common.preprocess import PIPELINE_VERSION

MODEL_TYPE = os.environ.get("MODEL_TYPE", "ML").upper()
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
_cache = {}
log = logging.getLogger("model_service")


class IntegrityError(Exception):
    pass


def _registry():
    with open(os.path.join(MODELS_DIR, "registry.json")) as f:
        return json.load(f)


def load_model(version):
    if version in _cache:
        return _cache[version]
    entry = next((e for e in _registry() if e["type"] == MODEL_TYPE and e["version"] == version
                  and e["status"] == "validated"), None)
    if entry is None:
        raise IntegrityError("model not registered/validated")
    path = os.path.join(MODELS_DIR, os.path.basename(entry["file"]))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    if h.hexdigest() != entry["sha256"]:          # SEC-07 integrity check
        raise IntegrityError("checksum mismatch")
    _cache[version] = joblib.load(path)
    return _cache[version]


def valid_payload(d):
    """Strict schema check (SEC-06): right keys, right length, finite real numbers only."""
    return (isinstance(d, dict)
            and isinstance(d.get("features"), list)
            and len(d["features"]) == len(FEATURES)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in d["features"])
            and isinstance(d.get("pipeline_version"), str)
            and isinstance(d.get("model_version"), str))


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024      # a valid request is < 1 KB

    @app.get("/health")
    def health():
        return jsonify(status="ok", type=MODEL_TYPE)

    @app.get("/info")
    def info():
        """Which model versions this service can serve, with their recorded metrics."""
        try:
            models = [{"version": e["version"], "status": e["status"], "metrics": e.get("metrics"),
                       "pipeline_version": e["pipeline_version"]}
                      for e in _registry() if e["type"] == MODEL_TYPE]
        except OSError:
            return jsonify(error="registry unavailable"), 503
        return jsonify(type=MODEL_TYPE, pipeline_version=PIPELINE_VERSION,
                       features=FEATURES, models=models)

    @app.post("/predict")
    def predict():
        t0, rid = time.perf_counter(), uuid.uuid4().hex[:8]
        d = request.get_json(silent=True)
        if not valid_payload(d):
            return jsonify(error="invalid schema"), 400
        if d["pipeline_version"] != PIPELINE_VERSION:
            return jsonify(error="pipeline mismatch"), 409
        try:
            model = load_model(d["model_version"])
        except (IntegrityError, FileNotFoundError, OSError) as e:
            log.warning("req=%s load failed: %s", rid, e)
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
        ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("req=%s type=%s version=%s %.1fms", rid, MODEL_TYPE, d["model_version"], ms)  # no patient data (FR-18)
        return jsonify(label="Ischemia Likely" if p >= 0.5 else "Ischemia Unlikely",
                       probability=round(p * 100, 1), importances=[i[1] for i in imps[:5]],
                       model_version=d["model_version"], model_type=MODEL_TYPE, latency_ms=ms)
    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
