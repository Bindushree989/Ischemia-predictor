"""Prediction microservice (ML, DL or QML). Choose with MODEL_TYPE env var.
Internal API: POST /predict, GET /health. Stateless, read-only (SAD 2.8).

One instance serves one model type. The gateway (``web/app.py``) sends it
already-preprocessed features plus the model version to use; the service loads
that version, checks its integrity, and returns a probability with a short
feature-attribution list.

Configuration (environment variables)
-------------------------------------
``MODEL_TYPE``  ``ML`` | ``DL`` | ``QML`` (default ``ML``); only registry entries
                of this type can be served.
``MODELS_DIR``  Folder with ``registry.json`` and model files (default ``models``).
``PORT``        Port when started directly (default ``5001``).

Endpoints
---------
``GET /health``
    200 ``{"status": "ok", "type": MODEL_TYPE}``

``POST /predict``
    Body: ``{"features": [float x len(FEATURES)], "pipeline_version": str,
    "model_version": str}``. Features must already be transformed by the shared
    preprocessing pipeline.

    200 ``{"label", "probability" (percent, 1 decimal), "importances" (top 5),
    "model_version"}``
    400 invalid schema, 409 pipeline version mismatch,
    503 model not registered/validated, checksum mismatch, or file unreadable.

Security note: this is an internal service with no authentication. Run it only
on a trusted network and expose it to the gateway alone.
"""
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
# Loaded models keyed by version. Per process and never invalidated, so restart
# the service after changing a model file or its registry entry.
_cache = {}


class IntegrityError(Exception):
    """Raised when a model is not registered/validated or fails its checksum."""


def load_model(version):
    """Load a validated model by version, verifying its SHA-256 first.

    Looks up ``version`` in ``MODELS_DIR/registry.json`` (matching this
    service's ``MODEL_TYPE`` and status ``validated``), hashes the model file and
    compares it with the registered checksum (SEC-07) before unpickling it.
    Results are cached per version.

    Args:
        version: Model version string, as registered.

    Returns:
        The loaded model object (must support ``predict_proba``).

    Raises:
        IntegrityError: not registered/validated for this type, or checksum mismatch.
        OSError: registry or model file missing or unreadable
            (``FileNotFoundError`` is a subclass).

    Note:
        ``joblib.load`` runs pickle code. The checksum check is what makes
        loading acceptable, so never skip it.
    """
    if version in _cache:
        return _cache[version]
    with open(os.path.join(MODELS_DIR, "registry.json")) as f:
        reg = json.load(f)
    entry = next((e for e in reg if e["type"] == MODEL_TYPE and e["version"] == version
                  and e["status"] == "validated"), None)
    if entry is None:
        raise IntegrityError("model not registered/validated")
    path = os.path.join(MODELS_DIR, os.path.basename(entry["file"]))  # basename blocks path traversal
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    if h.hexdigest() != entry["sha256"]:          # SEC-07
        raise IntegrityError("checksum mismatch")
    _cache[version] = joblib.load(path)
    return _cache[version]


def create_app():
    """Application factory. Returns a Flask app exposing ``/health`` and ``/predict``."""
    app = Flask(__name__)

    @app.get("/health")
    def health():
        """Liveness check. Returns the status and the model type this instance serves."""
        return jsonify(status="ok", type=MODEL_TYPE)

    @app.post("/predict")
    def predict():
        """Score one preprocessed sample.

        Validates the request schema, checks the pipeline version, loads the
        requested model, then returns the label, probability (percent) and the
        top 5 features by leave-one-out attribution.
        """
        d = request.get_json(silent=True)
        # Strict schema check: right number of numeric (non-bool) features and
        # string versions.
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
        p = float(model.predict_proba(x)[0, 1])  # probability of the positive class (0..1)
        imps = []
        # Attribution: reset one feature at a time to 0.0 and see how the
        # probability moves. Inputs are standardised, so 0.0 is the training mean.
        # "direction" says whether the feature's actual value pushes risk up or down.
        for i, name in enumerate(FEATURES):         # leave-one-out (set to mean) attribution
            x2 = x.copy()
            x2[0, i] = 0.0
            delta = p - float(model.predict_proba(x2)[0, 1])
            imps.append((abs(delta), {"name": name,
                                      "direction": "increases" if delta >= 0 else "decreases"}))
        imps.sort(key=lambda t: -t[0])  # largest effect first
        return jsonify(label="Ischemia Likely" if p >= 0.5 else "Ischemia Unlikely",
                       probability=round(p * 100, 1), importances=[i[1] for i in imps[:5]],
                       model_version=d["model_version"])
    return app


# Module-level app for WSGI servers (e.g. ``gunicorn service:app``).
app = create_app()

if __name__ == "__main__":
    # Dev server only; 0.0.0.0 listens on all interfaces. Use a WSGI server in production.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
