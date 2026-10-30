"""Flask API gateway for the Ischemia Predictor (SAD sections 2-3).

The gateway is the single entry point for clients. It owns everything that is
not model inference itself:

* Authentication (JWT, bcrypt) and role-based access control (RBAC)
* Input validation and preprocessing (shared ``common.preprocess`` pipeline)
* Orchestration of the ML / DL / QML prediction microservices
* Encrypted prediction history (AES-256-GCM)
* Model registry (register, validate by checksum, activate)
* Audit logging and per-user rate limiting

Roles
-----
* ``Clinician``     - create predictions, view own history / single prediction
* ``Administrator`` - view history, delete predictions, activate models,
                      create users, read audit logs
* ``ML Engineer``   - create predictions, list / register models

Configuration (environment variables)
-------------------------------------
``DATABASE_URL``   SQLAlchemy URI            (default ``sqlite:///ischemia.db``)
``SECRET_KEY``     JWT signing secret        (default is a DEV-ONLY value)
``DATA_KEY``       64 hex chars, AES-256 key (default is a DEV-ONLY value)
``MODELS_DIR``     Model artifacts folder    (default ``models``)
``DEFAULT_MODEL``  ``ML`` | ``DL`` | ``QML`` (default ``ML``)
``RISK_LOW`` / ``RISK_HIGH``  Risk thresholds in percent (default 30 / 70)
``ML_URL`` / ``DL_URL`` / ``QML_URL``  Inference service base URLs
``FLASK_AUTOCREATE``  Set to ``0`` to skip creating the module-level ``app``

Error format
------------
Every error response is JSON::

    {"error_code": "...", "message": "...", "request_id": "...", "fields": {...}}

``fields`` is only present for validation errors.

Tags such as FR-xx / SEC-xx refer to requirement IDs in the SRS.
"""
import base64
import json
import os
import re
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps

import bcrypt
import jwt
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from flask import Flask, g, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy

from common.features import FEATURES, RANGES
from common.preprocess import PIPELINE_VERSION, load_pipeline, transform_one

db = SQLAlchemy()
DISCLAIMER = "Decision-support output; not a substitute for professional medical diagnosis."
ROLES = {"Clinician", "Administrator", "ML Engineer"}
# Model type -> env var holding that inference service's base URL.
SERVICE_URLS = {"ML": "ML_URL", "DL": "DL_URL", "QML": "QML_URL"}
# Fallback URLs used for local development when the env vars are not set.
DEFAULT_URLS = {"ML": "http://localhost:5001", "DL": "http://localhost:5002", "QML": "http://localhost:5003"}


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------
class User(db.Model):
    """Application user. Also tracks failed logins for account lockout (SEC-02)."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)  # bcrypt hash, never plaintext
    role = db.Column(db.String(20), nullable=False)            # one of ROLES
    failed_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)                      # UTC; None when not locked


class Prediction(db.Model):
    """A stored prediction result, owned by the user who requested it."""
    id = db.Column(db.String(40), primary_key=True)            # "p-" + 8 hex chars
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_ref = db.Column(db.String(40))                     # caller-supplied or generated reference
    inputs = db.Column(db.Text)  # AES-256-GCM encrypted JSON
    label = db.Column(db.String(30))
    probability = db.Column(db.Float)                          # percent, compared to RISK_LOW/RISK_HIGH
    risk = db.Column(db.String(10))                            # Low | Moderate | High
    model_version = db.Column(db.String(30))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Model(db.Model):
    """Registry entry for a trained model artifact."""
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(5))                             # ML | DL | QML
    version = db.Column(db.String(30), unique=True)
    status = db.Column(db.String(15))                          # "validated" (checksum OK) | "unvalidated"
    checksum = db.Column(db.String(64))                        # expected SHA-256 of the artifact
    file = db.Column(db.String(100))                           # file name inside MODELS_DIR
    pipeline_version = db.Column(db.String(30))                # preprocessing version it was trained with
    is_active = db.Column(db.Boolean, default=False)           # at most one active model per type


class AuditLog(db.Model):
    """Append-only record of security-relevant actions."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    action = db.Column(db.String(60))                          # e.g. login, prediction, deletion
    ip = db.Column(db.String(45))                              # fits IPv6
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


def call_service(mtype, payload, timeout):
    """POST ``payload`` to the inference service for ``mtype`` and return its JSON.

    Args:
        mtype: Model type key in ``SERVICE_URLS`` (``"ML"``, ``"DL"`` or ``"QML"``).
        payload: JSON-serialisable body (features, pipeline and model version).
        timeout: Seconds to wait for the service.

    Returns:
        The decoded JSON response (expects ``probability``, ``label``,
        ``importances`` and ``model_version``).

    Raises:
        requests.RequestException: on connection errors, timeouts or non-2xx status.
    """
    base = os.environ.get(SERVICE_URLS[mtype], DEFAULT_URLS[mtype])
    r = requests.post(base + "/predict", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def create_app(config=None):
    """Application factory.

    Args:
        config: Optional dict of Flask config overrides (used by tests).

    Returns:
        A configured Flask app with the database created and the model
        registry synced from ``MODELS_DIR/registry.json``.
    """
    app = Flask(__name__, static_folder="static")
    app.config.update(
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "sqlite:///ischemia.db"),
        # DEV-ONLY defaults below. Always set real values in production.
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-change-me-dev-secret-change-me"),
        DATA_KEY=os.environ.get("DATA_KEY", "00" * 32),  # 64 hex chars = AES-256 key
        MODELS_DIR=os.environ.get("MODELS_DIR", "models"),
        DEFAULT_MODEL=os.environ.get("DEFAULT_MODEL", "ML"),
        RISK_LOW=float(os.environ.get("RISK_LOW", 30)),
        RISK_HIGH=float(os.environ.get("RISK_HIGH", 70)),
        RATE_LIMIT=60,  # max requests per user per rolling 60 s
    )
    if config:
        app.config.update(config)
    db.init_app(app)
    # In-memory per-user request timestamps for rate limiting.
    # Per-process only: not shared across workers or restarts.
    hits = defaultdict(deque)

    # ---------- helpers ----------
    def aes():
        """Return an AES-GCM cipher built from the configured ``DATA_KEY``."""
        return AESGCM(bytes.fromhex(app.config["DATA_KEY"]))

    def encrypt(obj):
        """Encrypt a JSON-serialisable object with AES-256-GCM.

        Returns base64 of ``nonce (12 bytes) || ciphertext+tag``. A fresh random
        nonce is used for every call.
        """
        nonce = os.urandom(12)
        return base64.b64encode(nonce + aes().encrypt(nonce, json.dumps(obj).encode(), None)).decode()

    def error(code, http, message, fields=None, **extra):
        """Build the standard JSON error response.

        Args:
            code: Machine-readable ``error_code``.
            http: HTTP status code.
            message: Human-readable, patient-data-free message.
            fields: Optional ``{field: message}`` map for validation errors.
            **extra: Extra response headers (key -> value).
        """
        body = {"error_code": code, "message": message, "request_id": uuid.uuid4().hex[:12]}
        if fields:
            body["fields"] = fields
        app.logger.warning("%s %s %s", http, code, body["request_id"])  # no patient data (FR-18)
        resp = jsonify(body)
        resp.status_code = http
        for k, v in extra.items():
            resp.headers[k] = str(v)
        return resp

    def audit(action, user_id=None):
        """Write an audit-log row (action, user, client IP) and commit."""
        db.session.add(AuditLog(user_id=user_id, action=action, ip=request.remote_addr))
        db.session.commit()

    def make_token(user):
        """Issue a signed HS256 JWT for ``user`` that expires in 15 minutes."""
        return jwt.encode({"sub": str(user.id), "role": user.role,
                           "exp": datetime.utcnow() + timedelta(minutes=15)},
                          app.config["SECRET_KEY"], algorithm="HS256")

    def require(*roles):
        """Route decorator enforcing authentication, RBAC and rate limiting.

        Args:
            *roles: Allowed roles. If empty, any authenticated user is allowed.

        On success it sets ``g.user`` and ``g.new_token`` (a refreshed token,
        returned to the client in the ``X-Token`` header by ``refresh``).

        Error responses:
            401 UNAUTHENTICATED  missing / invalid / expired token, or unknown user
            403 FORBIDDEN        role not in ``roles``
            429 RATE_LIMITED     more than ``RATE_LIMIT`` requests in 60 s
        """
        def deco(fn):
            @wraps(fn)
            def wrapper(*a, **kw):
                h = request.headers.get("Authorization", "")
                if not h.startswith("Bearer "):
                    return error("UNAUTHENTICATED", 401, "Please log in.")
                try:
                    claims = jwt.decode(h[7:], app.config["SECRET_KEY"], algorithms=["HS256"])
                except jwt.PyJWTError:
                    return error("UNAUTHENTICATED", 401, "Session expired. Please log in.")
                user = db.session.get(User, int(claims["sub"]))
                if user is None:
                    return error("UNAUTHENTICATED", 401, "Please log in.")
                if roles and user.role not in roles:          # SEC-04
                    return error("FORBIDDEN", 403, "You are not allowed to do this.")
                # Sliding-window rate limit: drop timestamps older than 60 s first.
                now = time.time()
                q = hits[user.id]
                while q and now - q[0] > 60:
                    q.popleft()
                if len(q) >= app.config["RATE_LIMIT"]:        # SEC-09
                    return error("RATE_LIMITED", 429, "Too many requests.", Retry_After=60)
                q.append(now)
                g.user = user
                g.new_token = make_token(user)                # sliding 15-min inactivity (FR-03)
                return fn(*a, **kw)
            return wrapper
        return deco

    @app.after_request
    def refresh(resp):
        """Attach the refreshed JWT (if any) as the ``X-Token`` response header."""
        if g.get("new_token"):
            resp.headers["X-Token"] = g.new_token
        return resp

    def check_password_policy(pw):
        """Return truthy if ``pw`` has 10+ chars with lower, upper, digit and symbol."""
        return (len(pw) >= 10 and re.search(r"[a-z]", pw) and re.search(r"[A-Z]", pw)
                and re.search(r"\d", pw) and re.search(r"[^\w\s]", pw))

    def validate(data):
        """Validate and normalise the clinical input features.

        Every name in ``FEATURES`` must be present, numeric (not bool) and inside
        its ``RANGES`` bounds. ``sex`` may also be given as a string starting with
        "m" (-> 1) or "f" (-> 0).

        Returns:
            ``(values, errors)`` where ``values`` maps feature -> float and
            ``errors`` maps feature -> message. Empty ``errors`` means valid.
        """
        out, errs = {}, {}
        for f in FEATURES:
            lo, hi = RANGES[f]
            v = data.get(f)
            if f == "sex" and isinstance(v, str):
                v = {"m": 1, "f": 0}.get(v.strip().lower()[:1], v)
            if v is None or v == "" or isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not lo <= v <= hi:
                errs[f] = f"{f} is required and must be a number between {lo} and {hi}"
            else:
                out[f] = float(v)
        return out, errs

    def risk_of(p):
        """Map a probability (percent) to ``Low`` (< RISK_LOW), ``Moderate``
        (RISK_LOW..RISK_HIGH inclusive) or ``High`` (> RISK_HIGH)."""
        if p < app.config["RISK_LOW"]:
            return "Low"
        return "Moderate" if p <= app.config["RISK_HIGH"] else "High"

    def sync_registry():
        """Import entries from ``MODELS_DIR/registry.json`` into the Model table.

        Only versions not already in the database are added, so it is safe to
        run on every startup. A missing registry file is ignored.
        """
        path = os.path.join(app.config["MODELS_DIR"], "registry.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            for e in json.load(f):
                if not Model.query.filter_by(version=e["version"]).first():
                    db.session.add(Model(type=e["type"], version=e["version"], status=e["status"],
                                         checksum=e["sha256"], file=e["file"],
                                         pipeline_version=e["pipeline_version"],
                                         is_active=e.get("is_active", False)))
        db.session.commit()

    # Create tables and load the registry once at startup.
    with app.app_context():
        db.create_all()
        sync_registry()

    # ---------- routes ----------
    @app.get("/")
    def index():
        """Serve the single-page frontend (``static/index.html``)."""
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/v1/auth/login")
    def login():
        """Authenticate a user and issue a JWT.

        Body: ``{"username": str, "password": str}``
        200: ``{"token": str, "role": str}``
        400 VALIDATION_ERROR, 401 UNAUTHENTICATED (bad credentials),
        423 ACCOUNT_LOCKED (5 failed attempts -> locked for 15 minutes, SEC-02).
        """
        d = request.get_json(silent=True) or {}
        u, pw = d.get("username"), d.get("password")
        if not isinstance(u, str) or not isinstance(pw, str):
            return error("VALIDATION_ERROR", 400, "Username and password required.")
        user = User.query.filter_by(username=u).first()
        if user and user.locked_until and user.locked_until > datetime.utcnow():
            return error("ACCOUNT_LOCKED", 423, "Account locked for 15 minutes.")
        if user and bcrypt.checkpw(pw.encode(), user.password_hash.encode()):
            user.failed_attempts = 0
            db.session.commit()
            audit("login", user.id)
            return jsonify(token=make_token(user), role=user.role)
        if user:
            user.failed_attempts += 1
            if user.failed_attempts >= 5:                     # SEC-02
                user.locked_until = datetime.utcnow() + timedelta(minutes=15)
                user.failed_attempts = 0
                db.session.commit()
                return error("ACCOUNT_LOCKED", 423, "Account locked for 15 minutes.")
            db.session.commit()
        # Same message for unknown user and wrong password (no user enumeration).
        return error("UNAUTHENTICATED", 401, "Invalid username or password.")

    @app.post("/api/v1/auth/logout")
    @require()
    def logout():
        """Log out. Returns 204. Tokens are stateless, so the client must discard its token."""
        return "", 204

    @app.post("/api/v1/predictions")
    @require("Clinician", "ML Engineer")
    def predict():
        """Run a prediction and store the result.

        Roles: Clinician, ML Engineer.
        Body: all clinical features in ``FEATURES`` plus optional
        ``patient_ref`` (str) and ``model`` (``ML`` | ``DL`` | ``QML``,
        defaults to ``DEFAULT_MODEL``).

        Flow: validate inputs -> find the active validated model -> check the
        preprocessing version -> transform features -> call the inference
        service (one retry inside a 10 s budget, FR-17) -> encrypt and store
        inputs -> return the result.

        201: ``{prediction_id, label, probability, risk, top_features,
        model_version, disclaimer}``
        400 VALIDATION_ERROR, 500 PIPELINE_MISMATCH / INTERNAL_ERROR,
        503 SERVICE_UNAVAILABLE (no active model or service unreachable).
        """
        d = request.get_json(silent=True) or {}
        values, errs = validate(d)
        if errs:                                              # FR-05, FR-06
            return error("VALIDATION_ERROR", 400, "Some fields are invalid.", fields=errs)
        mtype = str(d.get("model", app.config["DEFAULT_MODEL"])).upper()
        if mtype not in SERVICE_URLS:
            return error("VALIDATION_ERROR", 400, "model must be ML, DL or QML.")
        row = Model.query.filter_by(type=mtype, is_active=True, status="validated").first()
        if row is None:
            return error("SERVICE_UNAVAILABLE", 503,
                         "Prediction service temporarily unavailable, please retry")
        if row.pipeline_version != PIPELINE_VERSION:          # FR-08
            return error("PIPELINE_MISMATCH", 500, "Preprocessing version mismatch.")
        try:
            pipe = load_pipeline(app.config["MODELS_DIR"])
        except OSError:
            return error("INTERNAL_ERROR", 500, "Preprocessing artifact missing.")
        feats = transform_one(pipe, [values[f] for f in FEATURES])
        payload = {"features": feats, "pipeline_version": PIPELINE_VERSION,
                   "model_version": row.version}
        # Up to 2 attempts, all within one shared 10 s deadline.
        deadline, res = time.time() + 10, None
        for _ in range(2):                                    # retry once inside 10 s (FR-17)
            left = deadline - time.time()
            if left <= 0:
                break
            try:
                res = call_service(mtype, payload, left)
                break
            except Exception:
                continue
        if res is None:
            return error("SERVICE_UNAVAILABLE", 503,
                         "Prediction service temporarily unavailable, please retry")
        p = float(res["probability"])
        pid = "p-" + uuid.uuid4().hex[:8]
        db.session.add(Prediction(
            id=pid, user_id=g.user.id, patient_ref=str(d.get("patient_ref") or "P-" + uuid.uuid4().hex[:6]),
            inputs=encrypt(values), label=res["label"], probability=p, risk=risk_of(p),
            model_version=res["model_version"]))
        db.session.commit()
        audit("prediction", g.user.id)
        return jsonify(prediction_id=pid, label=res["label"], probability=p, risk=risk_of(p),
                       top_features=res["importances"][:5], model_version=res["model_version"],
                       disclaimer=DISCLAIMER), 201

    def p_dict(p):
        """Serialise a Prediction for API responses (encrypted inputs are never returned)."""
        return {"prediction_id": p.id, "label": p.label, "probability": p.probability, "risk": p.risk,
                "model_version": p.model_version, "created_at": p.created_at.isoformat(),
                "disclaimer": DISCLAIMER}

    @app.get("/api/v1/predictions")
    @require("Clinician", "Administrator")
    def history():
        """List the caller's own predictions, newest first.

        Roles: Clinician, Administrator.
        Query: ``page`` (1-based, 20 items per page, FR-20).
        200: ``{page, items: [...]}``
        """
        page = max(int(request.args.get("page", 1) or 1), 1)
        q = Prediction.query.filter_by(user_id=g.user.id).order_by(Prediction.created_at.desc())
        items = q.offset((page - 1) * 20).limit(20).all()    # FR-20
        return jsonify(page=page, items=[p_dict(p) for p in items])

    @app.get("/api/v1/predictions/<pid>")
    @require("Clinician")
    def one(pid):
        """Fetch one of the caller's own predictions by id (Clinician only).

        200: prediction object. 404 NOT_FOUND if missing or owned by someone else.
        """
        p = Prediction.query.filter_by(id=pid, user_id=g.user.id).first()
        return jsonify(p_dict(p)) if p else error("NOT_FOUND", 404, "Not found.")

    @app.delete("/api/v1/predictions/<pid>")
    @require("Administrator")
    def delete(pid):
        """Delete any prediction by id (Administrator only). 204 on success, 404 if missing.
        The deletion is audit-logged."""
        p = db.session.get(Prediction, pid)
        if not p:
            return error("NOT_FOUND", 404, "Not found.")
        db.session.delete(p)
        db.session.commit()
        audit("deletion", g.user.id)
        return "", 204

    @app.get("/api/v1/models")
    @require("Administrator", "ML Engineer")
    def list_models():
        """List all registered models (Administrator, ML Engineer)."""
        return jsonify([{"id": m.id, "type": m.type, "version": m.version, "status": m.status,
                         "is_active": m.is_active} for m in Model.query.all()])

    @app.post("/api/v1/models")
    @require("ML Engineer")
    def register_model():
        """Register a model artifact (ML Engineer only).

        Body: ``{type, version, file, sha256, pipeline_version}`` (all strings).
        The file is looked up in ``MODELS_DIR`` by base name and its SHA-256 is
        compared with ``sha256``. A match gives status ``validated`` (and the
        entry is appended to ``registry.json`` for the inference services);
        otherwise the model is saved as ``unvalidated`` and cannot be activated.

        201: ``{id, status}``. 400 VALIDATION_ERROR (bad body or duplicate version).
        """
        d = request.get_json(silent=True) or {}
        if d.get("type") not in SERVICE_URLS or not all(isinstance(d.get(k), str) for k in
                                                          ("version", "file", "sha256", "pipeline_version")):
            return error("VALIDATION_ERROR", 400, "type, version, file, sha256, pipeline_version required.")
        fn = os.path.basename(d["file"])  # basename blocks path traversal
        path = os.path.join(app.config["MODELS_DIR"], fn)
        import hashlib
        ok = os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == d["sha256"]
        if Model.query.filter_by(version=d["version"]).first():
            return error("VALIDATION_ERROR", 400, "Version already registered.")
        m = Model(type=d["type"], version=d["version"], status="validated" if ok else "unvalidated",
                  checksum=d["sha256"], file=fn, pipeline_version=d["pipeline_version"])
        db.session.add(m)
        db.session.commit()
        if ok:  # keep the services' registry file in sync
            rp = os.path.join(app.config["MODELS_DIR"], "registry.json")
            reg = json.load(open(rp)) if os.path.exists(rp) else []
            reg.append({"type": m.type, "version": m.version, "file": fn, "sha256": m.checksum,
                        "pipeline_version": m.pipeline_version, "status": "validated", "is_active": False})
            json.dump(reg, open(rp, "w"), indent=2)
        return jsonify(id=m.id, status=m.status), 201

    @app.put("/api/v1/models/<int:mid>/activate")
    @require("Administrator")
    def activate(mid):
        """Make a validated model the active one for its type (Administrator only).

        Any previously active model of the same type is deactivated.
        200: ``{id, is_active}``. 404 NOT_FOUND, 409 MODEL_NOT_ACTIVATABLE
        (status is not ``validated``). Audit-logged as ``model_activation``.
        """
        m = db.session.get(Model, mid)
        if not m:
            return error("NOT_FOUND", 404, "Not found.")
        if m.status != "validated":
            return error("MODEL_NOT_ACTIVATABLE", 409, "Model is not validated.")
        Model.query.filter_by(type=m.type).update({"is_active": False})
        m.is_active = True
        db.session.commit()
        audit("model_activation", g.user.id)
        return jsonify(id=m.id, is_active=True)

    @app.post("/api/v1/admin/users")
    @require("Administrator")
    def create_user():
        """Create a user (Administrator only).

        Body: ``{username, password, role}``; ``role`` must be in ``ROLES`` and
        the password must satisfy ``check_password_policy``.
        201: ``{username, role}``. 400 VALIDATION_ERROR (invalid input or duplicate username).
        """
        d = request.get_json(silent=True) or {}
        u, pw, role = d.get("username"), d.get("password"), d.get("role")
        if not isinstance(u, str) or not u or role not in ROLES or not isinstance(pw, str) \
                or not check_password_policy(pw):
            return error("VALIDATION_ERROR", 400,
                         "Valid username/role required; password needs 10+ chars with upper, lower, digit and symbol.")
        if User.query.filter_by(username=u).first():
            return error("VALIDATION_ERROR", 400, "Username exists.")
        db.session.add(User(username=u, role=role,
                            password_hash=bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()))
        db.session.commit()
        audit("role_change", g.user.id)
        return jsonify(username=u, role=role), 201

    @app.get("/api/v1/admin/audit-logs")
    @require("Administrator")
    def logs():
        """Return the 200 most recent audit-log entries, newest first (Administrator only)."""
        rows = AuditLog.query.order_by(AuditLog.id.desc()).limit(200).all()
        return jsonify([{"user_id": r.user_id, "action": r.action, "ip": r.ip,
                         "timestamp": r.timestamp.isoformat()} for r in rows])

    return app


# Module-level app for `flask run` / WSGI servers. Tests can set
# FLASK_AUTOCREATE=0 and call create_app(config) themselves.
app = create_app() if os.environ.get("FLASK_AUTOCREATE", "1") == "1" else None
