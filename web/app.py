"""Flask gateway: auth + RBAC, validation, preprocessing, orchestration, history, model
registry, audit log (SAD sections 2-3)."""
import base64
import json
import os
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
SERVICE_URLS = {"ML": "ML_URL", "DL": "DL_URL", "QML": "QML_URL"}
DEFAULT_URLS = {"ML": "http://localhost:5001", "DL": "http://localhost:5002", "QML": "http://localhost:5003"}


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    failed_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)


class Prediction(db.Model):
    id = db.Column(db.String(40), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_ref = db.Column(db.String(40))
    inputs = db.Column(db.Text)  # AES-256-GCM encrypted JSON
    label = db.Column(db.String(30))
    probability = db.Column(db.Float)
    risk = db.Column(db.String(10))
    model_version = db.Column(db.String(30))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(5))
    version = db.Column(db.String(30), unique=True)
    status = db.Column(db.String(15))
    checksum = db.Column(db.String(64))
    file = db.Column(db.String(100))
    pipeline_version = db.Column(db.String(30))
    is_active = db.Column(db.Boolean, default=False)


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    action = db.Column(db.String(60))
    ip = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


def call_service(mtype, payload, timeout):
    base = os.environ.get(SERVICE_URLS[mtype], DEFAULT_URLS[mtype])
    r = requests.post(base + "/predict", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def create_app(config=None):
    app = Flask(__name__, static_folder="static")
    app.config.update(
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "sqlite:///ischemia.db"),
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-change-me-dev-secret-change-me"),
        DATA_KEY=os.environ.get("DATA_KEY", "00" * 32),  # 64 hex chars = AES-256 key
        MODELS_DIR=os.environ.get("MODELS_DIR", "models"),
        DEFAULT_MODEL=os.environ.get("DEFAULT_MODEL", "ML"),
        RISK_LOW=float(os.environ.get("RISK_LOW", 30)),
        RISK_HIGH=float(os.environ.get("RISK_HIGH", 70)),
        RATE_LIMIT=60,
    )
    if config:
        app.config.update(config)
    db.init_app(app)
    hits = defaultdict(deque)

    # ---------- helpers ----------
    def aes():
        return AESGCM(bytes.fromhex(app.config["DATA_KEY"]))

    def encrypt(obj):
        nonce = os.urandom(12)
        return base64.b64encode(nonce + aes().encrypt(nonce, json.dumps(obj).encode(), None)).decode()

    def error(code, http, message, fields=None, **extra):
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
        db.session.add(AuditLog(user_id=user_id, action=action, ip=request.remote_addr))
        db.session.commit()

    def make_token(user):
        return jwt.encode({"sub": str(user.id), "role": user.role,
                           "exp": datetime.utcnow() + timedelta(minutes=15)},
                          app.config["SECRET_KEY"], algorithm="HS256")

    def require(*roles):
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
        if g.get("new_token"):
            resp.headers["X-Token"] = g.new_token
        return resp

    def validate(data):
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
        if p < app.config["RISK_LOW"]:
            return "Low"
        return "Moderate" if p <= app.config["RISK_HIGH"] else "High"

    def sync_registry():
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

    with app.app_context():
        db.create_all()
        sync_registry()

    # ---------- routes ----------
    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/v1/auth/login")
    def login():
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
        return error("UNAUTHENTICATED", 401, "Invalid username or password.")

    @app.post("/api/v1/auth/logout")
    @require()
    def logout():
        return "", 204

    @app.post("/api/v1/predictions")
    @require("Clinician", "ML Engineer")
    def predict():
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
        return {"prediction_id": p.id, "label": p.label, "probability": p.probability, "risk": p.risk,
                "model_version": p.model_version, "created_at": p.created_at.isoformat(),
                "disclaimer": DISCLAIMER}

    @app.get("/api/v1/predictions")
    @require("Clinician", "Administrator")
    def history():
        page = max(int(request.args.get("page", 1) or 1), 1)
        q = Prediction.query.filter_by(user_id=g.user.id).order_by(Prediction.created_at.desc())
        items = q.offset((page - 1) * 20).limit(20).all()    # FR-20
        return jsonify(page=page, items=[p_dict(p) for p in items])

    @app.get("/api/v1/predictions/<pid>")
    @require("Clinician")
    def one(pid):
        p = Prediction.query.filter_by(id=pid, user_id=g.user.id).first()
        return jsonify(p_dict(p)) if p else error("NOT_FOUND", 404, "Not found.")

    @app.delete("/api/v1/predictions/<pid>")
    @require("Administrator")
    def delete(pid):
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
        return jsonify([{"id": m.id, "type": m.type, "version": m.version, "status": m.status,
                         "is_active": m.is_active} for m in Model.query.all()])

    @app.post("/api/v1/models")
    @require("ML Engineer")
    def register_model():
        d = request.get_json(silent=True) or {}
        if d.get("type") not in SERVICE_URLS or not all(isinstance(d.get(k), str) for k in
                                                          ("version", "file", "sha256", "pipeline_version")):
            return error("VALIDATION_ERROR", 400, "type, version, file, sha256, pipeline_version required.")
        fn = os.path.basename(d["file"])
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
        d = request.get_json(silent=True) or {}
        u, pw, role = d.get("username"), d.get("password"), d.get("role")
        if not isinstance(u, str) or not u or role not in ROLES or not isinstance(pw, str) or not pw:
            return error("VALIDATION_ERROR", 400,
                         "Username, role, and a non-empty password are required.")
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
        rows = AuditLog.query.order_by(AuditLog.id.desc()).limit(200).all()
        return jsonify([{"user_id": r.user_id, "action": r.action, "ip": r.ip,
                         "timestamp": r.timestamp.isoformat()} for r in rows])

    return app


app = create_app() if os.environ.get("FLASK_AUTOCREATE", "1") == "1" else None
