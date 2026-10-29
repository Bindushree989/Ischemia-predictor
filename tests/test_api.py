"""Automated checks for Test Plan cases (service call mocked). Run: pytest -q"""
import json
import numpy as np
import pytest

from common.features import FEATURES
from common.preprocess import PIPELINE_VERSION, build_pipeline, pipeline_path
import joblib
import bcrypt


@pytest.fixture()
def client(tmp_path, monkeypatch):
    X = np.random.default_rng(0).normal(size=(50, len(FEATURES)))
    joblib.dump(build_pipeline().fit(X), pipeline_path(str(tmp_path)))
    (tmp_path / "registry.json").write_text(json.dumps([{
        "type": "ML", "version": "ml-1.0", "file": "x", "sha256": "0", "pipeline_version": PIPELINE_VERSION,
        "status": "validated", "is_active": True}]))
    from web import app as webapp
    app = webapp.create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://",
                             "MODELS_DIR": str(tmp_path)})
    with app.app_context():
        for name, role in (("doc", "Clinician"), ("doc2", "Clinician"), ("adm", "Administrator")):
            webapp.db.session.add(webapp.User(username=name, role=role, password_hash=bcrypt.hashpw(
                b"Str0ng!Passw0rd", bcrypt.gensalt()).decode()))
        webapp.db.session.commit()
    monkeypatch.setattr(webapp, "call_service", lambda t, p, timeout: {
        "label": "Ischemia Likely", "probability": 72.4, "model_version": p["model_version"],
        "importances": [{"name": f, "direction": "increases"} for f in FEATURES[:5]]})
    return app.test_client(), webapp


def login(c, u="doc", pw="Str0ng!Passw0rd"):
    return c.post("/api/v1/auth/login", json={"username": u, "password": pw})


def hdr(c, u="doc"):
    return {"Authorization": "Bearer " + login(c, u).get_json()["token"]}


GOOD = dict(age=58, sex=1, chest_pain_type=3, resting_bp=140, cholesterol=240, fasting_bs=0,
            resting_ecg=1, max_hr=120, exercise_angina=1, oldpeak=1.5, st_slope=1)


def test_tc01_login(client):
    assert login(client[0]).status_code == 200


def test_tc02_lockout(client):
    c = client[0]
    for _ in range(5):
        r = login(c, pw="wrong")
    assert r.status_code == 423
    assert login(c).status_code == 423


def test_tc03_04_validation(client):
    c = client[0]
    r = c.post("/api/v1/predictions", json={**GOOD, "age": "abc", "max_hr": 999}, headers=hdr(c))
    j = r.get_json()
    assert r.status_code == 400 and {"age", "max_hr"} <= set(j["fields"])


def test_tc05_07_prediction(client):
    c = client[0]
    r = c.post("/api/v1/predictions", json=GOOD, headers=hdr(c))
    j = r.get_json()
    assert r.status_code == 201 and j["risk"] == "High" and len(j["top_features"]) == 5
    assert "not a substitute" in j["disclaimer"]


def test_tc10_history_isolation(client):
    c = client[0]
    c.post("/api/v1/predictions", json=GOOD, headers=hdr(c))
    assert len(c.get("/api/v1/predictions", headers=hdr(c)).get_json()["items"]) == 1
    assert len(c.get("/api/v1/predictions", headers=hdr(c, "doc2")).get_json()["items"]) == 0


def test_tc11_rbac(client):
    c = client[0]
    r = c.post("/api/v1/admin/users", json={"username": "x"}, headers=hdr(c))
    assert r.status_code == 403


def test_tc17_password_policy(client):
    c = client[0]
    r = c.post("/api/v1/admin/users", headers=hdr(c, "adm"),
               json={"username": "n", "password": "Ab1!xy", "role": "Clinician"})
    assert r.status_code == 400


def test_tc09_service_down(client, monkeypatch):
    c, webapp = client
    def boom(*a, **k): raise RuntimeError
    monkeypatch.setattr(webapp, "call_service", boom)
    r = c.post("/api/v1/predictions", json=GOOD, headers=hdr(c))
    assert r.status_code == 503


def test_tc21_rate_limit(client):
    c = client[0]
    h = hdr(c)
    codes = [c.get("/api/v1/predictions", headers=h).status_code for _ in range(65)]
    assert 429 in codes
