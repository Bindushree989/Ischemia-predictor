"""API and frontend contract tests for Ischemia Predictor.

Run from the repository root:
    python -m pytest -q

Prediction-service calls are mocked so tests do not require running ML/DL/QML
microservices. The tests still exercise Flask routing, authentication, RBAC,
validation, persistence, and response contracts.
"""
import json

import bcrypt
import joblib
import numpy as np
import pytest

from common.features import FEATURES
from common.preprocess import PIPELINE_VERSION, build_pipeline, pipeline_path


GOOD = {
    "age": 58, "sex": 1, "chest_pain_type": 3, "resting_bp": 140,
    "cholesterol": 240, "fasting_bs": 0, "resting_ecg": 1,
    "max_hr": 120, "exercise_angina": 1, "oldpeak": 1.5, "st_slope": 1,
}


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """Create an isolated SQLite app with a fitted preprocessing artifact."""
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(60, len(FEATURES)))
    joblib.dump(build_pipeline().fit(samples), pipeline_path(str(tmp_path)))

    registry = [{
        "type": "ML",
        "version": "ml-test-1",
        "file": "test-model.joblib",
        "sha256": "test-checksum",
        "pipeline_version": PIPELINE_VERSION,
        "status": "validated",
        "is_active": True,
    }]
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    from web import app as webapp

    app = webapp.create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "MODELS_DIR": str(tmp_path),
        "SECRET_KEY": "test-secret-key-that-is-long-enough",
        "DATA_KEY": "11" * 32,
        "RATE_LIMIT": 200,
    })

    with app.app_context():
        users = [
            ("doc", "Clinician"),
            ("doc2", "Clinician"),
            ("adm", "Administrator"),
            ("engineer", "ML Engineer"),
        ]
        for username, role in users:
            webapp.db.session.add(webapp.User(
                username=username,
                role=role,
                password_hash=bcrypt.hashpw(
                    b"Str0ng!Passw0rd", bcrypt.gensalt()
                ).decode(),
            ))
        webapp.db.session.commit()

    def fake_service(model_type, payload, timeout):
        return {
            "label": "Ischemia Likely",
            "probability": 72.4,
            "model_version": payload["model_version"],
            "importances": [
                {"name": feature, "direction": "increases"}
                for feature in FEATURES[:5]
            ],
        }

    monkeypatch.setattr(webapp, "call_service", fake_service)
    return app, app.test_client(), webapp


@pytest.fixture()
def client(app_client):
    return app_client[1]


def login(client, username="doc", password="Str0ng!Passw0rd"):
    return client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )


def auth_headers(client, username="doc"):
    response = login(client, username)
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.get_json()['token']}"}


def make_prediction(client, username="doc", payload=None):
    return client.post(
        "/api/v1/predictions",
        json=payload or GOOD,
        headers=auth_headers(client, username),
    )


# Authentication and session contracts

def test_login_returns_token_and_role(client):
    response = login(client)
    assert response.status_code == 200
    assert response.get_json()["role"] == "Clinician"
    assert response.get_json()["token"]


@pytest.mark.parametrize(
    ("username", "role"),
    [("adm", "Administrator"), ("engineer", "ML Engineer")],
)
def test_login_returns_each_account_role(client, username, role):
    response = login(client, username)
    assert response.status_code == 200
    assert response.get_json()["role"] == role


def test_invalid_login_is_rejected(client):
    response = login(client, password="incorrect")
    assert response.status_code == 401
    assert response.get_json()["error_code"] == "UNAUTHENTICATED"


def test_five_failed_logins_lock_account(client):
    for _ in range(5):
        response = login(client, password="wrong-password")
    assert response.status_code == 423
    assert login(client).status_code == 423


def test_logout_requires_authentication(client):
    assert client.post("/api/v1/auth/logout").status_code == 401
    response = client.post(
        "/api/v1/auth/logout", headers=auth_headers(client)
    )
    assert response.status_code == 204


# Prediction validation and response contracts

def test_prediction_rejects_missing_and_out_of_range_fields(client):
    invalid = {**GOOD, "age": None, "max_hr": 999}
    response = client.post(
        "/api/v1/predictions",
        json=invalid,
        headers=auth_headers(client),
    )
    body = response.get_json()
    assert response.status_code == 400
    assert {"age", "max_hr"} <= set(body["fields"])


def test_prediction_rejects_unknown_model(client):
    response = client.post(
        "/api/v1/predictions",
        json={**GOOD, "model": "UNKNOWN"},
        headers=auth_headers(client),
    )
    assert response.status_code == 400


def test_prediction_returns_frontend_contract(client):
    response = make_prediction(client)
    body = response.get_json()

    assert response.status_code == 201
    assert body["prediction_id"].startswith("p-")
    assert body["label"] == "Ischemia Likely"
    assert body["probability"] == pytest.approx(72.4)
    assert body["risk"] == "High"
    assert body["model_version"] == "ml-test-1"
    assert len(body["top_features"]) == 5
    assert "not a substitute" in body["disclaimer"].lower()


def test_prediction_service_failure_returns_503(app_client, monkeypatch):
    app, client, webapp = app_client

    def unavailable(*args, **kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(webapp, "call_service", unavailable)
    response = client.post(
        "/api/v1/predictions",
        json=GOOD,
        headers=auth_headers(client),
    )
    assert response.status_code == 503
    assert response.get_json()["error_code"] == "SERVICE_UNAVAILABLE"


def test_prediction_requires_clinician_or_engineer(client):
    response = client.post(
        "/api/v1/predictions",
        json=GOOD,
        headers=auth_headers(client, "adm"),
    )
    assert response.status_code == 403


# History, ownership and role-based access

def test_history_returns_saved_prediction_contract(client):
    make_prediction(client)
    response = client.get(
        "/api/v1/predictions",
        headers=auth_headers(client),
    )
    body = response.get_json()

    assert response.status_code == 200
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert {"prediction_id", "label", "probability", "risk",
            "model_version", "created_at", "disclaimer"} <= set(item)


def test_history_is_isolated_per_user(client):
    make_prediction(client, "doc")
    own = client.get(
        "/api/v1/predictions", headers=auth_headers(client, "doc")
    ).get_json()
    other = client.get(
        "/api/v1/predictions", headers=auth_headers(client, "doc2")
    ).get_json()

    assert len(own["items"]) == 1
    assert len(other["items"]) == 0


def test_administrator_can_view_history_but_not_predict(client):
    headers = auth_headers(client, "adm")
    assert client.get("/api/v1/predictions", headers=headers).status_code == 200
    assert client.post(
        "/api/v1/predictions", json=GOOD, headers=headers
    ).status_code == 403


def test_engineer_can_list_models_but_cannot_activate(client):
    headers = auth_headers(client, "engineer")
    response = client.get("/api/v1/models", headers=headers)
    assert response.status_code == 200
    model_id = response.get_json()[0]["id"]
    assert client.put(
        f"/api/v1/models/{model_id}/activate", headers=headers
    ).status_code == 403


# Administration and model registry

def test_non_admin_cannot_create_users(client):
    response = client.post(
        "/api/v1/admin/users",
        json={
            "username": "new-user",
            "password": "Str0ng!Passw0rd",
            "role": "Clinician",
        },
        headers=auth_headers(client, "doc"),
    )
    assert response.status_code == 403


def test_admin_can_create_user_with_short_password(client):
    response = client.post(
        "/api/v1/admin/users",
        json={
            "username": "weak-user",
            "password": "x",
            "role": "Clinician",
        },
        headers=auth_headers(client, "adm"),
    )
    assert response.status_code == 201
    login_response = login(client, "weak-user", "x")
    assert login_response.status_code == 200
    assert login_response.get_json()["role"] == "Clinician"


def test_admin_cannot_create_user_with_empty_password(client):
    response = client.post(
        "/api/v1/admin/users",
        json={"username": "empty-password", "password": "", "role": "Clinician"},
        headers=auth_headers(client, "adm"),
    )
    assert response.status_code == 400


def test_admin_can_create_user_with_valid_password(client):
    response = client.post(
        "/api/v1/admin/users",
        json={
            "username": "new-clinician",
            "password": "Str0ng!Passw0rd",
            "role": "Clinician",
        },
        headers=auth_headers(client, "adm"),
    )
    assert response.status_code == 201
    assert response.get_json()["role"] == "Clinician"


def test_model_list_has_fields_used_by_frontend(client):
    response = client.get(
        "/api/v1/models", headers=auth_headers(client, "adm")
    )
    assert response.status_code == 200
    model = response.get_json()[0]
    assert {"id", "type", "version", "status", "is_active"} <= set(model)


def test_admin_can_read_audit_logs(client):
    response = client.get(
        "/api/v1/admin/audit-logs",
        headers=auth_headers(client, "adm"),
    )
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)


# Frontend delivery contract

def test_homepage_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Ischemia Predictor" in response.data
    assert b"predictionForm" in response.data


def test_frontend_contains_all_primary_views(client):
    response = client.get("/")
    html = response.get_data(as_text=True)
    for marker in (
        "loginView", "predictPage", "historyPage",
        "adminPage", "modelsPage", "resultCard",
    ):
        assert marker in html
