"""API and frontend contract tests for Ischemia Predictor.

Run from the repository root:
    python -m pytest -q

Prediction-service calls are mocked so tests do not require running ML/DL/QML
microservices. The tests still exercise Flask routing, authentication, RBAC,
validation, persistence, and response contracts.
"""
import hashlib
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

    registry = [
        {
            "type": "ML",
            "version": "ml-test-1",
            "file": "test-model.joblib",
            "sha256": "test-checksum",
            "pipeline_version": PIPELINE_VERSION,
            "status": "validated",
            "is_active": True,
            "framework": "scikit-learn",
            "algorithm": "Random Forest",
            "metrics": {"accuracy": 0.9, "recall": 0.91, "roc_auc": 0.92},
        },
        {
            "type": "ML",
            "version": "ml-test-2",
            "file": "test-model-2.joblib",
            "sha256": "test-checksum-2",
            "pipeline_version": PIPELINE_VERSION,
            "status": "validated",
            "is_active": False,
            "framework": "XGBoost",
            "algorithm": "XGBoost Classifier",
            "metrics": {"accuracy": 0.92, "recall": 0.93, "roc_auc": 0.94},
        },
        {
            "type": "QML",
            "version": "qml-test-1",
            "file": "qml-test-1.joblib",
            "sha256": "qml-test-checksum",
            "pipeline_version": PIPELINE_VERSION,
            "status": "validated",
            "is_active": True,
            "framework": "PennyLane",
            "algorithm": "Variational Classifier",
            "metrics": {"accuracy": 0.77, "recall": 0.76, "roc_auc": 0.86},
        },
    ]
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


def test_prediction_requires_an_active_exact_model_version(client):
    unavailable = make_prediction(
        client, payload={**GOOD, "model": "ML", "model_version": "ml-test-2"}
    )
    mismatched_family = make_prediction(
        client, payload={**GOOD, "model": "DL", "model_version": "ml-test-1"}
    )
    assert unavailable.status_code == 400
    assert unavailable.get_json()["error_code"] == "MODEL_NOT_AVAILABLE"
    assert mismatched_family.status_code == 400


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
    assert client.put(
        f"/api/v1/models/{model_id}/deactivate", headers=headers
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
    assert model["framework"] == "scikit-learn"
    assert model["algorithm"] == "Random Forest"
    assert model["metrics"]["accuracy"] == 0.9


def test_prediction_model_options_only_include_active_validated_models(client):
    response = client.get(
        "/api/v1/prediction-models", headers=auth_headers(client)
    )
    assert response.status_code == 200
    assert response.get_json() == [{"type": "ML", "version": "ml-test-1"}]


def test_prediction_model_options_require_a_prediction_role(client):
    assert client.get("/api/v1/prediction-models").status_code == 401
    response = client.get(
        "/api/v1/prediction-models", headers=auth_headers(client, "adm")
    )
    assert response.status_code == 403
    assert client.get(
        "/api/v1/prediction-models", headers=auth_headers(client, "engineer")
    ).status_code == 200


def test_activating_alternative_model_changes_prediction_version(client):
    models = client.get(
        "/api/v1/models", headers=auth_headers(client, "adm")
    ).get_json()
    alternative = next(model for model in models if model["version"] == "ml-test-2")

    activation = client.put(
        f"/api/v1/models/{alternative['id']}/activate",
        headers=auth_headers(client, "adm"),
    )
    assert activation.status_code == 200

    options = client.get(
        "/api/v1/prediction-models", headers=auth_headers(client)
    ).get_json()
    assert options == [
        {"type": "ML", "version": "ml-test-1"},
        {"type": "ML", "version": "ml-test-2"},
    ]

    legacy_response = make_prediction(client)
    assert legacy_response.status_code == 201
    assert legacy_response.get_json()["model_version"] == "ml-test-1"

    response = make_prediction(
        client, payload={**GOOD, "model": "ML", "model_version": "ml-test-2"}
    )
    assert response.status_code == 201
    assert response.get_json()["model_version"] == "ml-test-2"


def test_admin_can_deactivate_one_version_without_affecting_siblings(client):
    models = client.get(
        "/api/v1/models", headers=auth_headers(client, "adm")
    ).get_json()
    original = next(model for model in models if model["version"] == "ml-test-1")
    alternative = next(model for model in models if model["version"] == "ml-test-2")
    admin_headers = auth_headers(client, "adm")

    assert client.put(
        f"/api/v1/models/{alternative['id']}/activate", headers=admin_headers
    ).status_code == 200
    response = client.put(
        f"/api/v1/models/{alternative['id']}/deactivate", headers=admin_headers
    )
    assert response.status_code == 200
    assert response.get_json()["is_active"] is False

    options = client.get(
        "/api/v1/prediction-models", headers=auth_headers(client)
    ).get_json()
    assert options == [{"type": "ML", "version": original["version"]}]


def test_model_below_metric_gate_cannot_be_activated(client):
    models = client.get(
        "/api/v1/models", headers=auth_headers(client, "adm")
    ).get_json()
    qml = next(model for model in models if model["version"] == "qml-test-1")
    assert qml["status"] == "unvalidated"
    assert qml["can_activate"] is False
    response = client.put(
        f"/api/v1/models/{qml['id']}/activate",
        headers=auth_headers(client, "adm"),
    )
    assert response.status_code == 409
    assert response.get_json()["error_code"] == "MODEL_NOT_ACTIVATABLE"


def test_checksum_registration_does_not_bypass_metric_validation(app_client, tmp_path):
    _, client, _ = app_client
    artifact = tmp_path / "registered-model.joblib"
    artifact.write_bytes(b"test artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    registration = client.post(
        "/api/v1/models",
        json={
            "type": "ML",
            "version": "ml-registered-test",
            "file": artifact.name,
            "sha256": digest,
            "pipeline_version": PIPELINE_VERSION,
        },
        headers=auth_headers(client, "engineer"),
    )
    assert registration.status_code == 201
    assert registration.get_json()["status"] == "unvalidated"

    model = next(
        item for item in client.get(
            "/api/v1/models", headers=auth_headers(client, "adm")
        ).get_json()
        if item["version"] == "ml-registered-test"
    )
    activation = client.put(
        f"/api/v1/models/{model['id']}/activate",
        headers=auth_headers(client, "adm"),
    )
    assert activation.status_code == 409


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
        "adminPage", "modelsPage", "resultCard", "Open MLflow",
        "Six candidate architectures", "ROC-AUC", "/prediction-models",
        "model_version", "Deactivate",
    ):
        assert marker in html
