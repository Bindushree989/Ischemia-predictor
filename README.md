# Ischemia Predictor

A clinical decision-support application that combines a Flask API gateway with ML, DL,
and QML prediction services. The gateway handles authentication, role-based access control,
input validation, preprocessing, prediction orchestration, history, model registry, and audit logs.

> **Clinical safety:** This application is a prototype decision-support tool. Its output is
> not a medical diagnosis and must not be used as the sole basis for clinical decisions.
> A qualified healthcare professional must interpret results in the appropriate clinical context.

## Project overview

- **Frontend:** HTML, CSS, and browser JavaScript served by Flask (`web/static/index.html`).
- **Gateway:** Flask application in `web/app.py`.
- **Prediction services:** ML, DL, and QML services in `services/model_service.py`.
- **Preprocessing:** Shared pipeline in `common/`.
- **Training:** Offline model training under `training/`.
- **Tests:** Pytest suite under `tests/`.
- **Documentation:** SRS, Software Architecture and Design Specification (SAD) and Test Plan,
  all version 1.1. Each lists the deviations from the first submission.

The dataset is `data/heart.csv` (918 records, 11 input features, target `HeartDisease`).
Heart-disease prediction is used as a proxy task; it must not be represented as a clinically
validated ischemia diagnostic model.

## User roles

| Role | Supported UI/API capabilities |
|---|---|
| Clinician | Create predictions and view own prediction history |
| Administrator | View own history, create users, inspect audit logs, activate validated models, delete predictions (API) |
| ML Engineer | Create predictions (test use) and inspect or register models (API) |

The backend enforces permissions. Hiding a navigation item in the browser is only a usability
measure, not an authorization mechanism.

## Requirements

- Python 3.10 or newer.
- Git.
- Windows, Linux, or macOS.
- Optional: Docker and Docker Compose.

## Local setup — Windows PowerShell

Run from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

## Train the models

```powershell
python -m training.train --data data/heart.csv --models models
```

Training takes a few minutes. It prints held-out accuracy, recall and ROC-AUC for each model,
plus 5-fold cross-validation accuracy for ML and DL.

A model is marked `validated` only if its held-out accuracy and recall are both at least 85%
(NFR-06). Only validated models can serve predictions. Do not use `--force-validate` to imply
that a model has passed a metric threshold when it has not; if you use it, report the real numbers.

### Training options

| Option | Default | Purpose |
|---|---|---|
| `--data` | `data/heart.csv` | Path to the dataset CSV |
| `--models` | `models` | Output folder |
| `--seed` | `42` | Random seed for the split and models |
| `--qubits` | `4` | Number of features fed to the QML circuit |
| `--qml-epochs` | `15` | QML training epochs |
| `--force-validate` | off | Mark models validated even if below 85% (report real numbers) |

### Training outputs

Files written to `models/`:

- `registry.json` — model versions, SHA-256 checksums, status and held-out metrics.
- `training_report.json` — dataset hash, training date, seed, DL settings, QML features and
  loss curve, held-out metrics (accuracy, recall, precision, F1, ROC-AUC, confusion matrix)
  and 5-fold cross-validation scores.
- `pipeline-1.0.joblib` — the shared preprocessing pipeline used for training and inference.
- `ml-1.0.joblib`, `dl-1.0.joblib`, `qml-1.0.joblib` — the trained models.
- `heldout.csv` — the test split (used for TC-22).

### Results (heart.csv, 80/20 stratified split, seed 42)

| Model | Held-out accuracy | Held-out recall | 5-fold CV accuracy | ROC-AUC | Status |
|---|---|---|---|---|---|
| ML (random forest) | 90.2% | 93.1% | 86.2% ± 3.5% | 0.927 | validated |
| DL (MLP, scikit-learn) | 87.0% | 91.2% | 85.7% ± 3.3% | 0.917 | validated |
| QML (4-qubit simulator) | 78.8% | 83.3% | not run | 0.863 | unvalidated (below 85%) |

Cross-validation is the more conservative estimate; a single held-out split can be optimistic.
Numbers can shift slightly with a different seed or library version. Re-run training and copy
the values from `models/training_report.json` to update this table.

### After retraining

Model versions keep the same names (`ml-1.0`, `dl-1.0`, `qml-1.0`), and the gateway imports
versions from `registry.json` only the first time it sees them. After retraining, either:

1. delete `instance/ischemia.db` and recreate the users, or
2. change the version names in `training/train.py` before training.

Then restart all prediction services and the gateway.

## Create user accounts

```powershell
python -m scripts.create_user admin Administrator "Adm1n!Passw0rd"
python -m scripts.create_user doc1 Clinician "Cl1nician!Passw0rd"
python -m scripts.create_user engineer1 "ML Engineer" "Eng1neer!Passw0rd"
```

Use passwords of 10 or more characters with an uppercase letter, a lowercase letter, a digit
and a symbol. The Administrator's user-creation API enforces this policy; the `create_user`
helper script does not, so follow it yourself. Use unique passwords in real deployments. Do not
commit credentials, `.env` files, database files, model artifacts containing sensitive data, or
production secrets to Git.

## Run the application

Start each service in a separate terminal from the project root, with the virtual environment
activated in each terminal.

Terminal 1 — ML:

```powershell
$env:MODEL_TYPE="ML"
$env:PORT="5001"
python -m services.model_service
```

Terminal 2 — DL:

```powershell
$env:MODEL_TYPE="DL"
$env:PORT="5002"
python -m services.model_service
```

Terminal 3 — QML:

```powershell
$env:MODEL_TYPE="QML"
$env:PORT="5003"
python -m services.model_service
```

Terminal 4 — Flask gateway:

```powershell
flask --app web.app run --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser and sign in using an account
created above. Select a prediction model, complete all 11 clinical inputs, and submit the form.

If a selected model is not registered, validated, active, or reachable, the API returns a
service-unavailable response. Start the relevant service and verify the model registry and
pipeline version before retrying. With the current training results the QML model is
`unvalidated`, so selecting it returns a service-unavailable response unless it was trained
with `--force-validate`.

### Server settings

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///ischemia.db` | Database connection |
| `SECRET_KEY` | development value | Signs session tokens; set a long random value outside development |
| `DATA_KEY` | development value | 64 hex characters; AES-256 key for stored inputs |
| `MODELS_DIR` | `models` | Folder with the registry, pipeline and models |
| `DEFAULT_MODEL` | `ML` | Model used when the request does not name one |
| `RISK_LOW` / `RISK_HIGH` | `30` / `70` | Risk category thresholds (percent) |
| `ML_URL`, `DL_URL`, `QML_URL` | `http://localhost:5001` / `5002` / `5003` | Addresses of the prediction services |

Generate a key with `python -c "import os; print(os.urandom(32).hex())"`.

## Model service endpoints (internal)

The prediction services are internal and are called only by the gateway.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Returns label, probability, top-5 feature contributions, model version, model type and latency. Rejects wrong length or type, `NaN`/`Infinity`, pipeline-version mismatch, and request bodies over 16 KB. |
| `GET /health` | Liveness check. |
| `GET /info` | Lists the model versions this service can serve, with their recorded metrics and the feature list. |

Model files are checked against the SHA-256 checksum in the registry before loading; a
tampered file is refused. Logs contain a request ID and timing only, never patient data.

## Errors

Every error returns `{"error_code", "message", "request_id", "fields"?}` with no stack traces,
SQL or patient data. Unexpected failures return `INTERNAL_ERROR`. Rate limiting returns
HTTP 429 with a `Retry-After: 60` header.

## Frontend guide

The frontend is served from `web/static/index.html`.

- **New prediction:** enter the 11 required clinical features and choose ML, DL, or QML.
  Numeric fields show their unit and valid range; categorical fields are dropdowns.
- **Prediction result:** review the returned label, probability, risk category, model version,
  top feature contributions, and disclaimer.
- **Prediction history:** review the authenticated user's saved prediction records.
- **Administration:** administrators can create users, inspect audit events, and activate
  validated models.
- **Model registry:** administrators and ML Engineers can inspect registered model versions.

The browser performs usability validation, while the Flask API remains authoritative for
input validation and access control. Never treat client-side validation as a security boundary.

## Docker (optional)

`Dockerfile` and `docker-compose.yml` start the gateway, the three prediction services and
PostgreSQL. Only the gateway publishes a port (5000).

1. Train the models first (they are mounted from `./models`).
2. In `docker-compose.yml`, replace `SECRET_KEY` and `DATA_KEY` with real values.
3. Run:

```bash
docker compose up --build
docker compose exec web python -m scripts.create_user doc1 Clinician "Cl1nician!Passw0rd"
```

Kubernetes manifests are not included in this release.

## Run tests

Install the project dependencies, then run:

```powershell
python -m pytest -q
```

The suite has 28 automated tests in three files:

- **`tests/test_api.py`** — gateway: login, lockout, validation, prediction, history isolation,
  role-based access, password policy, service failure and rate limiting. They use an isolated
  SQLite database, a temporary preprocessing pipeline, and a mocked prediction-service call, so
  no model services need to be running.
- **`tests/test_service.py`** — prediction service: successful prediction, rejection of
  `NaN`/`Infinity`, wrong length and wrong types, pipeline-version mismatch, oversized requests,
  tampered model files, and `/info`. They build a small temporary model.
- **`tests/test_tc_extra.py`** — cases written to match Test Plan IDs (TC-03, 04, 06, 09, 12,
  16, 17, 19, 21, 22, 24) plus the generic `INTERNAL_ERROR` handler.

For a focused run:

```powershell
python -m pytest tests/test_api.py -v
python -m pytest tests/test_service.py -v
python -m pytest tests/test_tc_extra.py -v
```

Coverage (measured at 73% of the gateway, services and shared code; training scripts are not
included):

```powershell
pip install pytest-cov
python -m pytest -q --cov=web --cov=services --cov=common
```

## Git workflow

Create a feature branch, make the change, run tests, and push the branch:

```bash
git switch main
git pull origin main
git switch -c feature/model-enhancements

python -m pytest -q

git add -A
git commit -m "Describe the change"
git push -u origin feature/model-enhancements
```

Open a pull request targeting the agreed integration branch. If another branch (for example
the UI redesign) has not yet been merged into `main`, coordinate the PR base with the team to
avoid losing that work.

## Deviations from the SRS and SAD

These are recorded in full in SRS Appendix D. Summary:

| # | Area | Documents said | What was built |
|---|---|---|---|
| D1 | Dataset | UCI heart disease, final choice pending (TBD-1) | `data/heart.csv`, 918 records; 80/20 stratified split, seed 42 |
| D2 | Inputs | 9 fields | 11 fields (adds fasting blood sugar and ST slope); categories are integer codes; cholesterol and resting BP of 0 treated as missing and imputed |
| D3 | Target | Myocardial ischemia | `HeartDisease` used as a proxy; the UI still says "Ischemia Likely/Unlikely" |
| D4 | DL service | TensorFlow/PyTorch | scikit-learn MLP |
| D5 | QML service | Quantum SDK, 85% target | PennyLane simulator, 4 qubits; 78.8% accuracy and 83.3% recall, so `unvalidated` and not served |
| D6 | Explanation | Top-5 features | Leave-one-out attribution for all models; internal feature names are shown |
| D7 | TLS | TLS 1.2+ | The development server uses HTTP; TLS is expected at an ingress or reverse proxy; stored inputs are AES-256-GCM encrypted |
| D8 | Kubernetes | Orchestrated by Kubernetes | Dockerfile and Docker Compose only (only the web port is published) |
| D9 | Database | PostgreSQL with migration scripts | SQLite locally, PostgreSQL via `DATABASE_URL` in Compose; tables created by the app; no migration scripts |
| D10 | Configuration | Administrator changes thresholds and default model | Server settings `RISK_LOW`, `RISK_HIGH`, `DEFAULT_MODEL` |
| D11 | Rate limit | 60 per minute per user | In-memory, per process, authenticated endpoints only |
| D12 | Pseudonymisation | Replace the patient identifier | `patient_ref` is auto-generated when omitted; no replace function |
| D13 | Audit log | Retained 1 year | Login, prediction, deletion, model activation and user creation are logged; nothing is purged and no retention job exists |
| D14 | Admin functions | Admin UI | Deleting a prediction and registering a model are API-only; no change-role call |
| D15 | Error handling | Offer another model when one is down | Entered data is kept; the user selects another model manually |
| D16 | UI | Separate result page, tooltips, HTML user guide | Result on the Predict page; units and ranges in labels; this README is the user guide |
| D17 | Verification | NFR-01, 02, 03, 08, 09, 11 | Not measured (load, browser and usability tests not run) |
| D18 | Retraining | Not specified | Version names are reused; see "After retraining" above |
| D19 | `create_user` script | Password policy (SEC-01) | The script skips the policy check; the admin API enforces it |
| D20 | Internal API | `/predict`, `/health` | Also `GET /info`, `latency_ms`, `NaN`/`Infinity` rejection and a 16 KB body limit |

## Important project limitations

- Heart-disease dataset labels are a proxy and do not establish clinical ischemia validity.
- Model performance depends on the training data, split strategy, preprocessing, and evaluation.
- QML does not reach the 85% target and is not served unless trained with `--force-validate`,
  in which case the real numbers must be reported.
- DL/QML implementations and their performance should be described accurately in the project demo.
- Load, TLS, network-isolation, browser-compatibility and usability tests have not been run
  (see the Test Plan, section 18.1).
- Docker support is provided but is not covered by the automated tests.
- The development Flask server is not a production deployment server.
- Production deployment requires secure secret management, HTTPS/TLS, database hardening,
  access logging and retention policy, monitoring, and clinical validation.
- Do not enter real patient identifiers or sensitive patient records into an unapproved deployment.
