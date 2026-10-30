# Ischemia Predictor

A clinical decision-support application that combines a Flask API gateway with ML, DL,
and QML prediction services. The gateway handles authentication, role-based access control,
input validation, preprocessing, prediction orchestration, history, model registry, and audit logs.

> **Clinical safety:** This application is a prototype decision-support tool. Its output is
> not a medical diagnosis and must not be used as the sole basis for clinical decisions.
> A qualified healthcare professional must interpret results in the appropriate clinical context.

## Project overview

- **Frontend:** HTML, CSS, and browser JavaScript served by Flask.
- **Gateway:** Flask application in `web/app.py`.
- **Prediction services:** ML, DL, and QML services in `services/model_service.py`.
- **Preprocessing:** Shared pipeline in `common/`.
- **Training:** Offline model training under `training/`.
- **Tests:** Pytest suite under `tests/`.

The dataset currently documented by the project is `data/heart.csv` (918 records, 11 input
features, target `HeartDisease`). Heart-disease prediction is being used as a proxy task;
it should not be represented as a clinically validated ischemia diagnostic model.

## User roles

| Role | Supported UI/API capabilities |
|---|---|
| Clinician | Create predictions and view own prediction history |
| Administrator | View own history, create users, inspect audit logs, activate validated models |
| ML Engineer | Create predictions and inspect the model registry |

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

Restart the prediction services after retraining, because they cache loaded models.

### Results (heart.csv, 80/20 stratified split, seed 42)

| Model | Held-out accuracy | 5-fold CV accuracy | ROC-AUC | Status |
|---|---|---|---|---|
| ML (random forest) | 90.2% | 86.2% ± 3.5% | 0.927 | validated |
| DL (MLP, scikit-learn) | 87.0% | 85.7% ± 3.3% | 0.917 | validated |
| QML (4-qubit simulator) | 78.8% | not run | 0.863 | unvalidated (below 85%) |

Cross-validation is the more conservative estimate; the single held-out split can be optimistic.
Numbers can shift slightly with a different seed or library version. Re-run training and copy
the values from `models/training_report.json` to update this table.

## Create user accounts

```powershell
python -m scripts.create_user admin Administrator "Adm1n!Passw0rd"
python -m scripts.create_user doc1 Clinician "Cl1nician!Passw0rd"
python -m scripts.create_user engineer1 "ML Engineer" "Eng1neer!Passw0rd"
```

Passwords need 10 or more characters with an uppercase letter, a lowercase letter, a digit and
a symbol. Use unique passwords in real deployments. Do not commit credentials, `.env` files,
database files, model artifacts containing sensitive data, or production secrets to Git.

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

If a selected model is not registered, validated, active, or reachable, the API may return a
service-unavailable response. Start the relevant service and verify the model registry and
pipeline version before retrying. With the current training results, the QML model is
`unvalidated`, so selecting it returns a service-unavailable response unless it was trained
with `--force-validate`.

## Model service endpoints (internal)

The prediction services are internal and are called only by the gateway.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Returns label, probability, top-5 feature contributions, model version and latency. Rejects wrong length or type, `NaN`/`Infinity`, pipeline-version mismatch, and request bodies over 16 KB. |
| `GET /health` | Liveness check. |
| `GET /info` | Lists the model versions this service can serve, with their recorded metrics and the feature list. |

Model files are checked against the SHA-256 checksum in the registry before loading; a
tampered file is refused. Logs contain a request ID and timing only, never patient data.

## Frontend guide

The frontend is served from `web/static/index.html`.

- **New prediction:** enter the 11 required clinical features and choose ML, DL, or QML.
- **Prediction result:** review the returned label, probability, risk category, model version,
  top feature contributions, and disclaimer.
- **Prediction history:** review the authenticated user's saved prediction records.
- **Administration:** administrators can create users, inspect audit events, and activate
  validated models.
- **Model registry:** administrators and ML Engineers can inspect registered model versions.
- **Theme and responsive layout:** use the theme control and the navigation adapts to smaller screens.

The browser performs usability validation, while the Flask API remains authoritative for
input validation and access control. Never treat client-side validation as a security boundary.

## Run tests

Install the project dependencies, then run:

```powershell
python -m pytest -q
```

The suite has two parts:

- **`tests/test_api.py`** — gateway checks: login, lockout, validation, prediction, history
  isolation, role-based access, password policy, service failure and rate limiting. They use an
  isolated SQLite database, a temporary preprocessing pipeline, and a mocked prediction-service
  call, so no model services need to be running.
- **`tests/test_service.py`** — prediction-service checks: successful prediction, rejection of
  `NaN`/`Infinity`, wrong length and wrong types, pipeline-version mismatch, oversized requests,
  tampered model files, and the `/info` endpoint. They build a small temporary model, so trained
  models are not required.

For a focused run:

```powershell
python -m pytest tests/test_api.py -v
python -m pytest tests/test_service.py -v
```

## Git workflow

Create a feature branch, make the change, run tests, and push the branch:

```bash
git switch main
git pull origin main
git switch -c feature/model-enhancements

python -m pytest -q

git add common/qml_model.py services/model_service.py training/train.py tests/test_service.py tests/util.py README.md
git commit -m "Add CV metrics, training report, calibrated mini-batch QML, hardened model service"
git push -u origin feature/model-enhancements
```

Open a pull request targeting the agreed integration branch. If another branch (for example
the UI redesign) has not yet been merged into `main`, coordinate the PR base with the team to
avoid losing that work.

## Deviations from the SRS and SAD

- **Dataset:** `heart.csv` (918 rows). The label "heart disease" is used as a proxy for ischemia.
- **Inputs:** categorical fields (sex, chest pain type, resting ECG, exercise angina, ST slope)
  are encoded as integers. Cholesterol and resting blood pressure values of 0 mean "not
  measured" and are treated as missing, then imputed.
- **DL:** a scikit-learn MLP, not TensorFlow or PyTorch.
- **QML:** a PennyLane simulator (4 qubits, 3 layers, mini-batch training, Platt-calibrated
  probabilities). It does not reach the 85% target.
- **Explanations:** the top-5 feature contributions use leave-one-out attribution, the same
  method for all three models.
- **Transport and limits:** the development server uses HTTP (TLS is expected at the ingress),
  and the rate limit is in-memory.
- **Not implemented:** pseudonymous-ID replacement (SEC-10), Kubernetes manifests, and the
  log-retention job.

## Important project limitations

- Heart-disease dataset labels are a proxy and do not establish clinical ischemia validity.
- Model performance depends on the training data, split strategy, preprocessing, and evaluation.
- DL/QML implementations and their performance should be described accurately in the project demo.
- The development Flask server is not a production deployment server.
- Production deployment requires secure secret management, HTTPS/TLS, database hardening,
  access logging and retention policy, monitoring, and clinical validation.
- Do not enter real patient identifiers or sensitive patient records into an unapproved deployment.
