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

Review the training output and generated model registry. A model must meet the project's
validation criteria before it can be activated. Do not use a force-validation option to imply
that a model has passed a metric threshold when it has not.

Create initial accounts using the project's helper:

```powershell
python -m scripts.create_user admin Administrator "Adm1n!Passw0rd"
python -m scripts.create_user doc1 Clinician "Cl1nician!Passw0rd"
python -m scripts.create_user engineer1 "ML Engineer" "Eng1neer!Passw0rd"
```

Use unique passwords in real deployments. Do not commit credentials, `.env` files, database
files, model artifacts containing sensitive data, or production secrets to Git.

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
pipeline version before retrying.

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

The API tests use an isolated SQLite database, a temporary preprocessing pipeline, and a mocked
prediction-service call. They do not require the ML, DL, or QML services to be running.

For a focused test run:

```powershell
python -m pytest tests/test_api.py -v
```

## Git workflow for UI contribution

Create a feature branch, make the change, run tests, and push the branch:

```bash
git switch main
git pull origin main
git switch -c feature/ui-testing-documentation

python -m pytest -q

git add tests/test_api.py README.md
git commit -m "test(docs): expand API coverage and document UI"
git push -u origin feature/ui-testing-documentation
```

Open a pull request targeting the agreed integration branch. If the UI redesign branch has not
yet been merged into `main`, coordinate the PR base with the team to avoid losing the UI work.

## Important project limitations

- Heart-disease dataset labels are a proxy and do not establish clinical ischemia validity.
- Model performance depends on the training data, split strategy, preprocessing, and evaluation.
- DL/QML implementations and their performance should be described accurately in the project demo.
- The development Flask server is not a production deployment server.
- Production deployment requires secure secret management, HTTPS/TLS, database hardening,
  access logging and retention policy, monitoring, and clinical validation.
- Do not enter real patient identifiers or sensitive patient records into an unapproved deployment.
