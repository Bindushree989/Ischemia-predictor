# Ischemia Predictor Project Guide

This guide explains the six trained model candidates, what the tests establish, how the GUI and
MLflow fit together, how to start the application, and what the maintained project folders contain.

> **Safety:** This is a research/demo decision-support prototype using the public heart-disease
> dataset as a proxy task. It is not a clinically validated ischemia diagnostic system and must
> not be used to diagnose or treat patients.

## Quick start

### Recommended: Docker Compose on Windows

1. Install and start Docker Desktop, then open PowerShell in the repository root.
2. Run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
   ```

3. The script creates local secrets if needed, starts MLflow, runs `dvc repro`, builds and starts
   the app stack, then securely prompts you to create an Administrator and Clinician account.
4. Open the GUI at [http://localhost:5000](http://localhost:5000). Open MLflow at
   [http://localhost:5001](http://localhost:5001).
5. To stop the stack, run `docker compose down`. This keeps the PostgreSQL and MLflow data volumes.

The GUI's **Model registry** view compares candidate metrics and links to MLflow. In the
prediction form, only currently active, validated versions are offered. At the recorded results,
that means `ml-1.0` and `dl-1.0`; QML candidates remain visible in the registry but are not
selectable because neither passed the validation gate. Administrators can activate `ml-2.0` and
`dl-2.0` independently without replacing the original versions; the Clinician picker then lists
all four active, validated versions.

### Python-only development run

Use the project virtual environment and create artifacts first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
dvc repro
```

Open five additional PowerShell terminals at the repository root. Activate the environment in
each with `.\.venv\Scripts\Activate.ps1` before running its process. Use the same database value
in the gateway and user-creation terminal.

**Terminal 1 — MLflow** (port 5001):

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

**Terminal 2 — ML service** (port 5101):

```powershell
$env:MODEL_TYPE="ML"; $env:PORT="5101"
python -m services.model_service
```

**Terminal 3 — DL service** (port 5102):

```powershell
$env:MODEL_TYPE="DL"; $env:PORT="5102"
python -m services.model_service
```

**Terminal 4 — QML service** (port 5103):

```powershell
$env:MODEL_TYPE="QML"; $env:PORT="5103"
python -m services.model_service
```

**Terminal 5 — Gateway and GUI** (port 5000):

```powershell
$dbPath = Join-Path $env:TEMP "ischemia-predictor-demo.db"
$env:DATABASE_URL = "sqlite:///$($dbPath -replace '\\','/')"
$env:ML_URL = "http://127.0.0.1:5101"
$env:DL_URL = "http://127.0.0.1:5102"
$env:QML_URL = "http://127.0.0.1:5103"
flask --app web.app run --host 127.0.0.1 --port 5000
```

The local Python example uses development-only default signing/encryption keys; never expose this
configuration outside localhost or reuse it for a durable/shared database.

To view the model registry, create an Administrator; to submit predictions, create a Clinician.
In another terminal, set the same `DATABASE_URL` and create either or both accounts:

```powershell
$dbPath = Join-Path $env:TEMP "ischemia-predictor-demo.db"
$env:DATABASE_URL = "sqlite:///$($dbPath -replace '\\','/')"
function New-LocalDemoUser($username, $role) {
  $securePassword = Read-Host "$role password" -AsSecureString
  $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
  try {
    [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer) |
      python -m scripts.create_user $username $role --password-stdin
    if ($LASTEXITCODE -ne 0) { throw "Could not create $role account." }
  }
  finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
  }
}
New-LocalDemoUser admin Administrator
New-LocalDemoUser clinician Clinician
```

The password is entered masked and passed to the account helper over standard input. The account
is created in the temporary database under `%TEMP%`, not in the tracked `instance/ischemia.db`.
Stop each process with Ctrl+C. The MLflow link in the app expects the tracking UI at port 5001.

### Train or retrain

Start MLflow first if you want to view the new runs, then run:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
dvc repro
```

If DVC reports that the stage is already current, no new run is created. For a deliberate fresh
training run, use:

```powershell
python -m training.train --data data/heart.csv --models models
```

Training replaces generated artifacts under `models/` and logs six model runs to the configured
MLflow experiment. Restart the gateway and all three services after changing model artifacts.
Avoid retraining over a model version already synchronized into a persistent application database;
assign new version identifiers before deployment.

## The six model candidates

All candidates use the shared preprocessing pipeline and the same stratified 80/20 split of
`data/heart.csv` (918 rows; 734 training and 184 held out, seed 42). Accuracy, recall, precision,
F1, ROC-AUC, confusion counts, and the model version are written to `models/training_report.json`.
ML/DL candidates also report five-fold cross-validation scores on the training partition.

| Family / version | Framework and model | How it works | Held-out accuracy / recall | Registry state |
|---|---|---|---:|---|
| ML `ml-1.0` | scikit-learn Random Forest | Averages votes from 400 decision trees; each tree sees bootstrapped data, with leaves constrained by `min_samples_leaf=2`. | 90.22% / 93.14% | Validated, active |
| ML `ml-2.0` | XGBoost Classifier | Boosts 300 shallow trees in sequence; each new tree reduces the ensemble's remaining classification error. | 85.87% / 85.29% | Validated, inactive |
| DL `dl-1.0` | scikit-learn tuned MLP | A feed-forward neural network; five-fold grid search selects hidden-layer size and regularization (`(16,)`, alpha 1.0 in this run). | 86.96% / 91.18% | Validated, active |
| DL `dl-2.0` | scikit-learn deep MLP | A feed-forward network with three hidden layers (64, 32, 16), early stopping, and regularization. | 87.50% / 90.20% | Validated, inactive |
| QML `qml-1.0` | PennyLane variational classifier | Encodes four selected features as qubit rotations, applies three trainable entangling layers, and calibrates outputs. | 77.72% / 76.47% | Unvalidated |
| QML `qml-2.0` | PennyLane variational classifier | Same feature encoding and training process with five trainable entangling layers. | 76.63% / 80.39% | Unvalidated |

The promotion gate requires **both** held-out accuracy and recall to be at least 85%. Do not use
`--force-validate` to represent a failed metric gate as a pass. XGBoost and the deeper MLP pass
but are inactive until an Administrator activates them in the registry. Activating one does not
deactivate its siblings, and the GUI lists each activated version separately. QML is inspectable
but deliberately absent from the prediction selector until a QML candidate passes the gate.

### Why the QML candidates are blocked

The QML artifacts are trained and can produce probabilities when loaded directly, but their
recorded held-out accuracy / recall are `77.72% / 76.47%` for `qml-1.0` and `76.63% / 80.39%` for
`qml-2.0`. Both are below the required 85% threshold for **each** metric. Their registry status is
therefore `unvalidated`; the model service rejects direct prediction requests with
`model not registered/validated`, and the Clinician picker omits them. This is intentional
validation behavior, not a missing file or Windows permission problem.

The Clinician role can use every active, validated model version (the four passing ML/DL versions),
but the Administrator role cannot promote a model that fails the gate. Windows Administrator
privileges do not change the app-level validation state. Do not edit the registry status or use
`--force-validate` to expose these candidates for patient-facing predictions. To make a QML model
eligible, retrain/tune it using the training data and only activate it if both held-out metrics
pass; these dataset metrics still do not establish clinical validity.

These are measurements on one public dataset split—not evidence of clinical validity or
generalization to real patients.

## What the GUI does

- **Sign-in:** authenticates the account and sets the available navigation according to its role.
- **Prediction:** validates 11 input features, submits them to the gateway, and displays the model
  version, probability, risk category, feature contributions, and safety disclaimer.
- **Available model picker:** requests `/api/v1/prediction-models`; the API only returns active,
  validated versions. A missing model is shown as a useful empty state, not as a selectable option.
- **History:** lets a Clinician view only their own saved prediction history.
- **Administration / Model registry:** shows candidate status and metrics; an Administrator can
  activate validated alternatives. ML Engineers can inspect the registry.
- **MLflow:** the registry page's **Open MLflow** link opens the tracking UI on port 5001.

The Python-only app is served at `http://127.0.0.1:5000`; Compose serves it at
`http://localhost:5000`. The form is not a clinical diagnostic tool.

## MLflow and DVC

MLflow records one run per candidate in the `ischemia-predictor` experiment. Each run includes
dataset hash, seed, framework/algorithm, relevant hyperparameters, held-out metrics, validation
and active flags, model artifact, preprocessing artifact, and a registry-entry JSON. Classical
ML and DL runs also include their cross-validation metrics.

DVC's `dvc.yaml` and `dvc.lock` describe the training stage and its generated artifacts. Local
training uses `mlflow.db`/`mlruns` by default; Compose runs a persistent MLflow server and keeps
its database and artifacts in the `mlflow_data` volume. The Model Registry GUI link needs the
tracking UI to be running.

## Test plan: what is checked

### Run the automated tests (Windows PowerShell)

From the repository root, use the project environment. Setting `FLASK_AUTOCREATE=0` prevents
test imports from initializing the default tracked SQLite database:

```powershell
$env:FLASK_AUTOCREATE = "0"
$env:DATABASE_URL = "sqlite://"
.\.venv\Scripts\python.exe -m pytest -q
```

Run an individual test file when debugging:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_service.py -q
```

`tests\test_api.py` covers the web/API and role contracts; `tests\test_service.py` covers the
prediction microservice. `dvc repro` separately runs model training/evaluation and MLflow logging;
it is not a replacement for the automated test suite.

### Run the Playwright browser test

Playwright automates a real browser, so it checks the GUI and live API/service wiring rather than
only calling Python test clients. The configuration is in `playwright.config.js`; the tests are in
`tests/e2e/model-selection.spec.js`. They check the unauthenticated login screen and required
credentials, Clinician access and model availability, required clinical inputs and age bounds,
theme switching, sign-out, and predictions for every active validated model version. Prediction
tests submit synthetic measurements and save results in the app database; use a disposable
local/test database, never real patient data.

Install the JavaScript test dependency and Chromium once:

```powershell
npm.cmd ci
npm.cmd run test:e2e:install
```

If the local Windows policy blocks the downloaded Chromium executable, do not run the test as
Administrator to bypass that policy. Use an organization-approved browser/runtime or ask the
machine administrator to approve the Playwright browser. If Microsoft Edge is installed and
approved, select it with `$env:PLAYWRIGHT_BROWSER_CHANNEL = "msedge"` instead of installing
Playwright's Chromium.

Start the application and its model services as described in the Python-only or Docker setup.
Then set credentials for an existing local Clinician account in a separate PowerShell terminal
and run the test:

```powershell
$env:PLAYWRIGHT_BASE_URL = "http://127.0.0.1:5000"
$env:PLAYWRIGHT_USERNAME = "clinician"
$securePassword = Read-Host "Test account password" -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
  $env:PLAYWRIGHT_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
  npm.cmd run test:e2e
  if ($LASTEXITCODE -ne 0) { throw "Playwright test failed." }
}
finally {
  Remove-Item Env:PLAYWRIGHT_PASSWORD -ErrorAction SilentlyContinue
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
}
```

Use `npm.cmd run test:e2e -- --headed` to watch the browser. The two login-page checks run without
credentials; Clinician-only tests skip if the username or password is not set. A failure trace is
retained under `test-results\`; generated reports and traces are ignored by Git. The HTML report is
written to `playwright-report\index.html`; open it with `npm.cmd run test:e2e:report`. Playwright
is a separate browser test runner and is not included in `pytest -q`.

| Test level | How to run / inspect | What it establishes | What it does not establish |
|---|---|---|---|
| Unit and API contract | `python -m pytest tests/test_api.py -q` | Login, role permissions, input validation, errors, history, model registry, activation, and frontend contract. Model-service calls are mocked. | Does not prove live service availability or real model performance. |
| Inference service | `python -m pytest tests/test_service.py -q` | Serialized model predictions, probabilities, all registered versions, input shape/type handling, pipeline checks, checksums, and service metadata. | Does not evaluate clinical safety or external validity. |
| Training/evaluation | `dvc repro` | Trains the six candidates, writes model artifacts and reports, evaluates the held-out split, runs ML/DL cross-validation, and logs MLflow runs. | A single split and cross-validation are not independent clinical validation. |
| Full automated suite | `python -m pytest -q` | Runs the repository's API and service regression tests together. | Does not exercise a live browser, Docker deployment, or MLflow UI. |
| Playwright browser E2E | `npm.cmd run test:e2e` | Checks login validation, model availability, clinical form constraints, theme, sign-out, and exact-version predictions in the live GUI. | Clinician checks require a running app/services and test account; does not establish clinical validity. |
| Manual system/UI | Sign in; inspect registry; activate/deactivate validated versions; inspect result/history and MLflow. | Confirms the full application can be exercised interactively. | This manual smoke test is not a substitute for clinical validation or a production readiness review. |

See [TEST_PLAN.md](TEST_PLAN.md) for pass criteria, recommended run order, recorded evidence, and
checks that still require separate validation.

For a simple manual GUI smoke test, verify that `ML (ml-1.0)` and `DL (dl-1.0)` appear with the
initial registry and that QML does not. Activate `ml-2.0` and `dl-2.0` in the Administrator
registry; verify the Clinician picker lists all four versions and can produce a result for each
exact selection. Deactivate an alternative and verify only that version disappears. Never
activate or select an unvalidated model.

## Folder map

The following summaries cover the maintained top-level project folders. Each folder is described
in five short lines; local environments and caches are listed separately below.

### `.dvc/`
- Stores DVC's local repository configuration.
- Holds DVC's small metadata files.
- The pipeline definition itself is in root `dvc.yaml`.
- The committed lock state is root `dvc.lock`.
- Large DVC cache data is local and is not committed.

### `.github/`
- Contains repository automation configuration.
- `workflows/python-ci.yml` defines CI checks.
- CI installs project dependencies.
- CI runs automated Python tests.
- Workflow changes affect pull-request validation.

### `.github/workflows/`
- Holds GitHub Actions workflow definitions.
- `python-ci.yml` is the current workflow.
- The workflow checks changes in CI.
- It installs dependencies before testing.
- It runs the automated test suite.

### `common/`
- Defines the 11 model input features and valid ranges.
- Loads and validates the heart dataset.
- Builds the shared preprocessing pipeline.
- Contains the PennyLane QML classifier.
- Is used by both training and inference.

### `data/`
- Contains the input heart-disease dataset.
- `heart.csv` has 918 records.
- `HeartDisease` is the binary prediction target.
- The data is split reproducibly during training.
- It is a proxy dataset, not clinical ischemia evidence.

### `instance/`
- Contains the Flask application's local SQLite database.
- The tracked `ischemia.db` may hold local account and app state.
- The gateway stores users, model registry rows, history, and audit events here by default.
- Automated API tests use isolated temporary databases instead.
- Avoid running demos or tests against this tracked database.

### `k8s/`
- Contains Kubernetes deployment configuration.
- `deployment.yaml` defines the app and model services.
- It configures internal service networking and probes.
- PostgreSQL storage and service configuration are included.
- Use only when a Kubernetes cluster is enabled.

### `models/`
- Holds generated serialized model candidates.
- Stores the shared fitted preprocessing pipeline.
- `registry.json` records versions, checksums, status, and metrics.
- `training_report.json` records the detailed evaluation results.
- DVC regenerates/updates these outputs; some artifacts are Git-ignored.

### `scripts/`
- Contains user/bootstrap and launch helpers.
- `create_user.py` creates the first app account.
- `start.ps1` starts the Docker Compose app and prompts for accounts.
- `run-all.ps1` combines DVC, Compose, and Kubernetes workflows.
- Use PowerShell from the repository root.

### `services/`
- Implements the model inference microservice.
- Loads registered model artifacts by version.
- Verifies model files against their SHA-256 checksums.
- Validates prediction request shapes and values.
- Exposes `/predict`, `/health`, and `/info`.

### `tests/`
- Contains automated API and service tests.
- `test_api.py` covers authentication, validation, history, and registry behavior.
- `test_service.py` covers inference and service error handling.
- `e2e/` contains the separate Playwright browser smoke test.
- Python tests use temporary databases and model files.

### `tests/e2e/`
- Contains Playwright end-to-end browser tests.
- `model-selection.spec.js` checks login, Clinician model selection, form validation, theme, and sign-out.
- Prediction checks confirm the result version matches each selected active version.
- Uses synthetic inputs and a live local application.
- Clinician checks require npm dependencies, a browser, and test-account environment variables.

### Playwright configuration
- `playwright.config.js` sets the E2E test folder and local base URL.
- `package.json` defines the Playwright test and browser-install commands.
- `package-lock.json` pins the JavaScript test dependencies.
- `PLAYWRIGHT_BROWSER_CHANNEL` can select an installed, approved browser channel.
- `npm.cmd run test:e2e` runs the browser test.
- Browser traces and reports are generated outputs, not source files.
- Prediction checks write to the configured app database; use a disposable local/test database.

### `tests/tests/`
- Contains a secondary test-helper package.
- `util.py` provides shared test support.
- It is not a production application package.
- Main test cases are stored directly in `tests/`.
- The package keeps reusable test utilities separate.

### `training/`
- Contains the offline training pipeline.
- `train.py` builds and evaluates all six candidates.
- Training writes model artifacts and the model registry.
- Each candidate is logged as an MLflow run.
- `training/README.md` documents the training process.

### `training/training/`
- Contains documentation for the training process.
- `README.md` describes the candidate families.
- It records the shared split and validation gate.
- It summarizes outputs and MLflow tracking.
- Update it when training behavior changes.

### `web/`
- Contains the Flask API gateway and app factory.
- Enforces authentication, roles, and input validation.
- Orchestrates inference calls to the selected family service.
- Serves the browser app from `static/index.html`.
- Persists encrypted prediction inputs and audit events.

### `web/static/`
- Contains browser-facing static assets.
- `index.html` holds the app markup and styles.
- Its JavaScript handles API requests and navigation.
- The browser uses the Flask gateway as its API.
- The model picker is populated from validated active versions.

Local-only or generated folders such as `.venv/`, `.pytest_cache/`, `mlruns/`, and DVC cache
contents are environment outputs, not maintained source folders. `models/` also contains generated
artifacts; use the registry/report and DVC pipeline rather than editing serialized model files.
