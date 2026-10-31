# Ischemia Predictor

Flask gateway (auth, RBAC, validation, preprocessing, orchestration, history, model registry,
audit log) + three prediction microservices (ML, DL, QML) + offline training pipeline.
Dataset: `data/heart.csv` (918 patients, 11 inputs, label `HeartDisease`). Python 3.10+.

## Setup (once) - Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
dvc repro
python -m scripts.create_user admin Administrator YOUR_STRONG_ADMIN_PASSWORD
python -m scripts.create_user doc1 Clinician YOUR_STRONG_CLINICIAN_PASSWORD
```
Replace the password placeholders with a non-empty password; the built-in role is
selected from the account, not from a separate login option.
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.
Training takes about 1-2 minutes and prints accuracy/recall per model. A model needs
>= 85% accuracy AND recall (NFR-06) to be `validated`; only validated models can predict.
(`--force-validate` overrides this; if you use it, report the real numbers.)

## Run (every time) - 4 terminals, each: `cd <project>` then `.venv\Scripts\activate`
```powershell
$env:MODEL_TYPE="ML";  $env:PORT="5001"; python -m services.model_service     # terminal 1
$env:MODEL_TYPE="DL";  $env:PORT="5002"; python -m services.model_service     # terminal 2
$env:MODEL_TYPE="QML"; $env:PORT="5003"; python -m services.model_service     # terminal 3
flask --app web.app run -p 5000                                               # terminal 4
```
Open http://localhost:5000, log in as `doc1`, fill the form, choose a model, press Predict.
Restart all terminals after retraining.

## Tests
`pytest -q`  (automated checks for TC-01..05, 07, 09, 10, 11, 17, 21)

## Start the whole application (Windows PowerShell)
Install Docker Desktop for Windows with Winget (`winget install --id Docker.DockerDesktop`)
or the official installer, then launch it and wait until its engine is running. Docker
Desktop includes Docker Compose and `kubectl`. Start the Compose application with one
command from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

The script checks Docker and the DVC configuration, creates `.env` with fresh secrets
if needed, starts MLflow, runs `dvc repro`, builds and starts PostgreSQL and the
web/ML/DL/QML services, waits for the UI to respond, and prompts for an Administrator
password for the `admin` username, then prompts for a Clinician username and password.
Passwords are sent to the container through standard input; existing accounts are left
unchanged. Any non-empty password is accepted. Open the app at http://localhost:5000 and MLflow at
http://localhost:5001.

To stop the app, run `docker compose down`. PostgreSQL data is kept in the
`postgres_data` named volume; `docker compose down -v` also deletes that database.

## Docker Compose details
The image contains the web gateway, all three model services, and the trained model
artifacts from `models/`. Compose starts distinct ML, DL, and QML containers, PostgreSQL,
the web gateway, and MLflow; the web UI is published on port 5000 and MLflow is restricted
to localhost on port 5001. The host `models/` directory is shared with the containers
(read-only in the prediction services), so the gateway and all model services see the
same registry and artifacts.

For manual configuration, copy `.env.example` to `.env` and replace its values with
unique secrets. `DATA_KEY` must be exactly 64 hexadecimal characters (32 bytes). Keep
`.env` private.

## DVC and MLflow status
DVC tracks the training stage and generated model artifacts in `dvc.yaml` and `dvc.lock`;
the dataset remains Git-versioned and is recorded as a training dependency. DVC's local
cache is under `.dvc/cache` and is ignored by Git. To reproduce training manually with the
Docker Compose environment, run `docker compose run --rm trainer`. No shared DVC remote
is configured, so sharing the cache across machines requires configuring and securing a
remote with `dvc remote add` and `dvc push`.

MLflow is a Compose service at http://localhost:5001. It stores run metadata in a
persistent local SQLite database and proxies run artifacts into the `mlflow_data` volume.
The training pipeline logs one MLflow run per ML, DL, and QML model, including dataset
hash, validation status, accuracy, recall, precision, F1, ROC AUC, model-specific
parameters, registry entry, model, and preprocessing artifacts. The startup script waits
for MLflow, runs `dvc repro`, then starts the
application. DVC and MLflow state are local-only; configure a shared DVC remote and
authenticated, durable MLflow backend/artifact storage before using them across machines
or in production. Existing runs made before artifact-proxy configuration are retained in
the previous SQLite file; new runs use the correctly proxied artifact store.

## Enable Kubernetes
Docker Desktop's Kubernetes cluster is optional for the one-command Compose workflow.
To run the Kubernetes deployment, open Docker Desktop **Settings > Kubernetes**, enable
Kubernetes, and apply/restart. Wait for the cluster to become Ready, then apply the
manifest below. This creates the `ischemia-app` ServiceAccount and assigns it to the app
pods; the service account has no Kubernetes API permissions and does not mount a token.

## Kubernetes deployment
`k8s/deployment.yaml` defines PostgreSQL with persistent storage, the web gateway, and
separate internal Services and Deployments for ML, DL, and QML. The model services have
readiness/liveness probes; the web service is a ClusterIP and is not exposed publicly
by default. Pods use the dedicated `ischemia-app` Kubernetes ServiceAccount, with
automatic API token mounting disabled because the app does not call the Kubernetes API.

With Docker Desktop Kubernetes enabled, build the image from the repository root; the
cluster uses the same Docker Desktop image store. Ensure model files exist first by
installing the project requirements and running `dvc repro`; a fresh checkout regenerates
them locally because no shared DVC remote is configured. Docker Desktop's `docker.exe` and
`kubectl.exe` may not be on `PATH`, so add their bundled directory in PowerShell:

```powershell
$dockerBin = Join-Path $env:ProgramFiles "Docker\Docker\resources\bin"
$env:PATH = "$dockerBin;$env:PATH"
docker build -t ischemia-predictor:local .
```

Create Kubernetes secrets (use fresh values; `DATA_KEY` must be 64 hex characters):

```powershell
kubectl create namespace ischemia
$postgresPassword = python -c "import secrets; print(secrets.token_hex(32))"
$secretKey = python -c "import secrets; print(secrets.token_hex(32))"
$dataKey = python -c "import secrets; print(secrets.token_hex(32))"
kubectl create secret generic ischemia-secrets -n ischemia `
  --from-literal=postgres-password=$postgresPassword `
  --from-literal=secret-key=$secretKey `
  --from-literal=data-key=$dataKey
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/web -n ischemia
kubectl port-forward service/web 8081:80 -n ischemia
```

Open http://localhost:8081. Port 8081 avoids conflicts with the Compose app on port 5000
and MLflow on port 5001.

### Update and verify a running Kubernetes deployment

After changing the UI, API, or model-service code, rebuild the image in Docker Desktop's
image store and restart the app deployments so their pods use it:

```powershell
docker build -t ischemia-predictor:local .
kubectl rollout restart deployment/web deployment/ml deployment/dl deployment/qml -n ischemia
kubectl rollout status deployment/web -n ischemia
kubectl rollout status deployment/ml -n ischemia
kubectl rollout status deployment/dl -n ischemia
kubectl rollout status deployment/qml -n ischemia
```

In another PowerShell window, keep the port-forward running and verify the UI and API
through the Kubernetes Service:

```powershell
kubectl port-forward service/web 8081:80 -n ischemia
```

```powershell
Invoke-WebRequest http://localhost:8081/ -UseBasicParsing
kubectl get pods,services,endpoints -n ischemia
kubectl logs -f deployment/web -n ischemia
```

Ready pods and populated Service endpoints confirm that Kubernetes has connected each
Service to a pod. The web pod can also reach the prediction services using their internal
DNS names (`ml`, `dl`, and `qml`). In Docker Desktop, select the `ischemia` namespace
under Kubernetes workloads, then select a pod to inspect its logs and events.

For a remote cluster, build and push the image to a registry
accessible to the cluster, then replace `ischemia-predictor:local` in the manifest with
the registry image reference. Configure ingress/TLS separately for public access.
Kubernetes secret resources are only base64-encoded by default; enable cluster secret
encryption and restrict RBAC access in production. PostgreSQL storage availability and
size depend on the cluster's default StorageClass.

Kubernetes packages model artifacts in the image. When replacing model artifacts, rebuild
and redeploy the image so the gateway and each model service receive the same
`models/registry.json` and model files. Use this image-based workflow for model updates
in Kubernetes; unlike Compose, the Kubernetes manifest does not mount a shared writable
host `models/` directory.

## Deviations from the SRS/SAD (explain in the demo)
- Dataset: heart.csv (918 rows). Its label is "heart disease", used as the proxy for ischemia.
- Categorical inputs (sex, chest pain, resting ECG, exercise angina, ST slope) are encoded as small
  integers; Cholesterol/RestingBP values of 0 (not measured) are treated as missing and imputed.
- DL = scikit-learn MLPClassifier (small neural net), not TensorFlow/PyTorch.
- QML = PennyLane simulator, 4 qubits, 3 layers; it is less accurate and may stay `unvalidated`.
- Top-5 feature explanation = leave-one-out attribution (same method for all models).
- Dev server uses HTTP; TLS is expected at the ingress/reverse proxy. Rate limit is in-memory.
- Not implemented: pseudonymous-ID replacement (SEC-10), log-retention job.
