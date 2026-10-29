# Ischemia Predictor

Flask gateway (auth, RBAC, validation, preprocessing, orchestration, history, model registry,
audit log) + three prediction microservices (ML, DL, QML) + offline training pipeline.
Dataset: `data/heart.csv` (918 patients, 11 inputs, label `HeartDisease`). Python 3.10+.

## Setup (once) - Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas scipy scikit-learn joblib flask flask-sqlalchemy bcrypt PyJWT requests cryptography pennylane pytest
python -m training.train --data data/heart.csv --models models
python -m scripts.create_user admin Administrator 'Adm1n!Passw0rd'
python -m scripts.create_user doc1 Clinician 'Cl1nician!Pass'
```
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

## Docker (optional)
`docker compose up --build`. Set real `SECRET_KEY` and `DATA_KEY` first
(`python -c "import os;print(os.urandom(32).hex())"`). Kubernetes manifests are not included.

## Deviations from the SRS/SAD (explain in the demo)
- Dataset: heart.csv (918 rows). Its label is "heart disease", used as the proxy for ischemia.
- Categorical inputs (sex, chest pain, resting ECG, exercise angina, ST slope) are encoded as small
  integers; Cholesterol/RestingBP values of 0 (not measured) are treated as missing and imputed.
- DL = scikit-learn MLPClassifier (small neural net), not TensorFlow/PyTorch.
- QML = PennyLane simulator, 4 qubits, 3 layers; it is less accurate and may stay `unvalidated`.
- Top-5 feature explanation = leave-one-out attribution (same method for all models).
- Dev server uses HTTP; TLS is expected at the ingress/reverse proxy. Rate limit is in-memory.
- Not implemented: pseudonymous-ID replacement (SEC-10), Kubernetes manifests, log-retention job.
