# Ischemia Predictor Test Plan

## Purpose and safety boundary

This plan checks software behavior, model packaging, and application wiring. Passing it does not
establish clinical validity, safety, regulatory compliance, or readiness for patient care. Use
synthetic data and a disposable local database for browser and system tests.

## Test levels and current evidence

| Level | Command or activity | Pass criteria | Current evidence |
|---|---|---|---|
| API and service regression | `.\.venv\Scripts\python.exe -m pytest -q` | All tests pass; no failures or unexpected skips. | Latest recorded Python run: 38 passed. Re-run after backend or model changes. |
| Browser E2E | `npm.cmd run test:e2e` | Login validation, role/model availability, field constraints, theme, sign-out, and exact-version predictions pass. | 7 passed in Edge on 2026-11-13. Report: `playwright-report\index.html`. |
| Training and evaluation | `dvc repro` | All six candidates train from the configured data; artifacts, held-out metrics, and MLflow runs are produced. | Run when training code, data, or dependencies change; inspect DVC outputs and MLflow rather than inferring success from a prior run. |
| Administrator / Clinician workflow | Manual test with a disposable database | Only active, validated versions are selectable; a selected version is the one returned; deactivation removes only that version. | Manual workflow; record the account role and versions checked. |
| MLflow inspection | Open the local MLflow UI | Expected runs and artifacts exist and metrics correspond to the current training run. | Manual verification; record experiment and run IDs. |
| Performance, deployment, compatibility | Environment-specific test | Meet explicit workload, deployment, and browser support targets set for the release. | Not established by the current unit or browser suite. |

## Recommended run order

1. **API and service regression:** use the project virtual environment and an isolated test
   database. Keep `FLASK_AUTOCREATE=0` and `DATABASE_URL=sqlite://` for the Python suite.
2. **Model build/evaluation:** run `dvc repro` when the model pipeline or its inputs changed.
   Review each model's held-out accuracy and recall, the generated artifacts, and the MLflow run.
3. **Live application checks:** start the application and inference services against a disposable
   database. Do not point browser tests at a database containing real patient data.
4. **Playwright E2E:** install dependencies once with `npm.cmd ci`, then run
   `npm.cmd run test:e2e`. Set credentials for a local Clinician test account; the two public
   login-page cases run without credentials, while Clinician cases skip without them. Use an
   approved browser channel if Windows policy blocks Playwright Chromium.
5. **Manual workflow:** verify Administrator activation/deactivation, Clinician exact-version
   selection, result/history display, and MLflow runs. Never promote a model solely to make a test
   pass.
6. **Record evidence:** capture date, command, environment/browser, pass/fail/skip counts, report
   location, model versions, and any manual steps. Do not include passwords or patient data.

## Model-release acceptance criteria

- An automated test failure blocks the software test gate until diagnosed and fixed.
- A model is selectable only when its registry state is validated and both held-out accuracy and
  recall meet the configured 85% threshold.
- Inference output identifies the exact selected model version and returns a valid probability.
- A model below the threshold remains unavailable to Clinicians. Administrative privileges do not
  bypass this policy.
- Threshold passage is a project-level screening gate, not evidence of clinical validation.

## Coverage gaps and follow-up

- No independent external or prospective clinical validation is established here.
- Performance under production load, deployment rollback, backup/recovery, and supported-browser
  coverage need environment-specific tests before release.
- Security testing must be performed and documented separately; passing functional tests is not a
  security review.
- QML candidates below the configured metric gate remain blocked. Retraining or tuning must be
  evaluated on held-out data before any status change.

## Related documentation

- [Project guide](PROJECT_GUIDE.md) has setup and command details.
- [README test matrix](README.md#how-to-read-the-test-plan) summarizes the test levels.
- `IEEE 829 Test Plan Ischemia Predictor .docx` is the earlier three-candidate plan; use this
  Markdown plan as the current test status and execution record.
