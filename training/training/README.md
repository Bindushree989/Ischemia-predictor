# ML Training Pipeline

## Purpose

This directory contains the model training and evaluation pipeline for the Ischemia Predictor project.

## Training and Evaluation

The training script is `train.py`. It fits and evaluates six candidates:

- ML: scikit-learn Random Forest and XGBoost Classifier.
- DL: tuned scikit-learn MLP and a three-hidden-layer MLP.
- QML: PennyLane variational classifiers with three and five layers.

All candidates use the shared preprocessing pipeline and a stratified held-out split. Accuracy
and recall must both reach 85% for a model to be marked validated. Only validated candidates can
be activated; QML candidates that miss either target remain available for inspection but cannot
serve predictions.

The evaluation results should be recorded alongside the dataset hash, model version, framework,
algorithm, validation status, held-out metrics, and cross-validation results where applicable.
Training writes one MLflow run per candidate with the model and preprocessing artifacts.

## Reproducibility

- Use the project's documented Python environment and dependencies.
- Keep the dataset and preprocessing steps consistent with the existing implementation.
- Record actual evaluation results; do not assume a model meets a target without verifying the results.
- Avoid committing credentials, private patient information, or local environment files.

## Notes

This document describes the training pipeline at a high level. Update it if the implementation or execution instructions change.
