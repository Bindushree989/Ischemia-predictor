"""Shared preprocessing pipeline (FR-07, FR-08). The SAME versioned artifact is used for
training and inference."""
import os
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PIPELINE_VERSION = "pipeline-1.0"


def build_pipeline():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])


def pipeline_path(models_dir):
    return os.path.join(models_dir, f"{PIPELINE_VERSION}.joblib")


def load_pipeline(models_dir):
    return joblib.load(pipeline_path(models_dir))


def transform_one(pipeline, values):
    return pipeline.transform(np.array([values], dtype=float))[0].tolist()
