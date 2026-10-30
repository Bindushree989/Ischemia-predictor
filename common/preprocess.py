"""Shared preprocessing pipeline (FR-07, FR-08). The SAME versioned artifact is used for
training and inference.

Training code builds the pipeline with ``build_pipeline``, fits it on the training
data and saves it at ``pipeline_path(models_dir)``. The gateway then loads that
same file with ``load_pipeline`` and applies ``transform_one`` to each request, so
inference sees exactly the transformation the models were trained on.

The pipeline is: median imputation -> standard scaling (zero mean, unit variance).

Versioning
----------
``PIPELINE_VERSION`` is part of the artifact file name and is stored with every
registered model. The gateway refuses to predict when a model's
``pipeline_version`` differs from this constant (FR-08). Bump it whenever the
pipeline steps or the feature order change, then retrain and re-register models.
"""
import os
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Version tag of the preprocessing artifact; must match each model's pipeline_version.
PIPELINE_VERSION = "pipeline-1.0"


def build_pipeline():
    """Create a new, unfitted preprocessing pipeline.

    Steps:
        impute: ``SimpleImputer`` filling missing values with the column median.
        scale:  ``StandardScaler`` standardising each feature.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``. Fit it on training data, then
        save it to ``pipeline_path(models_dir)``.
    """
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])


def pipeline_path(models_dir):
    """Return the path of the fitted pipeline artifact for the current version.

    Args:
        models_dir: Folder holding model artifacts.

    Returns:
        ``<models_dir>/<PIPELINE_VERSION>.joblib``, e.g. ``models/pipeline-1.0.joblib``.
    """
    return os.path.join(models_dir, f"{PIPELINE_VERSION}.joblib")


def load_pipeline(models_dir):
    """Load the fitted pipeline artifact for the current ``PIPELINE_VERSION``.

    Args:
        models_dir: Folder holding model artifacts.

    Returns:
        The fitted ``Pipeline``.

    Raises:
        OSError: if the artifact file is missing or unreadable (the gateway turns
            this into an ``INTERNAL_ERROR`` response).

    Note:
        ``joblib`` files are pickles and can run code on load. Only load
        artifacts from a trusted ``models_dir``.
    """
    return joblib.load(pipeline_path(models_dir))


def transform_one(pipeline, values):
    """Transform one sample with a fitted pipeline.

    Args:
        pipeline: A fitted pipeline from ``load_pipeline``.
        values: Feature values for a single sample, in the same order as
            ``common.features.FEATURES``.

    Returns:
        The transformed features as a plain Python ``list`` of floats
        (JSON-serialisable, ready to send to an inference service).
    """
    return pipeline.transform(np.array([values], dtype=float))[0].tolist()
