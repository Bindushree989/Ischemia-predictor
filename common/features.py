"""Feature definitions for the heart-disease CSV (heart.csv, 918 patients).

Categorical columns are encoded as small integers so the whole model input is a
numeric vector. The web form shows these as dropdowns.
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_bs",
            "resting_ecg", "max_hr", "exercise_angina", "oldpeak", "st_slope"]

# (min, max) accepted by the API
RANGES = {
    "age": (1, 120), "sex": (0, 1), "chest_pain_type": (0, 3), "resting_bp": (50, 250),
    "cholesterol": (50, 700), "fasting_bs": (0, 1), "resting_ecg": (0, 2), "max_hr": (40, 250),
    "exercise_angina": (0, 1), "oldpeak": (-3, 7), "st_slope": (0, 2),
}

ENCODING = {
    "Sex": {"F": 0, "M": 1},
    "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3},
    "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
    "ExerciseAngina": {"N": 0, "Y": 1},
    "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2},
}
COLUMN_MAP = {"Age": "age", "Sex": "sex", "ChestPainType": "chest_pain_type",
              "RestingBP": "resting_bp", "Cholesterol": "cholesterol", "FastingBS": "fasting_bs",
              "RestingECG": "resting_ecg", "MaxHR": "max_hr", "ExerciseAngina": "exercise_angina",
              "Oldpeak": "oldpeak", "ST_Slope": "st_slope"}


def load_csv(path):
    """Read heart.csv -> DataFrame with FEATURES + 'label'. Zeros that mean 'not measured'
    (Cholesterol, RestingBP) become NaN and are imputed by the shared preprocessing."""
    df = pd.read_csv(path)
    for col, mapping in ENCODING.items():
        df[col] = df[col].map(mapping)
    df = df.rename(columns=COLUMN_MAP)
    df["label"] = df["HeartDisease"].astype(int)
    for col in ("cholesterol", "resting_bp"):
        df.loc[df[col] <= 0, col] = np.nan
    df = df.dropna(subset=["age", "sex", "chest_pain_type", "resting_ecg", "exercise_angina",
                           "st_slope", "max_hr", "oldpeak", "fasting_bs"])
    return df[FEATURES + ["label"]].reset_index(drop=True)
