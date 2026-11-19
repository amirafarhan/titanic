"""Titanic survival prediction with the Keras network from ``Deep_Learning.ipynb``.

Preprocessing (fitted on the training split only):
  * Age -> median, Embarked -> most frequent, Fare -> median (missing values)
  * Sex, Embarked -> OrdinalEncoder
  * all features -> StandardScaler

Train and save:  python titanic_model.py   (artifacts go to ``models/``)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "titanic_mlp.keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "Survived"
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
RANDOM_STATE = 42


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    encode = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder()), CATEGORICAL),
    ])
    return Pipeline([("encode", encode), ("scale", StandardScaler())])


def build_model(n_features: int):
    """Architecture and optimiser from Deep_Learning.ipynb."""
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.regularizers import L1, L2

    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(64, activation="relu", kernel_regularizer=L1(0.01)),
        Dropout(0.7),
        Dense(32, activation="relu", kernel_regularizer=L2(0.001)),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=SGD(momentum=0.9), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train() -> dict:
    import joblib
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                                 roc_auc_score, roc_curve)
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=RANDOM_STATE)

    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    model = build_model(X_train_t.shape[1])
    history = model.fit(X_train_t, y_train, epochs=20, validation_split=0.1, batch_size=32,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=5)], verbose=2)

    proba = model.predict(X_test_t, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    metadata = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH, include_optimizer=False)
    joblib.dump(pre, PREPROCESSOR_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Test accuracy {metadata['accuracy']:.3f}, ROC AUC {metadata['roc_auc']:.3f}. Saved to {MODEL_DIR}/")
    return metadata


def load_artifacts():
    """Return ``(model, preprocessor, metadata)``; raises FileNotFoundError if not trained."""
    import joblib
    from tensorflow.keras.models import load_model

    for p in (MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Train first with: python titanic_model.py")
    return load_model(MODEL_PATH), joblib.load(PREPROCESSOR_PATH), json.loads(METADATA_PATH.read_text())


def predict_proba(model, pre, data: pd.DataFrame | dict) -> np.ndarray:
    """Survival probability for one passenger dict or a DataFrame of passengers."""
    frame = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame[FEATURES]
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in CATEGORICAL:  # None / blanks -> NaN so the imputer handles them
        frame[col] = frame[col].replace({None: np.nan, "": np.nan}).astype(object)
    return model.predict(pre.transform(frame), verbose=0).ravel()


if __name__ == "__main__":
    train()
