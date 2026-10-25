from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "placement_data.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "model"
)

MODEL_PATH = (
    MODEL_DIR
    / "placement_model.pkl"
)


if not DATASET_PATH.exists():

    raise FileNotFoundError(
        "placement_data.csv not found. "
        "Run create_dataset.py first."
    )


data = pd.read_csv(
    DATASET_PATH
)
print("\nDataset columns:")
print(data.columns.tolist())


X = data.drop(
    "Eligible",
    axis=1
)

y = data["Eligible"]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("Model trained successfully.")

print(
    f"Accuracy: {accuracy:.2%}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


print("\nModel saved successfully.")

print(
    MODEL_PATH
)