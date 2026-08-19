import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(
    "dataset/breast_cancer_wisconsin_diagnostic.csv"
)

MODEL_DIR = Path("backend/models")

MODEL_PATH = MODEL_DIR / "breast_cancer.pkl"

METADATA_PATH = MODEL_DIR / (
    "breast_cancer_metadata.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BREAST CANCER MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Original shape: {df.shape}")


# ============================================================
# PREPROCESSING
# ============================================================

# Remove ID because it does not contain useful medical information
df = df.drop(columns=["id"], errors="ignore")

# Target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]


# Encode:
# M = Malignant
# B = Benign

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


print("\nClasses:")

for index, label in enumerate(label_encoder.classes_):
    print(f"{index} = {label}")

print(f"\nFeatures: {X.shape}")
print(f"Target: {y.shape}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            )
        )
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            )
        )
    ])
}


# ============================================================
# TRAIN AND COMPARE
# ============================================================

best_model = None
best_model_name = None
best_accuracy = 0


for name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print(
        f"{name} Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"{name} F1 Score: "
        f"{f1:.4f}"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)

print(
    f"BEST MODEL: "
    f"{best_model_name}"
)

final_predictions = best_model.predict(X_test)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_f1 = f1_score(
    y_test,
    final_predictions
)

print(
    f"FINAL ACCURACY: "
    f"{final_accuracy:.4f}"
)

print(
    f"FINAL F1: "
    f"{final_f1:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print(
    f"\nModel saved to: "
    f"{MODEL_PATH}"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "target": "diagnosis",

    "model": best_model_name,

    "dataset": str(DATASET_PATH),

    "features": X.columns.tolist(),

    "classes": label_encoder.classes_.tolist(),

    "class_mapping": {
        "0": "B",
        "1": "M"
    },

    "test_accuracy": round(
        float(final_accuracy),
        4
    ),

    "test_macro_f1": round(
        float(
            f1_score(
                y_test,
                final_predictions,
                average="macro"
            )
        ),
        4
    )
}


with open(
    METADATA_PATH,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    f"Metadata saved to: "
    f"{METADATA_PATH}"
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
