import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "dataset/lung_disease_data.csv"

MODEL_DIR = Path("backend/models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LUNG DISEASE MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Original shape:", df.shape)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = df.dropna(how="all")

print(
    "Shape after removing completely empty rows:",
    df.shape
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Disease Type"


# ============================================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================================

missing_target = df[TARGET].isna().sum()

print(
    f"\nRows with missing target ({TARGET}):",
    missing_target
)

df = df.dropna(
    subset=[TARGET]
)

# Also remove blank target values if present
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
)

df = df[
    df[TARGET] != ""
]

print(
    "Shape after removing rows with missing target:",
    df.shape
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


print("\nTarget distribution:")

print(
    y.value_counts()
)


print("\nFeatures:")

print(
    X.columns.tolist()
)


print("\nFeature shape:", X.shape)

print(
    "Target shape:",
    y.shape
)


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nNumerical features:")

print(
    numerical_features
)


print("\nCategorical features:")

print(
    categorical_features
)


# ============================================================
# PREPROCESSING
# ============================================================

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


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


print("\nTraining samples:", X_train.shape[0])

print(
    "Testing samples:",
    X_test.shape[0]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

}


# ============================================================
# TRAIN MODELS
# ============================================================

best_model = None

best_name = None

best_accuracy = 0

best_f1 = 0


results = {}


for name, classifier in models.items():

    print("\n" + "-" * 70)

    print(
        f"Training {name}..."
    )


    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )


    # Train
    pipeline.fit(
        X_train,
        y_train
    )


    # Predict
    predictions = pipeline.predict(
        X_test
    )


    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )


    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )


    print(
        f"Accuracy: {accuracy:.4f}"
    )


    print(
        f"Macro F1: {macro_f1:.4f}"
    )


    results[name] = {
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        )
    }


    # Select best model
    if macro_f1 > best_f1:

        best_accuracy = accuracy

        best_f1 = macro_f1

        best_model = pipeline

        best_name = name


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "BEST MODEL:",
    best_name
)


print(
    f"FINAL ACCURACY: {best_accuracy:.4f}"
)


print(
    f"FINAL MACRO F1: {best_f1:.4f}"
)


final_predictions = best_model.predict(
    X_test
)


print(
    "\nClassification Report:\n"
)


print(
    classification_report(
        y_test,
        final_predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "lung_disease.pkl"
)


joblib.dump(
    best_model,
    model_path
)


print(
    f"\nModel saved to: {model_path}"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "dataset": "lung_disease",

    "dataset_file": "lung_disease_data.csv",

    "target": TARGET,

    "features": X.columns.tolist(),

    "numerical_features": numerical_features,

    "categorical_features": categorical_features,

    "classes": sorted(
        y.unique().tolist()
    ),

    "model": best_name,

    "test_accuracy": float(
        best_accuracy
    ),

    "test_macro_f1": float(
        best_f1
    ),

    "training_samples": int(
        X_train.shape[0]
    ),

    "testing_samples": int(
        X_test.shape[0]
    ),

    "results": results

}


metadata_path = (
    MODEL_DIR /
    "lung_disease_metadata.json"
)


with open(
    metadata_path,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    f"Metadata saved to: {metadata_path}"
)


print("\n" + "=" * 70)

print(
    "TRAINING COMPLETE"
)

print("=" * 70)
