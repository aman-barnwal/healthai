import os
import json
import joblib
import warnings

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("backend/models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET RULES
# ============================================================

# These are columns that are normally identifiers rather than
# useful predictive features.

ID_COLUMNS = {
    "id",
    "ID",
    "patient_id",
    "Patient_ID",
    "patientid",
    "PatientID",
    "record_id",
    "Record_ID",
    "index",
    "Index"
}


# ============================================================
# TARGET DETECTION
# ============================================================

TARGET_NAMES = [
    "target",
    "Target",
    "TARGET",

    "label",
    "Label",
    "LABEL",

    "class",
    "Class",
    "CLASS",

    "outcome",
    "Outcome",
    "OUTCOME",

    "diagnosis",
    "Diagnosis",

    "result",
    "Result",

    "prediction",
    "Prediction",

    "death_event",
    "DEATH_EVENT",

    "num",

    "NObeyesdad"
]


def detect_target(df):
    """
    Automatically determine the target column.

    Priority:
    1. Known target names
    2. Last column in CSV

    Convention:
    New datasets should have the target as the LAST column.
    """

    # --------------------------------------------------------
    # Look for known target names
    # --------------------------------------------------------

    for column in TARGET_NAMES:

        if column in df.columns:
            return column

    # --------------------------------------------------------
    # Otherwise use the last column
    # --------------------------------------------------------

    return df.columns[-1]


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def detect_id_columns(df, target):
    """
    Detect columns that are likely identifiers.
    """

    columns_to_remove = []

    for column in df.columns:

        if column == target:
            continue

        # Exact known ID name
        if column in ID_COLUMNS:
            columns_to_remove.append(column)
            continue

        # ID-like column names
        lower = column.lower()

        if (
            lower.endswith("_id")
            or lower.endswith("id")
            or "patient_id" in lower
            or "record_id" in lower
        ):
            columns_to_remove.append(column)

    return columns_to_remove


# ============================================================
# SPECIAL TARGET PROCESSING
# ============================================================

def process_target(dataset_name, target_name, y):
    """
    Apply safe target transformations.

    Heart disease dataset:
        0 = no disease
        1-4 = disease

    Convert it into:
        0 = no disease
        1 = disease
    """

    if (
        target_name == "num"
        and dataset_name.lower() == "heart_disease"
    ):

        print(
            "\nHeart disease target detected."
        )

        print(
            "Converting disease severity "
            "0-4 into binary classification."
        )

        y = (y > 0).astype(int)

    return y


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32"
        ]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "string",
            "category",
            "bool"
        ]
    ).columns.tolist()

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
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

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return (
        preprocessor,
        numerical_features,
        categorical_features
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():

    return {

        "Logistic Regression":

            LogisticRegression(
                max_iter=3000
            ),

        "Random Forest":

            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
    }


# ============================================================
# TRAIN ONE DATASET
# ============================================================

def train_dataset(csv_path):

    dataset_name = csv_path.stem

    print("\n")
    print("=" * 75)
    print(f"DATASET: {dataset_name}")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    print(
        f"\nLoaded: {csv_path}"
    )

    print(
        f"Original shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    print(
        f"Duplicate rows removed: {removed}"
    )

    print(
        f"Final dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = detect_target(df)

    print(
        f"\nDetected target: {target}"
    )

    # --------------------------------------------------------
    # Target must contain useful data
    # --------------------------------------------------------

    if df[target].nunique() < 2:

        print(
            "SKIPPED: target contains fewer than "
            "2 unique classes."
        )

        return None

    # --------------------------------------------------------
    # Split X / y
    # --------------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # --------------------------------------------------------
    # Target processing
    # --------------------------------------------------------

    y = process_target(
        dataset_name,
        target,
        y
    )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # Remove identifier columns
    # --------------------------------------------------------

    id_columns = detect_id_columns(
        X,
        target
    )

    if id_columns:

        print(
            "\nRemoving identifier columns:"
        )

        print(
            id_columns
        )

        X = X.drop(
            columns=id_columns
        )

    # --------------------------------------------------------
    # Remove completely empty columns
    # --------------------------------------------------------

    empty_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if empty_columns:

        print(
            "\nRemoving completely empty columns:"
        )

        print(
            empty_columns
        )

        X = X.drop(
            columns=empty_columns
        )

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    (
        preprocessor,
        numerical_features,
        categorical_features
    ) = create_preprocessor(X)

    print(
        f"\nNumber of features: {X.shape[1]}"
    )

    print(
        f"Numerical features: "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    # --------------------------------------------------------
    # Determine number of classes
    # --------------------------------------------------------

    number_of_classes = y.nunique()

    print(
        f"Number of target classes: "
        f"{number_of_classes}"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    except ValueError as error:

        print(
            "\nCould not perform stratified split:"
        )

        print(error)

        print(
            "\nSkipping dataset."
        )

        return None

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = get_models()

    best_pipeline = None
    best_model_name = None
    best_score = -1

    results = []

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for model_name, model in models.items():

        print(
            "\n" + "-" * 65
        )

        print(
            f"Training: {model_name}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        try:

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

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

            results.append(
                {
                    "model": model_name,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1
                }
            )

            # ------------------------------------------------
            # Select based on macro F1.
            #
            # This is safer than selecting purely on accuracy
            # when classes are imbalanced.
            # ------------------------------------------------

            if macro_f1 > best_score:

                best_score = macro_f1

                best_pipeline = pipeline

                best_model_name = model_name

        except Exception as error:

            print(
                f"Model failed: {error}"
            )

    # --------------------------------------------------------
    # No model succeeded
    # --------------------------------------------------------

    if best_pipeline is None:

        print(
            "\nNo model could be trained."
        )

        return None

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    predictions = best_pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 65
    )

    print(
        f"BEST MODEL: {best_model_name}"
    )

    print(
        f"TEST ACCURACY: {accuracy:.4f}"
    )

    print(
        f"TEST MACRO F1: {macro_f1:.4f}"
    )

    print(
        "=" * 65
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{dataset_name}.pkl"
    )

    joblib.dump(
        best_pipeline,
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "dataset": dataset_name,

        "dataset_file": csv_path.name,

        "target": target,

        "features": X.columns.tolist(),

        "removed_id_columns": id_columns,

        "numerical_features": numerical_features,

        "categorical_features": categorical_features,

        "classes": [
            str(value)
            for value in sorted(
                y.unique(),
                key=str
            )
        ],

        "model": best_model_name,

        "test_accuracy": float(
            accuracy
        ),

        "test_macro_f1": float(
            macro_f1
        ),

        "training_samples": len(
            X_train
        ),

        "testing_samples": len(
            X_test
        )
    }

    metadata_path = (
        MODEL_DIR
        / f"{dataset_name}_metadata.json"
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
        f"Metadata saved: {metadata_path}"
    )

    return metadata


# ============================================================
# DISCOVER ALL DATASETS
# ============================================================

def discover_datasets():

    datasets = sorted(
        DATASET_DIR.glob("*.csv")
    )

    # Ignore hidden files
    datasets = [
        file
        for file in datasets
        if not file.name.startswith(".")
    ]

    return datasets


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print(
        "#" * 75
    )

    print(
        "#"
    )

    print(
        "#              HEALTHCAREAI AUTO TRAINER"
    )

    print(
        "#"
    )

    print(
        "#" * 75
    )

    # --------------------------------------------------------
    # Discover datasets automatically
    # --------------------------------------------------------

    datasets = discover_datasets()

    if not datasets:

        print(
            "\nNo CSV datasets found."
        )

        print(
            f"Put CSV files inside: "
            f"{DATASET_DIR}"
        )

        return

    print(
        f"\nDiscovered {len(datasets)} dataset(s):"
    )

    for dataset in datasets:

        print(
            f"  ✓ {dataset.name}"
        )

    # --------------------------------------------------------
    # Train every dataset
    # --------------------------------------------------------

    successful = []

    failed = []

    for dataset in datasets:

        try:

            result = train_dataset(
                dataset
            )

            if result:

                successful.append(
                    result
                )

            else:

                failed.append(
                    dataset.name
                )

        except Exception as error:

            print(
                "\n"
                + "!" * 75
            )

            print(
                f"FAILED: {dataset.name}"
            )

            print(
                f"ERROR: {error}"
            )

            print(
                "!" * 75
            )

            failed.append(
                dataset.name
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n\n")

    print(
        "#" * 75
    )

    print(
        "#                  TRAINING SUMMARY"
    )

    print(
        "#" * 75
    )

    print()

    for result in successful:

        print(
            f"{result['dataset']:<40}"
            f"{result['model']:<22}"
            f"Accuracy: "
            f"{result['test_accuracy']:.4f}"
        )

    if failed:

        print(
            "\nDatasets skipped/failed:"
        )

        for name in failed:

            print(
                f"  ✗ {name}"
            )

    print(
        "\nModels available:"
    )

    for model in sorted(
        MODEL_DIR.glob("*.pkl")
    ):

        print(
            f"  ✓ {model}"
        )

    print(
        "\nHealthcareAI automatic training complete."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
