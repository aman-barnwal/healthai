from pathlib import Path
import json
import gc
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ==============================================================
# CONFIGURATION
# ==============================================================

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("backend/models")

TEST_SIZE = 0.20
RANDOM_STATE = 123

# Do not load models larger than this.
# This prevents the audit from exhausting RAM.
MAX_MODEL_SIZE_GB = 1.0


print("=" * 90)
print("HEALTHCAREAI MODEL AUDIT")
print("=" * 90)

results = []


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def model_size_bytes(path):
    """Return model size in bytes."""
    return path.stat().st_size


def model_size(path):
    """Return human-readable model size."""

    size = model_size_bytes(path)

    if size >= 1024 ** 3:
        return f"{size / (1024 ** 3):.2f} GB"

    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.2f} MB"

    if size >= 1024:
        return f"{size / 1024:.2f} KB"

    return f"{size} bytes"


def load_metadata(model_name):
    """Load metadata file if available."""

    metadata_path = (
        MODEL_DIR /
        f"{model_name}_metadata.json"
    )

    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)

    except Exception as e:
        print("Metadata error:", e)
        return {}


def find_dataset(model_name, metadata):
    """
    Try multiple methods to find the dataset used
    for training the model.
    """

    # ----------------------------------------------------------
    # METHOD 1: METADATA
    # ----------------------------------------------------------

    if metadata:

        dataset_file = metadata.get("dataset_file")

        if dataset_file:

            dataset_file = str(dataset_file)

            path = DATASET_DIR / dataset_file

            if path.exists():
                return path

            # Search recursively using filename
            for file in DATASET_DIR.rglob("*.csv"):

                if file.name == Path(dataset_file).name:
                    return file

    # ----------------------------------------------------------
    # METHOD 2: EXACT MODEL NAME MATCH
    # ----------------------------------------------------------

    for file in DATASET_DIR.rglob("*.csv"):

        if file.stem.lower() == model_name.lower():
            return file

    # ----------------------------------------------------------
    # METHOD 3: NORMALIZED NAME MATCH
    # ----------------------------------------------------------

    normalized_model = (
        model_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    for file in DATASET_DIR.rglob("*.csv"):

        normalized_file = (
            file.stem.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if normalized_file == normalized_model:
            return file

    # ----------------------------------------------------------
    # METHOD 4: PARTIAL MATCH
    # ----------------------------------------------------------

    for file in DATASET_DIR.rglob("*.csv"):

        normalized_file = (
            file.stem.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if (
            normalized_model in normalized_file
            or normalized_file in normalized_model
        ):
            return file

    return None


def detect_target(df, metadata):
    """
    Detect target column using metadata first.
    """

    if metadata:

        target = metadata.get("target")

        if target in df.columns:
            return target

    # Fallback: last column
    return df.columns[-1]


def remove_id_columns(X):
    """
    Remove likely identifier columns.
    """

    id_columns = []

    known_id_names = [
        "id",
        "patientid",
        "patient_id",
        "record_id",
        "index",
        "unnamed: 0"
    ]

    for col in X.columns:

        lower = str(col).lower().strip()

        if (
            lower in known_id_names
            or lower.endswith("_id")
        ):
            id_columns.append(col)

    if id_columns:
        X = X.drop(columns=id_columns)

    return X, id_columns


def get_class_imbalance(y):
    """
    Calculate class imbalance information.
    """

    counts = y.value_counts()

    if len(counts) <= 1:
        return None

    smallest = counts.min()
    largest = counts.max()

    ratio = smallest / largest

    return {
        "smallest_class": int(smallest),
        "largest_class": int(largest),
        "ratio": round(ratio, 4)
    }


def safe_result(
    model_name,
    status,
    size,
    accuracy=None,
    macro_f1=None,
    warning="",
    dataset=""
):
    """
    Create standardized result record.
    """

    results.append({
        "model": model_name,
        "status": status,
        "dataset": dataset,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "size": size,
        "warning": warning
    })


# ==============================================================
# GET MODELS
# ==============================================================

model_files = sorted(
    MODEL_DIR.glob("*.pkl")
)

print(f"\nModels found: {len(model_files)}")


# ==============================================================
# AUDIT EACH MODEL
# ==============================================================

for model_path in model_files:

    print("\n")
    print("-" * 90)

    print(
        f"MODEL: {model_path.name}"
    )

    model_name = model_path.stem

    size_text = model_size(model_path)

    size_gb = (
        model_size_bytes(model_path)
        / (1024 ** 3)
    )

    print(
        "Model size:",
        size_text
    )

    # ----------------------------------------------------------
    # LOAD METADATA
    # ----------------------------------------------------------

    metadata = load_metadata(
        model_name
    )

    if metadata:

        print(
            "Metadata:",
            "FOUND"
        )

        if metadata.get("target"):

            print(
                "Metadata target:",
                metadata.get("target")
            )

        if metadata.get("dataset_file"):

            print(
                "Metadata dataset:",
                metadata.get("dataset_file")
            )

    else:

        print(
            "Metadata:",
            "NOT FOUND"
        )

    # ----------------------------------------------------------
    # SKIP HUGE MODELS
    # ----------------------------------------------------------

    if size_gb > MAX_MODEL_SIZE_GB:

        warning = (
            f"MODEL SKIPPED: {size_gb:.2f} GB "
            f"is larger than the {MAX_MODEL_SIZE_GB:.1f} GB limit"
        )

        print(
            "WARNING:",
            warning
        )

        safe_result(
            model_name=model_name,
            status="SKIPPED - TOO LARGE",
            size=size_text,
            warning=warning
        )

        continue

    # ----------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------

    model = None

    try:

        print(
            "Loading model..."
        )

        model = joblib.load(
            model_path
        )

        print(
            "Load status: OK"
        )

        print(
            "Model type:",
            type(model).__name__
        )

    except MemoryError:

        warning = (
            "NOT ENOUGH MEMORY TO LOAD MODEL"
        )

        print(
            "Load status: FAILED"
        )

        print(
            "Error:",
            warning
        )

        safe_result(
            model_name=model_name,
            status="FAILED TO LOAD",
            size=size_text,
            warning=warning
        )

        gc.collect()

        continue

    except Exception as e:

        print(
            "Load status: FAILED"
        )

        print(
            "Error:",
            e
        )

        safe_result(
            model_name=model_name,
            status="FAILED TO LOAD",
            size=size_text,
            warning=str(e)
        )

        gc.collect()

        continue

    # ----------------------------------------------------------
    # FIND DATASET
    # ----------------------------------------------------------

    dataset_path = find_dataset(
        model_name,
        metadata
    )

    if dataset_path is None:

        warning = (
            "Dataset could not be automatically matched"
        )

        print(
            "Dataset: NOT FOUND"
        )

        safe_result(
            model_name=model_name,
            status="MODEL OK / DATASET NOT FOUND",
            size=size_text,
            warning=warning
        )

        del model
        gc.collect()

        continue

    print(
        "Dataset:",
        dataset_path
    )

    # ----------------------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------------------

    try:

        df = pd.read_csv(
            dataset_path
        )

        print(
            "Dataset shape:",
            df.shape
        )

    except Exception as e:

        warning = (
            f"DATASET LOAD FAILED: {e}"
        )

        print(
            warning
        )

        safe_result(
            model_name=model_name,
            status="DATASET LOAD FAILED",
            size=size_text,
            dataset=str(dataset_path),
            warning=warning
        )

        del model
        gc.collect()

        continue

    # ----------------------------------------------------------
    # DETECT TARGET
    # ----------------------------------------------------------

    target = detect_target(
        df,
        metadata
    )

    print(
        "Target:",
        target
    )

    if target not in df.columns:

        warning = (
            "Target column not found"
        )

        print(
            "WARNING:",
            warning
        )

        safe_result(
            model_name=model_name,
            status="TARGET NOT FOUND",
            size=size_text,
            dataset=str(dataset_path),
            warning=warning
        )

        del model
        del df

        gc.collect()

        continue

    # ----------------------------------------------------------
    # REMOVE ROWS WITH MISSING TARGET
    # ----------------------------------------------------------

    original_rows = len(df)

    df = df.dropna(
        subset=[target]
    )

    removed_rows = (
        original_rows - len(df)
    )

    if removed_rows > 0:

        print(
            "Rows removed due to missing target:",
            removed_rows
        )

    # ----------------------------------------------------------
    # CHECK DUPLICATES
    # ----------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        "Duplicate rows:",
        duplicate_rows
    )

    # ----------------------------------------------------------
    # PREPARE FEATURES
    # ----------------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # ----------------------------------------------------------
    # REMOVE ID COLUMNS
    # ----------------------------------------------------------

    X, id_columns = remove_id_columns(
        X
    )

    if id_columns:

        print(
            "Removing ID columns:",
            id_columns
        )

    print(
        "Features:",
        X.shape
    )

    print(
        "Classes:",
        y.nunique()
    )

    # ----------------------------------------------------------
    # TARGET LEAKAGE CHECK
    # ----------------------------------------------------------

    leakage_columns = []

    target_lower = (
        str(target)
        .lower()
        .strip()
    )

    for col in X.columns:

        col_lower = (
            str(col)
            .lower()
            .strip()
        )

        if (
            col_lower == target_lower
        ):
            leakage_columns.append(
                col
            )

    if leakage_columns:

        print(
            "WARNING: POSSIBLE TARGET LEAKAGE:",
            leakage_columns
        )

    # ----------------------------------------------------------
    # CLASS DISTRIBUTION
    # ----------------------------------------------------------

    class_info = get_class_imbalance(
        y
    )

    if class_info:

        print(
            "Class balance ratio:",
            class_info["ratio"]
        )

        print(
            "Smallest class:",
            class_info["smallest_class"]
        )

        print(
            "Largest class:",
            class_info["largest_class"]
        )

    # ----------------------------------------------------------
    # CHECK DATASET SIZE
    # ----------------------------------------------------------

    if len(df) < 100:

        print(
            "WARNING: VERY SMALL DATASET"
        )

    # ----------------------------------------------------------
    # VALIDATION SPLIT
    # ----------------------------------------------------------

    try:

        stratify_value = None

        if y.nunique() > 1:

            class_counts = y.value_counts()

            if class_counts.min() >= 2:

                stratify_value = y

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=stratify_value
            )
        )

    except Exception as e:

        warning = (
            f"TRAIN/TEST SPLIT FAILED: {e}"
        )

        print(
            warning
        )

        safe_result(
            model_name=model_name,
            status="SPLIT FAILED",
            size=size_text,
            dataset=str(dataset_path),
            warning=warning
        )

        del model
        del df
        del X
        del y

        gc.collect()

        continue

    # ----------------------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------------------

    try:

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        print()
        print(
            f"AUDIT ACCURACY: {accuracy:.4f}"
        )

        print(
            f"AUDIT MACRO F1: {macro_f1:.4f}"
        )

    except Exception as e:

        warning = (
            f"PREDICTION FAILED: {e}"
        )

        print(
            warning
        )

        safe_result(
            model_name=model_name,
            status="PREDICTION FAILED",
            size=size_text,
            dataset=str(dataset_path),
            warning=warning
        )

        del model
        del df
        del X
        del y

        gc.collect()

        continue

    # ----------------------------------------------------------
    # WARNINGS
    # ----------------------------------------------------------

    warnings_list = []

    # Huge model

    if size_gb > 0.5:

        warnings_list.append(
            "LARGE MODEL (> 500 MB)"
        )

    # Suspiciously high accuracy

    if accuracy >= 0.995:

        warning = (
            "SUSPICIOUSLY HIGH ACCURACY - "
            "CHECK FOR DATA LEAKAGE"
        )

        print(
            "WARNING:",
            warning
        )

        warnings_list.append(
            warning
        )

    # Very high but slightly below threshold

    elif accuracy >= 0.98:

        warning = (
            "VERY HIGH ACCURACY - "
            "VERIFY WITH CROSS VALIDATION"
        )

        print(
            "NOTICE:",
            warning
        )

        warnings_list.append(
            warning
        )

    # Low performance

    if accuracy < 0.50:

        warning = (
            "LOW PERFORMANCE - "
            "MODEL MAY NOT BE USEFUL"
        )

        print(
            "WARNING:",
            warning
        )

        warnings_list.append(
            warning
        )

    # Poor macro F1

    if macro_f1 < 0.50:

        warning = (
            "LOW MACRO F1 - "
            "MODEL MAY PERFORM POORLY ON SOME CLASSES"
        )

        print(
            "WARNING:",
            warning
        )

        warnings_list.append(
            warning
        )

    # Duplicate rows

    if duplicate_rows > 0:

        warning = (
            f"DUPLICATE ROWS DETECTED: "
            f"{duplicate_rows}"
        )

        warnings_list.append(
            warning
        )

    # Class imbalance

    if class_info:

        if class_info["ratio"] < 0.10:

            warning = (
                "SEVERE CLASS IMBALANCE"
            )

            print(
                "WARNING:",
                warning
            )

            warnings_list.append(
                warning
            )

        elif class_info["ratio"] < 0.30:

            warning = (
                "MODERATE CLASS IMBALANCE"
            )

            print(
                "NOTICE:",
                warning
            )

            warnings_list.append(
                warning
            )

    # Possible target leakage

    if leakage_columns:

        warnings_list.append(
            "POSSIBLE TARGET LEAKAGE"
        )

    # Small dataset

    if len(df) < 100:

        warnings_list.append(
            "VERY SMALL DATASET"
        )

    # ----------------------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------------------

    status = "PASS"

    if warnings_list:
        status = "REVIEW"

    safe_result(
        model_name=model_name,
        status=status,
        accuracy=round(
            accuracy,
            4
        ),
        macro_f1=round(
            macro_f1,
            4
        ),
        size=size_text,
        dataset=str(dataset_path),
        warning=" | ".join(
            warnings_list
        )
    )

    # ----------------------------------------------------------
    # CLEAN MEMORY
    # ----------------------------------------------------------

    del model
    del df
    del X
    del y
    del X_train
    del X_test
    del y_train
    del y_test
    del predictions

    gc.collect()


# ==============================================================
# FINAL REPORT
# ==============================================================

print("\n")
print("=" * 90)
print("FINAL MODEL AUDIT REPORT")
print("=" * 90)

results_df = pd.DataFrame(
    results
)

if not results_df.empty:

    # Sort models with valid accuracy first
    results_df = results_df.sort_values(
        by="accuracy",
        ascending=False,
        na_position="last"
    )

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # SAVE REPORT
    # ----------------------------------------------------------

    output_file = (
        MODEL_DIR /
        "model_audit_report.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print(
        "Report saved to:"
    )

    print(
        output_file
    )

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("AUDIT SUMMARY")
    print("=" * 90)

    print(
        "Total models:",
        len(results_df)
    )

    print(
        "PASS:",
        (
            results_df["status"]
            == "PASS"
        ).sum()
    )

    print(
        "REVIEW:",
        (
            results_df["status"]
            == "REVIEW"
        ).sum()
    )

    print(
        "SKIPPED:",
        results_df["status"]
        .astype(str)
        .str.contains(
            "SKIPPED",
            na=False
        )
        .sum()
    )

    print(
        "FAILED:",
        results_df["status"]
        .astype(str)
        .str.contains(
            "FAILED",
            na=False
        )
        .sum()
    )


print("\n")
print("=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)
