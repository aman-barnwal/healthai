from pathlib import Path
import json
import gc
import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "backend" / "models"

MAX_MODEL_SIZE_GB = 1.0
MAX_TEST_ROWS = 500
RANDOM_STATE = 42

print("=" * 90)
print("HEALTHCAREAI FULL SYSTEM TEST")
print("=" * 90)

print(f"Project root : {ROOT_DIR}")
print(f"Dataset dir  : {DATASET_DIR}")
print(f"Model dir    : {MODEL_DIR}")


# ============================================================
# HELPERS
# ============================================================

def format_size(size_bytes):
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} bytes"


def load_metadata(model_name):
    metadata_path = MODEL_DIR / f"{model_name}_metadata.json"

    if not metadata_path.exists():
        return {}, None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f), metadata_path
    except Exception as e:
        print(f"Metadata error: {e}")
        return {}, metadata_path


def find_dataset(model_name, metadata):
    """
    Find dataset using metadata first, then exact filename match.
    """

    if metadata:
        possible_keys = [
            "dataset_file",
            "dataset",
            "dataset_path",
            "source_dataset"
        ]

        for key in possible_keys:
            value = metadata.get(key)

            if not value:
                continue

            candidate = Path(value)

            # Absolute or project-relative path
            if candidate.exists():
                return candidate

            # dataset/filename.csv
            candidate = DATASET_DIR / Path(value).name

            if candidate.exists():
                return candidate

    # Exact filename/stem match
    for file in DATASET_DIR.rglob("*.csv"):
        if file.stem.lower() == model_name.lower():
            return file

    return None


def detect_target(df, metadata):
    """
    Use metadata target first.
    """

    if metadata:
        target = metadata.get("target")

        if target in df.columns:
            return target

    common_targets = [
        "target",
        "class",
        "label",
        "diagnosis",
        "disease",
        "outcome",
        "output",
        "DEATH_EVENT",
        "Diabetes_binary",
        "Diabetes_012",
        "NObeyesdad"
    ]

    for col in common_targets:
        if col in df.columns:
            return col

    return df.columns[-1]


def remove_id_columns(X):
    id_columns = []

    for col in X.columns:
        name = str(col).lower().strip()

        if (
            name == "id"
            or name == "index"
            or name.endswith("_id")
            or name in [
                "patientid",
                "patient_id",
                "record_id",
                "subject_id",
                "encounter_id",
                "unnamed: 0"
            ]
        ):
            id_columns.append(col)

    if id_columns:
        X = X.drop(columns=id_columns, errors="ignore")

    return X, id_columns


def sample_data(X, y, max_rows=MAX_TEST_ROWS):
    """
    Prevent huge prediction tests.
    """

    if len(X) <= max_rows:
        return X, y

    rng = np.random.RandomState(RANDOM_STATE)

    indices = rng.choice(
        len(X),
        size=max_rows,
        replace=False
    )

    return (
        X.iloc[indices],
        y.iloc[indices]
    )


def check_feature_compatibility(model, X):
    """
    Compare dataset features with model expected features when available.
    """

    expected = None

    # sklearn estimators
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

    # sklearn pipeline final estimator
    elif hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                expected = list(step.feature_names_in_)
                break

    if expected is None:
        return True, "Feature schema unavailable"

    actual = list(X.columns)

    missing = [col for col in expected if col not in actual]
    extra = [col for col in actual if col not in expected]

    if missing:
        return False, (
            f"Missing expected features: {missing[:10]}"
        )

    return True, (
        f"Features compatible"
        + (
            f" | Extra dataset columns: {extra[:10]}"
            if extra else ""
        )
    )


# ============================================================
# RESULTS
# ============================================================

results = []

model_files = sorted(MODEL_DIR.glob("*.pkl"))

if not model_files:
    print("\nERROR: No .pkl models found.")
    sys.exit(1)

print(f"\nModels discovered: {len(model_files)}")


# ============================================================
# MAIN TEST LOOP
# ============================================================

for model_path in model_files:

    print("\n" + "-" * 90)
    print(f"MODEL: {model_path.name}")
    print("-" * 90)

    model_name = model_path.stem
    size_bytes = model_path.stat().st_size
    size_gb = size_bytes / (1024 ** 3)

    result = {
        "model": model_name,
        "model_file": str(model_path),
        "size": format_size(size_bytes),
        "size_gb": round(size_gb, 4),
        "dataset": None,
        "target": None,
        "dataset_rows": None,
        "dataset_columns": None,
        "classes": None,
        "test_rows": None,
        "accuracy": None,
        "macro_f1": None,
        "status": "UNKNOWN",
        "warnings": ""
    }

    warnings_list = []

    print(f"Size: {format_size(size_bytes)}")

    # --------------------------------------------------------
    # HUGE MODEL SAFETY
    # --------------------------------------------------------

    if size_gb > MAX_MODEL_SIZE_GB:

        warning = (
            f"VERY LARGE MODEL ({size_gb:.2f} GB) - "
            f"SKIPPED TO PREVENT MEMORY CRASH"
        )

        print(f"WARNING: {warning}")

        result["status"] = "SKIPPED"
        result["warnings"] = warning

        results.append(result)

        continue

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata, metadata_path = load_metadata(model_name)

    if metadata_path is None:
        print("Metadata: NOT FOUND")
        warnings_list.append("Metadata missing")
    elif metadata:
        print(f"Metadata: {metadata_path.name}")
    else:
        print("Metadata: FOUND BUT INVALID")
        warnings_list.append("Metadata invalid")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = None

    try:

        print("Loading model...")

        model = joblib.load(model_path)

        print("Load status: OK")
        print(f"Type: {type(model).__name__}")

    except MemoryError:

        warning = "MemoryError while loading model"

        print(f"FAILED: {warning}")

        result["status"] = "FAILED"
        result["warnings"] = warning

        results.append(result)

        continue

    except Exception as e:

        warning = f"Model load failed: {e}"

        print(f"FAILED: {warning}")

        result["status"] = "FAILED"
        result["warnings"] = warning

        results.append(result)

        continue

    # --------------------------------------------------------
    # FIND DATASET
    # --------------------------------------------------------

    dataset_path = find_dataset(model_name, metadata)

    if dataset_path is None:

        warning = "Training dataset not found"

        print(f"WARNING: {warning}")

        result["status"] = "MODEL OK / DATASET NOT FOUND"
        result["warnings"] = " | ".join(
            warnings_list + [warning]
        )

        results.append(result)

        del model
        gc.collect()

        continue

    print(f"Dataset: {dataset_path}")

    result["dataset"] = str(dataset_path)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        print("Loading dataset...")

        df = pd.read_csv(
            dataset_path,
            low_memory=False
        )

        print(f"Dataset shape: {df.shape}")

        result["dataset_rows"] = len(df)
        result["dataset_columns"] = len(df.columns)

    except Exception as e:

        warning = f"Dataset load failed: {e}"

        print(f"FAILED: {warning}")

        result["status"] = "DATASET FAILED"
        result["warnings"] = " | ".join(
            warnings_list + [warning]
        )

        results.append(result)

        del model
        gc.collect()

        continue

    # --------------------------------------------------------
    # TARGET DETECTION
    # --------------------------------------------------------

    target = detect_target(df, metadata)

    print(f"Target: {target}")

    result["target"] = target

    if target not in df.columns:

        warning = "Target column not found"

        print(f"FAILED: {warning}")

        result["status"] = "TARGET FAILED"
        result["warnings"] = " | ".join(
            warnings_list + [warning]
        )

        results.append(result)

        del df
        del model
        gc.collect()

        continue

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    X, id_columns = remove_id_columns(X)

    if id_columns:
        print(f"Removed ID columns: {id_columns}")

    print(f"Features: {X.shape}")
    print(f"Classes: {y.nunique()}")

    result["classes"] = int(y.nunique())

    # --------------------------------------------------------
    # FEATURE COMPATIBILITY
    # --------------------------------------------------------

    compatible, message = check_feature_compatibility(
        model,
        X
    )

    print(f"Feature check: {message}")

    if not compatible:
        warnings_list.append(message)

    # --------------------------------------------------------
    # SAMPLE DATA
    # --------------------------------------------------------

    X_test, y_test = sample_data(X, y)

    print(
        f"Prediction test rows: {len(X_test)} "
        f"(max allowed: {MAX_TEST_ROWS})"
    )

    result["test_rows"] = len(X_test)

    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    try:

        print("Running prediction...")

        predictions = model.predict(X_test)

        print("Prediction status: OK")

    except Exception as e:

        warning = f"Prediction failed: {e}"

        print(f"FAILED: {warning}")

        result["status"] = "PREDICTION FAILED"
        result["warnings"] = " | ".join(
            warnings_list + [warning]
        )

        results.append(result)

        del df
        del X
        del y
        del X_test
        del y_test
        del model

        gc.collect()

        continue

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    try:

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

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Macro F1:      {macro_f1:.4f}")

        result["accuracy"] = round(float(accuracy), 4)
        result["macro_f1"] = round(float(macro_f1), 4)

    except Exception as e:

        warning = f"Metric calculation failed: {e}"

        print(f"WARNING: {warning}")

        warnings_list.append(warning)

    # --------------------------------------------------------
    # HEALTH CHECKS
    # --------------------------------------------------------

    if result["accuracy"] is not None:

        if accuracy >= 0.995:

            warning = (
                "Extremely high accuracy - "
                "investigate possible data leakage"
            )

            print(f"WARNING: {warning}")

            warnings_list.append(warning)

        elif accuracy < 0.50:

            warning = (
                "Low accuracy - model may require retraining"
            )

            print(f"WARNING: {warning}")

            warnings_list.append(warning)

    if size_gb > 0.5:

        warning = "Large model size (> 500 MB)"

        print(f"WARNING: {warning}")

        warnings_list.append(warning)

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if warnings_list:

        result["status"] = "REVIEW"

    else:

        result["status"] = "PASS"

    result["warnings"] = " | ".join(warnings_list)

    results.append(result)

    # --------------------------------------------------------
    # MEMORY CLEANUP
    # --------------------------------------------------------

    del df
    del X
    del y
    del X_test
    del y_test
    del predictions
    del model

    gc.collect()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 90)
print("FINAL HEALTHCAREAI SYSTEM REPORT")
print("=" * 90)

results_df = pd.DataFrame(results)

if not results_df.empty:

    results_df = results_df.sort_values(
        by=["status", "accuracy"],
        ascending=[True, False],
        na_position="last"
    )

    display_columns = [
        "model",
        "status",
        "accuracy",
        "macro_f1",
        "classes",
        "dataset_rows",
        "test_rows",
        "size",
        "warnings"
    ]

    print(
        results_df[
            [col for col in display_columns if col in results_df.columns]
        ].to_string(
            index=False
        )
    )

    report_path = MODEL_DIR / "full_system_test_report.csv"

    results_df.to_csv(
        report_path,
        index=False
    )

    print("\n" + "=" * 90)
    print("SYSTEM SUMMARY")
    print("=" * 90)

    total = len(results_df)
    passed = int(
        (results_df["status"] == "PASS").sum()
    )
    review = int(
        (results_df["status"] == "REVIEW").sum()
    )
    skipped = int(
        (results_df["status"] == "SKIPPED").sum()
    )
    failed = int(
        results_df["status"].astype(str)
        .str.contains(
            "FAILED",
            case=False,
            na=False
        )
        .sum()
    )

    print(f"Total models : {total}")
    print(f"PASS         : {passed}")
    print(f"REVIEW       : {review}")
    print(f"SKIPPED      : {skipped}")
    print(f"FAILED       : {failed}")

    print("\nReport saved:")
    print(report_path)

else:

    print("No results generated.")


print("\n" + "=" * 90)
print("FULL SYSTEM TEST COMPLETE")
print("=" * 90)
