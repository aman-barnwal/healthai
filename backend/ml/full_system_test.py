"""
HealthcareAI - Full System Test

This script performs a broad audit of all locally stored ML models.

Tests include:
1. Model discovery
2. Safe model loading
3. Metadata validation
4. Dataset discovery
5. Dataset quality checks
6. Duplicate row detection
7. Possible target leakage checks
8. Feature compatibility checks
9. Holdout prediction testing
10. Cross-validation of cloned estimators when practical
11. Prediction latency
12. Model size and memory-risk detection
13. Final leaderboard/report

This is an engineering audit tool, not a clinical validation system.
"""

from pathlib import Path
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

REPORT_PATH = MODEL_DIR / "full_system_test_report.csv"
JSON_REPORT_PATH = MODEL_DIR / "full_system_test_report.json"

RANDOM_STATE = 42

# Skip loading models larger than this.
# Prevents huge models from killing available RAM.
MAX_MODEL_SIZE_GB = 1.0

# Cross-validation is skipped for very large datasets.
MAX_CV_ROWS = 50_000

# Maximum number of rows used for prediction timing.
MAX_TIMING_ROWS = 100

# Test split.
TEST_SIZE = 0.20

# Number of CV folds.
CV_FOLDS = 5


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subsection(title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def format_bytes(size):
    if size >= 1024 ** 3:
        return f"{size / (1024 ** 3):.2f} GB"

    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.2f} MB"

    if size >= 1024:
        return f"{size / 1024:.2f} KB"

    return f"{size} bytes"


def safe_float(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

        return float(value)

    except Exception:
        return None


def safe_round(value, digits=4):
    value = safe_float(value)

    if value is None:
        return None

    return round(value, digits)


# =============================================================================
# MODEL / METADATA DISCOVERY
# =============================================================================

def load_metadata(model_path):
    """
    Load metadata associated with a model.

    Supports:
        model.pkl
        model_metadata.json
    """

    metadata_path = (
        MODEL_DIR /
        f"{model_path.stem}_metadata.json"
    )

    if not metadata_path.exists():
        return {}, None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata, metadata_path

    except Exception as error:
        print(f"Metadata error: {error}")
        return {}, metadata_path


def find_dataset(model_name, metadata):
    """
    Find the dataset associated with a model.

    Priority:
    1. dataset_file from metadata
    2. dataset path from metadata
    3. Exact filename match
    4. Normalized model-name match
    """

    possible_paths = []

    if metadata:
        for key in [
            "dataset_file",
            "dataset",
            "dataset_path",
        ]:
            value = metadata.get(key)

            if value:
                possible_paths.append(str(value))

    for value in possible_paths:
        path = Path(value)

        if path.exists():
            return path

        path = DATASET_DIR / path.name

        if path.exists():
            return path

    model_normalized = (
        model_name
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    csv_files = list(DATASET_DIR.rglob("*.csv"))

    # Exact stem match
    for file in csv_files:
        if file.stem.lower() == model_name.lower():
            return file

    # Normalized match
    for file in csv_files:
        file_normalized = (
            file.stem
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if file_normalized == model_normalized:
            return file

    return None


def detect_target(df, metadata):
    """
    Detect target column.

    Priority:
    1. Metadata target
    2. Common target names
    3. Last column fallback
    """

    if metadata:
        target = metadata.get("target")

        if target in df.columns:
            return target

    common_targets = [
        "target",
        "label",
        "class",
        "diagnosis",
        "disease",
        "outcome",
        "result",
        "prediction",
        "Diabetes_binary",
        "Diabetes_012",
        "DEATH_EVENT",
        "NObeyesdad",
        "Dry Eye Disease",
        "Dialysis_Needed",
        "stroke",
        "status",
    ]

    for column in common_targets:
        if column in df.columns:
            return column

    return df.columns[-1]


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def detect_id_columns(X):
    """
    Detect likely ID/index columns.

    These can artificially inflate accuracy.
    """

    id_columns = []

    exact_names = {
        "id",
        "index",
        "patientid",
        "patient_id",
        "record_id",
        "sample_id",
        "subject_id",
        "case_id",
        "unnamed: 0",
    }

    for column in X.columns:

        name = str(column).strip().lower()

        if (
            name in exact_names
            or name.endswith("_id")
            or name.startswith("unnamed")
        ):
            id_columns.append(column)

    return id_columns


def detect_possible_target_leakage(X, y):
    """
    Look for columns suspiciously related to the target.

    This is heuristic only.
    """

    suspicious = []

    target_name = str(y.name).lower()

    for column in X.columns:

        column_name = str(column).lower()

        if column_name == target_name:
            suspicious.append(
                f"{column}: exact target-name match"
            )

        if target_name in column_name:
            suspicious.append(
                f"{column}: contains target name"
            )

    return suspicious


def analyze_dataset(df, target):
    """
    Return basic dataset quality statistics.
    """

    total_missing = int(df.isna().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    duplicate_ratio = (
        duplicate_rows / len(df)
        if len(df) > 0
        else 0
    )

    target_missing = int(df[target].isna().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": duplicate_ratio,
        "target_missing": target_missing,
        "classes": int(df[target].nunique()),
    }


# =============================================================================
# FEATURE COMPATIBILITY
# =============================================================================

def get_model_feature_names(model):
    """
    Try to determine expected feature names.
    """

    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)

    except Exception:
        pass

    try:
        if hasattr(model, "named_steps"):

            for step in model.named_steps.values():

                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)

    except Exception:
        pass

    return None


def check_feature_compatibility(model, X):
    """
    Compare dataset features with expected model features.
    """

    expected_features = get_model_feature_names(model)

    if expected_features is None:
        return {
            "status": "UNKNOWN",
            "expected_count": None,
            "actual_count": int(X.shape[1]),
            "missing_features": [],
            "extra_features": [],
        }

    actual_features = list(X.columns)

    missing_features = [
        feature
        for feature in expected_features
        if feature not in actual_features
    ]

    extra_features = [
        feature
        for feature in actual_features
        if feature not in expected_features
    ]

    status = "PASS"

    if missing_features:
        status = "FAIL"

    return {
        "status": status,
        "expected_count": len(expected_features),
        "actual_count": len(actual_features),
        "missing_features": missing_features,
        "extra_features": extra_features,
    }


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def create_split(X, y):
    """
    Create a safe train/test split.
    """

    if y.nunique() < 2:
        raise ValueError(
            "Target contains fewer than 2 classes."
        )

    class_counts = y.value_counts()

    can_stratify = (
        len(class_counts) > 1
        and class_counts.min() >= 2
    )

    stratify = y if can_stratify else None

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def evaluate_holdout(model, X, y):
    """
    Evaluate the existing trained model on a reproducible holdout split.

    Important:
    This does NOT necessarily reproduce the model's original training split.
    It is an audit score, not necessarily an unbiased external-validation score.
    """

    X_train, X_test, y_train, y_test = create_split(
        X,
        y,
    )

    start = time.perf_counter()

    predictions = model.predict(X_test)

    prediction_time = (
        time.perf_counter() - start
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "prediction_time": prediction_time,
        "test_samples": len(X_test),
    }


def evaluate_prediction_latency(model, X):
    """
    Measure approximate prediction latency.
    """

    if len(X) == 0:
        return None

    sample = X.head(
        min(
            MAX_TIMING_ROWS,
            len(X),
        )
    )

    start = time.perf_counter()

    model.predict(sample)

    elapsed = (
        time.perf_counter() - start
    )

    return {
        "samples": len(sample),
        "seconds": elapsed,
        "milliseconds_per_sample": (
            elapsed * 1000 / len(sample)
        ),
    }


def evaluate_cross_validation(model, X, y):
    """
    Retrain cloned estimator using cross-validation.

    Skipped for:
    - Huge datasets
    - Models that cannot be cloned
    - Models where CV would be too expensive
    """

    if len(X) > MAX_CV_ROWS:

        return {
            "status": "SKIPPED",
            "reason": (
                f"Dataset has {len(X)} rows "
                f"(limit: {MAX_CV_ROWS})"
            ),
            "mean_accuracy": None,
            "std_accuracy": None,
        }

    class_counts = y.value_counts()

    if (
        y.nunique() < 2
        or class_counts.min() < 2
    ):
        return {
            "status": "SKIPPED",
            "reason": "Insufficient class samples",
            "mean_accuracy": None,
            "std_accuracy": None,
        }

    n_splits = min(
        CV_FOLDS,
        int(class_counts.min()),
    )

    if n_splits < 2:
        return {
            "status": "SKIPPED",
            "reason": "Cannot create CV folds",
            "mean_accuracy": None,
            "std_accuracy": None,
        }

    try:

        estimator = clone(model)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

        return {
            "status": "PASS",
            "reason": "",
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        }

    except Exception as error:

        return {
            "status": "FAILED",
            "reason": str(error),
            "mean_accuracy": None,
            "std_accuracy": None,
        }


# =============================================================================
# STATUS CLASSIFICATION
# =============================================================================

def classify_model(
    holdout,
    cv_result,
    size_gb,
    dataset_info,
    leakage_flags,
    feature_check,
):
    """
    Produce PASS / REVIEW / FAIL.
    """

    warnings_list = []

    status = "PASS"

    if size_gb > MAX_MODEL_SIZE_GB:
        status = "REVIEW"

        warnings_list.append(
            f"Model exceeds {MAX_MODEL_SIZE_GB} GB"
        )

    if dataset_info["duplicate_ratio"] > 0.10:
        status = "REVIEW"

        warnings_list.append(
            "More than 10% duplicate rows"
        )

    if leakage_flags:
        status = "REVIEW"

        warnings_list.append(
            "Possible target leakage"
        )

    if feature_check["status"] == "FAIL":
        status = "FAIL"

        warnings_list.append(
            "Feature mismatch"
        )

    if holdout:

        accuracy = holdout["accuracy"]

        if accuracy >= 0.995:
            status = "REVIEW"

            warnings_list.append(
                "Suspiciously high audit accuracy"
            )

        elif accuracy < 0.50:
            status = "REVIEW"

            warnings_list.append(
                "Low audit accuracy"
            )

    if (
        cv_result
        and cv_result["status"] == "PASS"
        and holdout
        and cv_result["mean_accuracy"] is not None
    ):

        difference = abs(
            holdout["accuracy"]
            - cv_result["mean_accuracy"]
        )

        if difference > 0.15:

            status = "REVIEW"

            warnings_list.append(
                "Large holdout/CV performance difference"
            )

    return status, warnings_list


# =============================================================================
# SINGLE MODEL TEST
# =============================================================================

def test_model(model_path):
    """
    Perform complete testing for one model.
    """

    subsection(
        f"MODEL: {model_path.name}"
    )

    model_name = model_path.stem

    result = {
        "model": model_name,
        "model_file": model_path.name,
    }

    # -------------------------------------------------------------------------
    # MODEL SIZE
    # -------------------------------------------------------------------------

    size_bytes = model_path.stat().st_size

    size_gb = (
        size_bytes / (1024 ** 3)
    )

    result["model_size"] = format_bytes(
        size_bytes
    )

    result["model_size_gb"] = safe_round(
        size_gb,
        4,
    )

    print(
        "Model size:",
        result["model_size"],
    )

    # -------------------------------------------------------------------------
    # LARGE MODEL PROTECTION
    # -------------------------------------------------------------------------

    if size_gb > MAX_MODEL_SIZE_GB:

        print(
            "STATUS: SKIPPED"
        )

        print(
            f"Reason: model larger than "
            f"{MAX_MODEL_SIZE_GB} GB"
        )

        result.update({
            "status": "SKIPPED",
            "accuracy": None,
            "macro_f1": None,
            "precision": None,
            "recall": None,
            "cv_accuracy": None,
            "cv_std": None,
            "prediction_time_sec": None,
            "latency_ms_per_sample": None,
            "warning": (
                f"Skipped for memory safety: "
                f"{result['model_size']}"
            ),
        })

        return result

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    metadata, metadata_path = load_metadata(
        model_path
    )

    result["metadata_found"] = bool(
        metadata
    )

    result["metadata_file"] = (
        metadata_path.name
        if metadata_path
        else None
    )

    if metadata:
        print(
            "Metadata:",
            metadata_path.name,
        )
    else:
        print(
            "Metadata: NOT FOUND"
        )

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    try:

        load_start = time.perf_counter()

        model = joblib.load(
            model_path
        )

        load_time = (
            time.perf_counter()
            - load_start
        )

        result["load_time_sec"] = safe_round(
            load_time,
            4,
        )

        result["model_type"] = (
            type(model).__name__
        )

        print(
            "Load status: OK"
        )

        print(
            "Model type:",
            result["model_type"],
        )

        print(
            f"Load time: "
            f"{load_time:.4f}s"
        )

    except Exception as error:

        print(
            "Load status: FAILED"
        )

        print(
            "Error:",
            error,
        )

        result.update({
            "status": "FAILED",
            "accuracy": None,
            "macro_f1": None,
            "precision": None,
            "recall": None,
            "cv_accuracy": None,
            "cv_std": None,
            "prediction_time_sec": None,
            "latency_ms_per_sample": None,
            "warning": str(error),
        })

        return result

    # -------------------------------------------------------------------------
    # FIND DATASET
    # -------------------------------------------------------------------------

    dataset_path = find_dataset(
        model_name,
        metadata,
    )

    if dataset_path is None:

        print(
            "Dataset: NOT FOUND"
        )

        result.update({
            "status": "REVIEW",
            "dataset": None,
            "accuracy": None,
            "macro_f1": None,
            "precision": None,
            "recall": None,
            "cv_accuracy": None,
            "cv_std": None,
            "prediction_time_sec": None,
            "latency_ms_per_sample": None,
            "warning": (
                "Model loads but training dataset "
                "could not be found"
            ),
        })

        del model
        gc.collect()

        return result

    result["dataset"] = str(
        dataset_path.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        "Dataset:",
        result["dataset"],
    )

    # -------------------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------------------

    try:

        df = pd.read_csv(
            dataset_path
        )

    except Exception as error:

        print(
            "Dataset load failed:",
            error,
        )

        result.update({
            "status": "FAILED",
            "warning": (
                f"Dataset load failed: {error}"
            ),
        })

        del model
        gc.collect()

        return result

    # -------------------------------------------------------------------------
    # TARGET DETECTION
    # -------------------------------------------------------------------------

    target = detect_target(
        df,
        metadata,
    )

    result["target"] = target

    if target not in df.columns:

        print(
            "Target: NOT FOUND"
        )

        result.update({
            "status": "FAILED",
            "warning": (
                "Target column could not "
                "be identified"
            ),
        })

        del model
        del df
        gc.collect()

        return result

    print(
        "Target:",
        target,
    )

    # -------------------------------------------------------------------------
    # DATASET QUALITY
    # -------------------------------------------------------------------------

    dataset_info = analyze_dataset(
        df,
        target,
    )

    result.update(
        dataset_info
    )

    print(
        "Dataset shape:",
        (
            dataset_info["rows"],
            dataset_info["columns"],
        ),
    )

    print(
        "Classes:",
        dataset_info["classes"],
    )

    print(
        "Missing values:",
        dataset_info["missing_values"],
    )

    print(
        "Duplicate rows:",
        dataset_info["duplicate_rows"],
    )

    # -------------------------------------------------------------------------
    # PREPARE FEATURES
    # -------------------------------------------------------------------------

    df = df.dropna(
        subset=[target]
    )

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    id_columns = detect_id_columns(
        X
    )

    if id_columns:

        print(
            "Removing ID columns:",
            id_columns,
        )

        X = X.drop(
            columns=id_columns
        )

    result["id_columns_removed"] = (
        ", ".join(
            map(str, id_columns)
        )
    )

    result["feature_count"] = int(
        X.shape[1]
    )

    # -------------------------------------------------------------------------
    # TARGET LEAKAGE CHECK
    # -------------------------------------------------------------------------

    leakage_flags = (
        detect_possible_target_leakage(
            X,
            y,
        )
    )

    result["leakage_flags"] = (
        " | ".join(
            leakage_flags
        )
    )

    if leakage_flags:

        print(
            "Possible leakage:",
            leakage_flags,
        )

    # -------------------------------------------------------------------------
    # FEATURE COMPATIBILITY
    # -------------------------------------------------------------------------

    feature_check = (
        check_feature_compatibility(
            model,
            X,
        )
    )

    result["feature_check"] = (
        feature_check["status"]
    )

    result["expected_features"] = (
        feature_check[
            "expected_count"
        ]
    )

    result["actual_features"] = (
        feature_check[
            "actual_count"
        ]
    )

    result["missing_features"] = (
        ", ".join(
            map(
                str,
                feature_check[
                    "missing_features"
                ],
            )
        )
    )

    result["extra_features"] = (
        ", ".join(
            map(
                str,
                feature_check[
                    "extra_features"
                ],
            )
        )
    )

    print(
        "Feature compatibility:",
        feature_check["status"],
    )

    if (
        feature_check[
            "missing_features"
        ]
    ):
        print(
            "Missing features:",
            feature_check[
                "missing_features"
            ],
        )

    # -------------------------------------------------------------------------
    # HOLDOUT EVALUATION
    # -------------------------------------------------------------------------

    holdout = None

    try:

        holdout = evaluate_holdout(
            model,
            X,
            y,
        )

        result["accuracy"] = safe_round(
            holdout["accuracy"]
        )

        result["macro_f1"] = safe_round(
            holdout["macro_f1"]
        )

        result["precision"] = safe_round(
            holdout["precision"]
        )

        result["recall"] = safe_round(
            holdout["recall"]
        )

        result["prediction_time_sec"] = (
            safe_round(
                holdout[
                    "prediction_time"
                ],
                6,
            )
        )

        result["test_samples"] = (
            holdout[
                "test_samples"
            ]
        )

        print(
            f"\nAUDIT ACCURACY: "
            f"{holdout['accuracy']:.4f}"
        )

        print(
            f"AUDIT MACRO F1: "
            f"{holdout['macro_f1']:.4f}"
        )

        print(
            f"AUDIT PRECISION: "
            f"{holdout['precision']:.4f}"
        )

        print(
            f"AUDIT RECALL: "
            f"{holdout['recall']:.4f}"
        )

    except Exception as error:

        print(
            "Holdout evaluation failed:",
            error,
        )

        result["accuracy"] = None
        result["macro_f1"] = None
        result["precision"] = None
        result["recall"] = None
        result["prediction_time_sec"] = None

        result["holdout_error"] = str(
            error
        )

    # -------------------------------------------------------------------------
    # PREDICTION LATENCY
    # -------------------------------------------------------------------------

    latency = None

    try:

        latency = (
            evaluate_prediction_latency(
                model,
                X,
            )
        )

        if latency:

            result[
                "latency_ms_per_sample"
            ] = safe_round(
                latency[
                    "milliseconds_per_sample"
                ],
                6,
            )

            print(
                f"Prediction latency: "
                f"{latency['milliseconds_per_sample']:.6f} "
                f"ms/sample"
            )

    except Exception as error:

        result[
            "latency_error"
        ] = str(error)

        print(
            "Latency test failed:",
            error,
        )

    # -------------------------------------------------------------------------
    # CROSS VALIDATION
    # -------------------------------------------------------------------------

    cv_result = None

    try:

        print(
            "\nRunning cross-validation..."
        )

        cv_result = (
            evaluate_cross_validation(
                model,
                X,
                y,
            )
        )

        result["cv_status"] = (
            cv_result["status"]
        )

        result["cv_accuracy"] = safe_round(
            cv_result[
                "mean_accuracy"
            ]
        )

        result["cv_std"] = safe_round(
            cv_result[
                "std_accuracy"
            ]
        )

        if (
            cv_result["status"]
            == "PASS"
        ):

            print(
                f"CV ACCURACY: "
                f"{cv_result['mean_accuracy']:.4f} "
                f"± "
                f"{cv_result['std_accuracy']:.4f}"
            )

        else:

            print(
                "CV:",
                cv_result["status"],
            )

            print(
                "Reason:",
                cv_result["reason"],
            )

    except Exception as error:

        result["cv_status"] = "FAILED"

        result["cv_accuracy"] = None
        result["cv_std"] = None

        result["cv_error"] = str(
            error
        )

        print(
            "Cross-validation failed:",
            error,
        )

    # -------------------------------------------------------------------------
    # FINAL CLASSIFICATION
    # -------------------------------------------------------------------------

    status, warnings_list = classify_model(
        holdout,
        cv_result,
        size_gb,
        dataset_info,
        leakage_flags,
        feature_check,
    )

    result["status"] = status

    result["warning"] = (
        " | ".join(
            warnings_list
        )
    )

    print(
        "\nFINAL STATUS:",
        status,
    )

    if warnings_list:

        print(
            "WARNINGS:"
        )

        for warning in warnings_list:

            print(
                f"  - {warning}"
            )

    # -------------------------------------------------------------------------
    # CLEAN MEMORY
    # -------------------------------------------------------------------------

    del model
    del df
    del X
    del y

    gc.collect()

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    section(
        "HEALTHCAREAI FULL SYSTEM TEST"
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Dataset directory:",
        DATASET_DIR,
    )

    print(
        "Model directory:",
        MODEL_DIR,
    )

    print(
        "Maximum model size:",
        f"{MAX_MODEL_SIZE_GB} GB",
    )

    if not MODEL_DIR.exists():

        print(
            "\nERROR: Model directory not found."
        )

        return

    model_files = sorted(
        MODEL_DIR.glob("*.pkl")
    )

    if not model_files:

        print(
            "\nNo .pkl models found."
        )

        return

    print(
        f"\nModels discovered: "
        f"{len(model_files)}"
    )

    results = []

    start_time = time.perf_counter()

    for index, model_path in enumerate(
        model_files,
        start=1,
    ):

        print(
            f"\n[{index}/{len(model_files)}]"
        )

        try:

            result = test_model(
                model_path
            )

            results.append(
                result
            )

        except KeyboardInterrupt:

            print(
                "\n\nTest interrupted by user."
            )

            break

        except Exception as error:

            print(
                f"\nUNEXPECTED ERROR "
                f"FOR {model_path.name}"
            )

            print(
                error
            )

            results.append({
                "model": model_path.stem,
                "model_file": model_path.name,
                "status": "FAILED",
                "warning": (
                    f"Unexpected error: {error}"
                ),
            })

        gc.collect()

    total_time = (
        time.perf_counter()
        - start_time
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    section(
        "FINAL HEALTHCAREAI SYSTEM REPORT"
    )

    if not results:

        print(
            "No models were tested."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    if "accuracy" in results_df.columns:

        results_df = (
            results_df.sort_values(
                by="accuracy",
                ascending=False,
                na_position="last",
            )
        )

    display_columns = [
        "model",
        "status",
        "accuracy",
        "macro_f1",
        "cv_accuracy",
        "model_size",
        "feature_check",
        "warning",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results_df.columns
    ]

    print(
        results_df[
            available_columns
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    total_models = len(
        results_df
    )

    passed = int(
        (
            results_df["status"]
            == "PASS"
        ).sum()
    )

    review = int(
        (
            results_df["status"]
            == "REVIEW"
        ).sum()
    )

    skipped = int(
        (
            results_df["status"]
            == "SKIPPED"
        ).sum()
    )

    failed = int(
        (
            results_df["status"]
            == "FAILED"
        ).sum()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SYSTEM SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Total models:       "
        f"{total_models}"
    )

    print(
        f"PASS:               "
        f"{passed}"
    )

    print(
        f"REVIEW:             "
        f"{review}"
    )

    print(
        f"SKIPPED:            "
        f"{skipped}"
    )

    print(
        f"FAILED:             "
        f"{failed}"
    )

    if (
        "accuracy"
        in results_df.columns
    ):

        valid_scores = (
            results_df["accuracy"]
            .dropna()
        )

        if not valid_scores.empty:

            print(
                f"Average audit accuracy: "
                f"{valid_scores.mean():.4f}"
            )

            print(
                f"Best audit accuracy:    "
                f"{valid_scores.max():.4f}"
            )

            print(
                f"Lowest audit accuracy:  "
                f"{valid_scores.min():.4f}"
            )

    print(
        f"\nTotal test time: "
        f"{total_time:.2f} seconds"
    )

    # =========================================================================
    # SAVE CSV REPORT
    # =========================================================================

    try:

        results_df.to_csv(
            REPORT_PATH,
            index=False,
        )

        print(
            f"\nCSV report saved:"
        )

        print(
            REPORT_PATH
        )

    except Exception as error:

        print(
            "\nCould not save CSV report:",
            error,
        )

    # =========================================================================
    # SAVE JSON REPORT
    # =========================================================================

    try:

        json_results = (
            results_df
            .replace(
                {
                    np.nan: None,
                }
            )
            .to_dict(
                orient="records"
            )
        )

        with open(
            JSON_REPORT_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                json_results,
                f,
                indent=4,
                default=str,
            )

        print(
            "\nJSON report saved:"
        )

        print(
            JSON_REPORT_PATH
        )

    except Exception as error:

        print(
            "\nCould not save JSON report:",
            error,
        )

    section(
        "FULL SYSTEM TEST COMPLETE"
    )

    print(
        "NOTE:"
    )

    print(
        "High accuracy on the original training dataset "
        "is not equivalent to real-world clinical accuracy."
    )

    print(
        "Models marked REVIEW should be inspected for "
        "data leakage, duplicates, feature mismatch, "
        "or unrealistic validation performance."
    )


if __name__ == "__main__":
    main()
