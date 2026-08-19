from pathlib import Path
import json
import gc
import os
import traceback

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ==============================================================
# CONFIGURATION
# ==============================================================

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("backend/models")

# Skip models larger than this by default.
# Your 6.8 GB diabetes models will be skipped instead of killing RAM.
MAX_MODEL_SIZE_GB = 1.0

# Maximum rows used during audit.
# Prevents unnecessarily huge prediction jobs.
MAX_AUDIT_ROWS = 50000

TEST_SIZE = 0.20
RANDOM_STATE = 123

# Set to True only if you intentionally want to try loading huge models.
FORCE_LOAD_HUGE_MODELS = False


# ==============================================================
# HELPERS
# ==============================================================

def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def model_size_bytes(path):
    return path.stat().st_size


def model_size_human(path):
    size = model_size_bytes(path)

    if size >= 1024 ** 3:
        return f"{size / (1024 ** 3):.2f} GB"

    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.2f} MB"

    if size >= 1024:
        return f"{size / 1024:.2f} KB"

    return f"{size} bytes"


def load_metadata(model_name):
    """
    Load metadata for a model if available.
    """

    metadata_path = MODEL_DIR / f"{model_name}_metadata.json"

    if not metadata_path.exists():
        return {}, None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f), metadata_path

    except Exception as e:
        print(f"Metadata load failed: {e}")
        return {}, metadata_path


def find_dataset(model_name, metadata):
    """
    Find dataset using:
    1. metadata dataset_file
    2. exact model-name match
    3. normalized name match
    """

    # ----------------------------------------------------------
    # 1. METADATA
    # ----------------------------------------------------------

    dataset_file = metadata.get("dataset_file")

    if dataset_file:
        candidate = DATASET_DIR / dataset_file

        if candidate.exists():
            return candidate

        # Search recursively
        matches = list(DATASET_DIR.rglob(dataset_file))

        if matches:
            return matches[0]

    # ----------------------------------------------------------
    # 2. EXACT MODEL NAME
    # ----------------------------------------------------------

    candidates = list(DATASET_DIR.rglob("*.csv"))

    for file in candidates:
        if file.stem.lower() == model_name.lower():
            return file

    # ----------------------------------------------------------
    # 3. NORMALIZED MATCH
    # ----------------------------------------------------------

    normalized_model = (
        model_name.lower()
        .replace("_pipeline", "")
        .replace("_model", "")
        .replace("-", "_")
    )

    for file in candidates:

        normalized_file = (
            file.stem.lower()
            .replace("-", "_")
        )

        if (
            normalized_file == normalized_model
            or normalized_model in normalized_file
            or normalized_file in normalized_model
        ):
            return file

    return None


def detect_target(df, metadata):
    """
    Determine target column.
    """

    metadata_target = metadata.get("target")

    if metadata_target in df.columns:
        return metadata_target

    # Common target names
    possible_targets = [
        "target",
        "label",
        "class",
        "diagnosis",
        "disease",
        "outcome",
        "result",
        "num",
        "stroke",
        "DEATH_EVENT",
        "Diabetes_binary",
        "Diabetes_012",
    ]

    for target in possible_targets:

        if target in df.columns:
            return target

    # Last-column fallback
    return df.columns[-1]


def remove_id_columns(X):
    """
    Remove likely ID/index columns.
    """

    id_columns = []

    for col in X.columns:

        lower = str(col).lower().strip()

        if (
            lower == "id"
            or lower == "index"
            or lower == "unnamed: 0"
            or lower.endswith("_id")
            or lower in {
                "patientid",
                "patient_id",
                "record_id",
                "case_id",
                "subject_id",
            }
        ):
            id_columns.append(col)

    if id_columns:
        X = X.drop(columns=id_columns)

    return X, id_columns


def sample_dataset(df, target):
    """
    Limit audit dataset size while preserving classes.
    """

    if len(df) <= MAX_AUDIT_ROWS:
        return df

    print(
        f"Dataset has {len(df):,} rows. "
        f"Sampling {MAX_AUDIT_ROWS:,} rows for audit."
    )

    try:

        if df[target].nunique() > 1:

            return (
                df.groupby(
                    target,
                    group_keys=False
                )
                .apply(
                    lambda x: x.sample(
                        n=max(
                            1,
                            int(
                                MAX_AUDIT_ROWS
                                * len(x)
                                / len(df)
                            )
                        ),
                        random_state=RANDOM_STATE
                    )
                )
                .reset_index(drop=True)
            )

    except Exception as e:
        print(f"Stratified sampling warning: {e}")

    return df.sample(
        n=MAX_AUDIT_ROWS,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


def get_expected_features(model):
    """
    Try to discover features expected by the trained model.
    """

    possible_objects = [model]

    # Pipeline final estimator
    try:

        if hasattr(model, "steps"):

            for _, step in model.steps:
                possible_objects.append(step)

    except Exception:
        pass

    for obj in possible_objects:

        if hasattr(obj, "feature_names_in_"):

            try:
                return list(obj.feature_names_in_)
            except Exception:
                pass

    return None


def align_features(X, expected_features):
    """
    Align audit dataset features with model expectations.
    """

    if not expected_features:
        return X, [], []

    missing = [
        col
        for col in expected_features
        if col not in X.columns
    ]

    extra = [
        col
        for col in X.columns
        if col not in expected_features
    ]

    # If missing features exist, don't silently fabricate them.
    if missing:
        return X, missing, extra

    X = X[expected_features]

    return X, missing, extra


def safe_value(value):
    """
    Convert NumPy/Pandas values to JSON-safe Python values.
    """

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        )
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        )
    ):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return None

    return value


# ==============================================================
# MAIN AUDIT
# ==============================================================

print_header("HEALTHCAREAI ADVANCED MODEL AUDIT")

print("Dataset directory:", DATASET_DIR.resolve())
print("Model directory:", MODEL_DIR.resolve())
print("Maximum model size:", f"{MAX_MODEL_SIZE_GB} GB")
print("Maximum audit rows:", f"{MAX_AUDIT_ROWS:,}")
print("Force load huge models:", FORCE_LOAD_HUGE_MODELS)


results = []

model_files = sorted(
    MODEL_DIR.glob("*.pkl")
)

print(f"\nModels discovered: {len(model_files)}")


for model_path in model_files:

    print("\n" + "-" * 100)
    print(f"MODEL: {model_path.name}")
    print("-" * 100)

    model_name = model_path.stem

    size_bytes = model_size_bytes(model_path)
    size_gb = size_bytes / (1024 ** 3)
    size_human = model_size_human(model_path)

    warnings_list = []

    print("Model size:", size_human)

    # ----------------------------------------------------------
    # HUGE MODEL PROTECTION
    # ----------------------------------------------------------

    if size_gb > MAX_MODEL_SIZE_GB:

        warning = (
            f"MODEL TOO LARGE ({size_gb:.2f} GB). "
            f"Skipped to protect system memory."
        )

        print("WARNING:", warning)

        if not FORCE_LOAD_HUGE_MODELS:

            results.append({
                "model": model_name,
                "status": "SKIPPED - TOO LARGE",
                "size": size_human,
                "dataset": None,
                "dataset_rows": None,
                "target": None,
                "features": None,
                "classes": None,
                "accuracy": None,
                "macro_f1": None,
                "precision_macro": None,
                "recall_macro": None,
                "warning": warning,
            })

            continue

    # ----------------------------------------------------------
    # METADATA
    # ----------------------------------------------------------

    metadata, metadata_path = load_metadata(model_name)

    if metadata_path and metadata:
        print("Metadata:", metadata_path.name)

    else:
        print("Metadata: NOT FOUND")
        warnings_list.append("METADATA NOT FOUND")

    # ----------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------

    model = None

    try:

        print("Loading model...")

        model = joblib.load(model_path)

        print("Load status: OK")
        print("Model type:", type(model).__name__)

    except MemoryError:

        warning = "INSUFFICIENT MEMORY TO LOAD MODEL"

        print("Load status: FAILED")
        print("WARNING:", warning)

        results.append({
            "model": model_name,
            "status": "FAILED - MEMORY",
            "size": size_human,
            "dataset": None,
            "dataset_rows": None,
            "target": None,
            "features": None,
            "classes": None,
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": warning,
        })

        continue

    except Exception as e:

        warning = str(e)

        print("Load status: FAILED")
        print("Error:", warning)

        results.append({
            "model": model_name,
            "status": "FAILED TO LOAD",
            "size": size_human,
            "dataset": None,
            "dataset_rows": None,
            "target": None,
            "features": None,
            "classes": None,
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": warning,
        })

        continue

    # ----------------------------------------------------------
    # FIND DATASET
    # ----------------------------------------------------------

    dataset_path = find_dataset(
        model_name,
        metadata
    )

    if dataset_path is None:

        warning = "DATASET NOT FOUND"

        print("Dataset:", warning)

        results.append({
            "model": model_name,
            "status": "MODEL OK - DATASET NOT FOUND",
            "size": size_human,
            "dataset": None,
            "dataset_rows": None,
            "target": None,
            "features": None,
            "classes": None,
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": " | ".join(
                warnings_list + [warning]
            ),
        })

        del model
        gc.collect()

        continue

    print("Dataset:", dataset_path)

    # ----------------------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------------------

    try:

        df = pd.read_csv(
            dataset_path,
            low_memory=False
        )

        print("Original dataset shape:", df.shape)

    except Exception as e:

        warning = f"DATASET LOAD FAILED: {e}"

        print(warning)

        results.append({
            "model": model_name,
            "status": "DATASET LOAD FAILED",
            "size": size_human,
            "dataset": str(dataset_path),
            "dataset_rows": None,
            "target": None,
            "features": None,
            "classes": None,
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": warning,
        })

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

    print("Detected target:", target)

    if target not in df.columns:

        warning = "TARGET COLUMN NOT FOUND"

        print("WARNING:", warning)

        results.append({
            "model": model_name,
            "status": "TARGET NOT FOUND",
            "size": size_human,
            "dataset": str(dataset_path),
            "dataset_rows": len(df),
            "target": target,
            "features": None,
            "classes": None,
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": " | ".join(
                warnings_list + [warning]
            ),
        })

        del model
        del df
        gc.collect()

        continue

    # ----------------------------------------------------------
    # CLEAN DATA
    # ----------------------------------------------------------

    before_rows = len(df)

    df = df.dropna(
        subset=[target]
    )

    dropped_rows = before_rows - len(df)

    if dropped_rows > 0:

        print(
            f"Rows dropped due to missing target: "
            f"{dropped_rows}"
        )

    # Sample if needed
    df = sample_dataset(
        df,
        target
    )

    print("Audit dataset shape:", df.shape)

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # ----------------------------------------------------------
    # REMOVE ID COLUMNS
    # ----------------------------------------------------------

    X, id_columns = remove_id_columns(X)

    if id_columns:
        print(
            "Removed ID columns:",
            id_columns
        )

    # ----------------------------------------------------------
    # FEATURE ALIGNMENT
    # ----------------------------------------------------------

    expected_features = get_expected_features(model)

    if expected_features:

        print(
            "Model expected features:",
            len(expected_features)
        )

        X, missing_features, extra_features = align_features(
            X,
            expected_features
        )

        if missing_features:

            warning = (
                f"FEATURE MISMATCH - "
                f"{len(missing_features)} expected "
                f"features missing"
            )

            print("WARNING:", warning)
            print(
                "Missing features:",
                missing_features[:20]
            )

            results.append({
                "model": model_name,
                "status": "FEATURE MISMATCH",
                "size": size_human,
                "dataset": str(dataset_path),
                "dataset_rows": len(df),
                "target": target,
                "features": X.shape[1],
                "classes": int(y.nunique()),
                "accuracy": None,
                "macro_f1": None,
                "precision_macro": None,
                "recall_macro": None,
                "warning": " | ".join(
                    warnings_list + [warning]
                ),
            })

            del model
            del df
            del X
            del y

            gc.collect()

            continue

        if extra_features:

            print(
                f"Extra dataset features ignored: "
                f"{len(extra_features)}"
            )

    # ----------------------------------------------------------
    # DATA SUMMARY
    # ----------------------------------------------------------

    print("Features:", X.shape)
    print("Classes:", y.nunique())

    if y.nunique() < 2:

        warning = "ONLY ONE TARGET CLASS FOUND"

        print("WARNING:", warning)

        results.append({
            "model": model_name,
            "status": "INVALID TARGET",
            "size": size_human,
            "dataset": str(dataset_path),
            "dataset_rows": len(df),
            "target": target,
            "features": X.shape[1],
            "classes": int(y.nunique()),
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": " | ".join(
                warnings_list + [warning]
            ),
        })

        del model
        del df
        del X
        del y

        gc.collect()

        continue

    # ----------------------------------------------------------
    # SPLIT DATA
    # ----------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        print(
            "Audit test samples:",
            len(X_test)
        )

    except Exception as e:

        warning = f"SPLIT FAILED: {e}"

        print(warning)

        results.append({
            "model": model_name,
            "status": "SPLIT FAILED",
            "size": size_human,
            "dataset": str(dataset_path),
            "dataset_rows": len(df),
            "target": target,
            "features": X.shape[1],
            "classes": int(y.nunique()),
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": " | ".join(
                warnings_list + [warning]
            ),
        })

        del model
        del df
        del X
        del y

        gc.collect()

        continue

    # ----------------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------------

    try:

        print("Running predictions...")

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

        precision_macro = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall_macro = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        print(
            f"\nAUDIT ACCURACY: "
            f"{accuracy:.4f}"
        )

        print(
            f"AUDIT MACRO F1: "
            f"{macro_f1:.4f}"
        )

        print(
            f"AUDIT MACRO PRECISION: "
            f"{precision_macro:.4f}"
        )

        print(
            f"AUDIT MACRO RECALL: "
            f"{recall_macro:.4f}"
        )

        # ------------------------------------------------------
        # WARNINGS
        # ------------------------------------------------------

        if accuracy >= 0.995:

            warning = (
                "SUSPICIOUSLY HIGH ACCURACY - "
                "CHECK FOR DATA LEAKAGE OR DATA OVERLAP"
            )

            print("WARNING:", warning)

            warnings_list.append(warning)

        if accuracy < 0.50:

            warning = (
                "LOW PERFORMANCE - "
                "MODEL MAY NOT BE RELIABLE"
            )

            print("WARNING:", warning)

            warnings_list.append(warning)

        if macro_f1 < 0.50:

            warning = (
                "LOW MACRO F1 - "
                "CHECK CLASS IMBALANCE"
            )

            print("WARNING:", warning)

            warnings_list.append(warning)

        status = "PASS"

        if warnings_list:
            status = "REVIEW"

        results.append({
            "model": model_name,
            "status": status,
            "size": size_human,
            "dataset": str(dataset_path),
            "dataset_rows": int(len(df)),
            "target": target,
            "features": int(X.shape[1]),
            "classes": int(y.nunique()),
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "precision_macro": round(
                float(precision_macro),
                4
            ),
            "recall_macro": round(
                float(recall_macro),
                4
            ),
            "warning": " | ".join(
                warnings_list
            ),
        })

    except Exception as e:

        warning = (
            f"PREDICTION FAILED: {e}"
        )

        print(warning)

        traceback.print_exc()

        results.append({
            "model": model_name,
            "status": "PREDICTION FAILED",
            "size": size_human,
            "dataset": str(dataset_path),
            "dataset_rows": int(len(df)),
            "target": target,
            "features": int(X.shape[1]),
            "classes": int(y.nunique()),
            "accuracy": None,
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "warning": " | ".join(
                warnings_list + [warning]
            ),
        })

    # ----------------------------------------------------------
    # MEMORY CLEANUP
    # ----------------------------------------------------------

    del model
    del df
    del X
    del y

    try:
        del X_train
        del X_test
        del y_train
        del y_test
    except Exception:
        pass

    gc.collect()


# ==============================================================
# FINAL REPORT
# ==============================================================

print_header("FINAL HEALTHCAREAI MODEL AUDIT REPORT")

results_df = pd.DataFrame(results)

if results_df.empty:

    print("No models were audited.")

else:

    results_df = results_df.sort_values(
        by=[
            "accuracy",
            "macro_f1"
        ],
        ascending=False,
        na_position="last"
    )

    display_columns = [
        "model",
        "status",
        "accuracy",
        "macro_f1",
        "precision_macro",
        "recall_macro",
        "features",
        "classes",
        "dataset_rows",
        "size",
        "warning",
    ]

    existing_columns = [
        col
        for col in display_columns
        if col in results_df.columns
    ]

    print(
        results_df[
            existing_columns
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    total = len(results_df)

    passed = (
        results_df["status"] == "PASS"
    ).sum()

    review = (
        results_df["status"] == "REVIEW"
    ).sum()

    failed = results_df[
        results_df["status"].str.contains(
            "FAILED|MISMATCH|INVALID",
            case=False,
            na=False
        )
    ].shape[0]

    skipped = results_df[
        results_df["status"].str.contains(
            "SKIPPED",
            case=False,
            na=False
        )
    ].shape[0]

    print("\n" + "-" * 100)
    print("SUMMARY")
    print("-" * 100)

    print("Total models:", total)
    print("PASS:", passed)
    print("REVIEW:", review)
    print("FAILED / INVALID:", failed)
    print("SKIPPED:", skipped)

    valid_scores = results_df[
        results_df["accuracy"].notna()
    ]

    if not valid_scores.empty:

        best_model = valid_scores.iloc[0]

        print("\nBEST AUDITED MODEL:")

        print(
            f"{best_model['model']} "
            f"| Accuracy: {best_model['accuracy']:.4f} "
            f"| Macro F1: {best_model['macro_f1']:.4f}"
        )

    # ----------------------------------------------------------
    # SAVE CSV
    # ----------------------------------------------------------

    csv_path = (
        MODEL_DIR /
        "model_audit_report.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    # ----------------------------------------------------------
    # SAVE JSON
    # ----------------------------------------------------------

    json_path = (
        MODEL_DIR /
        "model_audit_report.json"
    )

    safe_results = []

    for row in results_df.to_dict(
        orient="records"
    ):

        safe_row = {
            key: safe_value(value)
            for key, value in row.items()
        }

        safe_results.append(
            safe_row
        )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            safe_results,
            f,
            indent=4
        )

    print("\nReports saved:")

    print(csv_path)
    print(json_path)


print("\n" + "=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)
