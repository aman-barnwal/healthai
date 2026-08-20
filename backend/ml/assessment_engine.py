import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "backend" / "models"


# ============================================================
# CACHES
# ============================================================

_MODEL_CACHE = {}

_FEATURE_CACHE = {}

_METADATA_CACHE = {}


# ============================================================
# INTERNAL MODEL FILTER
# ============================================================

SKIP_MODELS = {
    "heart_disease_pipeline",
    "processed-data",
    "train_folds",
}


# ============================================================
# GET MODEL PATH
# ============================================================

def get_model_path(model_name: str) -> Path:

    if not isinstance(model_name, str) or not model_name.strip():

        raise ValueError(
            "A valid model name is required."
        )

    model_name = model_name.strip()

    model_path = MODELS_DIR / f"{model_name}.pkl"

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not model_path.is_file():

        raise FileNotFoundError(
            f"Model path is not a valid file: {model_path}"
        )

    return model_path


# ============================================================
# GET METADATA PATH
# ============================================================

def get_metadata_path(model_name: str) -> Path:

    return (
        MODELS_DIR
        / f"{model_name}_metadata.json"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name: str):

    model_name = model_name.strip()

    if model_name in _MODEL_CACHE:

        return _MODEL_CACHE[model_name]

    model_path = get_model_path(
        model_name
    )

    try:

        model = joblib.load(
            model_path
        )

    except Exception as error:

        raise RuntimeError(
            f"Failed to load model "
            f"'{model_name}': {error}"
        ) from error

    _MODEL_CACHE[
        model_name
    ] = model

    return model


# ============================================================
# CLEAR MODEL CACHE
# ============================================================

def clear_model_cache():

    _MODEL_CACHE.clear()

    _FEATURE_CACHE.clear()


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(model_name: str) -> dict:

    model_name = model_name.strip()

    if model_name in _METADATA_CACHE:

        return _METADATA_CACHE[
            model_name
        ]

    metadata_path = get_metadata_path(
        model_name
    )

    if not metadata_path.exists():

        _METADATA_CACHE[
            model_name
        ] = {}

        return {}

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(
                file
            )

        if not isinstance(
            metadata,
            dict
        ):

            metadata = {}

    except (
        OSError,
        json.JSONDecodeError
    ):

        metadata = {}

    _METADATA_CACHE[
        model_name
    ] = metadata

    return metadata


# ============================================================
# EXTRACT FEATURES FROM MODEL
# ============================================================

def extract_features_from_model(model):

    # --------------------------------------------------------
    # DIRECT MODEL FEATURES
    # --------------------------------------------------------

    features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if features is not None:

        return list(
            features
        )

    # --------------------------------------------------------
    # PIPELINE FEATURES
    # --------------------------------------------------------

    if hasattr(
        model,
        "named_steps"
    ):

        for step in model.named_steps.values():

            features = getattr(
                step,
                "feature_names_in_",
                None
            )

            if features is not None:

                return list(
                    features
                )

    # --------------------------------------------------------
    # COLUMN TRANSFORMER FEATURES
    # --------------------------------------------------------

    if hasattr(
        model,
        "transformers_"
    ):

        for transformer in model.transformers_:

            if len(transformer) < 3:

                continue

            columns = transformer[2]

            if isinstance(
                columns,
                (list, tuple)
            ):

                return list(
                    columns
                )

    return []


# ============================================================
# GET EXACT FEATURE ORDER
# ============================================================

def get_expected_features(
    model_name: str
) -> list:

    model_name = model_name.strip()

    if model_name in _FEATURE_CACHE:

        return _FEATURE_CACHE[
            model_name
        ]

    # --------------------------------------------------------
    # LOAD MODEL ONLY ONCE
    # --------------------------------------------------------

    model = load_model(
        model_name
    )

    # --------------------------------------------------------
    # FIRST PRIORITY:
    # MODEL FEATURE NAMES
    # --------------------------------------------------------

    features = extract_features_from_model(
        model
    )

    if features:

        features = list(
            features
        )

        _FEATURE_CACHE[
            model_name
        ] = features

        return features

    # --------------------------------------------------------
    # SECOND PRIORITY:
    # METADATA
    # --------------------------------------------------------

    metadata = load_metadata(
        model_name
    )

    metadata_features = metadata.get(
        "features",
        []
    )

    if isinstance(
        metadata_features,
        list
    ) and metadata_features:

        features = list(
            metadata_features
        )

        _FEATURE_CACHE[
            model_name
        ] = features

        return features

    raise ValueError(
        f"Could not determine feature order "
        f"for model '{model_name}'."
    )


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def validate_input_data(
    model_name: str,
    data: dict
) -> dict:

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "Input data must be a dictionary."
        )

    expected_features = get_expected_features(
        model_name
    )

    missing_features = [

        feature

        for feature in expected_features

        if feature not in data

    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(
                missing_features
            )
        )

    ordered_data = {

        feature: data[feature]

        for feature in expected_features

    }

    return ordered_data


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input(
    model_name: str,
    data: dict
) -> pd.DataFrame:

    ordered_data = validate_input_data(
        model_name,
        data
    )

    return pd.DataFrame(
        [ordered_data]
    )


# ============================================================
# CONVERT NUMPY VALUES TO PYTHON
# ============================================================

def convert_to_python(value: Any):

    if hasattr(
        value,
        "item"
    ):

        try:

            return value.item()

        except Exception:

            pass

    return value


# ============================================================
# GET PREDICTION PROBABILITY
# ============================================================

def get_prediction_probability(
    model,
    input_df,
    prediction
):

    if not hasattr(
        model,
        "predict_proba"
    ):

        return None

    try:

        probabilities = model.predict_proba(
            input_df
        )[0]

        if len(probabilities) == 0:

            return None

        # ----------------------------------------------------
        # BINARY CLASSIFICATION
        # ----------------------------------------------------

        if len(probabilities) == 2:

            return float(
                probabilities[1]
            )

        # ----------------------------------------------------
        # MULTI-CLASS
        #
        # Return probability of predicted class
        # ----------------------------------------------------

        classes = getattr(
            model,
            "classes_",
            None
        )

        if classes is not None:

            for index, class_value in enumerate(
                classes
            ):

                if class_value == prediction:

                    return float(
                        probabilities[index]
                    )

        return float(
            max(probabilities)
        )

    except Exception:

        return None


# ============================================================
# RUN PREDICTION
# ============================================================

def predict(
    model_name: str,
    data: dict
) -> dict:

    model_name = model_name.strip()

    # --------------------------------------------------------
    # LOAD MODEL ONCE
    # --------------------------------------------------------

    model = load_model(
        model_name
    )

    # --------------------------------------------------------
    # BUILD INPUT
    # --------------------------------------------------------

    input_df = build_model_input(
        model_name,
        data
    )

    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_df
        )[0]

    except Exception as error:

        raise RuntimeError(
            f"Prediction failed for "
            f"'{model_name}': {error}"
        ) from error

    prediction = convert_to_python(
        prediction
    )

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probability = get_prediction_probability(
        model,
        input_df,
        prediction
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "success": True,

        "model": model_name,

        "prediction": prediction,

        "probability": probability,

        "features": list(
            input_df.columns
        ),

        "input": input_df.iloc[
            0
        ].to_dict()

    }


# ============================================================
# LIST AVAILABLE MODELS
# ============================================================

def list_available_models() -> list:

    models = []

    if not MODELS_DIR.exists():

        return models

    for model_path in MODELS_DIR.glob(
        "*.pkl"
    ):

        model_name = model_path.stem

        # ----------------------------------------------------
        # SKIP GENERATED / NON-PREDICTION FILES
        # ----------------------------------------------------

        if model_name.endswith(
            "_pipeline"
        ):

            continue

        if model_name in SKIP_MODELS:

            continue

        models.append(
            model_name
        )

    return sorted(
        models
    )


# ============================================================
# GET MODEL INFORMATION
# ============================================================

def get_model_info(
    model_name: str
) -> dict:

    metadata = load_metadata(
        model_name
    )

    features = []

    # Prefer metadata first for information display.
    # This avoids loading heavy models unnecessarily.
    metadata_features = metadata.get(
        "features",
        []
    )

    if isinstance(
        metadata_features,
        list
    ):

        features = metadata_features

    return {

        "model": model_name,

        "feature_count": len(
            features
        ),

        "features": features,

        "metadata": metadata

    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nAvailable HealthcareAI models:\n"
    )

    models = list_available_models()

    for model_name in models:

        try:

            metadata = load_metadata(
                model_name
            )

            features = metadata.get(
                "features",
                []
            )

            print(
                f"✓ {model_name}"
            )

            if features:

                print(
                    f"  Features: "
                    f"{len(features)}"
                )

            else:

                print(
                    "  Features: "
                    "metadata unavailable"
                )

        except Exception as error:

            print(
                f"✗ {model_name}"
            )

            print(
                f"  Error: {error}"
            )

    print(
        f"\nTotal models: {len(models)}"
    )

    print(
        "\nDone."
    )
