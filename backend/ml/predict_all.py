import json
import joblib
import pandas as pd

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("backend/models")


# ============================================================
# MODEL REGISTRY
# ============================================================

def discover_models():
    """
    Automatically discover every trained model that has
    corresponding metadata.
    """

    registry = {}

    for model_path in MODEL_DIR.glob("*.pkl"):

        metadata_path = MODEL_DIR / (
            f"{model_path.stem}_metadata.json"
        )

        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, "r") as file:
                metadata = json.load(file)

            registry[model_path.stem] = {
                "model_path": model_path,
                "metadata_path": metadata_path,
                "metadata": metadata
            }

        except Exception as error:

            print(
                f"Could not load metadata for "
                f"{model_path.name}: {error}"
            )

    return registry


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name):
    """
    Load a trained model from the registry.
    """

    registry = discover_models()

    if model_name not in registry:

        raise ValueError(
            f"Model '{model_name}' not found."
        )

    model_path = registry[
        model_name
    ]["model_path"]

    return joblib.load(model_path)


# ============================================================
# GET MODEL INFORMATION
# ============================================================

def get_model_info(model_name):

    registry = discover_models()

    if model_name not in registry:

        raise ValueError(
            f"Model '{model_name}' not found."
        )

    return registry[
        model_name
    ]["metadata"]


# ============================================================
# LIST ALL MODELS
# ============================================================

def list_models():

    registry = discover_models()

    results = []

    for name, data in registry.items():

        metadata = data["metadata"]

        results.append({
            "name": name,
            "target": metadata.get("target"),
            "model": metadata.get("model"),
            "accuracy": metadata.get(
                "test_accuracy"
            ),
            "macro_f1": metadata.get(
                "test_macro_f1"
            )
        })

    return results


# ============================================================
# PREDICT
# ============================================================

def predict(model_name, patient_data):
    """
    Make a prediction using one of the trained models.

    patient_data must be a dictionary whose keys match the
    training features.
    """

    registry = discover_models()

    if model_name not in registry:

        raise ValueError(
            f"Model '{model_name}' not found."
        )

    model = joblib.load(
        registry[model_name]["model_path"]
    )

    metadata = registry[
        model_name
    ]["metadata"]

    # --------------------------------------------------------
    # Convert dictionary to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        [patient_data]
    )

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(df)[0]

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "model": model_name,
        "prediction": str(prediction),
        "target": metadata.get("target"),
        "model_type": metadata.get("model"),
        "test_accuracy": metadata.get(
            "test_accuracy"
        )
    }

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = (
            model.predict_proba(df)[0]
        )

        classes = model.classes_

        probability_map = {}

        for class_name, probability in zip(
            classes,
            probabilities
        ):

            probability_map[
                str(class_name)
            ] = round(
                float(probability) * 100,
                2
            )

        result[
            "probabilities"
        ] = probability_map

        result[
            "confidence"
        ] = round(
            float(max(probabilities)) * 100,
            2
        )

    return result


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("HEALTHCAREAI MODEL REGISTRY")
    print("=" * 70)

    models = list_models()

    if not models:

        print(
            "\nNo trained models found."
        )

    else:

        print(
            "\nAvailable models:\n"
        )

        for model in models:

            print(
                f"Model: {model['name']}"
            )

            print(
                f"Target: {model['target']}"
            )

            print(
                f"Algorithm: {model['model']}"
            )

            print(
                f"Test accuracy: "
                f"{model['accuracy']:.4f}"
            )

            print(
                f"Macro F1: "
                f"{model['macro_f1']:.4f}"
            )

            print("-" * 50)

    print(
        "\nModel registry loaded successfully."
    )
