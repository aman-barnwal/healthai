import joblib
import pandas as pd


MODEL_PATH = "backend/models/heart_disease.pkl"

def load_model():
    """Load the trained healthcare model."""
    return joblib.load(MODEL_PATH)


def predict_heart_disease(patient_data):
    """Predict heart disease risk for one patient."""

    model = load_model()

    # Convert patient data into a DataFrame
    patient_df = pd.DataFrame([patient_data])

    # Prediction
    prediction = model.predict(patient_df)[0]

    # Probability
    probabilities = model.predict_proba(patient_df)[0]

    disease_probability = probabilities[1]

    if prediction == 1:
        result = "Heart disease detected by the model"
    else:
        result = "No heart disease detected by the model"

    return {
        "prediction": int(prediction),
        "probability": float(disease_probability),
        "result": result
    }


if __name__ == "__main__":

    patient = {
        "age": 55,
        "sex": 1,
        "cp": 4,
        "trestbps": 150,
        "chol": 280,
        "fbs": 0,
        "restecg": 2,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 2.0,
        "slope": 2,
        "ca": 1,
        "thal": 7
    }

    result = predict_heart_disease(patient)

    print("\n========== HEALTHCAREAI PREDICTION ==========")
    print("Prediction:", result["prediction"])
    print("Probability:", f'{result["probability"]:.2%}')
    print("Result:", result["result"])
