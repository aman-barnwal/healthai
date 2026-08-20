import re


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {

    "diabetes_binary": {
        "display_name": "Diabetes Risk",
        "keywords": [
            "diabetes",
            "diabetic",
            "blood sugar",
            "sugar level",
            "glucose",
            "diabetes risk"
        ]
    },

    "pima_diabetes": {
        "display_name": "Diabetes Risk",
        "keywords": [
            "pima diabetes",
            "pregnancy diabetes"
        ]
    },

    "heart_disease": {
        "display_name": "Heart Disease Risk",
        "keywords": [
            "heart disease",
            "heart risk",
            "cardiac risk",
            "coronary disease",
            "heart attack risk"
        ]
    },

    "heart_failure_clinical_records": {
        "display_name": "Heart Failure Risk",
        "keywords": [
            "heart failure",
            "cardiac failure"
        ]
    },

    "stroke_prediction": {
        "display_name": "Stroke Risk",
        "keywords": [
            "stroke",
            "stroke risk"
        ]
    },

    "chronic_kidney_alternate": {
        "display_name": "Kidney Disease Risk",
        "keywords": [
            "kidney disease",
            "kidney risk",
            "chronic kidney",
            "renal disease"
        ]
    },

    "kidney_disease_dataset": {
        "display_name": "Kidney Disease Risk",
        "keywords": [
            "kidney problem",
            "renal problem"
        ]
    },

    "breast_cancer": {
        "display_name": "Breast Cancer Assessment",
        "keywords": [
            "breast cancer",
            "breast tumor",
            "breast lump"
        ]
    },

    "breast_cancer_wisconsin_diagnostic": {
        "display_name": "Breast Cancer Assessment",
        "keywords": [
            "wisconsin breast cancer"
        ]
    },

    "thyroid": {
        "display_name": "Thyroid Assessment",
        "keywords": [
            "thyroid",
            "hypothyroid",
            "hyperthyroid"
        ]
    },

    "indian_liver": {
        "display_name": "Liver Disease Assessment",
        "keywords": [
            "liver disease",
            "liver problem",
            "liver risk"
        ]
    },

    "hepatitis": {
        "display_name": "Hepatitis Assessment",
        "keywords": [
            "hepatitis",
            "hep b",
            "hep c"
        ]
    },

    "maternal_health_risk": {
        "display_name": "Maternal Health Risk",
        "keywords": [
            "pregnancy risk",
            "maternal health",
            "pregnant",
            "pregnancy health"
        ]
    },

    "obesity_levels": {
        "display_name": "Obesity Assessment",
        "keywords": [
            "obesity",
            "overweight",
            "weight risk",
            "obese"
        ]
    },

    "parkinsons_classification": {
        "display_name": "Parkinson's Assessment",
        "keywords": [
            "parkinson",
            "parkinson's",
            "tremor"
        ]
    },

    "mental_health_survey": {
        "display_name": "Mental Health Assessment",
        "keywords": [
            "mental health",
            "depression",
            "anxiety",
            "stress"
        ]
    },

    "Dry_Eye_Dataset": {
        "display_name": "Dry Eye Assessment",
        "keywords": [
            "dry eye",
            "dry eyes",
            "eye dryness"
        ]
    },

    "symptom_disease_basic": {
        "display_name": "Symptom-Based Disease Assessment",
        "keywords": [
            "my symptoms",
            "what disease",
            "what could i have",
            "what illness",
            "diagnose my symptoms"
        ]
    },

    "disease_symptoms": {
        "display_name": "Disease Symptom Analysis",
        "keywords": [
            "disease symptoms",
            "symptom analysis"
        ]
    }
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# ROUTE USER MESSAGE TO A MODEL
# ============================================================

def detect_model(user_message):

    text = normalize_text(
        user_message
    )

    if not text:
        return None

    best_model = None
    best_score = 0

    for model_name, config in MODEL_REGISTRY.items():

        keywords = config.get(
            "keywords",
            []
        )

        score = 0

        for keyword in keywords:

            keyword = keyword.lower()

            if keyword in text:

                # Longer matches are generally
                # more specific.
                score += len(
                    keyword.split()
                )

        if score > best_score:

            best_score = score
            best_model = model_name

    return best_model


# ============================================================
# GET MODEL DISPLAY NAME
# ============================================================

def get_model_display_name(model_name):

    config = MODEL_REGISTRY.get(
        model_name
    )

    if not config:
        return model_name.replace(
            "_",
            " "
        ).title()

    return config.get(
        "display_name",
        model_name
    )


# ============================================================
# GET MODEL INFORMATION
# ============================================================

def get_model_config(model_name):

    return MODEL_REGISTRY.get(
        model_name
    )


# ============================================================
# CHECK IF MESSAGE IS A HEALTH ASSESSMENT REQUEST
# ============================================================

def is_assessment_request(user_message):

    text = normalize_text(
        user_message
    )

    assessment_words = [

        "check",

        "risk",

        "assess",

        "assessment",

        "predict",

        "prediction",

        "do i have",

        "could i have",

        "am i at risk",

        "health check"

    ]

    if detect_model(text):

        return True

    return any(
        word in text
        for word in assessment_words
    )


# ============================================================
# GET SUPPORTED MODELS
# ============================================================

def get_supported_models():

    models = []

    for model_name, config in MODEL_REGISTRY.items():

        models.append({

            "model": model_name,

            "name": config.get(
                "display_name",
                model_name
            )

        })

    return models


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_messages = [

        "I want to check my diabetes risk",

        "Can you assess my heart disease risk?",

        "Do I have a stroke risk?",

        "I am worried about kidney disease",

        "Check my thyroid",

        "I have dry eyes",

        "I want to know about obesity risk",

        "I have tremors, could it be Parkinson's?"

    ]

    print(
        "\nHealthcareAI Model Router Test\n"
    )

    for message in test_messages:

        model = detect_model(
            message
        )

        display_name = None

        if model:

            display_name = get_model_display_name(
                model
            )

        print(
            f"User: {message}"
        )

        print(
            f"Model: {model}"
        )

        print(
            f"Assessment: {display_name}\n"
        )
