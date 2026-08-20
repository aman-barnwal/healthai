import re

from backend.ml.assessment_engine import (
    get_expected_features,
    get_model_info,
)


# ============================================================
# FEATURE QUESTION DEFINITIONS
# ============================================================

FEATURE_QUESTIONS = {

    # --------------------------------------------------------
    # COMMON NUMERIC FEATURES
    # --------------------------------------------------------

    "age": {
        "question": "How old are you?",
        "type": "number",
        "min": 1,
        "max": 120,
    },

    "Age": {
        "question": "How old are you?",
        "type": "number",
        "min": 1,
        "max": 120,
    },

    "age_category": {
        "question": "How old are you?",
        "type": "number",
        "min": 1,
        "max": 120,
    },

    "BMI": {
        "question": (
            "What is your height and weight? "
            "For example: 177 cm, 85 kg."
        ),
        "type": "height_weight",
    },

    "height": {
        "question": "What is your height?",
        "type": "number",
        "min": 50,
        "max": 250,
    },

    "Height": {
        "question": "What is your height in centimeters?",
        "type": "number",
        "min": 50,
        "max": 250,
    },

    "weight": {
        "question": "What is your weight in kilograms?",
        "type": "number",
        "min": 20,
        "max": 400,
    },

    "Weight": {
        "question": "What is your weight in kilograms?",
        "type": "number",
        "min": 20,
        "max": 400,
    },

    "gender": {
        "question": "What is your sex: male or female?",
        "type": "sex",
    },

    "Gender": {
        "question": "What is your sex: male or female?",
        "type": "sex",
    },

    "Sex": {
        "question": "What is your sex: male or female?",
        "type": "sex",
    },

    # --------------------------------------------------------
    # DIABETES FEATURES
    # --------------------------------------------------------

    "HighBP": {
        "question": "Do you have high blood pressure?",
        "type": "yes_no",
    },

    "HighChol": {
        "question": "Do you have high cholesterol?",
        "type": "yes_no",
    },

    "CholCheck": {
        "question": (
            "Have you had your cholesterol checked "
            "in the last 5 years?"
        ),
        "type": "yes_no",
    },

    "Smoker": {
        "question": (
            "Have you smoked at least 100 cigarettes "
            "during your lifetime?"
        ),
        "type": "yes_no",
    },

    "Stroke": {
        "question": "Have you ever had a stroke?",
        "type": "yes_no",
    },

    "HeartDiseaseorAttack": {
        "question": (
            "Have you ever been diagnosed with heart disease "
            "or had a heart attack?"
        ),
        "type": "yes_no",
    },

    "PhysActivity": {
        "question": (
            "Have you done physical activity or exercise "
            "during the last 30 days?"
        ),
        "type": "yes_no",
    },

    "Fruits": {
        "question": "Do you usually eat fruit regularly?",
        "type": "yes_no",
    },

    "Veggies": {
        "question": "Do you usually eat vegetables regularly?",
        "type": "yes_no",
    },

    "HvyAlcoholConsump": {
        "question": "Do you consume alcohol heavily?",
        "type": "yes_no",
    },

    "AnyHealthcare": {
        "question": (
            "Do you currently have healthcare coverage "
            "or health insurance?"
        ),
        "type": "yes_no",
    },

    "NoDocbcCost": {
        "question": (
            "During the last year, was there a time you needed "
            "to see a doctor but could not because of cost?"
        ),
        "type": "yes_no",
    },

    "GenHlth": {
        "question": (
            "How would you rate your general health: "
            "excellent, very good, good, fair, or poor?"
        ),
        "type": "health_rating",
    },

    "MentHlth": {
        "question": (
            "During the last 30 days, for how many days "
            "was your mental health not good?"
        ),
        "type": "days",
    },

    "PhysHlth": {
        "question": (
            "During the last 30 days, for how many days "
            "was your physical health not good?"
        ),
        "type": "days",
    },

    "DiffWalk": {
        "question": (
            "Do you have serious difficulty walking "
            "or climbing stairs?"
        ),
        "type": "yes_no",
    },

    "Education": {
        "question": (
            "What is your highest education level? "
            "For example: school, high school, college, "
            "or postgraduate."
        ),
        "type": "education",
    },

    "Income": {
        "question": (
            "Which broad income category best describes you: "
            "low, lower-middle, middle, upper-middle, or high?"
        ),
        "type": "income",
    },

    # --------------------------------------------------------
    # HEART DISEASE DATASET FEATURES
    # --------------------------------------------------------

    "cp": {
        "question": (
            "What type of chest pain do you have? "
            "If you know the medical category, provide "
            "a number from 0 to 3."
        ),
        "type": "number",
        "min": 0,
        "max": 3,
    },

    "trestbps": {
        "question": (
            "What is your resting blood pressure "
            "in mm Hg?"
        ),
        "type": "number",
        "min": 50,
        "max": 250,
    },

    "chol": {
        "question": (
            "What is your serum cholesterol level "
            "in mg/dL?"
        ),
        "type": "number",
        "min": 50,
        "max": 700,
    },

    "fbs": {
        "question": (
            "Is your fasting blood sugar greater than "
            "120 mg/dL? Answer yes or no."
        ),
        "type": "yes_no",
    },

    "restecg": {
        "question": (
            "If you know your resting ECG category, "
            "enter 0, 1, or 2."
        ),
        "type": "number",
        "min": 0,
        "max": 2,
    },

    "thalach": {
        "question": (
            "What is your maximum heart rate achieved?"
        ),
        "type": "number",
        "min": 40,
        "max": 250,
    },

    "exang": {
        "question": (
            "Do you experience exercise-induced angina? "
            "Answer yes or no."
        ),
        "type": "yes_no",
    },

    "oldpeak": {
        "question": (
            "If known, enter your ST depression value "
            "from an exercise ECG."
        ),
        "type": "number",
        "min": 0,
        "max": 10,
    },

    "slope": {
        "question": (
            "If known, enter your ST slope category: "
            "0, 1, or 2."
        ),
        "type": "number",
        "min": 0,
        "max": 2,
    },

    "ca": {
        "question": (
            "If known from medical testing, enter the number "
            "of major vessels: 0 to 4."
        ),
        "type": "number",
        "min": 0,
        "max": 4,
    },

    "thal": {
        "question": (
            "If known from your medical report, enter "
            "your thalassemia category."
        ),
        "type": "number",
        "min": 0,
        "max": 7,
    },
}


# ============================================================
# GENERATE A FALLBACK QUESTION
# ============================================================

def humanize_feature_name(feature_name):

    text = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        str(feature_name),
    )

    text = text.replace(
        "_",
        " "
    )

    return text.strip().lower()


def create_fallback_question(feature_name):

    readable_name = humanize_feature_name(
        feature_name
    )

    return {
        "question": (
            f"Please provide your value for "
            f"'{readable_name}'."
        ),
        "type": "number",
        "feature": feature_name,
    }


# ============================================================
# GET QUESTION FOR A FEATURE
# ============================================================

def get_question_for_feature(feature_name):

    question = FEATURE_QUESTIONS.get(
        feature_name
    )

    if question:

        result = question.copy()

        result["feature"] = feature_name

        return result

    return create_fallback_question(
        feature_name
    )


# ============================================================
# BUILD ASSESSMENT QUESTIONS
# ============================================================

def build_assessment_questions(model_name):

    features = get_expected_features(
        model_name
    )

    questions = []

    for feature in features:

        question = get_question_for_feature(
            feature
        )

        questions.append(
            question
        )

    return questions


# ============================================================
# GET ASSESSMENT INFORMATION
# ============================================================

def get_assessment_info(model_name):

    model_info = get_model_info(
        model_name
    )

    questions = build_assessment_questions(
        model_name
    )

    return {
        "model": model_name,
        "features": model_info.get(
            "features",
            []
        ),
        "metadata": model_info.get(
            "metadata",
            {}
        ),
        "questions": questions,
        "total_questions": len(questions),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_models = [

        "diabetes_binary",

        "heart_disease",

        "stroke_prediction",

    ]

    print(
        "\nHealthcareAI Assessment Manager Test\n"
    )

    for model_name in test_models:

        print(
            f"Model: {model_name}"
        )

        try:

            questions = build_assessment_questions(
                model_name
            )

            print(
                f"Questions: {len(questions)}"
            )

            for question in questions[:5]:

                print(
                    f"  • {question['feature']}"
                )

                print(
                    f"    {question['question']}"
                )

        except Exception as error:

            print(
                f"Error: {error}"
            )

        print()

    print(
        "Done."
    )
