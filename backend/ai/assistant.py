import os
import json
import re
from pathlib import Path

import joblib
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from backend.ml.schemas import MODEL_SCHEMAS


# ============================================================
# PROJECT / ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        f"GROQ_API_KEY not found in {ENV_PATH}"
    )


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are HealthcareAI.

You are a conversational healthcare assistant.

You provide clear, concise health information.

You also work with local machine-learning models that
can provide AI-based risk estimates.

Important rules:

- Never claim an ML prediction is a medical diagnosis.
- Never invent patient information.
- Never invent laboratory results.
- Never invent ML predictions.
- Local ML models perform predictions.
- Never modify or fabricate a local model result.
- Explain model results in simple language.
- Do not overwhelm users with technical details.
- Ask questions naturally, one at a time.
- Remember information already provided by the user.
- Do not ask the user for the same information twice.
- If the user says "yes it was normal", interpret that
  naturally instead of forcing them to answer again.
- If the user makes a small typo such as "make" instead
  of "male", infer the intended answer when obvious.
- Keep the conversation natural and friendly.
- Avoid asking for numeric dataset codes unless absolutely
  necessary.
- Convert natural answers into model-compatible values internally.
- A model estimate is not medical certainty.
- For severe or emergency symptoms recommend urgent medical care.
"""


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIRECTORY = PROJECT_ROOT / "backend" / "models"


def get_model_path(model_name):

    possible_paths = [

        MODEL_DIRECTORY / f"{model_name}.pkl",

        MODEL_DIRECTORY / f"{model_name.lower()}.pkl",

        MODEL_DIRECTORY / f"{model_name}_pipeline.pkl",

    ]

    for path in possible_paths:

        if path.exists():

            return path

    return None


# ============================================================
# MODEL CACHE
# ============================================================

MODEL_CACHE = {}


def load_local_model(model_name):

    if model_name in MODEL_CACHE:

        return MODEL_CACHE[model_name]

    model_path = get_model_path(model_name)

    if not model_path:

        raise FileNotFoundError(
            f"Model file not found for '{model_name}'"
        )

    model = joblib.load(model_path)

    MODEL_CACHE[model_name] = model

    return model


# ============================================================
# PATIENT SESSION
# ============================================================

patient_session = {

    "active": False,

    "model": None,

    "data": {},

    "question_index": 0,

    "questions": []

}


# ============================================================
# JSON HELPER
# ============================================================

def extract_json(text):

    if not text:

        return None

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:

        return json.loads(text)

    except Exception:

        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:

        try:

            return json.loads(
                match.group()
            )

        except Exception:

            return None

    return None


# ============================================================
# MODEL DETECTION
# ============================================================

def detect_model(user_message):

    text = user_message.lower()

    matches = []

    for model_name, schema in MODEL_SCHEMAS.items():

        for keyword in schema.get(
            "keywords",
            []
        ):

            keyword = keyword.lower()

            if keyword in text:

                matches.append(
                    (
                        len(keyword),
                        model_name
                    )
                )

    if not matches:

        return None

    matches.sort(
        reverse=True
    )

    return matches[0][1]


# ============================================================
# PREDICTION INTENT
# ============================================================

def is_prediction_request(user_message):

    text = user_message.lower()

    patterns = [

        r"\bcheck my\b",

        r"\bcheck.*risk\b",

        r"\bmy risk\b",

        r"\bpredict\b",

        r"\bprediction\b",

        r"\bassess\b",

        r"\brisk estimate\b",

        r"\bdo i have\b",

        r"\bam i at risk\b",

        r"\bi want to check\b",

        r"\bcan you check\b",

        r"\bcalculate.*risk\b"

    ]

    return any(
        re.search(
            pattern,
            text
        )
        for pattern in patterns
    )


# ============================================================
# CANCEL ASSESSMENT
# ============================================================

def is_cancel_request(message):

    text = message.lower().strip()

    cancel_words = [

        "cancel",

        "stop",

        "reset",

        "start over",

        "never mind",

        "nevermind",

        "quit"

    ]

    return any(
        word in text
        for word in cancel_words
    )


# ============================================================
# RESET SESSION
# ============================================================

def reset_session():

    patient_session["active"] = False

    patient_session["model"] = None

    patient_session["data"] = {}

    patient_session["question_index"] = 0

    patient_session["questions"] = []


# ============================================================
# DIABETES QUESTIONS
# ============================================================

DIABETES_QUESTIONS = [

    {
        "field": "Age",
        "question": (
            "Sure. I can give you an AI-based diabetes "
            "risk estimate, not a medical diagnosis. "
            "Let's start with a few basics — how old are you?"
        ),
        "type": "age"
    },

    {
        "field": "height_weight",
        "question": (
            "Got it. What's your height and weight? "
            "For example: 177 cm, 85 kg."
        ),
        "type": "height_weight"
    },

    {
        "field": "HighBP",
        "question": (
            "Do you have high blood pressure?"
        ),
        "type": "yes_no"
    },

    {
        "field": "HighChol",
        "question": (
            "Do you have high cholesterol?"
        ),
        "type": "yes_no"
    },

    {
        "field": "CholCheck",
        "question": (
            "Have you had your cholesterol checked "
            "in the last 5 years?"
        ),
        "type": "yes_no"
    },

    {
        "field": "Smoker",
        "question": (
            "Have you smoked at least 100 cigarettes "
            "in your lifetime?"
        ),
        "type": "yes_no"
    },

    {
        "field": "Stroke",
        "question": (
            "Have you ever had a stroke?"
        ),
        "type": "yes_no"
    },

    {
        "field": "HeartDiseaseorAttack",
        "question": (
            "Have you ever been diagnosed with coronary "
            "heart disease or had a heart attack?"
        ),
        "type": "yes_no"
    },

    {
        "field": "PhysActivity",
        "question": (
            "Have you done any physical activity or exercise "
            "during the last 30 days?"
        ),
        "type": "yes_no"
    },

    {
        "field": "Fruits",
        "question": (
            "Do you usually eat fruit regularly?"
        ),
        "type": "yes_no"
    },

    {
        "field": "Veggies",
        "question": (
            "And do you usually eat vegetables regularly?"
        ),
        "type": "yes_no"
    },

    {
        "field": "HvyAlcoholConsump",
        "question": (
            "Do you consume alcohol heavily?"
        ),
        "type": "yes_no"
    },

    {
        "field": "AnyHealthcare",
        "question": (
            "Do you currently have healthcare coverage "
            "or health insurance?"
        ),
        "type": "yes_no"
    },

    {
        "field": "NoDocbcCost",
        "question": (
            "During the last year, was there a time you "
            "needed to see a doctor but couldn't because "
            "of the cost?"
        ),
        "type": "yes_no"
    },

    {
        "field": "GenHlth",
        "question": (
            "How would you rate your general health: "
            "excellent, very good, good, fair, or poor?"
        ),
        "type": "health_rating"
    },

    {
        "field": "MentHlth",
        "question": (
            "In the last 30 days, for about how many days "
            "was your mental health not good?"
        ),
        "type": "days"
    },

    {
        "field": "PhysHlth",
        "question": (
            "And for about how many days in the last 30 "
            "was your physical health not good?"
        ),
        "type": "days"
    },

    {
        "field": "DiffWalk",
        "question": (
            "Do you have serious difficulty walking "
            "or climbing stairs?"
        ),
        "type": "yes_no"
    },

    {
        "field": "Sex",
        "question": (
            "What is your sex: male or female?"
        ),
        "type": "sex"
    },

    {
        "field": "Education",
        "question": (
            "What is your highest level of education? "
            "You can simply say school, high school, "
            "college, or postgraduate."
        ),
        "type": "education"
    },

    {
        "field": "Income",
        "question": (
            "Finally, which broad income category best "
            "describes you: low, lower-middle, middle, "
            "upper-middle, or high?"
        ),
        "type": "income"
    }

]


# ============================================================
# YES / NO PARSER
# ============================================================

def parse_yes_no(text):

    text = text.lower().strip()

    yes_words = [

        "yes",

        "yeah",

        "yep",

        "yup",

        "correct",

        "sure",

        "i do",

        "i have",

        "normal",

        "okay",

        "ok",

        "fine"

    ]

    no_words = [

        "no",

        "nope",

        "nah",

        "never",

        "i don't",

        "i dont",

        "not"

    ]

    if any(
        word in text
        for word in yes_words
    ):

        return 1

    if any(
        word in text
        for word in no_words
    ):

        return 0

    return None


# ============================================================
# PARSE AGE
# ============================================================

def parse_age(text):

    match = re.search(
        r"\b(\d{1,3})\b",
        text
    )

    if not match:

        return None

    age = int(
        match.group(1)
    )

    if age < 1 or age > 120:

        return None

    return age


# ============================================================
# CONVERT REAL AGE TO DATASET AGE CATEGORY
# ============================================================

def convert_age_to_category(age):

    categories = [

        (18, 24, 1),

        (25, 29, 2),

        (30, 34, 3),

        (35, 39, 4),

        (40, 44, 5),

        (45, 49, 6),

        (50, 54, 7),

        (55, 59, 8),

        (60, 64, 9),

        (65, 69, 10),

        (70, 74, 11),

        (75, 79, 12),

        (80, 120, 13)

    ]

    for minimum, maximum, category in categories:

        if minimum <= age <= maximum:

            return category

    return 1


# ============================================================
# PARSE HEIGHT / WEIGHT AND CALCULATE BMI
# ============================================================

def parse_height_weight(text):

    text = text.lower()

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        text
    )

    if len(numbers) < 2:

        return None

    height = float(
        numbers[0]
    )

    weight = float(
        numbers[1]
    )

    # Convert meters to centimeters if needed
    if height < 3:

        height = height * 100

    if height < 50 or height > 250:

        return None

    if weight < 20 or weight > 400:

        return None

    height_m = height / 100

    bmi = weight / (
        height_m ** 2
    )

    return {

        "height": round(height, 2),

        "weight": round(weight, 2),

        "BMI": round(bmi, 2)

    }


# ============================================================
# GENERAL HEALTH PARSER
# ============================================================

def parse_health_rating(text):

    text = text.lower()

    mapping = {

        "excellent": 1,

        "very good": 2,

        "good": 3,

        "fair": 4,

        "poor": 5

    }

    for word, value in mapping.items():

        if word in text:

            return value

    return None


# ============================================================
# DAYS PARSER
# ============================================================

def parse_days(text):

    match = re.search(
        r"\b(\d{1,2})\b",
        text
    )

    if not match:

        return None

    days = int(
        match.group(1)
    )

    if days < 0 or days > 30:

        return None

    return days


# ============================================================
# SEX PARSER
# ============================================================

def parse_sex(text):

    text = text.lower().strip()

    if (
        "male" in text
        or text == "m"
        or text == "make"
    ):

        return 1

    if (
        "female" in text
        or text == "f"
    ):

        return 0

    return None


# ============================================================
# EDUCATION PARSER
# ============================================================

def parse_education(text):

    text = text.lower()

    if (
        "college" in text
        or "university" in text
        or "bachelor" in text
    ):

        return 5

    if (
        "postgraduate" in text
        or "master" in text
        or "phd" in text
    ):

        return 6

    if (
        "high school" in text
        or "12" in text
    ):

        return 4

    if "school" in text:

        return 3

    return None


# ============================================================
# INCOME PARSER
# ============================================================

def parse_income(text):

    text = text.lower()

    mapping = {

        "low": 2,

        "lower-middle": 3,

        "lower middle": 3,

        "middle": 5,

        "upper-middle": 6,

        "upper middle": 6,

        "high": 7,

        "very high": 8

    }

    for key, value in mapping.items():

        if key in text:

            return value

    return None


# ============================================================
# PARSE CURRENT ANSWER
# ============================================================

def parse_answer(question, user_message):

    question_type = question["type"]

    if question_type == "age":

        return parse_age(
            user_message
        )

    if question_type == "height_weight":

        return parse_height_weight(
            user_message
        )

    if question_type == "yes_no":

        return parse_yes_no(
            user_message
        )

    if question_type == "health_rating":

        return parse_health_rating(
            user_message
        )

    if question_type == "days":

        return parse_days(
            user_message
        )

    if question_type == "sex":

        return parse_sex(
            user_message
        )

    if question_type == "education":

        return parse_education(
            user_message
        )

    if question_type == "income":

        return parse_income(
            user_message
        )

    return None


# ============================================================
# START ASSESSMENT
# ============================================================

def start_assessment(model_name):

    reset_session()

    patient_session["active"] = True

    patient_session["model"] = model_name

    if model_name == "diabetes_binary":

        patient_session[
            "questions"
        ] = DIABETES_QUESTIONS.copy()

    else:

        return (
            f"The local model '{model_name}' is available, "
            "but its conversational assessment flow has not "
            "been configured yet."
        )

    patient_session[
        "question_index"
    ] = 0

    return patient_session[
        "questions"
    ][0]["question"]


# ============================================================
# RUN LOCAL PREDICTION
# ============================================================

def run_prediction(model_name, patient_data):

    model = load_local_model(
        model_name
    )

    # --------------------------------------------------------
    # REMOVE NON-MODEL VALUES
    # --------------------------------------------------------

    clean_data = {}

    for key, value in patient_data.items():

        if key in [

            "height",

            "weight"

        ]:

            continue

        clean_data[key] = value

    # --------------------------------------------------------
    # IMPORTANT:
    # READ THE EXACT FEATURE ORDER USED DURING TRAINING
    #
    # This prevents:
    #
    # "Feature names should match those passed during fit"
    # --------------------------------------------------------

    expected_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if expected_features is not None:

        expected_features = list(
            expected_features
        )

        missing_features = [

            feature

            for feature in expected_features

            if feature not in clean_data

        ]

        if missing_features:

            raise ValueError(
                "Missing required model features: "
                + ", ".join(
                    missing_features
                )
            )

        ordered_data = {

            feature: clean_data[feature]

            for feature in expected_features

        }

        input_df = pd.DataFrame(
            [ordered_data],
            columns=expected_features
        )

    else:

        input_df = pd.DataFrame(
            [clean_data]
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_df
    )[0]

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            input_df
        )[0]

        if len(probabilities) > 1:

            probability = float(
                probabilities[1]
            )

    return {

        "prediction": int(
            prediction
        ),

        "probability": probability,

        "features": list(
            input_df.columns
        )

    }


# ============================================================
# COMPLETE ASSESSMENT
# ============================================================

def complete_assessment():

    model_name = patient_session[
        "model"
    ]

    data = patient_session[
        "data"
    ].copy()

    try:

        print(
            "\n[HealthcareAI] Running local ML model..."
        )

        result = run_prediction(
            model_name,
            data
        )

        reset_session()

        prediction = result[
            "prediction"
        ]

        probability = result[
            "probability"
        ]

        if probability is not None:

            probability_percent = round(
                probability * 100,
                1
            )

        else:

            probability_percent = None

        # ----------------------------------------------------
        # HUMAN RESPONSE
        # ----------------------------------------------------

        if prediction == 1:

            response = (
                "Based on the information you provided, "
                "the model classified your profile as having "
                "a higher estimated diabetes risk."
            )

        else:

            response = (
                "Based on the information you provided, "
                "the model classified your profile as having "
                "a lower estimated diabetes risk."
            )

        if probability_percent is not None:

            response += (
                f"\n\nThe model's estimated probability "
                f"for the positive class was "
                f"approximately {probability_percent}%."
            )

        response += (

            "\n\nThis is an AI-based model estimate, "
            "not a medical diagnosis. A proper medical "
            "evaluation may include blood glucose or HbA1c "
            "testing."

        )

        return response

    except Exception as error:

        print(
            "[HealthcareAI] Prediction error:",
            error
        )

        reset_session()

        return (
            "I couldn't complete the local model assessment "
            "because of a backend input issue. "
            "The assessment has been reset."
        )


# ============================================================
# HANDLE ACTIVE ASSESSMENT
# ============================================================

def handle_assessment(user_message):

    if is_cancel_request(
        user_message
    ):

        reset_session()

        return (
            "No problem — I've cancelled the assessment."
        )

    index = patient_session[
        "question_index"
    ]

    questions = patient_session[
        "questions"
    ]

    if index >= len(
        questions
    ):

        return complete_assessment()

    current_question = questions[
        index
    ]

    value = parse_answer(
        current_question,
        user_message
    )

    if value is None:

        return (
            "I didn't quite catch that. "
            + current_question["question"]
        )

    field = current_question[
        "field"
    ]

    # --------------------------------------------------------
    # HEIGHT + WEIGHT
    # --------------------------------------------------------

    if field == "height_weight":

        patient_session[
            "data"
        ].update(
            value
        )

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    elif field == "Age":

        patient_session[
            "data"
        ]["Age"] = convert_age_to_category(
            value
        )

        patient_session[
            "data"
        ]["actual_age"] = value

    # --------------------------------------------------------
    # NORMAL FIELD
    # --------------------------------------------------------

    else:

        patient_session[
            "data"
        ][field] = value

    # --------------------------------------------------------
    # MOVE TO NEXT QUESTION
    # --------------------------------------------------------

    patient_session[
        "question_index"
    ] += 1

    next_index = patient_session[
        "question_index"
    ]

    if next_index >= len(
        questions
    ):

        return complete_assessment()

    # --------------------------------------------------------
    # BMI RESPONSE
    # --------------------------------------------------------

    if field == "height_weight":

        bmi = patient_session[
            "data"
        ]["BMI"]

        next_question = questions[
            next_index
        ]["question"]

        return (
            f"Got it. Your BMI is approximately {bmi}. "
            f"{next_question}"
        )

    return questions[
        next_index
    ]["question"]


# ============================================================
# GROQ RESPONSE
# ============================================================

def ask_groq(user_message):

    completion = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": user_message
            }

        ],

        temperature=0.4,

        max_tokens=800

    )

    return completion.choices[
        0
    ].message.content


# ============================================================
# MAIN CHAT ROUTER
# ============================================================

def healthcare_ai(user_message):

    user_message = user_message.strip()

    if not user_message:

        return (
            "Ask me anything about health, "
            "or tell me if you'd like to check a health risk."
        )

    # --------------------------------------------------------
    # ACTIVE ASSESSMENT
    # --------------------------------------------------------

    if patient_session[
        "active"
    ]:

        return handle_assessment(
            user_message
        )

    # --------------------------------------------------------
    # NEW ML REQUEST
    # --------------------------------------------------------

    if is_prediction_request(
        user_message
    ):

        model_name = detect_model(
            user_message
        )

        if model_name:

            print(
                f"\n[HealthcareAI] Starting "
                f"{model_name} assessment..."
            )

            return start_assessment(
                model_name
            )

    # --------------------------------------------------------
    # GENERAL GROQ RESPONSE
    # --------------------------------------------------------

    return ask_groq(
        user_message
    )


# ============================================================
# CLI
# ============================================================

def main():

    print(
        "HealthcareAI is connected to Groq."
    )

    print(
        f"Model: {GROQ_MODEL}"
    )

    print(
        f"Using .env: {ENV_PATH}"
    )

    print(
        "Local ML model registry ready."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        try:

            user_message = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nHealthcareAI stopped."
            )

            break

        if user_message.lower() in [

            "exit",

            "quit"

        ]:

            print(
                "HealthcareAI stopped."
            )

            break

        response = healthcare_ai(
            user_message
        )

        print(
            "\nHealthcareAI:"
        )

        print(
            response
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
