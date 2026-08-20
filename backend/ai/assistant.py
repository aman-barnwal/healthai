import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.ml.schemas import MODEL_SCHEMAS
from backend.ml.predict_all import predict, list_models


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

MODEL = GROQ_MODEL


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are HealthcareAI, a helpful AI health assistant.

You provide clear, concise health information and support
local machine-learning health assessments.

Important rules:

- Never invent patient information.
- Never invent medical history.
- Never invent laboratory results.
- Never invent machine-learning predictions.
- Local ML models perform all predictions.
- Never alter or replace a local model result.
- A machine-learning prediction is not a medical diagnosis.
- Clearly describe ML results as estimates.
- Do not claim probability or model confidence equals medical certainty.
- Keep responses natural and conversational.
- Avoid unnecessary medical jargon.
- Ask one or two related questions at a time during assessments.
- Do not dump a long questionnaire on the user.
- Remember information already provided during the current assessment.
- For potentially serious symptoms, recommend medical evaluation.
- For possible emergency symptoms, recommend urgent medical care.
"""


# ============================================================
# PATIENT SESSION
# ============================================================

patient_session = {
    "model": None,
    "data": {},
    "profile": {},
    "current_field": None,
    "started": False,
    "original_request": ""
}


# ============================================================
# JSON HELPERS
# ============================================================

def extract_json(text):
    """
    Extract JSON safely from an LLM response.
    """

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

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
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

        except json.JSONDecodeError:
            return None

    return None


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    return str(value).strip().lower()


def extract_first_number(text):

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        str(text)
    )

    if not match:
        return None

    try:
        return float(match.group())

    except ValueError:
        return None


def parse_yes_no(value):

    text = normalize_text(value)

    yes_values = [
        "yes",
        "y",
        "yeah",
        "yep",
        "haan",
        "ha",
        "han",
        "true",
        "1"
    ]

    no_values = [
        "no",
        "n",
        "nope",
        "nah",
        "nahi",
        "nahin",
        "false",
        "0"
    ]

    if text in yes_values:
        return 1

    if text in no_values:
        return 0

    return None


def parse_sex(value):

    text = normalize_text(value)

    if text in [
        "male",
        "m",
        "man",
        "boy"
    ]:
        return 1

    if text in [
        "female",
        "f",
        "woman",
        "girl"
    ]:
        return 0

    return None


# ============================================================
# BMI
# ============================================================

def calculate_bmi(height_cm, weight_kg):

    try:

        height_m = float(height_cm) / 100

        weight = float(weight_kg)

        if height_m <= 0 or weight <= 0:
            return None

        bmi = weight / (height_m ** 2)

        return round(bmi, 2)

    except Exception:
        return None


def extract_height_weight(text):

    text = normalize_text(text)

    height = None
    weight = None

    # Example:
    # 177 cm, 85 kg
    # 177cm 85kg

    height_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)",
        text
    )

    weight_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilogram|kilograms)",
        text
    )

    if height_match:
        height = float(
            height_match.group(1)
        )

    if weight_match:
        weight = float(
            weight_match.group(1)
        )

    # Support:
    # 5'10"
    feet_match = re.search(
        r"(\d+)\s*(?:ft|feet|foot|')\s*(\d+)?",
        text
    )

    if height is None and feet_match:

        feet = float(
            feet_match.group(1)
        )

        inches = float(
            feet_match.group(2) or 0
        )

        height = round(
            (feet * 30.48) +
            (inches * 2.54),
            2
        )

    # Support:
    # 180 lbs
    pound_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?)",
        text
    )

    if weight is None and pound_match:

        pounds = float(
            pound_match.group(1)
        )

        weight = round(
            pounds * 0.453592,
            2
        )

    return height, weight


# ============================================================
# AGE CONVERSION
# ============================================================

def age_to_brfss_category(age):
    """
    Converts real age into the 1-13 BRFSS age category
    used by the diabetes_binary dataset.

    1  = 18-24
    2  = 25-29
    3  = 30-34
    ...
    12 = 75-79
    13 = 80+
    """

    try:

        age = float(age)

    except Exception:
        return None

    if age < 18:
        return None

    if age <= 24:
        return 1

    if age <= 29:
        return 2

    if age <= 34:
        return 3

    if age <= 39:
        return 4

    if age <= 44:
        return 5

    if age <= 49:
        return 6

    if age <= 54:
        return 7

    if age <= 59:
        return 8

    if age <= 64:
        return 9

    if age <= 69:
        return 10

    if age <= 74:
        return 11

    if age <= 79:
        return 12

    return 13


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_available_models():

    try:

        models = list_models()

        if isinstance(models, dict):
            return list(models.keys())

        if isinstance(models, list):
            return models

        return []

    except Exception as error:

        print(
            "[HealthcareAI] Model registry error:",
            error
        )

        return []


# ============================================================
# NORMALIZE MODEL NAME
# ============================================================

def normalize_model_name(model_name):

    if not model_name:
        return None

    model_name = str(
        model_name
    ).strip()

    if model_name in MODEL_SCHEMAS:
        return model_name

    for schema_name in MODEL_SCHEMAS:

        if (
            schema_name.lower()
            ==
            model_name.lower()
        ):
            return schema_name

    for schema_name, schema in MODEL_SCHEMAS.items():

        aliases = schema.get(
            "aliases",
            []
        )

        for alias in aliases:

            if (
                alias.lower()
                ==
                model_name.lower()
            ):
                return schema_name

    return None


# ============================================================
# DETECT MODEL
# ============================================================

def detect_model(user_message):

    text = user_message.lower()

    matches = []

    for model_name, schema in MODEL_SCHEMAS.items():

        keywords = schema.get(
            "keywords",
            []
        )

        for keyword in keywords:

            keyword = keyword.lower().strip()

            if not keyword:
                continue

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
        reverse=True,
        key=lambda item: item[0]
    )

    return matches[0][1]


# ============================================================
# DETECT PREDICTION INTENT
# ============================================================

def is_prediction_request(user_message):

    text = user_message.lower().strip()

    patterns = [

        r"\bpredict\b",
        r"\bprediction\b",
        r"\bcan you predict\b",

        r"\bassess\b",
        r"\bassessment\b",
        r"\bassess me\b",
        r"\bassess my\b",

        r"\bclassify\b",
        r"\bclassification\b",

        r"\bcheck my\b",
        r"\bcheck whether i\b",
        r"\bcheck if i\b",

        r"\bmy risk\b",
        r"\brisk of\b",
        r"\brisk for\b",
        r"\bam i at risk\b",
        r"\bwhat is my risk\b",
        r"\bcalculate my risk\b",

        r"\banalyze my\b",
        r"\banalyse my\b",
        r"\bevaluate my\b",

        r"\brun the model\b",
        r"\brun a prediction\b",
        r"\buse the model\b",

        r"\bdo i have\b",
        r"\bcould i have\b",
        r"\bcan you determine if i\b",

        r"\bi want to check\b",
        r"\bi want to know my\b"
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            text
        ):
            return True

    return False


# ============================================================
# CANCEL REQUEST
# ============================================================

def is_cancel_request(user_message):

    text = user_message.lower().strip()

    cancel_phrases = [

        "cancel",
        "stop",
        "stop assessment",
        "cancel assessment",
        "reset",
        "start over",
        "never mind",
        "nevermind",
        "quit assessment"
    ]

    return any(
        phrase in text
        for phrase in cancel_phrases
    )


# ============================================================
# ROUTE REQUEST
# ============================================================

def route_request(user_message):

    if not is_prediction_request(
        user_message
    ):

        return {
            "use_ml": False,
            "model": None
        }

    detected_model = detect_model(
        user_message
    )

    if detected_model:

        return {
            "use_ml": True,
            "model": detected_model
        }

    available_models = get_available_models()

    if not available_models:

        return {
            "use_ml": False,
            "model": None
        }

    prompt = f"""
The user explicitly requested a personal health prediction
or risk assessment.

USER MESSAGE:

{user_message}

AVAILABLE LOCAL MODELS:

{json.dumps(available_models, indent=2)}

Return ONLY valid JSON.

If a model clearly matches:

{{
    "use_ml": true,
    "model": "exact_model_name"
}}

Otherwise:

{{
    "use_ml": false,
    "model": null
}}

Rules:

- Use only an available model.
- Never invent model names.
- Do not guess if no model clearly matches.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        result = extract_json(
            content
        )

        if not isinstance(
            result,
            dict
        ):
            return {
                "use_ml": False,
                "model": None
            }

        if not result.get(
            "use_ml"
        ):
            return {
                "use_ml": False,
                "model": None
            }

        model_name = normalize_model_name(
            result.get("model")
        )

        if model_name:

            return {
                "use_ml": True,
                "model": model_name
            }

    except Exception as error:

        print(
            "[HealthcareAI] Routing error:",
            error
        )

    return {
        "use_ml": False,
        "model": None
    }


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def model_is_configured(model_name):

    schema = MODEL_SCHEMAS.get(
        model_name
    )

    if not schema:
        return False

    fields = schema.get(
        "fields",
        {}
    )

    return bool(fields)


# ============================================================
# SESSION RESET
# ============================================================

def reset_session():

    patient_session["model"] = None
    patient_session["data"] = {}
    patient_session["profile"] = {}
    patient_session["current_field"] = None
    patient_session["started"] = False
    patient_session["original_request"] = ""


# ============================================================
# DIABETES QUESTION FLOW
# ============================================================

DIABETES_QUESTIONS = [

    {
        "field": "age",
        "profile_field": "age",
        "question": (
            "Sure. I can give you an AI-based diabetes risk "
            "estimate, not a medical diagnosis. Let's start "
            "with a few basics — how old are you?"
        )
    },

    {
        "field": "height_weight",
        "question": (
            "Got it. What's your height and weight? "
            "For example: 177 cm, 85 kg."
        )
    },

    {
        "field": "HighBP",
        "question": (
            "Do you have high blood pressure?"
        )
    },

    {
        "field": "HighChol",
        "question": (
            "Do you have high cholesterol?"
        )
    },

    {
        "field": "CholCheck",
        "question": (
            "Have you had your cholesterol checked in the "
            "last 5 years?"
        )
    },

    {
        "field": "Smoker",
        "question": (
            "Have you smoked at least 100 cigarettes in your lifetime?"
        )
    },

    {
        "field": "Stroke",
        "question": (
            "Have you ever had a stroke?"
        )
    },

    {
        "field": "HeartDiseaseorAttack",
        "question": (
            "Have you ever been diagnosed with coronary heart "
            "disease or had a heart attack?"
        )
    },

    {
        "field": "PhysActivity",
        "question": (
            "Have you done any physical activity or exercise "
            "during the last 30 days?"
        )
    },

    {
        "field": "Fruits",
        "question": (
            "Do you usually eat fruit regularly?"
        )
    },

    {
        "field": "Veggies",
        "question": (
            "And do you usually eat vegetables regularly?"
        )
    },

    {
        "field": "HvyAlcoholConsump",
        "question": (
            "Do you consume alcohol heavily?"
        )
    },

    {
        "field": "AnyHealthcare",
        "question": (
            "Do you currently have any healthcare coverage "
            "or health insurance?"
        )
    },

    {
        "field": "NoDocbcCost",
        "question": (
            "During the last year, was there a time you needed "
            "to see a doctor but couldn't because of the cost?"
        )
    },

    {
        "field": "GenHlth",
        "question": (
            "How would you rate your general health: "
            "excellent, very good, good, fair, or poor?"
        )
    },

    {
        "field": "MentHlth",
        "question": (
            "In the last 30 days, for about how many days was "
            "your mental health not good?"
        )
    },

    {
        "field": "PhysHlth",
        "question": (
            "And for about how many days in the last 30 was "
            "your physical health not good?"
        )
    },

    {
        "field": "DiffWalk",
        "question": (
            "Do you have serious difficulty walking or climbing stairs?"
        )
    },

    {
        "field": "Sex",
        "question": (
            "What is your sex: male or female?"
        )
    },

    {
        "field": "Education",
        "question": (
            "What is your highest level of education? "
            "You can simply say something like school, "
            "high school, college, or postgraduate."
        )
    },

    {
        "field": "Income",
        "question": (
            "For the final model input, I need an approximate "
            "income category. You can share a broad category "
            "or range — no exact amount is required."
        )
    }
]


# ============================================================
# GENERAL FIELD LABEL
# ============================================================

def format_field_name(model_name, field):

    schema = MODEL_SCHEMAS.get(
        model_name,
        {}
    )

    fields = schema.get(
        "fields",
        {}
    )

    field_info = fields.get(
        field,
        {}
    )

    return field_info.get(
        "label",
        field.replace(
            "_",
            " "
        ).title()
    )


# ============================================================
# DIABETES AGE / HEALTH PARSERS
# ============================================================

def parse_general_health(value):

    text = normalize_text(value)

    mapping = {
        "excellent": 1,
        "very good": 2,
        "verygood": 2,
        "good": 3,
        "fair": 4,
        "poor": 5
    }

    if text in mapping:
        return mapping[text]

    number = extract_first_number(
        value
    )

    if number is not None and 1 <= number <= 5:
        return int(number)

    return None


def parse_days(value):

    number = extract_first_number(
        value
    )

    if number is None:
        return None

    number = int(number)

    if 0 <= number <= 30:
        return number

    return None


def parse_education(value):

    text = normalize_text(value)

    # Dataset categories:
    # 1 = Never attended school / kindergarten
    # 2 = Elementary
    # 3 = Some high school
    # 4 = High school graduate
    # 5 = Some college / technical school
    # 6 = College graduate

    number = extract_first_number(
        text
    )

    if number is not None and 1 <= number <= 6:
        return int(number)

    if any(
        word in text
        for word in [
            "postgraduate",
            "post graduate",
            "master",
            "bachelor",
            "graduate",
            "college degree",
            "university degree"
        ]
    ):
        return 6

    if any(
        word in text
        for word in [
            "college",
            "university",
            "technical",
            "diploma"
        ]
    ):
        return 5

    if "high school" in text:
        return 4

    if "secondary" in text:
        return 4

    if "school" in text:
        return 3

    return None


def parse_income(value):

    text = normalize_text(value)

    number = extract_first_number(
        text
    )

    if number is not None and 1 <= number <= 8:
        return int(number)

    # Because the original dataset uses US BRFSS income
    # categories, direct conversion from Indian income should
    # not pretend to be exact.

    if any(
        word in text
        for word in [
            "very low",
            "very poor",
            "lowest"
        ]
    ):
        return 1

    if any(
        word in text
        for word in [
            "low",
            "lower"
        ]
    ):
        return 3

    if any(
        word in text
        for word in [
            "middle",
            "average",
            "medium"
        ]
    ):
        return 5

    if any(
        word in text
        for word in [
            "high",
            "upper middle",
            "upper-middle"
        ]
    ):
        return 7

    if any(
        word in text
        for word in [
            "very high",
            "rich",
            "highest"
        ]
    ):
        return 8

    return None


# ============================================================
# GET NEXT DIABETES QUESTION
# ============================================================

def get_next_diabetes_question():

    data = patient_session["data"]

    for item in DIABETES_QUESTIONS:

        field = item["field"]

        if field == "age":

            if (
                "age"
                not in patient_session["profile"]
            ):
                return item

            continue

        if field == "height_weight":

            if (
                "BMI"
                not in data
            ):
                return item

            continue

        if field not in data:
            return item

    return None


# ============================================================
# SAVE DIABETES ANSWER
# ============================================================

def process_diabetes_answer(
    user_message
):

    question = get_next_diabetes_question()

    if not question:
        return True, None

    field = question["field"]

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if field == "age":

        age = extract_first_number(
            user_message
        )

        if age is None or age < 18:

            return False, (
                "Please tell me your age in years. "
                "For example: 20."
            )

        patient_session["profile"]["age"] = age

        age_category = age_to_brfss_category(
            age
        )

        if age_category is None:

            return False, (
                "Please enter a valid age in years."
            )

        patient_session["data"]["Age"] = age_category

        return True, None

    # --------------------------------------------------------
    # HEIGHT + WEIGHT
    # --------------------------------------------------------

    if field == "height_weight":

        height, weight = extract_height_weight(
            user_message
        )

        if height is None or weight is None:

            return False, (
                "I need both your height and weight. "
                "For example: 177 cm, 85 kg."
            )

        if (
            height < 100
            or height > 250
            or weight < 20
            or weight > 400
        ):

            return False, (
                "That doesn't look like a valid height or weight. "
                "Please try again, for example: 177 cm, 85 kg."
            )

        bmi = calculate_bmi(
            height,
            weight
        )

        if bmi is None:

            return False, (
                "I couldn't calculate your BMI. "
                "Please try again with your height and weight."
            )

        patient_session["profile"]["height_cm"] = height
        patient_session["profile"]["weight_kg"] = weight

        patient_session["data"]["BMI"] = bmi

        return True, (
            f"Got it. Your BMI is approximately {bmi}."
        )

    # --------------------------------------------------------
    # YES / NO FIELDS
    # --------------------------------------------------------

    yes_no_fields = [

        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk"
    ]

    if field in yes_no_fields:

        value = parse_yes_no(
            user_message
        )

        if value is None:

            return False, (
                "Just answer yes or no."
            )

        patient_session["data"][field] = value

        return True, None

    # --------------------------------------------------------
    # GENERAL HEALTH
    # --------------------------------------------------------

    if field == "GenHlth":

        value = parse_general_health(
            user_message
        )

        if value is None:

            return False, (
                "You can answer with: excellent, very good, "
                "good, fair, or poor."
            )

        patient_session["data"][field] = value

        return True, None

    # --------------------------------------------------------
    # DAYS
    # --------------------------------------------------------

    if field in [
        "MentHlth",
        "PhysHlth"
    ]:

        value = parse_days(
            user_message
        )

        if value is None:

            return False, (
                "Please enter a number between 0 and 30."
            )

        patient_session["data"][field] = value

        return True, None

    # --------------------------------------------------------
    # SEX
    # --------------------------------------------------------

    if field == "Sex":

        value = parse_sex(
            user_message
        )

        if value is None:

            return False, (
                "Please answer male or female."
            )

        patient_session["data"][field] = value

        return True, None

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    if field == "Education":

        value = parse_education(
            user_message
        )

        if value is None:

            return False, (
                "Could you describe your highest education level? "
                "For example: high school, college, "
                "or postgraduate."
            )

        patient_session["data"][field] = value

        return True, None

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    if field == "Income":

        value = parse_income(
            user_message
        )

        if value is None:

            return False, (
                "You can give a broad category such as "
                "low, middle, high, or very high."
            )

        patient_session["data"][field] = value

        return True, None

    return False, (
        "I couldn't understand that answer. "
        "Could you try again?"
    )


# ============================================================
# GENERIC INFORMATION EXTRACTION
# ============================================================

def extract_information(
    user_message,
    model_name
):

    schema = MODEL_SCHEMAS.get(
        model_name
    )

    if not schema:
        return {}

    fields = schema.get(
        "fields",
        {}
    )

    if not fields:
        return {}

    prompt = f"""
Extract only patient information explicitly provided by the user.

USER MESSAGE:

{user_message}

MODEL:

{model_name}

REQUIRED FIELD SCHEMA:

{json.dumps(fields, indent=2)}

Return ONLY valid JSON:

{{
    "provided_data": {{}}
}}

Rules:

1. Use only exact field names from the schema.
2. Never guess.
3. Never invent values.
4. Extract only explicitly provided values.
5. Convert numeric values where appropriate.
6. Do not include missing fields.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        result = extract_json(
            content
        )

        if not isinstance(
            result,
            dict
        ):
            return {}

        data = result.get(
            "provided_data",
            {}
        )

        if isinstance(
            data,
            dict
        ):
            return data

    except Exception as error:

        print(
            "[HealthcareAI] Extraction error:",
            error
        )

    return {}


# ============================================================
# MERGE PATIENT DATA
# ============================================================

def merge_patient_data(new_data):

    if not isinstance(
        new_data,
        dict
    ):
        return

    patient_session["data"].update(
        new_data
    )


# ============================================================
# GENERIC MISSING FIELDS
# ============================================================

def get_missing_fields(model_name):

    schema = MODEL_SCHEMAS.get(
        model_name
    )

    if not schema:
        return []

    fields = schema.get(
        "fields",
        {}
    )

    current_data = patient_session[
        "data"
    ]

    return [

        field
        for field in fields

        if field not in current_data
    ]


# ============================================================
# GENERIC NEXT QUESTION
# ============================================================

def ask_next_generic_question(
    model_name
):

    missing = get_missing_fields(
        model_name
    )

    if not missing:
        return None

    field = missing[0]

    patient_session[
        "current_field"
    ] = field

    label = format_field_name(
        model_name,
        field
    )

    return (
        f"Okay. Next, I need your {label}. "
        f"What would you enter for that?"
    )


# ============================================================
# EXPLAIN PREDICTION WITH GROQ
# ============================================================

def explain_prediction(
    result,
    model_name
):

    description = MODEL_SCHEMAS.get(
        model_name,
        {}
    ).get(
        "description",
        model_name
    )

    prompt = f"""
A local machine-learning model has completed a health assessment.

ASSESSMENT:

{description}

LOCAL MODEL RESULT:

{json.dumps(result, indent=2)}

Write a natural, clear response for the user.

Rules:

- Do not change the prediction.
- Do not invent medical findings.
- Call this an AI or machine-learning model estimate.
- Never call it a confirmed diagnosis.
- Mention probability only if it exists in the result.
- Explain the result simply.
- Keep the response concise.
- Give sensible next steps if appropriate.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as error:

        print(
            "[HealthcareAI] Explanation error:",
            error
        )

        return (
            "The local machine-learning model completed "
            "the assessment. This is an AI-based estimate, "
            "not a medical diagnosis.\n\n"
            f"Model result: {result}"
        )


# ============================================================
# GENERAL HEALTH QUESTION
# ============================================================

def answer_general_question(
    user_message
):

    try:

        response = client.chat.completions.create(
            model=MODEL,
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
            temperature=0.2
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as error:

        print(
            "[HealthcareAI] Groq error:",
            error
        )

        return (
            "I'm having trouble connecting to the health "
            "assistant right now. Please try again."
        )


# ============================================================
# RUN LOCAL MODEL
# ============================================================

def run_current_assessment():

    model_name = patient_session[
        "model"
    ]

    print(
        "\n[HealthcareAI] All required inputs received."
    )

    print(
        "[HealthcareAI] Running local ML model..."
    )

    print(
        json.dumps(
            patient_session["data"],
            indent=2
        )
    )

    try:

        result = predict(
            model_name,
            patient_session["data"]
        )

    except Exception as error:

        print(
            "[HealthcareAI] Prediction error:",
            error
        )

        reset_session()

        return (
            "I couldn't run the local model with those inputs. "
            "The model or input schema may need adjustment."
        )

    answer = explain_prediction(
        result,
        model_name
    )

    reset_session()

    return answer


# ============================================================
# START DIABETES ASSESSMENT
# ============================================================

def start_diabetes_assessment():

    patient_session[
        "current_field"
    ] = "age"

    return (
        "Sure. I can give you an AI-based diabetes risk "
        "estimate, not a medical diagnosis. Let's start "
        "with a few basics — how old are you?"
    )


# ============================================================
# CONTINUE DIABETES ASSESSMENT
# ============================================================

def continue_diabetes_assessment(
    user_message
):

    success, response = process_diabetes_answer(
        user_message
    )

    if not success:
        return response

    next_question = get_next_diabetes_question()

    if next_question:

        question = next_question[
            "question"
        ]

        # Add BMI acknowledgement naturally.
        if (
            "BMI"
            in patient_session["data"]
            and next_question["field"] == "HighBP"
        ):

            bmi = patient_session["data"]["BMI"]

            return (
                f"Got it. Your BMI is approximately {bmi}. "
                f"{question}"
            )

        return question

    return run_current_assessment()


# ============================================================
# MAIN HEALTHCARE AI
# ============================================================

def ask_healthcare_ai(
    user_message
):

    user_message = str(
        user_message
    ).strip()

    if not user_message:

        return (
            "Ask me a health question or tell me what "
            "you'd like to assess."
        )

    # --------------------------------------------------------
    # CANCEL ACTIVE ASSESSMENT
    # --------------------------------------------------------

    if (
        patient_session["model"]
        and is_cancel_request(user_message)
    ):

        reset_session()

        return (
            "No problem — I've cancelled that assessment. "
            "You can start a new one anytime."
        )

    # --------------------------------------------------------
    # CONTINUE ACTIVE ASSESSMENT
    # --------------------------------------------------------

    if patient_session["model"]:

        model_name = patient_session[
            "model"
        ]

        if model_name == "diabetes_binary":

            return continue_diabetes_assessment(
                user_message
            )

        new_data = extract_information(
            user_message,
            model_name
        )

        merge_patient_data(
            new_data
        )

        missing = get_missing_fields(
            model_name
        )

        if missing:

            return ask_next_generic_question(
                model_name
            )

        return run_current_assessment()

    # --------------------------------------------------------
    # NEW REQUEST
    # --------------------------------------------------------

    routing = route_request(
        user_message
    )

    # General health question
    if not routing.get(
        "use_ml"
    ):

        return answer_general_question(
            user_message
        )

    model_name = normalize_model_name(
        routing.get("model")
    )

    if not model_name:

        return (
            "I couldn't match that request to one of my "
            "available local health assessments."
        )

    # --------------------------------------------------------
    # CHECK MODEL SCHEMA
    # --------------------------------------------------------

    if not model_is_configured(
        model_name
    ):

        return (
            f"I have a local model for "
            f"{MODEL_SCHEMAS[model_name]['description']}, "
            f"but its input flow hasn't been configured yet."
        )

    # --------------------------------------------------------
    # START SESSION
    # --------------------------------------------------------

    patient_session["model"] = model_name
    patient_session["data"] = {}
    patient_session["profile"] = {}
    patient_session["current_field"] = None
    patient_session["started"] = True
    patient_session["original_request"] = user_message

    print(
        f"\n[HealthcareAI] Starting "
        f"{model_name} assessment..."
    )

    # --------------------------------------------------------
    # NATURAL DIABETES FLOW
    # --------------------------------------------------------

    if model_name == "diabetes_binary":

        return start_diabetes_assessment()

    # --------------------------------------------------------
    # GENERIC FLOW FOR OTHER MODELS
    # --------------------------------------------------------

    new_data = extract_information(
        user_message,
        model_name
    )

    merge_patient_data(
        new_data
    )

    missing = get_missing_fields(
        model_name
    )

    if missing:

        return ask_next_generic_question(
            model_name
        )

    return run_current_assessment()


# ============================================================
# CLI
# ============================================================

def main():

    print(
        "HealthcareAI is connected to Groq."
    )

    print(
        f"Model: {MODEL}"
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

            print("\nHealthcareAI stopped.")

            break

        if user_message.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "HealthcareAI stopped."
            )

            break

        answer = ask_healthcare_ai(
            user_message
        )

        print(
            "\nHealthcareAI:"
        )

        print(
            answer
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
