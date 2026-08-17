import os
import json
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
from groq import Groq

from backend.ml.predict_all import (
    list_models,
    get_model_info,
    predict
)

from backend.ml.schemas import MODEL_SCHEMAS


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=API_KEY
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are HealthcareAI.

You are a concise health-information and
decision-support assistant.

RULES:

- Give fact-based, medically grounded information.
- Keep responses short and focused.
- Never invent patient information.
- Never invent test results.
- Never invent medical history.
- Never invent ML predictions.
- An ML prediction is NOT a medical diagnosis.
- Clearly call ML results "model estimates".
- Ask only for information that is actually required.
- Do not overwhelm the user with unnecessary questions.
- For potentially serious symptoms, recommend appropriate
  medical evaluation.
- For possible emergency symptoms, clearly recommend
  urgent medical care.

ML RULES:

- Local machine-learning models perform the predictions.
- Groq must NOT create or alter ML predictions.
- Never claim that model confidence represents medical certainty.
- Never convert a model estimate into a diagnosis.
- If the local model gives a prediction, report it accurately.
- Keep the explanation concise.

STYLE:

- Natural
- Direct
- Factual
- Concise
- Easy to understand
- No unnecessary tables
- No unnecessary repetition
- Do not sound like a terminal or API
"""


# ============================================================
# PATIENT SESSION
# ============================================================

patient_session = {
    "model": None,
    "data": {}
}


# ============================================================
# AVAILABLE MODELS
# ============================================================

def get_available_models():

    models = list_models()

    return [
        {
            "name": model["name"],
            "target": model["target"],
            "accuracy": model["accuracy"]
        }
        for model in models
    ]


# ============================================================
# ROUTE USER REQUEST
# ============================================================

def route_request(user_message):

    models = get_available_models()

    prompt = f"""
Determine whether the user's message is:

1. A general health question

OR

2. A request to use an available machine-learning model.

USER:
{user_message}

AVAILABLE MODELS:
{json.dumps(models, indent=2)}

Return ONLY valid JSON.

For a general question:

{{
    "use_ml": false,
    "model": null
}}

For an ML request:

{{
    "use_ml": true,
    "model": "exact_model_name"
}}

Rules:

- Model name must exactly match an available model.
- Do not invent models.
- Do not select an ML model for a general medical question.
"""

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

    content = response.choices[0].message.content.strip()

    # Remove markdown JSON fences if Groq returns them.
    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

    try:

        return json.loads(content)

    except json.JSONDecodeError:

        return {
            "use_ml": False,
            "model": None
        }


# ============================================================
# EXTRACT PATIENT INFORMATION
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

    fields = schema["fields"]

    prompt = f"""
Extract patient information from the user's message.

USER MESSAGE:
{user_message}

MODEL:
{model_name}

MODEL FIELD SCHEMA:
{json.dumps(fields, indent=2)}

Return ONLY valid JSON:

{{
    "provided_data": {{}}
}}

IMPORTANT RULES:

1. Extract ONLY information explicitly provided by the user.
2. Never guess missing values.
3. Never invent measurements.
4. Use the exact field names from the schema.
5. Convert numeric values to numbers.
6. For categorical fields, convert the user's words
   to the exact numeric value defined in "values".
7. For yes/no fields, use the values defined in the schema.
8. Never create a value if the user did not provide it.
9. Do not include fields that were not provided.
10. If the user uses natural language, map it to the
    appropriate schema value.

Examples:

"male" -> sex = 1
"female" -> sex = 0

"typical angina" -> cp = 1
"atypical angina" -> cp = 2
"non-anginal pain" -> cp = 3
"asymptomatic" -> cp = 4

"normal ECG" -> restecg = 0

"no exercise-induced angina" -> exang = 0
"exercise-induced angina" -> exang = 1

"upsloping" -> slope = 1
"flat" -> slope = 2
"downsloping" -> slope = 3

"normal thalassemia" -> thal = 3
"fixed defect" -> thal = 6
"reversible defect" -> thal = 7

Return JSON only.
"""

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

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

    try:

        result = json.loads(content)

    except json.JSONDecodeError:

        return {}

    return result.get(
        "provided_data",
        {}
    )


# ============================================================
# MERGE PATIENT DATA
# ============================================================

def merge_patient_data(new_data):

    global patient_session

    if not isinstance(new_data, dict):
        return

    patient_session["data"].update(
        new_data
    )


# ============================================================
# FIND MISSING FIELDS
# ============================================================

def get_missing_fields(
    model_name
):

    schema = MODEL_SCHEMAS.get(
        model_name
    )

    if not schema:
        return []

    required_fields = list(
        schema["fields"].keys()
    )

    current_data = (
        patient_session["data"]
    )

    missing = []

    for field in required_fields:

        if field not in current_data:

            missing.append(field)

    return missing


# ============================================================
# FIELD DISPLAY NAMES
# ============================================================

def format_field_name(field):

    names = {

        "age":
            "Age (years)",

        "sex":
            "Sex (male/female)",

        "cp":
            "Chest pain type "
            "(typical angina / atypical angina / "
            "non-anginal pain / asymptomatic)",

        "trestbps":
            "Resting blood pressure (mmHg)",

        "chol":
            "Cholesterol (mg/dL)",

        "fbs":
            "Fasting blood sugar > 120 mg/dL (yes/no)",

        "restecg":
            "Resting ECG result "
            "(normal / ST-T wave abnormality / "
            "left ventricular hypertrophy)",

        "thalach":
            "Maximum heart rate achieved",

        "exang":
            "Exercise-induced angina (yes/no)",

        "oldpeak":
            "ST depression induced by exercise",

        "slope":
            "Exercise ST-segment slope "
            "(upsloping / flat / downsloping)",

        "ca":
            "Number of major vessels (0-3)",

        "thal":
            "Thalassemia result "
            "(normal / fixed defect / reversible defect)"
    }

    return names.get(
        field,
        field
    )


# ============================================================
# ASK FOR MISSING DATA
# ============================================================

def ask_for_missing(
    missing_fields
):

    readable = [
        format_field_name(field)
        for field in missing_fields
    ]

    prompt = f"""
Ask the user for these missing health-assessment values:

{json.dumps(readable, indent=2)}

Rules:

- Keep it very concise.
- Use bullet points.
- Do not explain each field.
- Do not invent values.
- Do not mention internal dataset codes.
- Do not mention ML implementation details.
- Ask only for the missing values.
"""

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

    return response.choices[0].message.content


# ============================================================
# EXPLAIN ML RESULT
# ============================================================

def explain_prediction(
    user_message,
    result
):

    prompt = f"""
You are HealthcareAI responding after a local
machine-learning model has completed an assessment.

USER MESSAGE:
{user_message}

LOCAL ML RESULT:
{json.dumps(result, indent=2)}

Write a natural, human-sounding response.

IMPORTANT RULES:

1. Do not sound like a terminal, API, database, or technical log.
2. Do not output JSON.
3. Do not mention internal field names.
4. Do not repeat every patient input.
5. Do not invent patient information.
6. Do not invent medical results.
7. Do not change the local ML prediction.
8. Clearly describe the result as a machine-learning model estimate.
9. Never describe the prediction as a confirmed diagnosis.
10. Mention model confidence if available.
11. Explain the result in simple language.
12. Keep the response short: 2-4 sentences.
13. Do not give unnecessary generic health advice.
14. If the user's message contains a potentially concerning
    symptom, acknowledge it appropriately.
15. Never claim that model confidence represents medical certainty.

For binary classification:

- prediction "0" means the model's negative class.
- prediction "1" means the model's positive class.

Do NOT automatically translate these into a confirmed
medical diagnosis.

For heart disease, prefer wording such as:

"Based on the information provided, the model estimates
a negative heart-disease classification..."

instead of:

"You don't have heart disease."

For other models, describe the model's output accurately
without inventing what the class means.

Respond naturally and conversationally.
"""

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

    return response.choices[0].message.content


# ============================================================
# GENERAL HEALTH QUESTION
# ============================================================

def answer_general_question(
    user_message
):

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

    return response.choices[0].message.content


# ============================================================
# RESET SESSION
# ============================================================

def reset_session():

    global patient_session

    patient_session = {
        "model": None,
        "data": {}
    }


# ============================================================
# MAIN HEALTHCAREAI
# ============================================================

def ask_healthcare_ai(
    user_message
):

    global patient_session

    # ========================================================
    # CONTINUE EXISTING ML SESSION
    # ========================================================

    if patient_session["model"] is not None:

        model_name = patient_session["model"]

        print(
            f"\n[HealthcareAI] Continuing "
            f"{model_name} assessment..."
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

            return ask_for_missing(
                missing
            )

        # All required data received.
        # Continue to model prediction.

    else:

        # ====================================================
        # NEW REQUEST
        # ====================================================

        routing = route_request(
            user_message
        )

        use_ml = routing.get(
            "use_ml",
            False
        )

        model_name = routing.get(
            "model"
        )

        # ====================================================
        # GENERAL HEALTH QUESTION
        # ====================================================

        if not use_ml:

            return answer_general_question(
                user_message
            )

        # ====================================================
        # START NEW ML SESSION
        # ====================================================

        if model_name not in MODEL_SCHEMAS:

            return (
                "I don't have a configured assessment "
                "for that condition yet."
            )

        patient_session["model"] = model_name
        patient_session["data"] = {}

        print(
            f"\n[HealthcareAI] Starting "
            f"{model_name} assessment..."
        )

        # Extract information already present
        # in the first user message.

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

            return ask_for_missing(
                missing
            )

    # ========================================================
    # RUN LOCAL ML MODEL
    # ========================================================

    print(
        "\n[HealthcareAI] All required "
        "model inputs received."
    )

    print(
        "[HealthcareAI] Running local ML model..."
    )

    # Keep detailed patient data in terminal logs
    # but never send this raw technical output to the user.

    print(
        "[HealthcareAI] Patient data:"
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
            "I couldn't run the model with "
            "the provided information."
        )

    # ========================================================
    # EXPLAIN RESULT NATURALLY
    # ========================================================

    answer = explain_prediction(
        user_message,
        result
    )

    # ========================================================
    # RESET SESSION AFTER COMPLETION
    # ========================================================

    reset_session()

    return answer


# ============================================================
# TERMINAL INTERFACE
# ============================================================

if __name__ == "__main__":

    print(
        "HealthcareAI is connected to Groq."
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        "Local ML models loaded."
    )

    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        try:

            user_message = input(
                "You: "
            )

        except KeyboardInterrupt:

            print(
                "\nExiting HealthcareAI."
            )

            break

        if user_message.lower().strip() == "exit":

            break

        if not user_message.strip():

            continue

        try:

            answer = ask_healthcare_ai(
                user_message
            )

            print(
                "\nHealthcareAI:"
            )

            print(
                answer
            )

            print()

        except Exception as error:

            print(
                "\nError:",
                error
            )

            print()
