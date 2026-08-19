import os
import json
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
You are HealthcareAI.

You provide concise health information and machine-learning
decision support.

Rules:

- Never invent patient information.
- Never invent medical history.
- Never invent laboratory results.
- Never invent machine-learning predictions.
- Local ML models perform predictions.
- Never alter a local model result.
- A machine-learning prediction is not a medical diagnosis.
- Clearly describe ML results as model estimates.
- Do not claim model confidence equals medical certainty.
- Keep responses concise and easy to understand.
- For potentially serious symptoms, recommend appropriate
  medical evaluation.
- For possible emergency symptoms, recommend urgent medical care.
"""


# ============================================================
# PATIENT SESSION
# ============================================================

patient_session = {
    "model": None,
    "data": {}
}


# ============================================================
# GET AVAILABLE MODELS
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

    # Exact schema match
    if model_name in MODEL_SCHEMAS:
        return model_name

    # Check aliases
    for schema_name, schema in MODEL_SCHEMAS.items():

        aliases = schema.get(
            "aliases",
            []
        )

        if model_name in aliases:
            return schema_name

    return None


# ============================================================
# DETECT MODEL LOCALLY
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

            if keyword.lower() in text:

                matches.append(
                    (
                        len(keyword),
                        model_name
                    )
                )

    if not matches:
        return None

    # Longest matching keyword wins.
    matches.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    return matches[0][1]


# ============================================================
# ROUTE REQUEST
# ============================================================

def route_request(user_message):

    # --------------------------------------------------------
    # FIRST: LOCAL DETERMINISTIC ROUTING
    # --------------------------------------------------------

    detected_model = detect_model(
        user_message
    )

    if detected_model:

        return {
            "use_ml": True,
            "model": detected_model
        }

    # --------------------------------------------------------
    # SECOND: GROQ FALLBACK
    # --------------------------------------------------------

    available_models = get_available_models()

    prompt = f"""
Determine whether this message is requesting a prediction from
a local machine-learning model.

USER MESSAGE:

{user_message}

AVAILABLE LOCAL MODELS:

{json.dumps(available_models, indent=2)}

Return ONLY JSON.

For a general health question:

{{
    "use_ml": false,
    "model": null
}}

For a local ML prediction request:

{{
    "use_ml": true,
    "model": "exact_model_name"
}}

Do not invent model names.
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

        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        result = json.loads(
            content
        )

        model_name = result.get(
            "model"
        )

        normalized = normalize_model_name(
            model_name
        )

        if normalized:

            return {
                "use_ml": True,
                "model": normalized
            }

        return {
            "use_ml": False,
            "model": None
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
# EXTRACT INFORMATION
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

    # No schema yet.
    if not fields:
        return {}

    prompt = f"""
Extract only explicitly provided patient information.

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
4. Extract only values explicitly provided.
5. Convert numeric values to numbers.
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

        content = content.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        result = json.loads(
            content
        )

        data = result.get(
            "provided_data",
            {}
        )

        if isinstance(data, dict):
            return data

        return {}

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
# FIND MISSING FIELDS
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

    missing = []

    for field in fields:

        if field not in current_data:

            missing.append(
                field
            )

    return missing


# ============================================================
# FORMAT FIELD NAME
# ============================================================

def format_field_name(
    model_name,
    field
):

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
# ASK FOR MISSING DATA
# ============================================================

def ask_for_missing(
    model_name,
    missing_fields
):

    readable_fields = []

    for field in missing_fields:

        readable_fields.append(
            format_field_name(
                model_name,
                field
            )
        )

    lines = []

    for field in readable_fields:

        lines.append(
            f"- {field}"
        )

    return (
        "To run the local model, I need:\n\n"
        + "\n".join(lines)
    )


# ============================================================
# EXPLAIN PREDICTION
# ============================================================

def explain_prediction(
    user_message,
    result,
    model_name
):

    prompt = f"""
A local machine-learning model has completed an assessment.

MODEL:

{model_name}

USER MESSAGE:

{user_message}

LOCAL MODEL RESULT:

{json.dumps(result, indent=2)}

Write a concise response.

Rules:

- Do not change the prediction.
- Do not invent medical findings.
- Call it a machine-learning model estimate.
- Do not call it a confirmed diagnosis.
- Mention probability or confidence only if present.
- Keep the answer between 2 and 4 sentences.
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
        )

    except Exception:

        return (
            "The local machine-learning model completed "
            "the assessment. This result is a model estimate "
            "and not a medical diagnosis.\n\n"
            f"Result: {result}"
        )


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

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# RESET SESSION
# ============================================================

def reset_session():

    patient_session["model"] = None

    patient_session["data"] = {}


# ============================================================
# CHECK MODEL CONFIGURATION
# ============================================================

def model_is_configured(
    model_name
):

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
# MAIN HEALTHCARE AI
# ============================================================

def ask_healthcare_ai(
    user_message
):

    # ========================================================
    # CONTINUE EXISTING ASSESSMENT
    # ========================================================

    if patient_session["model"]:

        model_name = patient_session[
            "model"
        ]

        new_data = extract_information(
            user_message,
            model_name
        )

        merge_patient_data(
            new_data
        )

    else:

        # ====================================================
        # NEW REQUEST
        # ====================================================

        routing = route_request(
            user_message
        )

        if not routing.get(
            "use_ml"
        ):

            return answer_general_question(
                user_message
            )

        model_name = routing.get(
            "model"
        )

        model_name = normalize_model_name(
            model_name
        )

        if not model_name:

            return (
                "I could not match that request to one of "
                "my configured local assessments."
            )

        # ----------------------------------------------------
        # MODEL EXISTS BUT INPUT SCHEMA NOT CONFIGURED
        # ----------------------------------------------------

        if not model_is_configured(
            model_name
        ):

            return (
                f"The local model for "
                f"'{MODEL_SCHEMAS[model_name]['description']}' "
                f"is available, but its patient input schema "
                f"has not been configured yet."
            )

        patient_session["model"] = (
            model_name
        )

        patient_session["data"] = {}

        print(
            f"\n[HealthcareAI] Starting "
            f"{model_name} assessment..."
        )

        new_data = extract_information(
            user_message,
            model_name
        )

        merge_patient_data(
            new_data
        )


    # ========================================================
    # CHECK MISSING DATA
    # ========================================================

    missing = get_missing_fields(
        model_name
    )

    if missing:

        return ask_for_missing(
            model_name,
            missing
        )


    # ========================================================
    # RUN LOCAL MODEL
    # ========================================================

    print(
        "\n[HealthcareAI] All required model "
        "inputs received."
    )

    print(
        "[HealthcareAI] Running local ML model..."
    )

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
            "I couldn't run the local model with "
            "the provided information. The model input "
            "format may need adjustment."
        )


    # ========================================================
    # EXPLAIN RESULT
    # ========================================================

    answer = explain_prediction(
        user_message,
        result,
        model_name
    )


    # ========================================================
    # RESET AFTER COMPLETION
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
        f"Using .env: {ENV_PATH}"
    )

    print(
        "Local ML model registry ready."
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


        if (
            user_message
            .lower()
            .strip()
            == "exit"
        ):

            print(
                "Exiting HealthcareAI."
            )

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
