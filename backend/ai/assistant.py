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
You are HealthcareAI.

You provide concise, helpful health information and support
local machine-learning assessments.

Important rules:

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
# JSON HELPERS
# ============================================================

def extract_json(text):
    """
    Extract JSON safely from an LLM response.
    Handles plain JSON and Markdown code blocks.
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

    # Try to locate a JSON object inside extra text.
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:
        try:
            return json.loads(match.group())

        except json.JSONDecodeError:
            return None

    return None


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

    model_name = str(model_name).strip()

    # Exact schema match
    if model_name in MODEL_SCHEMAS:
        return model_name

    # Case-insensitive schema match
    for schema_name in MODEL_SCHEMAS:

        if schema_name.lower() == model_name.lower():
            return schema_name

    # Check aliases
    for schema_name, schema in MODEL_SCHEMAS.items():

        aliases = schema.get(
            "aliases",
            []
        )

        for alias in aliases:

            if alias.lower() == model_name.lower():
                return schema_name

    return None


# ============================================================
# DETECT MODEL BY DISEASE / MODEL KEYWORDS
# ============================================================

def detect_model(user_message):
    """
    Detect which model the user is referring to.

    IMPORTANT:
    This function only identifies a possible model.
    It does NOT decide whether ML should be used.
    """

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

    # More specific / longer keyword wins.
    matches.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    return matches[0][1]


# ============================================================
# DETECT EXPLICIT ML PREDICTION INTENT
# ============================================================

def is_prediction_request(user_message):
    """
    Returns True only when the user is explicitly asking
    for a personal prediction, assessment, classification,
    or analysis using their own information.

    A disease name alone is NOT enough.
    """

    text = user_message.lower().strip()

    explicit_patterns = [

        # Prediction
        r"\bpredict\b",
        r"\bprediction\b",
        r"\bcan you predict\b",

        # Assessment
        r"\bassessment\b",
        r"\bassess me\b",
        r"\bassess my\b",
        r"\brun an assessment\b",

        # Classification
        r"\bclassify\b",
        r"\bclassification\b",

        # Personal checking
        r"\bcheck my\b",
        r"\bcheck whether i\b",
        r"\bcheck if i\b",

        # Personal risk
        r"\bmy risk\b",
        r"\bam i at risk\b",
        r"\bwhat is my risk\b",
        r"\bcalculate my risk\b",

        # Personal analysis
        r"\banalyze my\b",
        r"\banalyse my\b",
        r"\bevaluate my\b",
        r"\buse my data\b",
        r"\buse my health data\b",
        r"\buse my values\b",

        # Model usage
        r"\brun the model\b",
        r"\brun a prediction\b",
        r"\buse the model\b",

        # Direct personal questions
        r"\bdo i have\b",
        r"\bcould i have\b",
        r"\bcan you determine if i\b",
    ]

    for pattern in explicit_patterns:

        if re.search(
            pattern,
            text
        ):
            return True

    return False


# ============================================================
# DETECT ASSESSMENT CANCELLATION
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
        "quit assessment",
    ]

    return any(
        phrase in text
        for phrase in cancel_phrases
    )


# ============================================================
# ROUTE REQUEST
# ============================================================

def route_request(user_message):
    """
    Safe routing strategy:

    1. General health questions default to Groq.
    2. A disease keyword alone NEVER activates ML.
    3. Explicit prediction intent is required.
    4. If prediction intent exists, detect the local model.
    5. If deterministic detection fails, ask Groq to select
       from the existing local model registry.
    """

    # --------------------------------------------------------
    # GENERAL QUESTION BY DEFAULT
    # --------------------------------------------------------

    if not is_prediction_request(user_message):

        return {
            "use_ml": False,
            "model": None
        }

    # --------------------------------------------------------
    # EXPLICIT PREDICTION + LOCAL MODEL DETECTION
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
    # GROQ FALLBACK:
    # ONLY USED WHEN USER EXPLICITLY REQUESTED A PREDICTION
    # BUT THE MODEL COULD NOT BE DETERMINED LOCALLY.
    # --------------------------------------------------------

    available_models = get_available_models()

    if not available_models:

        return {
            "use_ml": False,
            "model": None
        }

    prompt = f"""
The user explicitly requested a personal machine-learning
prediction or assessment.

Determine whether one of the available local ML models matches
the request.

USER MESSAGE:

{user_message}

AVAILABLE LOCAL MODELS:

{json.dumps(available_models, indent=2)}

Return ONLY valid JSON.

If one model clearly matches:

{{
    "use_ml": true,
    "model": "exact_model_name"
}}

If no model clearly matches:

{{
    "use_ml": false,
    "model": null
}}

Rules:

1. Use ONLY a model from AVAILABLE LOCAL MODELS.
2. Do not invent model names.
3. Do not choose a model merely because the user mentions
   a disease indirectly.
4. The user has already expressed prediction intent.
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

        result = extract_json(content)

        if not isinstance(
            result,
            dict
        ):
            return {
                "use_ml": False,
                "model": None
            }

        if not result.get("use_ml"):

            return {
                "use_ml": False,
                "model": None
            }

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

    fields = schema.get(
        "fields",
        {}
    )

    # Model schema not configured.
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
5. Convert numeric values to numbers where appropriate.
6. Do not include missing fields.
7. Do not infer values from general health questions.
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

        result = extract_json(content)

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
        "To run the local machine-learning model, "
        "I need the following information:\n\n"
        + "\n".join(lines)
        + "\n\nYou can provide the values in one message "
          "or across multiple messages. Type 'cancel' to "
          "stop this assessment."
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

USER REQUEST:

{user_message}

LOCAL MODEL RESULT:

{json.dumps(result, indent=2)}

Write a concise response.

Rules:

- Do not change the prediction.
- Do not invent medical findings.
- Call it a machine-learning model estimate.
- Do not call it a confirmed diagnosis.
- Mention probability or confidence only if present
  in the local model result.
- Keep the answer between 2 and 4 sentences.
- If appropriate, recommend discussing concerning
  results with a qualified healthcare professional.
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

    except Exception as error:

        print(
            "[HealthcareAI] Explanation error:",
            error
        )

        return (
            "The local machine-learning model completed "
            "the assessment. This is a model estimate and "
            "not a medical diagnosis.\n\n"
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

    # --------------------------------------------------------
    # CANCEL ACTIVE ASSESSMENT
    # --------------------------------------------------------

    if (
        patient_session["model"]
        and is_cancel_request(user_message)
    ):

        reset_session()

        return (
            "The assessment has been cancelled. "
            "You can ask me a general health question or "
            "start a new assessment anytime."
        )

    # --------------------------------------------------------
    # CONTINUE EXISTING ASSESSMENT
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # NEW REQUEST
        # ----------------------------------------------------

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

        model_name = routing.get(
            "model"
        )

        model_name = normalize_model_name(
            model_name
        )

        if not model_name:

            return (
                "I could not match that prediction request "
                "to one of my available local assessments."
            )

        # ----------------------------------------------------
        # MODEL EXISTS BUT SCHEMA IS NOT CONFIGURED
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

        # ----------------------------------------------------
        # START NEW ASSESSMENT
        # ----------------------------------------------------

        patient_session["model"] = (
            model_name
        )

        patient_session["data"] = {}

        print(
            f"\n[HealthcareAI] Starting "
            f"{model_name} assessment..."
        )

        # The initial request may already contain patient data.
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
            "I couldn't run the local model with the "
            "provided information. The model input format "
            "may need adjustment."
        )

    # ========================================================
    # EXPLAIN RESULT USING GROQ
    # ========================================================

    answer = explain_prediction(
        user_message,
        result,
        model_name
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
