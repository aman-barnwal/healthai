import os
import re
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.ai.model_router import detect_model

from backend.ai.conversation_manager import (
    start_assessment,
    handle_answer,
    cancel_session,
    get_session,
)


# ============================================================
# PROJECT / ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1",
)


if not GROQ_API_KEY:

    raise RuntimeError(
        f"GROQ_API_KEY not found in {ENV_PATH}"
    )


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are HealthcareAI, a health-information and
decision-support assistant.

Your role is to provide useful, concise, medically
responsible health information.

IMPORTANT CONVERSATION RULES:

1. Maintain context from the conversation history.
2. Never ask the user for information they have already given.
3. Never repeat a question unless the previous answer was
   genuinely unclear or contradictory.
4. When the user corrects you, acknowledge the correction
   and continue from the information already provided.
5. If the user says something like:
   "I already told you"
   "I mentioned that"
   "as I said"
   "you already know"
   review the conversation context before responding.
6. Do not restart the conversation unnecessarily.
7. Ask only the minimum number of useful follow-up questions.
8. Prefer one focused question at a time.

MEDICAL SAFETY:

- Do not claim to diagnose a disease.
- Do not invent symptoms, history, test results, or diagnoses.
- Do not present probabilities as medical certainty.
- Encourage appropriate professional medical evaluation when
  symptoms could be serious.
- If symptoms suggest a possible emergency, clearly recommend
  urgent medical care.
- Do not minimize severe or worsening symptoms.

LOCAL ML MODELS:

- HealthcareAI contains local machine-learning models.
- Only the local ML system performs ML predictions.
- You must NEVER invent, simulate, or modify a model prediction.
- A model prediction is an AI-based estimate, not a diagnosis.
- If no local prediction result was provided to you, do not
  claim that a model produced one.

STYLE:

- Natural
- Clear
- Direct
- Calm
- Concise
- Context-aware
- Avoid unnecessary repetition
- Do not sound like an API or terminal
"""


# ============================================================
# IN-MEMORY CHAT HISTORY
# ============================================================

MAX_HISTORY_MESSAGES = 20

chat_history = deque(
    maxlen=MAX_HISTORY_MESSAGES
)


# ============================================================
# ACTIVE ASSESSMENT SESSION
# ============================================================

active_assessment_session_id = None


# ============================================================
# CANCEL DETECTION
# ============================================================

def is_cancel_request(user_message):

    if not user_message:

        return False

    text = user_message.lower().strip()

    cancel_phrases = [

        "cancel",

        "cancel assessment",

        "stop assessment",

        "stop the assessment",

        "quit assessment",

        "end assessment",

        "reset assessment",

        "never mind",

        "nevermind",

        "stop",

    ]

    return any(
        phrase == text
        or phrase in text
        for phrase in cancel_phrases
    )


# ============================================================
# RESET GENERAL CHAT HISTORY
# ============================================================

def clear_chat_history():

    chat_history.clear()


# ============================================================
# ADD CHAT MESSAGE
# ============================================================

def add_to_history(
    role,
    content
):

    if not content:

        return

    chat_history.append(

        {
            "role": role,
            "content": str(content),
        }

    )


# ============================================================
# BUILD GROQ MESSAGES
# ============================================================

def build_groq_messages(
    user_message
):

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }

    ]

    for message in chat_history:

        messages.append(
            message
        )

    messages.append(

        {
            "role": "user",
            "content": user_message,
        }

    )

    return messages


# ============================================================
# EMERGENCY / URGENT SYMPTOM CHECK
# ============================================================

def detect_urgent_symptoms(
    user_message
):

    text = user_message.lower().strip()

    emergency_patterns = [

        r"\bchest pain\b",

        r"\bsevere chest pain\b",

        r"\bcan't breathe\b",

        r"\bcannot breathe\b",

        r"\bdifficulty breathing\b",

        r"\btrouble breathing\b",

        r"\bshortness of breath\b",

        r"\bunconscious\b",

        r"\bpassed out\b",

        r"\bfainting\b",

        r"\bseizure\b",

        r"\bface drooping\b",

        r"\bslurred speech\b",

        r"\bsudden weakness\b",

        r"\bsevere bleeding\b",

        r"\bvomiting blood\b",

        r"\bcoughing blood\b",

        r"\bsuicidal\b",

        r"\bkill myself\b",

        r"\bend my life\b",

    ]

    for pattern in emergency_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            return True

    return False


# ============================================================
# HIGH-SEVERITY ABDOMINAL PAIN CHECK
# ============================================================

def detect_severe_abdominal_pain(
    user_message
):

    text = user_message.lower()

    abdomen_words = [

        "abdomen",

        "abdominal",

        "stomach",

        "belly",

        "lower abdomen",

    ]

    severity_words = [

        "severe",

        "extreme",

        "unbearable",

        "8/10",

        "9/10",

        "10/10",

        "getting worse",

        "increasing",

        "suddenly worse",

    ]

    has_location = any(
        word in text
        for word in abdomen_words
    )

    has_severity = any(
        word in text
        for word in severity_words
    )

    return (
        has_location
        and has_severity
    )


# ============================================================
# URGENT RESPONSE
# ============================================================

def get_urgent_response():

    return (
        "Some of the symptoms you described could require "
        "urgent medical evaluation. Please seek urgent medical "
        "care now, especially if the symptoms are severe, "
        "sudden, worsening, or accompanied by trouble breathing, "
        "fainting, severe weakness, confusion, or significant "
        "bleeding. HealthcareAI cannot diagnose an emergency "
        "through chat."
    )


# ============================================================
# FORMAT ASSESSMENT START RESPONSE
# ============================================================

def format_assessment_start(
    result
):

    if not isinstance(
        result,
        dict
    ):

        return str(
            result
        )

    if not result.get(
        "success"
    ):

        return result.get(
            "error",
            "I couldn't start that assessment."
        )

    message = result.get(
        "message",
        "Let's begin the assessment."
    )

    question = result.get(
        "question"
    )

    if question:

        return (
            f"{message}\n\n"
            f"{question}"
        )

    return message


# ============================================================
# FORMAT ASSESSMENT RESPONSE
# ============================================================

def format_assessment_response(
    result
):

    if not isinstance(
        result,
        dict
    ):

        return str(
            result
        )

    if not result.get(
        "success"
    ):

        return result.get(
            "error",
            "I couldn't process that answer."
        )

    if result.get(
        "completed"
    ):

        prediction = result.get(
            "prediction"
        )

        probability_percent = result.get(
            "probability_percent"
        )

        response = (
            "The assessment is complete.\n\n"
            f"Model result: {prediction}"
        )

        if probability_percent is not None:

            response += (
                f"\nEstimated positive-class probability: "
                f"{probability_percent}%"
            )

        response += (

            "\n\nThis is an AI-based model estimate, "
            "not a medical diagnosis."
        )

        return response

    message = result.get(
        "message"
    )

    question = result.get(
        "question"
    )

    if message and question:

        return (
            f"{message}\n\n"
            f"{question}"
        )

    if question:

        return question

    if message:

        return message

    return (
        "Please answer the current assessment question."
    )


# ============================================================
# START LOCAL ML ASSESSMENT
# ============================================================

def start_local_assessment(
    model_name
):

    global active_assessment_session_id

    result = start_assessment(
        model_name
    )

    if (
        isinstance(result, dict)
        and result.get("success")
    ):

        active_assessment_session_id = result.get(
            "session_id"
        )

        return format_assessment_start(
            result
        )

    active_assessment_session_id = None

    return format_assessment_start(
        result
    )


# ============================================================
# HANDLE ACTIVE ASSESSMENT
# ============================================================

def handle_active_assessment(
    user_message
):

    global active_assessment_session_id

    if not active_assessment_session_id:

        return (
            "The assessment session is no longer active. "
            "You can start a new assessment anytime."
        )

    if is_cancel_request(
        user_message
    ):

        result = cancel_session(
            active_assessment_session_id
        )

        active_assessment_session_id = None

        if isinstance(
            result,
            dict
        ):

            return result.get(
                "message",
                "Your assessment has been cancelled."
            )

        return (
            "Your assessment has been cancelled."
        )

    session = get_session(
        active_assessment_session_id
    )

    if not session:

        active_assessment_session_id = None

        return (
            "That assessment session has expired or completed. "
            "You can start another assessment anytime."
        )

    result = handle_answer(
        active_assessment_session_id,
        user_message,
    )

    response = format_assessment_response(
        result
    )

    if (
        isinstance(result, dict)
        and (
            result.get("completed")
            or result.get("active") is False
        )
    ):

        active_assessment_session_id = None

    else:

        session = get_session(
            active_assessment_session_id
        )

        if session is None:

            active_assessment_session_id = None

    return response


# ============================================================
# ASK GROQ WITH CONTEXT
# ============================================================

def ask_groq(
    user_message
):

    try:

        messages = build_groq_messages(
            user_message
        )

        completion = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=messages,

            temperature=0.3,

            max_tokens=700,

        )

        response = (
            completion
            .choices[0]
            .message
            .content
        )

        if not response:

            return (
                "I couldn't generate a response right now. "
                "Please try again."
            )

        response = response.strip()

        add_to_history(
            "user",
            user_message,
        )

        add_to_history(
            "assistant",
            response,
        )

        return response

    except Exception as error:

        print(
            "[HealthcareAI] Groq error:",
            repr(error),
        )

        return (
            "I couldn't process that request right now. "
            "Please try again."
        )


# ============================================================
# ROUTE ML REQUEST
# ============================================================

def route_ml_request(user_message):

    try:

        model_name = detect_model(
            user_message
        )

        return model_name

    except Exception as error:

        print(
            "[HealthcareAI] Model router error:",
            repr(error),
        )

        return None

# ============================================================
# MAIN HEALTHCARE AI
# ============================================================

def healthcare_ai(
    user_message
):

    global active_assessment_session_id

    if user_message is None:

        return (
            "Please enter a health question or "
            "tell me what assessment you would like to start."
        )

    user_message = str(
        user_message
    ).strip()

    if not user_message:

        return (
            "Ask me a health question, describe what you're "
            "experiencing, or tell me if you'd like a health "
            "risk assessment."
        )

    # --------------------------------------------------------
    # ACTIVE ML ASSESSMENT
    # --------------------------------------------------------

    if active_assessment_session_id:

        return handle_active_assessment(
            user_message
        )

    # --------------------------------------------------------
    # URGENT SYMPTOMS
    #
    # Safety response takes priority over normal routing.
    # --------------------------------------------------------

    if detect_urgent_symptoms(
        user_message
    ):

        response = get_urgent_response()

        add_to_history(
            "user",
            user_message,
        )

        add_to_history(
            "assistant",
            response,
        )

        return response

    # --------------------------------------------------------
    # SEVERE / WORSENING ABDOMINAL PAIN
    # --------------------------------------------------------

    if detect_severe_abdominal_pain(
        user_message
    ):

        response = (
            "Severe or worsening abdominal pain can need prompt "
            "medical evaluation. Because the pain is severe or "
            "getting worse, please seek urgent medical care rather "
            "than relying only on this chat. If you also have fever, "
            "persistent vomiting, fainting, severe weakness, or "
            "significant bleeding, seek emergency care immediately."
        )

        add_to_history(
            "user",
            user_message,
        )

        add_to_history(
            "assistant",
            response,
        )

        return response

    # --------------------------------------------------------
    # NEW LOCAL ML ASSESSMENT
    # --------------------------------------------------------

    model_name = route_ml_request(
        user_message
    )

    if model_name:

        print(
            f"[HealthcareAI] Starting "
            f"{model_name} assessment..."
        )

        return start_local_assessment(
            model_name
        )

    # --------------------------------------------------------
    # GENERAL HEALTH CONVERSATION
    # --------------------------------------------------------

    return ask_groq(
        user_message
    )


# ============================================================
# BACKWARD-COMPATIBLE API ENTRY POINT
# ============================================================

def chat_with_healthcare_ai(
    user_message
):

    return healthcare_ai(
        user_message
    )


# ============================================================
# OPTIONAL RESET FUNCTION
# ============================================================

def reset_healthcare_ai():

    global active_assessment_session_id

    if active_assessment_session_id:

        try:

            cancel_session(
                active_assessment_session_id
            )

        except Exception as error:

            print(
                "[HealthcareAI] Session reset error:",
                repr(error),
            )

    active_assessment_session_id = None

    clear_chat_history()

    return True


# ============================================================
# CLI
# ============================================================

def main():

    print(
        "\nHealthcareAI is ready."
    )

    print(
        f"Model: {GROQ_MODEL}"
    )

    print(
        f"Using .env: {ENV_PATH}"
    )

    print(
        "Local ML assessment engine ready."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        try:

            user_message = input(
                "\nYou: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\nHealthcareAI stopped."
            )

            break

        except EOFError:

            print(
                "\nHealthcareAI stopped."
            )

            break

        if user_message.lower() in [

            "exit",

            "quit",

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
