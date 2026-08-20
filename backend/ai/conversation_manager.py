import re
import uuid

from backend.ai.assessment_manager import (
    build_assessment_questions,
    get_assessment_info,
)

from backend.ml.assessment_engine import (
    predict,
)


# ============================================================
# ACTIVE CONVERSATION SESSIONS
# ============================================================

SESSIONS = {}


# ============================================================
# SESSION STRUCTURE
# ============================================================

def create_session(model_name):

    assessment_info = get_assessment_info(
        model_name
    )

    session_id = str(
        uuid.uuid4()
    )

    session = {

        "session_id": session_id,

        "model": model_name,

        "questions": assessment_info[
            "questions"
        ],

        "total_questions": assessment_info[
            "total_questions"
        ],

        "current_index": 0,

        "answers": {},

        "active": True,
    }

    SESSIONS[
        session_id
    ] = session

    return session


# ============================================================
# GET SESSION
# ============================================================

def get_session(session_id):

    if not session_id:

        return None

    return SESSIONS.get(
        session_id
    )


# ============================================================
# DELETE SESSION
# ============================================================

def delete_session(session_id):

    if session_id in SESSIONS:

        del SESSIONS[
            session_id
        ]

        return True

    return False


# ============================================================
# CANCEL SESSION
# ============================================================

def cancel_session(session_id):

    session = get_session(
        session_id
    )

    if not session:

        return {

            "success": False,

            "error": "Assessment session not found."
        }

    delete_session(
        session_id
    )

    return {

        "success": True,

        "cancelled": True,

        "message": (
            "Your assessment has been cancelled."
        )
    }


# ============================================================
# GET CURRENT QUESTION
# ============================================================

def get_current_question(session):

    current_index = session[
        "current_index"
    ]

    questions = session[
        "questions"
    ]

    if current_index >= len(
        questions
    ):

        return None

    return questions[
        current_index
    ]


# ============================================================
# FORMAT QUESTION RESPONSE
# ============================================================

def format_question_response(session):

    question = get_current_question(
        session
    )

    if question is None:

        return {

            "success": False,

            "error": (
                "No more questions available."
            )
        }

    question_number = (
        session["current_index"] + 1
    )

    total_questions = session[
        "total_questions"
    ]

    return {

        "success": True,

        "session_id": session[
            "session_id"
        ],

        "active": True,

        "model": session[
            "model"
        ],

        "question_number": question_number,

        "total_questions": total_questions,

        "feature": question[
            "feature"
        ],

        "question": question[
            "question"
        ],
    }


# ============================================================
# START ASSESSMENT
# ============================================================

def start_assessment(model_name):

    try:

        session = create_session(
            model_name
        )

        response = format_question_response(
            session
        )

        response[
            "message"
        ] = (
            f"Let's begin the assessment. "
            f"I'll ask {session['total_questions']} "
            f"questions."
        )

        return response

    except Exception as error:

        return {

            "success": False,

            "error": (
                f"Could not start assessment: "
                f"{str(error)}"
            )
        }


# ============================================================
# EXTRACT NUMBER
# ============================================================

def parse_number(text):

    if text is None:

        return None

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        str(text)
    )

    if not match:

        return None

    value = float(
        match.group()
    )

    if value.is_integer():

        return int(
            value
        )

    return value


# ============================================================
# PARSE YES / NO
# ============================================================

def parse_yes_no(text):

    text = str(
        text
    ).lower().strip()

    yes_words = [

        "yes",
        "yeah",
        "yep",
        "yup",
        "true",
        "correct",
        "i do",
        "i have",
    ]

    no_words = [

        "no",
        "nope",
        "nah",
        "false",
        "never",
        "i don't",
        "i dont",
    ]

    if text in [

        "1",
        "yes",
        "true"
    ]:

        return 1

    if text in [

        "0",
        "no",
        "false"
    ]:

        return 0

    for word in yes_words:

        if word in text:

            return 1

    for word in no_words:

        if word in text:

            return 0

    return None


# ============================================================
# PARSE SEX
# ============================================================

def parse_sex(text):

    text = str(
        text
    ).lower().strip()

    if text in [

        "male",
        "m"
    ]:

        return "Male"

    if text in [

        "female",
        "f"
    ]:

        return "Female"

    return None


# ============================================================
# PARSE HEIGHT AND WEIGHT
# ============================================================

def parse_height_weight(text):

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        str(text)
    )

    if len(numbers) < 2:

        return None

    height = float(
        numbers[0]
    )

    weight = float(
        numbers[1]
    )

    # Convert meters to centimeters

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

        "height": round(
            height,
            2
        ),

        "weight": round(
            weight,
            2
        ),

        "BMI": round(
            bmi,
            2
        ),
    }


# ============================================================
# PARSE HEALTH RATING
# ============================================================

def parse_health_rating(text):

    text = str(
        text
    ).lower()

    mapping = {

        "excellent": 1,

        "very good": 2,

        "good": 3,

        "fair": 4,

        "poor": 5,
    }

    for word, value in mapping.items():

        if word in text:

            return value

    return None


# ============================================================
# PARSE EDUCATION
# ============================================================

def parse_education(text):

    text = str(
        text
    ).lower()

    if (
        "postgraduate" in text
        or "master" in text
        or "phd" in text
    ):

        return 6

    if (
        "college" in text
        or "university" in text
        or "bachelor" in text
    ):

        return 5

    if (
        "high school" in text
        or "12" in text
    ):

        return 4

    if "school" in text:

        return 3

    return None


# ============================================================
# PARSE INCOME
# ============================================================

def parse_income(text):

    text = str(
        text
    ).lower()

    mapping = {

        "very high": 8,

        "upper-middle": 6,

        "upper middle": 6,

        "middle": 5,

        "lower-middle": 3,

        "lower middle": 3,

        "low": 2,

        "high": 7,
    }

    for word, value in mapping.items():

        if word in text:

            return value

    return None


# ============================================================
# PARSE ANSWER
# ============================================================

def parse_answer(question, user_message):

    question_type = question.get(
        "type",
        "number"
    )

    # --------------------------------------------------------
    # YES / NO
    # --------------------------------------------------------

    if question_type == "yes_no":

        return parse_yes_no(
            user_message
        )

    # --------------------------------------------------------
    # SEX
    # --------------------------------------------------------

    if question_type == "sex":

        return parse_sex(
            user_message
        )

    # --------------------------------------------------------
    # HEIGHT / WEIGHT
    # --------------------------------------------------------

    if question_type == "height_weight":

        return parse_height_weight(
            user_message
        )

    # --------------------------------------------------------
    # HEALTH RATING
    # --------------------------------------------------------

    if question_type == "health_rating":

        return parse_health_rating(
            user_message
        )

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    if question_type == "education":

        return parse_education(
            user_message
        )

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    if question_type == "income":

        return parse_income(
            user_message
        )

    # --------------------------------------------------------
    # DEFAULT NUMBER
    # --------------------------------------------------------

    return parse_number(
        user_message
    )


# ============================================================
# VALIDATE ANSWER
# ============================================================

def validate_answer(question, value):

    if value is None:

        return False, (
            "I couldn't understand that answer. "
            "Please try again."
        )

    question_type = question.get(
        "type"
    )

    # --------------------------------------------------------
    # NUMERIC VALIDATION
    # --------------------------------------------------------

    if question_type == "number":

        minimum = question.get(
            "min"
        )

        maximum = question.get(
            "max"
        )

        if minimum is not None:

            if value < minimum:

                return False, (
                    f"Please enter a value of at least "
                    f"{minimum}."
                )

        if maximum is not None:

            if value > maximum:

                return False, (
                    f"Please enter a value no greater than "
                    f"{maximum}."
                )

    # --------------------------------------------------------
    # HEIGHT / WEIGHT VALIDATION
    # --------------------------------------------------------

    if question_type == "height_weight":

        if not isinstance(
            value,
            dict
        ):

            return False, (
                "Please provide both height and weight. "
                "For example: 177 cm, 85 kg."
            )

    return True, None


# ============================================================
# STORE ANSWER
# ============================================================

def store_answer(
    session,
    question,
    value
):

    feature = question[
        "feature"
    ]

    # --------------------------------------------------------
    # BMI QUESTION
    # --------------------------------------------------------

    if question.get(
        "type"
    ) == "height_weight":

        session[
            "answers"
        ]["BMI"] = value[
            "BMI"
        ]

        # Store extra values too.
        # These will only be used if needed.

        session[
            "answers"
        ]["height"] = value[
            "height"
        ]

        session[
            "answers"
        ]["weight"] = value[
            "weight"
        ]

        return

    # --------------------------------------------------------
    # NORMAL FEATURE
    # --------------------------------------------------------

    session[
        "answers"
    ][feature] = value


# ============================================================
# COMPLETE ASSESSMENT
# ============================================================

def complete_assessment(session):

    model_name = session[
        "model"
    ]

    answers = session[
        "answers"
    ]

    try:

        result = predict(
            model_name,
            answers
        )

        probability = result.get(
            "probability"
        )

        if probability is not None:

            probability_percent = round(
                probability * 100,
                2
            )

        else:

            probability_percent = None

        session[
            "active"
        ] = False

        delete_session(
            session[
                "session_id"
            ]
        )

        return {

            "success": True,

            "completed": True,

            "active": False,

            "model": model_name,

            "prediction": result.get(
                "prediction"
            ),

            "probability": probability,

            "probability_percent": probability_percent,

            "features": result.get(
                "features"
            ),

            "message": (
                "Assessment completed successfully. "
                "This result is an AI-based model estimate "
                "and is not a medical diagnosis."
            ),
        }

    except Exception as error:

        return {

            "success": False,

            "completed": False,

            "error": (
                f"Prediction failed: {str(error)}"
            )
        }


# ============================================================
# HANDLE USER ANSWER
# ============================================================

def handle_answer(
    session_id,
    user_message
):

    session = get_session(
        session_id
    )

    if session is None:

        return {

            "success": False,

            "error": (
                "Assessment session not found."
            )
        }

    if not session.get(
        "active"
    ):

        return {

            "success": False,

            "error": (
                "This assessment is no longer active."
            )
        }

    # --------------------------------------------------------
    # CURRENT QUESTION
    # --------------------------------------------------------

    question = get_current_question(
        session
    )

    if question is None:

        return complete_assessment(
            session
        )

    # --------------------------------------------------------
    # PARSE ANSWER
    # --------------------------------------------------------

    value = parse_answer(
        question,
        user_message
    )

    # --------------------------------------------------------
    # VALIDATE ANSWER
    # --------------------------------------------------------

    valid, error_message = validate_answer(
        question,
        value
    )

    if not valid:

        return {

            "success": False,

            "session_id": session_id,

            "active": True,

            "error": error_message,

            "question": question[
                "question"
            ],

            "question_number": (
                session[
                    "current_index"
                ] + 1
            ),

            "total_questions": session[
                "total_questions"
            ],
        }

    # --------------------------------------------------------
    # STORE ANSWER
    # --------------------------------------------------------

    store_answer(
        session,
        question,
        value
    )

    # --------------------------------------------------------
    # NEXT QUESTION
    # --------------------------------------------------------

    session[
        "current_index"
    ] += 1

    if session[
        "current_index"
    ] >= session[
        "total_questions"
    ]:

        return complete_assessment(
            session
        )

    return format_question_response(
        session
    )


# ============================================================
# GET SESSION STATUS
# ============================================================

def get_session_status(session_id):

    session = get_session(
        session_id
    )

    if session is None:

        return {

            "success": False,

            "error": (
                "Assessment session not found."
            )
        }

    return {

        "success": True,

        "session_id": session[
            "session_id"
        ],

        "model": session[
            "model"
        ],

        "active": session[
            "active"
        ],

        "current_question": (
            session[
                "current_index"
            ] + 1
        ),

        "total_questions": session[
            "total_questions"
        ],

        "answers_collected": len(
            session[
                "answers"
            ]
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nHealthcareAI Conversation Manager Test\n"
    )

    response = start_assessment(
        "diabetes_binary"
    )

    print(
        "START RESPONSE:"
    )

    print(
        response
    )

    if response.get(
        "success"
    ):

        session_id = response[
            "session_id"
        ]

        print(
            "\nSESSION STATUS:"
        )

        print(
            get_session_status(
                session_id
            )
        )

        print(
            "\nCANCELLING TEST SESSION:"
        )

        print(
            cancel_session(
                session_id
            )
        )

    print(
        "\nDone."
    )
