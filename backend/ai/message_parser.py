import re


# ============================================================
# MESSAGE PARSER
# ============================================================
#
# Converts natural language answers from the user into values
# that can be passed to the ML models.
#
# Examples:
#
# "yes"                  -> 1
# "no"                   -> 0
# "20"                   -> 20
# "177 cm, 85 kg"        -> BMI
# "male"                 -> Male
# "female"               -> Female
#
# The parser is intentionally generic so it can support
# multiple HealthcareAI models.
# ============================================================


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):

    if value is None:

        return ""

    return str(
        value
    ).strip().lower()


# ============================================================
# EXTRACT NUMBERS
# ============================================================

def extract_numbers(text):

    text = normalize_text(
        text
    )

    matches = re.findall(
        r"-?\d+(?:\.\d+)?",
        text
    )

    numbers = []

    for match in matches:

        try:

            number = float(
                match
            )

            if number.is_integer():

                number = int(
                    number
                )

            numbers.append(
                number
            )

        except ValueError:

            continue

    return numbers


# ============================================================
# PARSE YES / NO
# ============================================================

def parse_yes_no(text):

    text = normalize_text(
        text
    )

    yes_patterns = [

        "yes",

        "yeah",

        "yep",

        "yup",

        "sure",

        "correct",

        "true",

        "i do",

        "i have",

        "i am",

        "positive"

    ]

    no_patterns = [

        "no",

        "nope",

        "nah",

        "false",

        "never",

        "negative",

        "i don't",

        "i dont",

        "do not",

        "haven't",

        "havent"

    ]

    for pattern in no_patterns:

        if pattern in text:

            return 0

    for pattern in yes_patterns:

        if pattern in text:

            return 1

    return None


# ============================================================
# PARSE INTEGER
# ============================================================

def parse_integer(
    text,
    minimum=None,
    maximum=None
):

    numbers = extract_numbers(
        text
    )

    if not numbers:

        return None

    value = int(
        numbers[0]
    )

    if (
        minimum is not None
        and value < minimum
    ):

        return None

    if (
        maximum is not None
        and value > maximum
    ):

        return None

    return value


# ============================================================
# PARSE FLOAT
# ============================================================

def parse_float(
    text,
    minimum=None,
    maximum=None
):

    numbers = extract_numbers(
        text
    )

    if not numbers:

        return None

    value = float(
        numbers[0]
    )

    if (
        minimum is not None
        and value < minimum
    ):

        return None

    if (
        maximum is not None
        and value > maximum
    ):

        return None

    return value


# ============================================================
# PARSE AGE
# ============================================================

def parse_age(text):

    return parse_integer(
        text,
        minimum=1,
        maximum=120
    )


# ============================================================
# PARSE HEIGHT AND WEIGHT
# ============================================================

def parse_height_weight(text):

    text = normalize_text(
        text
    )

    height_match = re.search(

        r"(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters|m|meter|meters)",

        text

    )

    weight_match = re.search(

        r"(\d+(?:\.\d+)?)\s*(kg|kilogram|kilograms)",

        text

    )

    # --------------------------------------------------------
    # CASE 1:
    # Explicit units
    #
    # Example:
    # 177 cm, 85 kg
    # --------------------------------------------------------

    if (
        height_match
        and weight_match
    ):

        height = float(
            height_match.group(1)
        )

        height_unit = (
            height_match.group(2)
            .lower()
        )

        weight = float(
            weight_match.group(1)
        )

        if height_unit in [

            "m",

            "meter",

            "meters"

        ]:

            height = height * 100

    else:

        # ----------------------------------------------------
        # CASE 2:
        # Two numbers without units
        #
        # Example:
        # 177 85
        # ----------------------------------------------------

        numbers = extract_numbers(
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

        if height < 3:

            height = height * 100

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        height < 50
        or height > 250
    ):

        return None

    if (
        weight < 20
        or weight > 400
    ):

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
        )

    }


# ============================================================
# PARSE SEX
# ============================================================

def parse_sex(text):

    text = normalize_text(
        text
    )

    if (

        "female" in text

        or text == "f"

        or "woman" in text

        or "girl" in text

    ):

        return "Female"

    if (

        "male" in text

        or text == "m"

        or "man" in text

        or "boy" in text

    ):

        return "Male"

    return None


# ============================================================
# PARSE GENDER
# ============================================================

def parse_gender(text):

    value = parse_sex(
        text
    )

    if value is None:

        return None

    return value


# ============================================================
# PARSE BINARY FEATURE
# ============================================================

def parse_binary(text):

    return parse_yes_no(
        text
    )


# ============================================================
# PARSE GENERAL CATEGORY
# ============================================================

def parse_category(
    text,
    allowed_values
):

    text = normalize_text(
        text
    )

    if not allowed_values:

        return None

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    for value in allowed_values:

        value_text = normalize_text(
            value
        )

        if text == value_text:

            return value

    # --------------------------------------------------------
    # Text contains category
    # --------------------------------------------------------

    for value in allowed_values:

        value_text = normalize_text(
            value
        )

        if value_text in text:

            return value

    # --------------------------------------------------------
    # Category appears inside allowed value
    # --------------------------------------------------------

    for value in allowed_values:

        value_text = normalize_text(
            value
        )

        if text in value_text:

            return value

    return None


# ============================================================
# PARSE CHEST PAIN TYPE
# ============================================================

def parse_chest_pain(text):

    text = normalize_text(
        text
    )

    numbers = extract_numbers(
        text
    )

    if numbers:

        value = int(
            numbers[0]
        )

        if 0 <= value <= 3:

            return value

    mappings = {

        "typical angina": 0,

        "atypical angina": 1,

        "non-anginal": 2,

        "non anginal": 2,

        "asymptomatic": 3

    }

    for keyword, value in mappings.items():

        if keyword in text:

            return value

    return None


# ============================================================
# PARSE GENERAL HEALTH
# ============================================================

def parse_general_health(text):

    text = normalize_text(
        text
    )

    mapping = {

        "excellent": 1,

        "very good": 2,

        "good": 3,

        "fair": 4,

        "poor": 5

    }

    for keyword, value in mapping.items():

        if keyword in text:

            return value

    return None


# ============================================================
# PARSE DAYS
# ============================================================

def parse_days(text):

    return parse_integer(
        text,
        minimum=0,
        maximum=30
    )


# ============================================================
# PARSE BLOOD PRESSURE
# ============================================================

def parse_blood_pressure(text):

    numbers = extract_numbers(
        text
    )

    if not numbers:

        return None

    value = float(
        numbers[0]
    )

    if (
        value < 40
        or value > 300
    ):

        return None

    return value


# ============================================================
# PARSE CHOLESTEROL
# ============================================================

def parse_cholesterol(text):

    numbers = extract_numbers(
        text
    )

    if not numbers:

        return None

    value = float(
        numbers[0]
    )

    if (
        value < 50
        or value > 1000
    ):

        return None

    return value


# ============================================================
# PARSE GLUCOSE
# ============================================================

def parse_glucose(text):

    numbers = extract_numbers(
        text
    )

    if not numbers:

        return None

    value = float(
        numbers[0]
    )

    if (
        value < 20
        or value > 1000
    ):

        return None

    return value


# ============================================================
# PARSE GENERIC NUMERIC VALUE
# ============================================================

def parse_numeric_feature(
    text,
    feature_name=None
):

    value = parse_float(
        text
    )

    if value is None:

        return None

    return value


# ============================================================
# FEATURE-SPECIFIC PARSER
# ============================================================

def parse_feature_answer(
    feature,
    user_message,
    allowed_values=None
):

    if not isinstance(
        user_message,
        str
    ):

        user_message = str(
            user_message
        )

    feature_lower = normalize_text(
        feature
    )

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if feature_lower == "age":

        return parse_age(
            user_message
        )

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    if feature_lower == "bmi":

        result = parse_height_weight(
            user_message
        )

        if result is None:

            return parse_float(
                user_message,
                minimum=10,
                maximum=100
            )

        return result[
            "BMI"
        ]

    # --------------------------------------------------------
    # SEX / GENDER
    # --------------------------------------------------------

    if feature_lower in [

        "sex",

        "gender"

    ]:

        parsed = parse_sex(
            user_message
        )

        if parsed is None:

            return None

        if allowed_values:

            category = parse_category(
                parsed,
                allowed_values
            )

            if category is not None:

                return category

        return parsed

    # --------------------------------------------------------
    # CHEST PAIN
    # --------------------------------------------------------

    if feature_lower == "cp":

        return parse_chest_pain(
            user_message
        )

    # --------------------------------------------------------
    # GENERAL HEALTH
    # --------------------------------------------------------

    if feature_lower == "genhlth":

        return parse_general_health(
            user_message
        )

    # --------------------------------------------------------
    # DAYS FEATURES
    # --------------------------------------------------------

    if feature_lower in [

        "menthlth",

        "physhlth"

    ]:

        return parse_days(
            user_message
        )

    # --------------------------------------------------------
    # HEIGHT / WEIGHT RELATED
    # --------------------------------------------------------

    if feature_lower in [

        "height",

        "weight"

    ]:

        result = parse_height_weight(
            user_message
        )

        if result is not None:

            return result.get(
                feature_lower
            )

    # --------------------------------------------------------
    # BLOOD PRESSURE
    # --------------------------------------------------------

    if feature_lower in [

        "trestbps",

        "bloodpressure",

        "blood_pressure",

        "avg_glucose_level"

    ]:

        return parse_numeric_feature(
            user_message,
            feature
        )

    # --------------------------------------------------------
    # CHOLESTEROL
    # --------------------------------------------------------

    if feature_lower in [

        "chol",

        "cholesterol"

    ]:

        return parse_cholesterol(
            user_message
        )

    # --------------------------------------------------------
    # COMMON BINARY FEATURES
    # --------------------------------------------------------

    binary_features = [

        "highbp",

        "highchol",

        "cholcheck",

        "smoker",

        "stroke",

        "heartdiseaseorattack",

        "physactivity",

        "fruits",

        "veggies",

        "hvyalcoholconsump",

        "anyhealthcare",

        "nodocbccost",

        "diffwalk",

        "hypertension",

        "heart_disease",

        "ever_married",

        "work_type",

        "residence_type"

    ]

    if feature_lower in binary_features:

        value = parse_binary(
            user_message
        )

        if value is not None:

            return value

    # --------------------------------------------------------
    # ALLOWED CATEGORIES
    # --------------------------------------------------------

    if allowed_values:

        category = parse_category(
            user_message,
            allowed_values
        )

        if category is not None:

            return category

    # --------------------------------------------------------
    # TRY YES / NO
    # --------------------------------------------------------

    binary_value = parse_yes_no(
        user_message
    )

    if binary_value is not None:

        return binary_value

    # --------------------------------------------------------
    # TRY NUMERIC
    # --------------------------------------------------------

    numeric_value = parse_numeric_feature(
        user_message,
        feature
    )

    if numeric_value is not None:

        return numeric_value

    # --------------------------------------------------------
    # RAW TEXT FALLBACK
    # --------------------------------------------------------

    cleaned = user_message.strip()

    if cleaned:

        return cleaned

    return None


# ============================================================
# VALIDATE PARSED VALUE
# ============================================================

def is_valid_parsed_value(value):

    if value is None:

        return False

    if isinstance(
        value,
        str
    ):

        return bool(
            value.strip()
        )

    return True


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def parse_message(
    feature,
    user_message,
    allowed_values=None
):

    value = parse_feature_answer(

        feature=feature,

        user_message=user_message,

        allowed_values=allowed_values

    )

    return {

        "success": is_valid_parsed_value(
            value
        ),

        "feature": feature,

        "value": value,

        "raw_message": user_message

    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    tests = [

        (
            "HighBP",
            "yes",
            None
        ),

        (
            "HighChol",
            "no",
            None
        ),

        (
            "age",
            "I am 20 years old",
            None
        ),

        (
            "BMI",
            "177 cm and 85 kg",
            None
        ),

        (
            "sex",
            "male",
            ["Male", "Female"]
        ),

        (
            "cp",
            "atypical angina",
            None
        ),

        (
            "trestbps",
            "120",
            None
        ),

        (
            "chol",
            "180 mg/dL",
            None
        ),

        (
            "MentHlth",
            "5 days",
            None
        )

    ]

    print(
        "\nHealthcareAI Message Parser Test\n"
    )

    for feature, message, allowed_values in tests:

        result = parse_message(

            feature,

            message,

            allowed_values

        )

        print(
            f"Feature: {feature}"
        )

        print(
            f"User: {message}"
        )

        print(
            f"Parsed: {result['value']}"
        )

        print()

    print(
        "Done."
    )
