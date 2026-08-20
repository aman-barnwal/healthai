"""
HealthcareAI Message Parser

Converts natural-language user answers into safe values for ML models.

Design goals:
- Never crash on bad user input.
- Never pass NaN, infinity, or arbitrary strings accidentally.
- Handle common yes/no, numeric, BMI, age, sex, health,
  education, income, chest pain, and categorical answers.
- Return None when an answer cannot be safely interpreted.
"""

import math
import re
from numbers import Number


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Convert input into a clean lowercase string.

    Returns an empty string for None.
    """

    if value is None:
        return ""

    try:
        return str(value).strip().lower()
    except Exception:
        return ""


# ============================================================
# SAFE NUMERIC CHECK
# ============================================================

def is_valid_number(value):
    """
    Return True only for finite numeric values.
    """

    if isinstance(value, bool):
        return False

    if not isinstance(value, Number):
        return False

    try:
        return math.isfinite(float(value))
    except Exception:
        return False


# ============================================================
# YES / NO PARSER
# ============================================================

def parse_yes_no(value):

    text = normalize_text(value)

    yes_values = {
        "yes",
        "y",
        "yeah",
        "yep",
        "true",
        "1",
        "sure",
        "correct",
        "of course",
        "affirmative",
        "haan",
        "ha",
        "han",
        "h",
    }

    no_values = {
        "no",
        "n",
        "nah",
        "nope",
        "false",
        "0",
        "ni",
        "na",
        "nahi",
        "nahin",
    }

    if text in yes_values:
        return 1

    if text in no_values:
        return 0

    return None


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_number(value):
    """
    Extract the first valid number from input.

    Examples:
        "I am 20 years old" -> 20.0
        "120 mm Hg" -> 120.0
        "85.5 kg" -> 85.5
    """

    if value is None:
        return None

    if is_valid_number(value):
        return float(value)

    text = normalize_text(value)

    if not text:
        return None

    match = re.search(
        r"(?<![\w.])-?\d+(?:\.\d+)?",
        text
    )

    if not match:
        return None

    try:
        number = float(match.group())

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError, OverflowError):
        return None


# ============================================================
# INTEGER PARSER
# ============================================================

def parse_integer(value):

    number = extract_number(value)

    if number is None:
        return None

    try:
        return int(round(number))
    except (TypeError, ValueError, OverflowError):
        return None


# ============================================================
# FLOAT PARSER
# ============================================================

def parse_float(value):

    number = extract_number(value)

    if number is None:
        return None

    try:
        number = float(number)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError, OverflowError):
        return None


# ============================================================
# AGE PARSER
# ============================================================

def parse_age(value):

    age = parse_integer(value)

    if age is None:
        return None

    if age < 0 or age > 120:
        return None

    return age


# ============================================================
# AGE CATEGORY PARSER
# ============================================================

def parse_age_category(value):
    """
    Supports datasets where age is represented as a category.

    If the user enters a normal age, converts approximately to
    the common BRFSS age category scheme.

    BRFSS categories:
        1 = 18-24
        2 = 25-29
        3 = 30-34
        ...
        13 = 80+
    """

    age = parse_age(value)

    if age is None:
        return None

    if age < 18:
        return None

    if age <= 24:
        return 1

    if age >= 80:
        return 13

    return min(13, ((age - 25) // 5) + 2)


# ============================================================
# BMI PARSER
# ============================================================

def parse_bmi(value):

    text = normalize_text(value)

    if not text:
        return None

    # Direct BMI
    bmi_match = re.search(
        r"\bbmi\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        text
    )

    if bmi_match:

        try:
            bmi = float(bmi_match.group(1))

            if math.isfinite(bmi) and 5 <= bmi <= 100:
                return round(bmi, 2)

        except (TypeError, ValueError, OverflowError):
            pass

    # Metric height
    height_cm_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)",
        text
    )

    # Weight
    weight_kg_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilogram|kilograms)",
        text
    )

    if height_cm_match and weight_kg_match:

        try:

            height_cm = float(
                height_cm_match.group(1)
            )

            weight_kg = float(
                weight_kg_match.group(1)
            )

            if (
                height_cm <= 0
                or weight_kg <= 0
                or height_cm > 300
                or weight_kg > 500
            ):
                return None

            height_m = height_cm / 100

            bmi = weight_kg / (height_m ** 2)

            if math.isfinite(bmi) and 5 <= bmi <= 100:
                return round(bmi, 2)

        except (
            TypeError,
            ValueError,
            OverflowError,
            ZeroDivisionError,
        ):
            return None

    # Direct number as BMI
    number = extract_number(text)

    if number is not None:

        if 5 <= number <= 100:
            return round(number, 2)

    return None


# ============================================================
# SEX / GENDER PARSER
# ============================================================

def parse_sex(value):
    """
    Numeric encoding used by numeric datasets:

        male   -> 1
        female -> 0
    """

    text = normalize_text(value)

    male_values = {
        "male",
        "m",
        "man",
        "boy",
    }

    female_values = {
        "female",
        "f",
        "woman",
        "girl",
    }

    if text in male_values:
        return 1

    if text in female_values:
        return 0

    # Already numeric
    number = parse_integer(value)

    if number in {0, 1}:
        return number

    return None


# ============================================================
# GENERAL HEALTH
# ============================================================

def parse_general_health(value):

    text = normalize_text(value)

    mapping = {

        "excellent": 1,

        "very good": 2,
        "verygood": 2,

        "good": 3,

        "fair": 4,

        "poor": 5,
        "bad": 5,

    }

    return mapping.get(text)


# ============================================================
# EDUCATION
# ============================================================

def parse_education(value):

    text = normalize_text(value)

    mapping = {

        "never attended school": 1,
        "no schooling": 1,

        "elementary": 2,
        "primary": 2,

        "middle school": 3,

        "high school": 4,
        "school": 4,
        "secondary": 4,

        "college": 5,
        "undergraduate": 5,
        "bachelor": 5,
        "bachelors": 5,

        "postgraduate": 6,
        "masters": 6,
        "master": 6,
        "phd": 6,
        "doctorate": 6,

    }

    if text in mapping:
        return mapping[text]

    number = parse_integer(value)

    if number is not None and 1 <= number <= 6:
        return number

    return None


# ============================================================
# INCOME
# ============================================================

def parse_income(value):

    text = normalize_text(value)

    mapping = {

        "low": 1,
        "lower": 1,

        "lower middle": 2,
        "lower-middle": 2,
        "lower middle class": 2,

        "middle": 3,
        "middle class": 3,

        "upper middle": 4,
        "upper-middle": 4,
        "upper middle class": 4,

        "high": 5,
        "rich": 5,
        "upper": 5,

    }

    if text in mapping:
        return mapping[text]

    number = parse_integer(value)

    if number is not None and 1 <= number <= 5:
        return number

    return None


# ============================================================
# CHEST PAIN TYPE
# ============================================================

def parse_chest_pain(value):

    text = normalize_text(value)

    direct_number = parse_integer(text)

    if direct_number is not None:

        if 0 <= direct_number <= 3:
            return direct_number

    mapping = {

        "typical angina": 0,
        "typical": 0,

        "atypical angina": 1,
        "atypical": 1,

        "non anginal pain": 2,
        "non-anginal pain": 2,
        "nonanginal pain": 2,

        "asymptomatic": 3,
        "no chest pain": 3,

    }

    return mapping.get(text)


# ============================================================
# DAYS PARSER
# ============================================================

def parse_days(value):

    number = parse_integer(value)

    if number is None:
        return None

    if number < 0:
        return None

    if number > 30:
        return None

    return number


# ============================================================
# EVER MARRIED
# ============================================================

def parse_ever_married(value):

    text = normalize_text(value)

    mapping = {
        "yes": 1,
        "married": 1,
        "currently married": 1,

        "no": 0,
        "single": 0,
        "unmarried": 0,
    }

    if text in mapping:
        return mapping[text]

    return parse_yes_no(value)


# ============================================================
# RESIDENCE TYPE
# ============================================================

def parse_residence_type(value):

    text = normalize_text(value)

    mapping = {

        "urban": 1,
        "city": 1,
        "town": 1,

        "rural": 0,
        "village": 0,

    }

    return mapping.get(text)


# ============================================================
# WORK TYPE
# ============================================================

def parse_work_type(value):
    """
    Numeric fallback encoding.

    IMPORTANT:
    If a specific model was trained using OneHotEncoder and
    expects strings, the assessment layer should pass the
    original category instead of this numeric value.
    """

    text = normalize_text(value)

    mapping = {

        "children": 0,
        "child": 0,

        "government": 1,
        "govt": 1,
        "govt job": 1,

        "private": 2,
        "private job": 2,

        "self employed": 3,
        "self-employed": 3,
        "business": 3,

        "never worked": 4,
        "unemployed": 4,

    }

    return mapping.get(text)


# ============================================================
# SMOKING STATUS
# ============================================================

def parse_smoking_status(value):

    text = normalize_text(value)

    mapping = {

        "never smoked": 0,
        "never": 0,
        "no": 0,

        "formerly smoked": 1,
        "former smoker": 1,
        "quit smoking": 1,

        "smokes": 2,
        "smoker": 2,
        "yes": 2,

        "unknown": 3,

    }

    return mapping.get(text)


# ============================================================
# BINARY FEATURE LIST
# ============================================================

YES_NO_FEATURES = {

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

}


# ============================================================
# FEATURE-SPECIFIC PARSER
# ============================================================

def parse_feature(feature, value):

    if value is None:
        return None

    try:

        feature_name = str(feature).strip()

        if not feature_name:
            return None

        feature_lower = feature_name.lower()

    except Exception:
        return None

    # --------------------------------------------------------
    # YES / NO
    # --------------------------------------------------------

    if feature_lower in YES_NO_FEATURES:
        return parse_yes_no(value)

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if feature_lower == "age":
        return parse_age(value)

    if feature_lower in {
        "age_category",
        "agecategory",
    }:
        return parse_age_category(value)

    # --------------------------------------------------------
    # SEX / GENDER
    # --------------------------------------------------------

    if feature_lower in {
        "sex",
        "gender",
    }:
        return parse_sex(value)

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    if feature_lower == "bmi":
        return parse_bmi(value)

    # --------------------------------------------------------
    # GENERAL HEALTH
    # --------------------------------------------------------

    if feature_lower in {
        "genhlth",
        "general_health",
        "generalhealth",
    }:
        return parse_general_health(value)

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    if feature_lower in {
        "education",
        "educationlevel",
        "education_level",
    }:
        return parse_education(value)

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    if feature_lower in {
        "income",
        "income_category",
        "incomelevel",
    }:
        return parse_income(value)

    # --------------------------------------------------------
    # CHEST PAIN
    # --------------------------------------------------------

    if feature_lower in {
        "cp",
        "chest_pain",
        "chestpaintype",
    }:
        return parse_chest_pain(value)

    # --------------------------------------------------------
    # DAYS
    # --------------------------------------------------------

    if feature_lower in {
        "menthlth",
        "physhlth",
        "mental_health_days",
        "physical_health_days",
    }:
        return parse_days(value)

    # --------------------------------------------------------
    # MARRIAGE
    # --------------------------------------------------------

    if feature_lower == "ever_married":
        return parse_ever_married(value)

    # --------------------------------------------------------
    # RESIDENCE
    # --------------------------------------------------------

    if feature_lower in {
        "residence_type",
        "residencetype",
    }:
        return parse_residence_type(value)

    # --------------------------------------------------------
    # WORK TYPE
    # --------------------------------------------------------

    if feature_lower in {
        "work_type",
        "worktype",
    }:
        return parse_work_type(value)

    # --------------------------------------------------------
    # SMOKING STATUS
    # --------------------------------------------------------

    if feature_lower in {
        "smoking_status",
        "smokingstatus",
    }:
        return parse_smoking_status(value)

    # --------------------------------------------------------
    # NUMERIC FALLBACK
    # --------------------------------------------------------

    number = extract_number(value)

    if number is not None:
        return number

    return None


# ============================================================
# PARSE USER MESSAGE
# ============================================================

def parse_user_message(feature, message):

    try:
        return parse_feature(
            feature,
            message,
        )

    except Exception:
        # Absolute safety barrier.
        return None


# ============================================================
# VALIDATE PARSED VALUE
# ============================================================

def validate_parsed_value(feature, value):

    if value is None:
        return False

    try:
        feature_lower = str(feature).lower()
    except Exception:
        return False

    # Reject all strings.
    if isinstance(value, str):
        return False

    # Reject booleans.
    if isinstance(value, bool):
        return False

    # Only finite numeric values are allowed.
    if not is_valid_number(value):
        return False

    numeric_value = float(value)

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if feature_lower == "age":
        return 0 <= numeric_value <= 120

    # --------------------------------------------------------
    # AGE CATEGORY
    # --------------------------------------------------------

    if feature_lower in {
        "age_category",
        "agecategory",
    }:
        return 1 <= numeric_value <= 13

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    if feature_lower == "bmi":
        return 5 <= numeric_value <= 100

    # --------------------------------------------------------
    # DAYS
    # --------------------------------------------------------

    if feature_lower in {
        "menthlth",
        "physhlth",
        "mental_health_days",
        "physical_health_days",
    }:
        return 0 <= numeric_value <= 30

    # --------------------------------------------------------
    # BINARY
    # --------------------------------------------------------

    if feature_lower in YES_NO_FEATURES:
        return numeric_value in {0, 1}

    return True


# ============================================================
# MAIN SAFE PARSE + VALIDATE FUNCTION
# ============================================================

def parse_and_validate(feature, message):
    """
    Main public parser.

    Always returns:
        int
        float
        or None

    Never returns a string.
    Never raises parsing exceptions.
    """

    try:

        value = parse_user_message(
            feature,
            message,
        )

        if not validate_parsed_value(
            feature,
            value,
        ):
            return None

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        if numeric_value.is_integer():
            return int(numeric_value)

        return numeric_value

    except Exception:
        return None


# ============================================================
# TEST SUITE
# ============================================================

if __name__ == "__main__":

    tests = [

        ("HighBP", "yes", 1),
        ("HighChol", "No", 0),
        ("HighChol", "ni", 0),

        ("age", "I am 20 years old", 20),
        ("age", "999 years", None),

        ("BMI", "177 cm and 85 kg", 27.13),
        ("BMI", "bmi is 25.5", 25.5),
        ("BMI", "hello", None),

        ("sex", "male", 1),
        ("sex", "female", 0),
        ("sex", "banana", None),

        ("cp", "typical angina", 0),
        ("cp", "atypical angina", 1),
        ("cp", "non-anginal pain", 2),
        ("cp", "asymptomatic", 3),

        ("trestbps", "120 mm Hg", 120),
        ("chol", "180 mg/dL", 180),

        ("MentHlth", "5 days", 5),
        ("MentHlth", "45 days", None),

        ("Education", "college", 5),
        ("Income", "middle", 3),
        ("GenHlth", "very good", 2),

        ("ever_married", "married", 1),
        ("ever_married", "single", 0),

        ("residence_type", "urban", 1),
        ("residence_type", "rural", 0),

        ("work_type", "private job", 2),

        ("smoking_status", "never smoked", 0),
        ("smoking_status", "formerly smoked", 1),

        ("unknown_numeric", "abc", None),
        ("unknown_numeric", "NaN", None),

    ]

    print("\nHealthcareAI Message Parser Test\n")

    passed = 0
    failed = 0

    for feature, user_input, expected in tests:

        parsed = parse_and_validate(
            feature,
            user_input,
        )

        success = (
            parsed == expected
            or (
                isinstance(parsed, float)
                and isinstance(expected, float)
                and abs(parsed - expected) < 0.01
            )
        )

        status = "PASS" if success else "FAIL"

        print(f"[{status}]")
        print(f"Feature : {feature}")
        print(f"Input   : {user_input}")
        print(f"Parsed  : {parsed}")
        print(f"Expected: {expected}")
        print()

        if success:
            passed += 1
        else:
            failed += 1

    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total : {passed + failed}")
    print("=" * 50)

    if failed == 0:
        print("\nALL TESTS PASSED.\n")
    else:
        print("\nSOME TESTS FAILED.\n")
