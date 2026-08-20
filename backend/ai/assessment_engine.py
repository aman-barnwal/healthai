import math
import re


class DiabetesAssessment:

    def __init__(self):
        self.data = {}
        self.current_step = "age"

        self.steps = [
            "age",
            "height_weight",
            "sex",
            "high_bp",
            "high_chol",
            "smoker",
            "phys_activity",
            "general_health",
            "stroke",
            "heart_disease",
            "diff_walk",
            "mental_health",
            "physical_health",
            "fruits_veggies",
            "alcohol",
            "healthcare",
            "doctor_cost",
            "education",
            "income",
        ]

    def start(self):
        return (
            "Sure. I can give you an **AI-based diabetes risk "
            "estimate**, but it is not a medical diagnosis.\n\n"
            "Let's start with a few basics. How old are you?"
        )

    def is_complete(self):
        return self.current_step == "complete"

    def process(self, message):

        message = message.strip()

        if self.current_step == "age":
            return self._process_age(message)

        if self.current_step == "height_weight":
            return self._process_height_weight(message)

        if self.current_step == "sex":
            return self._process_sex(message)

        if self.current_step == "high_bp":
            return self._process_binary(
                message,
                "HighBP",
                "Got it. Do you have high cholesterol?",
                "high_chol"
            )

        if self.current_step == "high_chol":
            return self._process_binary(
                message,
                "HighChol",
                "Do you currently smoke, or have you smoked regularly?",
                "smoker"
            )

        if self.current_step == "smoker":
            return self._process_binary(
                message,
                "Smoker",
                "Do you usually do any physical activity or exercise?",
                "phys_activity"
            )

        if self.current_step == "phys_activity":
            return self._process_binary(
                message,
                "PhysActivity",
                "How would you rate your general health: excellent, very good, good, fair, or poor?",
                "general_health"
            )

        if self.current_step == "general_health":
            return self._process_general_health(message)

        if self.current_step == "stroke":
            return self._process_binary(
                message,
                "Stroke",
                "Have you ever had coronary heart disease or a heart attack?",
                "heart_disease"
            )

        if self.current_step == "heart_disease":
            return self._process_binary(
                message,
                "HeartDiseaseorAttack",
                "Do you have serious difficulty walking or climbing stairs?",
                "diff_walk"
            )

        if self.current_step == "diff_walk":
            return self._process_binary(
                message,
                "DiffWalk",
                "During the last 30 days, for about how many days was your mental health not good?",
                "mental_health"
            )

        if self.current_step == "mental_health":
            return self._process_days(
                message,
                "MentHlth",
                "How many days in the last 30 was your physical health not good?",
                "physical_health"
            )

        if self.current_step == "physical_health":
            return self._process_days(
                message,
                "PhysHlth",
                "Do you regularly eat fruits?",
                "fruits"
            )

        if self.current_step == "fruits_veggies":
            return self._process_fruits_veggies(message)

        if self.current_step == "alcohol":
            return self._process_binary(
                message,
                "HvyAlcoholConsump",
                "Do you currently have any healthcare coverage or health insurance?",
                "healthcare"
            )

        if self.current_step == "healthcare":
            return self._process_binary(
                message,
                "AnyHealthcare",
                "In the past year, was there a time you needed a doctor but couldn't see one because of cost?",
                "doctor_cost"
            )

        if self.current_step == "doctor_cost":
            return self._process_binary(
                message,
                "NoDocbcCost",
                "What is your highest education level? For example: school, high school, college, or postgraduate.",
                "education"
            )

        if self.current_step == "education":
            return self._process_education(message)

        if self.current_step == "income":
            return self._process_income(message)

        return "I didn't understand that. Could you try again?"

    # ============================================================
    # AGE
    # ============================================================

    def _process_age(self, message):

        numbers = re.findall(r"\d+", message)

        if not numbers:
            return "I didn't catch your age. How old are you?"

        age = int(numbers[0])

        if age < 1 or age > 120:
            return "That doesn't look like a valid age. How old are you?"

        self.data["actual_age"] = age
        self.data["Age"] = self._convert_age(age)

        self.current_step = "height_weight"

        return "Got it. What's your height and weight?"

    # ============================================================
    # HEIGHT + WEIGHT + BMI
    # ============================================================

    def _process_height_weight(self, message):

        numbers = re.findall(r"\d+(?:\.\d+)?", message)

        if len(numbers) < 2:
            return (
                "Please tell me both your height and weight, "
                "for example: **177 cm and 85 kg**."
            )

        first = float(numbers[0])
        second = float(numbers[1])

        if first > 100:
            height_cm = first
            weight_kg = second
        else:
            height_cm = second
            weight_kg = first

        if height_cm < 100 or height_cm > 250:
            return "I couldn't understand your height. Please give it in cm."

        if weight_kg < 20 or weight_kg > 400:
            return "I couldn't understand your weight. Please give it in kg."

        bmi = weight_kg / ((height_cm / 100) ** 2)

        self.data["height_cm"] = height_cm
        self.data["weight_kg"] = weight_kg
        self.data["BMI"] = round(bmi, 2)

        self.current_step = "sex"

        return (
            f"Got it. Your calculated BMI is **{bmi:.1f}**. "
            "What is your sex: male or female?"
        )

    # ============================================================
    # SEX
    # ============================================================

    def _process_sex(self, message):

        text = message.lower()

        if "male" in text:
            self.data["Sex"] = 1

        elif "female" in text:
            self.data["Sex"] = 0

        else:
            return "Please tell me male or female."

        self.current_step = "high_bp"

        return "Do you have high blood pressure?"

    # ============================================================
    # YES / NO
    # ============================================================

    def _process_binary(
        self,
        message,
        field,
        next_question,
        next_step
    ):

        value = self._parse_yes_no(message)

        if value is None:
            return "Please answer with yes or no."

        self.data[field] = value

        self.current_step = next_step

        return next_question

    def _parse_yes_no(self, message):

        text = message.lower().strip()

        yes_words = [
            "yes",
            "yeah",
            "yep",
            "yup",
            "i do",
            "i have",
            "correct",
            "sure"
        ]

        no_words = [
            "no",
            "nope",
            "nah",
            "not really",
            "i don't",
            "i do not",
            "never"
        ]

        if any(word in text for word in yes_words):
            return 1

        if any(word in text for word in no_words):
            return 0

        return None

    # ============================================================
    # GENERAL HEALTH
    # ============================================================

    def _process_general_health(self, message):

        text = message.lower()

        mapping = {
            "excellent": 1,
            "very good": 2,
            "good": 3,
            "fair": 4,
            "poor": 5,
        }

        for key, value in mapping.items():
            if key in text:
                self.data["GenHlth"] = value
                self.current_step = "stroke"

                return "Have you ever had a stroke?"

        return (
            "Please choose one: excellent, very good, "
            "good, fair, or poor."
        )

    # ============================================================
    # DAYS
    # ============================================================

    def _process_days(
        self,
        message,
        field,
        next_question,
        next_step
    ):

        numbers = re.findall(r"\d+", message)

        if not numbers:
            return "Please tell me a number between 0 and 30."

        days = int(numbers[0])

        if days < 0 or days > 30:
            return "Please give a number between 0 and 30."

        self.data[field] = days
        self.current_step = next_step

        return next_question

    # ============================================================
    # FRUITS + VEGETABLES
    # ============================================================

    def _process_fruits_veggies(self, message):

        value = self._parse_yes_no(message)

        if value is None:
            return "Just answer yes or no. Do you regularly eat fruits?"

        self.data["Fruits"] = value

        self.current_step = "alcohol"

        return "Do you regularly eat vegetables?"

    # ============================================================
    # EDUCATION
    # ============================================================

    def _process_education(self, message):

        text = message.lower()

        if any(x in text for x in [
            "postgraduate",
            "master",
            "phd",
            "doctorate"
        ]):
            value = 6

        elif any(x in text for x in [
            "college",
            "university",
            "bachelor",
            "degree"
        ]):
            value = 5

        elif "high school" in text:
            value = 4

        elif "secondary" in text:
            value = 3

        else:
            value = 2

        self.data["Education"] = value
        self.current_step = "income"

        return (
            "Finally, what is your approximate annual household "
            "income range?"
        )

    # ============================================================
    # INCOME
    # ============================================================

    def _process_income(self, message):

        numbers = re.findall(r"\d+", message)

        if numbers:
            income = int(numbers[0])

            self.data["Income"] = self._convert_income(
                income
            )

        else:
            self.data["Income"] = 4

        self.current_step = "complete"

        return None

    # ============================================================
    # AGE CONVERSION
    # ============================================================

    def _convert_age(self, age):

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
    # INCOME CONVERSION
    # ============================================================

    def _convert_income(self, income):

        if income < 10000:
            return 1

        if income < 15000:
            return 2

        if income < 20000:
            return 3

        if income < 25000:
            return 4

        if income < 35000:
            return 5

        if income < 50000:
            return 6

        if income < 75000:
            return 7

        return 8

    # ============================================================
    # GET MODEL DATA
    # ============================================================

    def get_model_data(self):

        return {
            key: value
            for key, value in self.data.items()
            if key not in [
                "actual_age",
                "height_cm",
                "weight_kg"
            ]
        }
