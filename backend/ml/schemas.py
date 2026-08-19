MODEL_SCHEMAS = {

    # ============================================================
    # HEART DISEASE
    # ============================================================

    "heart_disease_pipeline": {

        "description": "Heart disease risk classification",

        "fields": {

            "age": {
                "type": "number",
                "description": "Age in years"
            },

            "sex": {
                "type": "category",
                "description": "Sex",
                "values": {
                    "female": 0,
                    "male": 1
                }
            },

            "cp": {
                "type": "category",
                "description": "Chest pain type",
                "values": {
                    "typical angina": 1,
                    "atypical angina": 2,
                    "non-anginal pain": 3,
                    "asymptomatic": 4
                }
            },

            "trestbps": {
                "type": "number",
                "description": "Resting blood pressure in mmHg"
            },

            "chol": {
                "type": "number",
                "description": "Serum cholesterol in mg/dL"
            },

            "fbs": {
                "type": "category",
                "description": "Fasting blood sugar greater than 120 mg/dL",
                "values": {
                    "no": 0,
                    "yes": 1
                }
            },

            "restecg": {
                "type": "category",
                "description": "Resting ECG result",
                "values": {
                    "normal": 0,
                    "st-t wave abnormality": 1,
                    "left ventricular hypertrophy": 2
                }
            },

            "thalach": {
                "type": "number",
                "description": "Maximum heart rate achieved"
            },

            "exang": {
                "type": "category",
                "description": "Exercise-induced angina",
                "values": {
                    "no": 0,
                    "yes": 1
                }
            },

            "oldpeak": {
                "type": "number",
                "description": "ST depression induced by exercise"
            },

            "slope": {
                "type": "category",
                "description": "Slope of peak exercise ST segment",
                "values": {
                    "upsloping": 1,
                    "flat": 2,
                    "downsloping": 3
                }
            },

            "ca": {
                "type": "number",
                "description": "Number of major vessels colored by fluoroscopy",
                "range": [0, 3]
            },

            "thal": {
                "type": "category",
                "description": "Thalassemia result",
                "values": {
                    "normal": 3,
                    "fixed defect": 6,
                    "reversible defect": 7
                }
            }
        }
    }
}
