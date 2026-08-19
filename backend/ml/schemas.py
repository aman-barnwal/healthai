MODEL_SCHEMAS = {

    "heart_disease_pipeline": {
        "description": "Heart disease risk classification",

        "keywords": [
            "heart disease",
            "heart risk",
            "cardiac",
            "heart attack",
            "chest pain"
        ],

        "fields": {
            "age": {
                "type": "number",
                "label": "Age (years)"
            },

            "sex": {
                "type": "category",
                "label": "Sex (male/female)",
                "values": {
                    "female": 0,
                    "male": 1
                }
            },

            "cp": {
                "type": "number",
                "label": "Chest pain type (1-4)"
            },

            "trestbps": {
                "type": "number",
                "label": "Resting blood pressure (mmHg)"
            },

            "chol": {
                "type": "number",
                "label": "Cholesterol (mg/dL)"
            },

            "fbs": {
                "type": "number",
                "label": "Fasting blood sugar > 120 mg/dL (0/1)"
            },

            "restecg": {
                "type": "number",
                "label": "Resting ECG result (0-2)"
            },

            "thalach": {
                "type": "number",
                "label": "Maximum heart rate achieved"
            },

            "exang": {
                "type": "number",
                "label": "Exercise-induced angina (0/1)"
            },

            "oldpeak": {
                "type": "number",
                "label": "ST depression induced by exercise"
            },

            "slope": {
                "type": "number",
                "label": "Exercise ST-segment slope (1-3)"
            },

            "ca": {
                "type": "number",
                "label": "Number of major vessels (0-3)"
            },

            "thal": {
                "type": "number",
                "label": "Thalassemia result (3/6/7)"
            }
        }
    },


    "breast_cancer": {
        "description": "Breast cancer classification",

        "keywords": [
            "breast cancer",
            "breast tumour",
            "breast tumor",
            "mammogram"
        ],

        "aliases": [
            "breast_cancer_wisconsin_diagnostic"
        ],

        "fields": {}
    },


    "diabetes_binary": {
        "description": "Diabetes risk classification",

        "keywords": [
            "diabetes",
            "diabetic",
            "diabetes risk",
            "blood sugar",
            "sugar disease"
        ],

        "fields": {}
    },


    "diabetes_health_indicators": {
        "description": "Diabetes health classification",

        "keywords": [
            "diabetes health indicators",
            "diabetes indicators"
        ],

        "fields": {}
    },


    "stroke_prediction": {
        "description": "Stroke risk prediction",

        "keywords": [
            "stroke",
            "brain stroke",
            "stroke risk"
        ],

        "fields": {}
    },


    "kidney_disease_dataset": {
        "description": "Kidney disease assessment",

        "keywords": [
            "kidney disease",
            "kidney problem",
            "renal disease",
            "renal problem"
        ],

        "fields": {}
    },


    "parkinsons_classification": {
        "description": "Parkinson's disease classification",

        "keywords": [
            "parkinson",
            "parkinson's",
            "parkinsons"
        ],

        "fields": {}
    },


    "thyroid": {
        "description": "Thyroid disease classification",

        "keywords": [
            "thyroid",
            "hypothyroidism",
            "hyperthyroidism"
        ],

        "fields": {}
    },


    "obesity_levels": {
        "description": "Obesity level classification",

        "keywords": [
            "obesity",
            "overweight",
            "body weight classification"
        ],

        "fields": {}
    },


    "maternal_health_risk": {
        "description": "Maternal health risk classification",

        "keywords": [
            "maternal health",
            "pregnancy risk",
            "pregnancy"
        ],

        "fields": {}
    },


    "Dry_Eye_Dataset": {
        "description": "Dry eye disease classification",

        "keywords": [
            "dry eye",
            "eye dryness",
            "dry eyes"
        ],

        "fields": {}
    },


    "heart_failure_clinical_records": {
        "description": "Heart failure classification",

        "keywords": [
            "heart failure",
            "cardiac failure"
        ],

        "fields": {}
    },


    "hepatitis": {
        "description": "Hepatitis classification",

        "keywords": [
            "hepatitis",
            "liver infection"
        ],

        "fields": {}
    },


    "indian_liver": {
        "description": "Liver disease classification",

        "keywords": [
            "liver disease",
            "liver problem",
            "liver risk"
        ],

        "fields": {}
    },


    "lung_disease": {
        "description": "Lung disease classification",

        "keywords": [
            "lung disease",
            "lung problem",
            "respiratory disease"
        ],

        "fields": {}
    }
}
