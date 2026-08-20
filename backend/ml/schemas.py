MODEL_SCHEMAS = {

    # ============================================================
    # HEART DISEASE
    # ============================================================

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


    # ============================================================
    # BREAST CANCER
    # ============================================================

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


    # ============================================================
    # DIABETES BINARY
    # Exact fields used by diabetes_binary.pkl
    # ============================================================

    "diabetes_binary": {
        "description": "Diabetes risk classification",

        "keywords": [
            "diabetes",
            "diabetic",
            "diabetes risk",
            "blood sugar",
            "sugar disease"
        ],

        "fields": {

            "HighBP": {
                "type": "number",
                "label": "Do you have high blood pressure? (0 = No, 1 = Yes)"
            },

            "HighChol": {
                "type": "number",
                "label": "Do you have high cholesterol? (0 = No, 1 = Yes)"
            },

            "CholCheck": {
                "type": "number",
                "label": "Have you had a cholesterol check in the last 5 years? (0 = No, 1 = Yes)"
            },

            "BMI": {
                "type": "number",
                "label": "What is your BMI?"
            },

            "Smoker": {
                "type": "number",
                "label": "Have you smoked at least 100 cigarettes in your lifetime? (0 = No, 1 = Yes)"
            },

            "Stroke": {
                "type": "number",
                "label": "Have you ever had a stroke? (0 = No, 1 = Yes)"
            },

            "HeartDiseaseorAttack": {
                "type": "number",
                "label": "Have you had coronary heart disease or a heart attack? (0 = No, 1 = Yes)"
            },

            "PhysActivity": {
                "type": "number",
                "label": "Have you done physical activity in the past 30 days? (0 = No, 1 = Yes)"
            },

            "Fruits": {
                "type": "number",
                "label": "Do you consume fruits regularly? (0 = No, 1 = Yes)"
            },

            "Veggies": {
                "type": "number",
                "label": "Do you consume vegetables regularly? (0 = No, 1 = Yes)"
            },

            "HvyAlcoholConsump": {
                "type": "number",
                "label": "Do you consume alcohol heavily? (0 = No, 1 = Yes)"
            },

            "AnyHealthcare": {
                "type": "number",
                "label": "Do you have any healthcare coverage? (0 = No, 1 = Yes)"
            },

            "NoDocbcCost": {
                "type": "number",
                "label": "Was there a time you needed a doctor but could not see one because of cost? (0 = No, 1 = Yes)"
            },

            "GenHlth": {
                "type": "number",
                "label": "Rate your general health (1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor)"
            },

            "MentHlth": {
                "type": "number",
                "label": "How many days in the last 30 was your mental health not good? (0-30)"
            },

            "PhysHlth": {
                "type": "number",
                "label": "How many days in the last 30 was your physical health not good? (0-30)"
            },

            "DiffWalk": {
                "type": "number",
                "label": "Do you have serious difficulty walking or climbing stairs? (0 = No, 1 = Yes)"
            },

            "Sex": {
                "type": "category",
                "label": "Sex (male/female)",
                "values": {
                    "female": 0,
                    "male": 1
                }
            },

            "Age": {
                "type": "number",
                "label": "Age category used by the model (1-13)"
            },

            "Education": {
                "type": "number",
                "label": "Education category (1-6)"
            },

            "Income": {
                "type": "number",
                "label": "Income category (1-8)"
            }
        }
    },


    # ============================================================
    # DIABETES HEALTH INDICATORS
    # ============================================================

    "diabetes_health_indicators": {
        "description": "Diabetes health classification",

        "keywords": [
            "diabetes health indicators",
            "diabetes indicators"
        ],

        "fields": {}
    },


    # ============================================================
    # STROKE
    # ============================================================

    "stroke_prediction": {
        "description": "Stroke risk prediction",

        "keywords": [
            "stroke",
            "brain stroke",
            "stroke risk"
        ],

        "fields": {}
    },


    # ============================================================
    # KIDNEY DISEASE
    # ============================================================

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


    # ============================================================
    # PARKINSON'S
    # ============================================================

    "parkinsons_classification": {
        "description": "Parkinson's disease classification",

        "keywords": [
            "parkinson",
            "parkinson's",
            "parkinsons"
        ],

        "fields": {}
    },


    # ============================================================
    # THYROID
    # ============================================================

    "thyroid": {
        "description": "Thyroid disease classification",

        "keywords": [
            "thyroid",
            "hypothyroidism",
            "hyperthyroidism"
        ],

        "fields": {}
    },


    # ============================================================
    # OBESITY
    # ============================================================

    "obesity_levels": {
        "description": "Obesity level classification",

        "keywords": [
            "obesity",
            "overweight",
            "body weight classification"
        ],

        "fields": {}
    },


    # ============================================================
    # MATERNAL HEALTH
    # ============================================================

    "maternal_health_risk": {
        "description": "Maternal health risk classification",

        "keywords": [
            "maternal health",
            "pregnancy risk",
            "pregnancy"
        ],

        "fields": {}
    },


    # ============================================================
    # DRY EYE
    # ============================================================

    "Dry_Eye_Dataset": {
        "description": "Dry eye disease classification",

        "keywords": [
            "dry eye",
            "eye dryness",
            "dry eyes"
        ],

        "fields": {}
    },


    # ============================================================
    # HEART FAILURE
    # ============================================================

    "heart_failure_clinical_records": {
        "description": "Heart failure classification",

        "keywords": [
            "heart failure",
            "cardiac failure"
        ],

        "fields": {}
    },


    # ============================================================
    # HEPATITIS
    # ============================================================

    "hepatitis": {
        "description": "Hepatitis classification",

        "keywords": [
            "hepatitis",
            "liver infection"
        ],

        "fields": {}
    },


    # ============================================================
    # INDIAN LIVER
    # ============================================================

    "indian_liver": {
        "description": "Liver disease classification",

        "keywords": [
            "liver disease",
            "liver problem",
            "liver risk"
        ],

        "fields": {}
    },


    # ============================================================
    # LUNG DISEASE
    # ============================================================

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
