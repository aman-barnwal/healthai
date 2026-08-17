import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

DATASET = "dataset/heart_disease.csv"

MODEL_DIR = "backend/models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("        HEALTHCAREAI HEART DISEASE MODEL TUNER")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print(
    f"Original dataset: {df.shape}"
)


# ============================================================
# CLEAN DATA
# ============================================================

df = df.drop_duplicates()

print(
    f"After duplicate removal: {df.shape}"
)


# ============================================================
# TARGET
# ============================================================

X = df.drop(
    columns=["num"]
)

y = df["num"].copy()


# Convert:
#
# 0 = no disease
# 1-4 = disease
#
# into binary classification.

y = (
    y > 0
).astype(int)


print("\nTarget distribution:")

print(
    y.value_counts()
    .sort_index()
)


# ============================================================
# TRAIN / FINAL TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Final test samples: {len(X_test)}"
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

    "Random Forest":
        Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        ),

    "Extra Trees":
        Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        ),

    "Gradient Boosting":
        Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

    "Hist Gradient Boosting":
        Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

    "SVM":
        Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    SVC(
                        probability=True,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        )
}


# ============================================================
# BASELINE CROSS-VALIDATION
# ============================================================

print()
print("=" * 70)
print("BASELINE 5-FOLD CROSS-VALIDATION")
print("=" * 70)

baseline_results = []


for name, model in models.items():

    print(
        f"\nTesting {name}..."
    )

    from sklearn.model_selection import cross_validate

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "f1",
            "precision",
            "recall"
        ],
        n_jobs=-1
    )

    accuracy = scores[
        "test_accuracy"
    ].mean()

    f1 = scores[
        "test_f1"
    ].mean()

    precision = scores[
        "test_precision"
    ].mean()

    recall = scores[
        "test_recall"
    ].mean()

    print(
        f"CV Accuracy : {accuracy:.4f}"
    )

    print(
        f"CV F1       : {f1:.4f}"
    )

    print(
        f"CV Precision: {precision:.4f}"
    )

    print(
        f"CV Recall   : {recall:.4f}"
    )

    baseline_results.append(
        {
            "name": name,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    )


# ============================================================
# RANDOM FOREST TUNING
# ============================================================

print()
print("=" * 70)
print("TUNING RANDOM FOREST")
print("=" * 70)


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


rf_params = {

    "model__n_estimators": [
        200,
        400,
        600,
        800,
        1000
    ],

    "model__max_depth": [
        None,
        3,
        5,
        7,
        10,
        15
    ],

    "model__min_samples_split": [
        2,
        4,
        6,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4,
        6
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        None
    ],

    "model__class_weight": [
        None,
        "balanced",
        "balanced_subsample"
    ]
}


rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_params,
    n_iter=50,
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)


rf_search.fit(
    X_train,
    y_train
)


print("\nBest Random Forest parameters:")

print(
    rf_search.best_params_
)

print(
    f"\nBest RF CV F1: "
    f"{rf_search.best_score_:.4f}"
)


# ============================================================
# EXTRA TREES TUNING
# ============================================================

print()
print("=" * 70)
print("TUNING EXTRA TREES")
print("=" * 70)


et_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


et_params = {

    "model__n_estimators": [
        200,
        400,
        600,
        800,
        1000
    ],

    "model__max_depth": [
        None,
        3,
        5,
        7,
        10,
        15
    ],

    "model__min_samples_split": [
        2,
        4,
        6,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4,
        6
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        None
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]
}


et_search = RandomizedSearchCV(
    estimator=et_pipeline,
    param_distributions=et_params,
    n_iter=50,
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)


et_search.fit(
    X_train,
    y_train
)


print("\nBest Extra Trees parameters:")

print(
    et_search.best_params_
)

print(
    f"\nBest Extra Trees CV F1: "
    f"{et_search.best_score_:.4f}"
)


# ============================================================
# SELECT BEST TUNED MODEL
# ============================================================

candidates = {

    "Random Forest Tuned":
        rf_search.best_estimator_,

    "Extra Trees Tuned":
        et_search.best_estimator_
}


best_name = None
best_model = None
best_cv_f1 = -1


for name, model in candidates.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    score = scores[
        "test_score"
    ].mean()

    print(
        f"\n{name} CV F1: {score:.4f}"
    )

    if score > best_cv_f1:

        best_cv_f1 = score
        best_name = name
        best_model = model


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print("FINAL UNTOUCHED TEST SET")
print("=" * 70)

print(
    f"\nSelected model: {best_name}"
)

print(
    f"Cross-validation F1: "
    f"{best_cv_f1:.4f}"
)


best_model.fit(
    X_train,
    y_train
)


predictions = best_model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)


print(
    f"\nFINAL TEST ACCURACY : {accuracy:.4f}"
)

print(
    f"FINAL TEST F1       : {f1:.4f}"
)

print(
    f"FINAL TEST PRECISION: {precision:.4f}"
)

print(
    f"FINAL TEST RECALL   : {recall:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        predictions
    )
)


# ============================================================
# SAVE
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "heart_disease_tuned.pkl"
)


joblib.dump(
    best_model,
    model_path
)


print()
print("=" * 70)

print(
    f"Model saved to: {model_path}"
)

print(
    f"FINAL ACCURACY: {accuracy * 100:.2f}%"
)

print(
    f"CV F1: {best_cv_f1 * 100:.2f}%"
)

print("=" * 70)
