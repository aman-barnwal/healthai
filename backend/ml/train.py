import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "dataset/heart_disease.csv"
MODEL_PATH = "backend/models/heart_disease_pipeline.pkl"


def load_and_prepare_data():
    df = pd.read_csv(DATASET_PATH)

    # Convert multi-class target into binary target
    # 0 = No heart disease
    # 1 = Heart disease
    df["num"] = (df["num"] > 0).astype(int)

    X = df.drop("num", axis=1)
    y = df["num"]

    return X, y


def main():

    print("Loading dataset...")

    X, y = load_and_prepare_data()

    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Logistic Regression pipeline
    logistic_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Random Forest pipeline
    forest_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ))
    ])

    print("\nTraining Logistic Regression...")
    logistic_pipeline.fit(X_train, y_train)

    logistic_predictions = logistic_pipeline.predict(X_test)
    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions
    )

    print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")

    print("\nTraining Random Forest...")
    forest_pipeline.fit(X_train, y_train)

    forest_predictions = forest_pipeline.predict(X_test)
    forest_accuracy = accuracy_score(
        y_test,
        forest_predictions
    )

    print(f"Random Forest Accuracy: {forest_accuracy:.4f}")

    # Choose the better model
    if forest_accuracy > logistic_accuracy:
        best_model = forest_pipeline
        best_name = "Random Forest"
        best_predictions = forest_predictions
    else:
        best_model = logistic_pipeline
        best_name = "Logistic Regression"
        best_predictions = logistic_predictions

    print(f"\nBest Model: {best_name}")

    print("\nClassification Report:")
    print(classification_report(y_test, best_predictions))

    # Save the complete pipeline
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
