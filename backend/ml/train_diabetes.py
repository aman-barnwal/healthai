import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "dataset/diabetes_binary.csv"
MODEL_PATH = "backend/models/diabetes_binary.pkl"


def main():
    print("Loading diabetes dataset...")

    df = pd.read_csv(DATASET_PATH)

    target_column = "Diabetes_binary"

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nFeature names:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("\nTraining Logistic Regression...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions
    ))

    print("\nSaving model...")

    joblib.dump(
        pipeline,
        MODEL_PATH,
        compress=3
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
