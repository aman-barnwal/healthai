import pandas as pd


DATASET_PATH = "dataset/heart_disease.csv"


def load_data():
    """Load the heart disease dataset."""
    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def clean_data(df):
    """Clean and prepare the dataset."""

    # Make a copy so the original DataFrame is not modified
    df = df.copy()

    # Convert target into binary classification
    # 0 = No heart disease
    # 1 = Heart disease
    df["num"] = (df["num"] > 0).astype(int)

    # Handle missing values
    # ca and thal contain missing values.
    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

    return df


if __name__ == "__main__":

    data = load_data()

    print("\nBefore cleaning:")
    print(data.isnull().sum())

    data = clean_data(data)

    print("\nAfter cleaning:")
    print(data.isnull().sum())

    print("\nTarget distribution:")
    print(data["num"].value_counts().sort_index())
