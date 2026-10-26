"""Train and evaluate a beginner-friendly customer churn model.

Expected input:
    data/customers.csv

The script detects common churn target names (for example, Churn or Exited),
prepares numeric and categorical columns, trains Logistic Regression, prints
test metrics, and saves the fitted pipeline plus its input-column metadata.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Build paths from this file so the script works from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "customers.csv"
MODEL_PATH = PROJECT_DIR / "model" / "churn_pipeline.joblib"

# Common target-column names used in public churn datasets.
TARGET_NAMES = (
    "churn",
    "churn label",
    "exited",
    "attrition",
    "customer churn",
    "is churn",
)


def load_data(file_path):
    """Read the customer CSV and stop early when the file is missing or empty."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place your customer churn CSV at data/customers.csv."
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"The dataset file is empty: {file_path}\n"
            "Add a real dataset before training the model."
        )

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("The dataset has a header but contains no customer rows.")

    # Remove accidental spaces from column names.
    data.columns = data.columns.astype(str).str.strip()

    if data.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    print("\n--- DATASET OVERVIEW ---")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print("\nColumn names:")
    print(list(data.columns))
    print("\nMissing values per column:")
    print(data.isnull().sum())

    return data


def find_target_column(data):
    """Find a likely churn target column without assuming exact capitalization."""
    normalized_columns = {
        str(column).strip().lower().replace("_", " ").replace("-", " "): column
        for column in data.columns
    }

    for target_name in TARGET_NAMES:
        if target_name in normalized_columns:
            return normalized_columns[target_name]

    available_columns = ", ".join(str(column) for column in data.columns)
    raise ValueError(
        "Could not identify the churn target column. Rename the target column "
        "to 'Churn' or update TARGET_NAMES in train_model.py. "
        f"Available columns: {available_columns}"
    )


def prepare_target(target):
    """Convert common two-class churn labels into numeric 0 and 1."""
    cleaned = target.astype("string").str.strip().str.lower()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    valid_values = set(cleaned.dropna().unique())

    positive_values = {
        "yes",
        "true",
        "1",
        "churn",
        "churned",
        "exited",
        "exit",
        "left",
        "attrited",
    }
    negative_values = {
        "no",
        "false",
        "0",
        "not churn",
        "not churned",
        "stayed",
        "stay",
        "retained",
        "active",
        "did not churn",
    }

    if not valid_values:
        raise ValueError("The target column contains no usable labels.")

    if valid_values.issubset(positive_values | negative_values):
        mapped = cleaned.map(
            lambda value: (
                1 if value in positive_values
                else 0 if value in negative_values
                else None
            )
        )
        return mapped.astype("Int64")

    # Numeric 0/1 targets are common in churn datasets.
    numeric_target = pd.to_numeric(target, errors="coerce")
    numeric_values = set(numeric_target.dropna().unique())
    if numeric_values.issubset({0, 1}) and numeric_values:
        return numeric_target.astype("Int64")

    raise ValueError(
        "The target labels could not be mapped safely to churn (1) and "
        "non-churn (0). Inspect the target values and update prepare_target(). "
        f"Found labels: {sorted(str(value) for value in valid_values)}"
    )


def prepare_features(data, target_column):
    """Separate the target, remove identifier columns, and identify data types."""
    target = prepare_target(data[target_column])

    # Drop rows with an unknown target because they cannot be used for training.
    usable_rows = target.notna()
    features = data.loc[usable_rows].drop(columns=[target_column]).copy()
    target = target.loc[usable_rows].astype(int)

    # Identifiers usually identify a person or row rather than describe churn.
    id_names = {
        "id",
        "customerid",
        "customer id",
        "customer_id",
        "row id",
        "row_id",
    }
    id_columns = [
        column
        for column in features.columns
        if str(column).strip().lower() in id_names
    ]
    if id_columns:
        print(f"\nDropping identifier columns: {id_columns}")
        features = features.drop(columns=id_columns)

    if features.empty or features.shape[1] == 0:
        raise ValueError("No usable input features remain after removing the target.")

    # Convert text columns to numeric when their non-empty values are numeric.
    for column in features.select_dtypes(include=["object", "string"]).columns:
        original = features[column]
        converted = pd.to_numeric(original, errors="coerce")
        non_empty_count = original.notna().sum()

        if non_empty_count > 0:
            numeric_ratio = converted.notna().sum() / non_empty_count
            if numeric_ratio >= 0.9:
                features[column] = converted

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable numeric or categorical features were found.")

    if target.nunique() != 2:
        raise ValueError(
            "The churn target must contain both classes: 0 (not churn) and "
            "1 (churn). Check the target labels and available rows."
        )

    print(f"\nTarget column: {target_column}")
    print(f"Numeric features: {numeric_columns}")
    print(f"Categorical features: {categorical_columns}")
    print("\nTarget distribution (0 = not churn, 1 = churn):")
    print(target.value_counts().sort_index())

    return features, target, numeric_columns, categorical_columns


def build_pipeline(numeric_columns, categorical_columns):
    """Create preprocessing and Logistic Regression as one reusable pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=None)),
        ]
    )


def evaluate_model(model, x_test, y_test):
    """Print common classification metrics on held-out test rows."""
    predictions = model.predict(x_test)

    print("\n--- TEST RESULTS ---")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not churn", "Churn"],
            zero_division=0,
        )
    )

    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        probabilities = model.predict_proba(x_test)[:, 1]
        print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.3f}")


def save_model(model, feature_columns, target_column):
    """Save the fitted pipeline and metadata needed by the API."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "pipeline": model,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "positive_class": 1,
        "negative_class": 0,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")


def train_model():
    """Run the complete training workflow from CSV to saved pipeline."""
    data = load_data(DATA_PATH)
    target_column = find_target_column(data)

    features, target, numeric_columns, categorical_columns = prepare_features(
        data, target_column
    )

    # Stratification preserves the class proportions in both data splits.
    class_counts = target.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each target class needs at least two rows to create a stratified "
            "train/test split. Add more examples for the smaller class."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_pipeline(numeric_columns, categorical_columns)
    model.fit(x_train, y_train)

    evaluate_model(model, x_test, y_test)
    save_model(model, features.columns, target_column)


if __name__ == "__main__":
    try:
        train_model()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\nTraining stopped: {error}")
