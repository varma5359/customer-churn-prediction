"""FastAPI backend for the Customer Churn Prediction System.

Run from the project root:
    uvicorn api:app --reload

Train the model first with:
    python train_model.py
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "churn_pipeline.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "A beginner-friendly API that predicts customer churn using a "
        "trained Logistic Regression pipeline."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    """Load the trained pipeline once and reuse it between requests."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run 'python train_model.py' first."
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {"pipeline", "feature_columns"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError(
            "The saved model file has an unexpected format. Retrain the model."
        )

    return bundle


def get_model_bundle() -> dict[str, Any]:
    """Convert model loading problems into a clear API response."""
    try:
        return load_model_bundle()
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_feature_schema(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe model input fields so the UI can build a matching form."""
    pipeline = bundle["pipeline"]
    feature_columns = list(bundle["feature_columns"])
    preprocessor = pipeline.named_steps.get("preprocessor")

    numeric_columns: set[str] = set()
    categorical_options: dict[str, list[str]] = {}

    if preprocessor is not None:
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "remainder" or transformer == "drop":
                continue

            column_names = [str(column) for column in columns]

            if transformer_name == "numeric":
                numeric_columns.update(column_names)

            elif transformer_name == "categorical":
                encoder = transformer.named_steps.get("encoder")
                if encoder is not None and hasattr(encoder, "categories_"):
                    for column_name, categories in zip(
                        column_names, encoder.categories_
                    ):
                        categorical_options[column_name] = [
                            str(category) for category in categories
                        ]

    schema = []
    for column_name in feature_columns:
        if column_name in numeric_columns:
            schema.append(
                {
                    "name": column_name,
                    "type": "number",
                    "options": [],
                }
            )
        elif column_name in categorical_options:
            schema.append(
                {
                    "name": column_name,
                    "type": "category",
                    "options": categorical_options[column_name],
                }
            )
        else:
            schema.append(
                {
                    "name": column_name,
                    "type": "text",
                    "options": [],
                }
            )

    return schema


@app.get("/")
def home() -> dict[str, str]:
    """Return basic information about this API."""
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> JSONResponse:
    """Report whether the API can access a trained model."""
    try:
        bundle = load_model_bundle()
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "model_loaded": True,
                "feature_count": len(bundle["feature_columns"]),
            },
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        return JSONResponse(
            status_code=503,
            content={
                "status": "model_unavailable",
                "model_loaded": False,
                "detail": str(error),
            },
        )


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Return the exact input fields and category choices expected by the model."""
    bundle = get_model_bundle()
    return {
        "features": get_feature_schema(bundle),
        "target": bundle.get("target_column", "Churn"),
    }


@app.post("/predict")
def predict(
    customer: dict[str, Any] = Body(
        ...,
        example={
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MonthlyCharges": 75.5,
        },
    ),
) -> dict[str, Any]:
    """Validate customer fields and return a churn prediction."""
    bundle = get_model_bundle()
    feature_columns = list(bundle["feature_columns"])

    if not isinstance(customer, dict) or not customer:
        raise HTTPException(
            status_code=422,
            detail="Send a JSON object containing customer feature values.",
        )

    missing_columns = [name for name in feature_columns if name not in customer]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some required customer fields are missing.",
                "missing_fields": missing_columns,
            },
        )

    # Ignore additional fields, and keep the same column order used in training.
    row = {name: customer[name] for name in feature_columns}
    input_data = pd.DataFrame([row])

    try:
        prediction = int(bundle["pipeline"].predict(input_data)[0])
        probabilities = bundle["pipeline"].predict_proba(input_data)[0]
        classes = list(bundle["pipeline"].named_steps["classifier"].classes_)
        churn_index = classes.index(1)
        churn_probability = float(probabilities[churn_index])
    except (TypeError, ValueError, KeyError, IndexError) as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "The supplied values could not be processed. Check the field "
                "types and category choices returned by GET /schema."
            ),
        ) from error

    label = "Likely to churn" if prediction == 1 else "Unlikely to churn"

    return {
        "prediction": prediction,
        "label": label,
        "churn_probability": round(churn_probability, 4),
        "message": (
            "This is a model estimate, not a guarantee of future customer behaviour."
        ),
    }
