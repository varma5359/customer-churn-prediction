"""Streamlit user interface for the Customer Churn Prediction System.

Start FastAPI in one terminal:
    uvicorn api:app --reload

Start Streamlit in another terminal:
    streamlit run app.py
"""

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 10

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


def get_api_schema() -> dict[str, Any] | None:
    """Ask the API which fields and category values the trained model expects."""
    try:
        response = requests.get(f"{API_URL}/schema", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(
            "Could not load the model input fields from FastAPI. "
            "Confirm that the API is running and that you trained the model first."
        )
        with st.expander("Connection details"):
            st.code(str(error))
        return None


def get_api_health() -> bool:
    """Check whether the backend and trained model are available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        return response.status_code == 200 and response.json().get("model_loaded") is True
    except (requests.exceptions.RequestException, ValueError):
        return False


def render_feature_input(feature: dict[str, Any]) -> Any:
    """Render a form field based on the schema returned by the training pipeline."""
    name = feature["name"]
    field_type = feature.get("type", "text")
    options = feature.get("options", [])

    if field_type == "category" and options:
        return st.selectbox(name, options=options, key=f"feature_{name}")

    if field_type == "number":
        return st.number_input(
            name,
            value=0.0,
            step=1.0,
            key=f"feature_{name}",
            help="Enter the value for this customer. Use the dataset's units.",
        )

    return st.text_input(name, key=f"feature_{name}")


def request_prediction(customer: dict[str, Any]) -> dict[str, Any] | None:
    """Send the submitted customer data to FastAPI and return its response."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError:
        detail = "The API rejected the submitted values."
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        st.error(f"Prediction failed: {detail}")
        return None
    except requests.exceptions.RequestException as error:
        st.error(
            "Could not contact FastAPI. Check that the backend is running "
            "and try again."
        )
        with st.expander("Connection details"):
            st.code(str(error))
        return None


st.title("📊 Customer Churn Prediction")
st.write(
    "Enter a customer's details to estimate whether they may leave the service. "
    "The form is generated from the features used by your trained model."
)

with st.sidebar:
    st.subheader("Backend status")
    if get_api_health():
        st.success("API and trained model are ready.")
    else:
        st.warning("API or trained model is unavailable.")
    st.caption(f"API address: {API_URL}")
    st.markdown("---")
    st.caption(
        "Educational prototype. Predictions are estimates and should not be "
        "treated as guarantees or used as the sole basis for customer decisions."
    )

schema_data = get_api_schema()

if schema_data is not None:
    features = schema_data.get("features", [])

    if not features:
        st.warning("The API did not return any model input fields.")
    else:
        st.subheader("Customer details")
        st.caption(
            "Choose values that match the dataset. Numeric fields start at 0; "
            "change them to the customer's actual values before predicting."
        )

        with st.form("customer_churn_form"):
            columns = st.columns(2)
            customer_values = {}

            for index, feature in enumerate(features):
                with columns[index % 2]:
                    customer_values[feature["name"]] = render_feature_input(feature)

            submitted = st.form_submit_button(
                "Predict customer churn",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            with st.spinner("Getting prediction from the model..."):
                result = request_prediction(customer_values)

            if result is not None:
                st.markdown("---")
                st.subheader("Prediction result")

                result_columns = st.columns(2)
                with result_columns[0]:
                    if result.get("prediction") == 1:
                        st.error("Likely to churn")
                    else:
                        st.success("Unlikely to churn")

                probability = result.get("churn_probability")
                with result_columns[1]:
                    if isinstance(probability, (int, float)):
                        st.metric("Estimated churn probability", f"{probability:.1%}")

                st.info(result.get("message", "Prediction completed."))
else:
    st.info(
        "Once the API is running and a model has been trained, the customer "
        "input form will appear here."
    )

with st.expander("How to start the project"):
    st.markdown(
        """
        1. Install dependencies with pip install -r requirements.txt.
        2. Add the dataset to data/customers.csv.
        3. Train the model with python train_model.py.
        4. Start the API in one terminal: uvicorn api:app --reload.
        5. Start this interface in another terminal: streamlit run app.py.
        6. Open the local URL printed by Streamlit.
        """
    )
