"""Streamlit dashboard for the Customer Churn Prediction System.

Run FastAPI in one terminal:
    uvicorn api:app --reload

Run Streamlit in another terminal:
    streamlit run app.py
"""

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 10

st.set_page_config(
    page_title="ChurnIQ | Customer Retention",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A lightweight design system: deep navy, clear typography, soft surfaces and
# restrained accent colors. The app remains usable if custom CSS is unavailable.
st.markdown(
    """
    <style>
    :root {
        --ink: #17233b;
        --muted: #718096;
        --line: #e7ecf3;
        --surface: #ffffff;
        --accent: #5b6cf9;
        --accent-soft: #eef0ff;
    }
    .stApp { background: #f5f7fb; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,247,251,.88); }
    [data-testid="stSidebar"] { background: #111b31; }
    [data-testid="stSidebar"] * { color: #edf2ff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .35rem 0; }
    .brand { font-size: 1.15rem; font-weight: 800; letter-spacing: -.03em; }
    .brand-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 2.25rem; height: 2.25rem; border-radius: .75rem;
        background: #6474ff; margin-right: .55rem; font-size: 1.15rem;
    }
    .hero {
        padding: 1.65rem 1.8rem; border-radius: 1.2rem; color: white;
        background: linear-gradient(115deg, #17233e 0%, #263b70 58%, #5969ed 100%);
        margin: .25rem 0 1.35rem 0;
    }
    .hero-kicker { color: #cbd5ff; font-size: .76rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { color: white; font-size: clamp(1.7rem, 3vw, 2.5rem);
        line-height: 1.15; margin: .5rem 0; letter-spacing: -.04em; }
    .hero p { color: #e1e8ff; margin: 0; max-width: 48rem; }
    .section-title { font-size: 1.18rem; font-weight: 750; color: #17233b;
        margin: .65rem 0 .2rem 0; }
    .section-caption { color: #718096; font-size: .9rem; margin-bottom: .85rem; }
    .panel {
        background: white; border: 1px solid #e7ecf3; border-radius: 1rem;
        padding: 1.1rem 1.2rem; height: 100%;
    }
    .panel-label { color: #718096; font-size: .8rem; font-weight: 650; }
    .panel-value { color: #17233b; font-size: 1.5rem; font-weight: 800;
        margin: .4rem 0 .15rem; }
    .panel-note { color: #718096; font-size: .8rem; }
    .step-number { display: inline-flex; width: 1.8rem; height: 1.8rem;
        border-radius: .6rem; align-items: center; justify-content: center;
        background: #eef0ff; color: #4d5ce5; font-weight: 800; margin-right: .45rem; }
    div[data-testid="stForm"] {
        background: white; padding: 1.2rem; border: 1px solid #e7ecf3;
        border-radius: 1rem;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border-radius: .7rem; font-weight: 700; min-height: 2.7rem;
    }
    [data-testid="stMetric"] {
        background: white; padding: 1rem 1.1rem; border: 1px solid #e7ecf3;
        border-radius: .9rem;
    }
    .footer { color: #8792a5; font-size: .78rem; padding: 1.3rem 0 .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_schema() -> dict[str, Any] | None:
    """Load the trained model's input schema from the API."""
    try:
        response = requests.get(f"{API_URL}/schema", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, ValueError):
        return None


def get_api_health() -> tuple[bool, str]:
    """Return backend readiness and a user-friendly status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        payload = response.json()
        ready = response.status_code == 200 and payload.get("model_loaded") is True
        if ready:
            return True, "API and trained model are ready"
        return False, payload.get("detail", "API is running, but the model is not ready")
    except (requests.exceptions.RequestException, ValueError):
        return False, "Cannot connect to the prediction API"


def friendly_label(name: str) -> str:
    """Turn raw dataframe column names into readable form labels."""
    return name.replace("_", " ").replace("-", " ").strip().title()


def render_feature_input(feature: dict[str, Any]) -> Any:
    """Render a control that matches the trained model's expected feature type."""
    name = feature["name"]
    label = friendly_label(name)
    field_type = feature.get("type", "text")
    options = feature.get("options", [])

    if field_type == "category" and options:
        return st.selectbox(label, options=options, key=f"feature_{name}")

    if field_type == "number":
        return st.number_input(
            label,
            value=0.0,
            step=1.0,
            key=f"feature_{name}",
            help="Enter the customer's actual value using the dataset's units.",
        )

    return st.text_input(label, key=f"feature_{name}")


def request_prediction(customer: dict[str, Any]) -> dict[str, Any] | None:
    """Send customer values to FastAPI and return a validated response."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get("detail", "The API rejected these values.")
        except ValueError:
            detail = "The API rejected these values."
        st.error(f"Prediction could not be completed: {detail}")
    except requests.exceptions.RequestException:
        st.error("Could not contact the API. Check that FastAPI is running.")
    return None


def render_hero(kicker: str, heading: str, description: str) -> None:
    """Render a consistent page header."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <h1>{heading}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(label: str, value: str, note: str) -> None:
    """Render a compact dashboard metric card."""
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">{label}</div>
            <div class="panel-value">{value}</div>
            <div class="panel-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load backend metadata once per Streamlit rerun.
api_ready, api_message = get_api_health()
schema_data = get_api_schema() if api_ready else None
features = schema_data.get("features", []) if schema_data else []
target_name = schema_data.get("target", "Churn") if schema_data else "Churn"

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">↗</span>ChurnIQ</div>',
        unsafe_allow_html=True,
    )
    st.caption("CUSTOMER RETENTION ANALYTICS")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Predict Churn", "Model Guide"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("SYSTEM STATUS")
    if api_ready:
        st.success("Model online")
    else:
        st.warning("Model unavailable")
    st.caption(api_message)
    st.markdown("---")
    st.caption("Built with Streamlit · FastAPI · scikit-learn")
    st.caption("Educational decision-support prototype")

if page == "Overview":
    render_hero(
        "CUSTOMER RETENTION WORKSPACE",
        "Understand churn. Take action earlier.",
        "Estimate customer churn risk with your trained machine-learning pipeline "
        "and turn a prediction into a clearer retention conversation.",
    )

    if api_ready:
        st.success("Your prediction service is connected and ready.")
    else:
        st.warning(
            "The prediction service is not ready yet. Start FastAPI and train the "
            "model to unlock the live prediction workflow."
        )

    st.markdown('<div class="section-title">System at a glance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Live technical status from your API — '
        'not customer-population statistics.</div>',
        unsafe_allow_html=True,
    )
    cards = st.columns(3)
    with cards[0]:
        render_panel(
            "PREDICTION API",
            "Online" if api_ready else "Offline",
            "Live backend health check",
        )
    with cards[1]:
        render_panel(
            "MODEL INPUTS",
            str(len(features)) if api_ready else "—",
            "Features expected per prediction",
        )
    with cards[2]:
        render_panel(
            "PREDICTION TARGET",
            friendly_label(str(target_name)),
            "Target learned during training",
        )

    st.markdown("")
    st.markdown('<div class="section-title">Your workflow</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">A simple path from customer details to a '
        'risk estimate.</div>',
        unsafe_allow_html=True,
    )
    workflow = st.columns(3)
    with workflow[0]:
        st.markdown(
            '<div class="panel"><span class="step-number">1</span><b>Enter customer data</b>'
            '<p class="panel-note">Use the fields learned from your training dataset.</p></div>',
            unsafe_allow_html=True,
        )
    with workflow[1]:
        st.markdown(
            '<div class="panel"><span class="step-number">2</span><b>Estimate churn risk</b>'
            '<p class="panel-note">The API applies the saved preprocessing and model pipeline.</p></div>',
            unsafe_allow_html=True,
        )
    with workflow[2]:
        st.markdown(
            '<div class="panel"><span class="step-number">3</span><b>Review the result</b>'
            '<p class="panel-note">Use the probability as a signal for further review.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### What this dashboard provides")
        st.markdown(
            "- A form generated from the trained model's actual input schema\n"
            "- A churn prediction and estimated probability\n"
            "- A simple explanation of the model workflow and its limitations"
        )
    with right:
        st.markdown("#### Quick start")
        st.info(
            "Choose **Predict Churn** in the sidebar to evaluate a customer. "
            "If the model is offline, follow the setup steps in the Model Guide."
        )

elif page == "Predict Churn":
    render_hero(
        "PREDICTION WORKSPACE",
        "Assess a customer's churn risk",
        "Enter the customer's details below. Required fields are generated from "
        "the same feature schema used by the trained model.",
    )

    if not api_ready or not features:
        st.error(
            "The model is not ready to accept predictions. Please complete the "
            "setup steps in Model Guide, then refresh this page."
        )
    else:
        st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Use real values from the dataset where possible. '
            'Numeric fields default to 0 and should be reviewed before submission.</div>',
            unsafe_allow_html=True,
        )
        with st.form("customer_churn_form"):
            customer_values: dict[str, Any] = {}
            columns = st.columns(2)
            for index, feature in enumerate(features):
                with columns[index % 2]:
                    customer_values[feature["name"]] = render_feature_input(feature)

            submitted = st.form_submit_button(
                "✦  Calculate churn risk",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            with st.spinner("Evaluating customer profile…"):
                result = request_prediction(customer_values)

            if result is not None:
                prediction = result.get("prediction")
                probability = result.get("churn_probability")
                label = result.get(
                    "label",
                    "Likely to churn" if prediction == 1 else "Unlikely to churn",
                )

                st.markdown("---")
                st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-caption">This result is an estimate from your '
                    'trained model, not a guarantee of future behaviour.</div>',
                    unsafe_allow_html=True,
                )
                result_left, result_right = st.columns([1, 1.25])
                with result_left:
                    if prediction == 1:
                        st.error(f"**{label}**")
                        st.markdown("Review this customer's circumstances before deciding on a retention action.")
                    else:
                        st.success(f"**{label}**")
                        st.markdown("The model estimates a lower likelihood of churn for this profile.")
                with result_right:
                    if isinstance(probability, (int, float)):
                        st.metric("Estimated churn probability", f"{probability:.1%}")
                        st.progress(max(0.0, min(1.0, float(probability))))
                        if probability >= 0.7:
                            st.caption("Risk band: Higher model-estimated risk")
                        elif probability >= 0.4:
                            st.caption("Risk band: Moderate model-estimated risk")
                        else:
                            st.caption("Risk band: Lower model-estimated risk")
                    st.caption("Risk bands are illustrative UI thresholds, not calibrated business policies.")

                st.info(result.get("message", "Prediction completed."))
                st.session_state["last_prediction"] = {
                    "label": label,
                    "probability": probability,
                    "prediction": prediction,
                }

elif page == "Model Guide":
    render_hero(
        "PROJECT GUIDE",
        "How the prediction system works",
        "A transparent overview of the beginner-friendly machine-learning workflow "
        "behind this dashboard.",
    )
    st.markdown("### Run the application locally")
    st.markdown("Open a terminal in the project folder and run these steps in order.")
    st.code(
        "pip install -r requirements.txt\n"
        "python download_data.py\n"
        "python train_model.py",
        language="bash",
    )
    st.markdown("Start the backend and dashboard in **two separate terminals**:")
    st.code("uvicorn api:app --reload", language="bash")
    st.code("streamlit run app.py", language="bash")
    st.markdown("### What happens behind the scenes")
    guide_columns = st.columns(2)
    with guide_columns[0]:
        st.markdown(
            '<div class="panel"><b>1 · Prepare the data</b>'
            '<p class="panel-note">The training script identifies the target, '
            'separates features, and builds preprocessing for numeric and categorical data.</p>'
            '<b>2 · Train the model</b><p class="panel-note">A Logistic Regression '
            'classifier is trained using a stratified train/test split.</p></div>',
            unsafe_allow_html=True,
        )
    with guide_columns[1]:
        st.markdown(
            '<div class="panel"><b>3 · Serve predictions</b>'
            '<p class="panel-note">FastAPI loads the saved preprocessing-and-model pipeline '
            'and validates incoming customer fields.</p>'
            '<b>4 · Review results</b><p class="panel-note">Streamlit displays the predicted '
            'class and model-estimated churn probability.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### Important limitations")
    st.warning(
        "This is an educational prototype. A probability is not a certainty, "
        "and model performance depends on data quality and representativeness. "
        "Do not use predictions as the sole basis for decisions about customers. "
        "Evaluate the model and validate its performance before real-world use."
    )
    st.markdown("### Backend connection")
    st.code(f"CHURN_API_URL={API_URL}", language="text")
    if api_ready:
        st.success(f"Connected · {len(features)} model input fields detected.")
    else:
        st.info("Not connected. Check that the API is running and the model has been trained.")

st.markdown(
    '<div class="footer">ChurnIQ · Customer Churn Prediction System · '
    'Model outputs are estimates and should be reviewed with business context.</div>',
    unsafe_allow_html=True,
)
