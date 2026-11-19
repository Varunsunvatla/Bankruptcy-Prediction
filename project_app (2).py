# -*- coding: utf-8 -*-
"""Bankruptcy Prediction Streamlit Application."""

import os
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Prevention System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)


# ---------------------------------------------------------
# BLACK & MAROON THEME
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #0B0B0F;
            --panel: #15151C;
            --panel-soft: #1D1B23;
            --maroon: #7A1838;
            --maroon-light: #A52A50;
            --text: #F5F1F3;
            --muted: #B7AEB3;
            --border: #35232C;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(11, 11, 15, 0.95);
        }

        [data-testid="stSidebar"] {
            background: #111015;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        .main-title {
            color: var(--text);
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.25rem;
        }

        .sub-title {
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: 1.6rem;
        }

        .section-title {
            color: #F5F1F3;
            font-size: 1.35rem;
            font-weight: 750;
            margin: 0.7rem 0 0.9rem 0;
        }

        .info-card {
            background: linear-gradient(145deg, #1B171D, #141319);
            border: 1px solid var(--border);
            border-left: 4px solid var(--maroon);
            border-radius: 13px;
            padding: 1.25rem 1.35rem;
            min-height: 145px;
        }

        .info-card h3 {
            color: #FFFFFF;
            margin-top: 0;
            font-size: 1.15rem;
        }

        .info-card p {
            color: var(--muted);
            line-height: 1.6;
        }

        .eyebrow {
            color: #D58AA1;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 18px;
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: var(--maroon);
            color: white;
            border: 1px solid var(--maroon-light);
            border-radius: 9px;
            min-height: 2.8rem;
            font-weight: 700;
            transition: 0.2s ease;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: var(--maroon-light);
            color: white;
            border-color: #C24A70;
        }

        .stSelectbox label,
        .stRadio label,
        .stTextInput label {
            color: #F0E9ED !important;
            font-weight: 600;
        }

        div[data-baseweb="select"] > div {
            background: #1A1820;
            border-color: #51303D;
            color: white;
        }

        div[data-baseweb="popover"] {
            color: #171219;
        }

        [data-testid="stAlert"] {
            border-radius: 10px;
        }

        hr {
            border-color: var(--border);
        }

        .result-card {
            background: #17151C;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.2rem;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 14px 0;">
            <div style="font-size: 2rem; font-weight: 850; color: #FFFFFF;">
                🏦 BPS
            </div>
            <div class="eyebrow">Bankruptcy Prevention System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "MAIN MENU",
        ["Dashboard", "Predict Bankruptcy", "Model Performance", "About Project"],
        label_visibility="visible",
    )

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Bankruptcy Classification")


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
if page == "Dashboard":
    st.markdown(
        '<div class="main-title">Business Risk Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Assess business risk indicators with your trained machine learning model.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input Features", "6")
    with col2:
        st.metric("Model Status", "Loaded" if model is not None else "Unavailable")
    with col3:
        st.metric("Prediction Type", "Classification")

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
                <h3>Intelligent Bankruptcy Prediction</h3>
                <p>
                    This application accepts six company risk indicators and uses
                    your trained model to classify the company as bankrupt or
                    non-bankrupt.
                </p>
                <p><b style="color:#D58AA1;">Model file:</b> model.pkl</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">Assessment Features</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
                <p>01 · Industrial Risk</p>
                <p>02 · Management Risk</p>
                <p>03 · Financial Flexibility</p>
                <p>04 · Credibility</p>
                <p>05 · Competitiveness</p>
                <p>06 · Operating Risk</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    if model_error:
        st.error(
            "The model could not be loaded. Make sure model.pkl is in the same "
            "folder as project_app.py."
        )
    else:
        st.success("Your model is loaded and ready. Open Predict Bankruptcy to assess a company.")

    st.info("Predictions are model outputs and should not be treated as guaranteed financial advice.")


# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
elif page == "Predict Bankruptcy":
    st.markdown(
        '<div class="main-title">Company Risk Assessment</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Choose a level for each indicator and generate a bankruptcy prediction.</div>',
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(
            "Unable to load model.pkl. Place model.pkl in the same directory "
            "as project_app.py, then restart the Streamlit app."
        )
        st.stop()

    st.markdown('<div class="section-title">Company Risk Indicators</div>', unsafe_allow_html=True)
    st.caption("Select Low, Medium, or High for each input. The numeric values are passed to the model.")

    levels = [0, 0.5, 1]
    risk_labels = {
        0: "0 - Low",
        0.5: "0.5 - Medium",
        1: "1 - High",
    }

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            industrial_risk = st.selectbox(
                "Industrial Risk", levels, format_func=lambda x: risk_labels[x]
            )
            management_risk = st.selectbox(
                "Management Risk", levels, format_func=lambda x: risk_labels[x]
            )
            financial_flexibility = st.selectbox(
                "Financial Flexibility", levels, format_func=lambda x: risk_labels[x]
            )

        with col2:
            credibility = st.selectbox(
                "Credibility", levels, format_func=lambda x: risk_labels[x]
            )
            competitiveness = st.selectbox(
                "Competitiveness", levels, format_func=lambda x: risk_labels[x]
            )
            operating_risk = st.selectbox(
                "Operating Risk", levels, format_func=lambda x: risk_labels[x]
            )

        submitted = st.form_submit_button("Generate Bankruptcy Prediction", use_container_width=True)

    if submitted:
        # Keep the same feature order used by the original project_app.py.
        input_data = np.array(
            [[
                industrial_risk,
                management_risk,
                financial_flexibility,
                credibility,
                competitiveness,
                operating_risk,
            ]],
            dtype=float,
        )

        try:
            prediction = model.predict(input_data)[0]

            st.divider()
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

            result_col, probability_col = st.columns(2)

            with result_col:
                if prediction == 1:
                    st.error("⚠️ Predicted Class: Bankruptcy")
                else:
                    st.success("✓ Predicted Class: Non-Bankruptcy")

            with probability_col:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0]
                    # For the original binary classification setup, class 1 is bankruptcy.
                    classes = list(getattr(model, "classes_", [0, 1]))
                    if 1 in classes:
                        bankruptcy_probability = float(probabilities[classes.index(1)])
                        st.metric(
                            "Estimated Bankruptcy Probability",
                            f"{bankruptcy_probability * 100:.2f}%",
                        )
                        st.progress(max(0.0, min(1.0, bankruptcy_probability)))
                    else:
                        st.metric("Prediction", str(prediction))
                else:
                    st.metric("Prediction", str(prediction))
                    st.caption("This model does not provide probability estimates.")

            st.caption(
                "This result is generated by the trained model and is not a guarantee "
                "of a company's actual financial condition."
            )

            st.markdown("#### Submitted Company Information")
            feature_names = [
                "Industrial Risk",
                "Management Risk",
                "Financial Flexibility",
                "Credibility",
                "Competitiveness",
                "Operating Risk",
            ]
            st.dataframe(
                { "Feature": feature_names, "Selected Value": input_data[0] },
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ---------------------------------------------------------
elif page == "Model Performance":
    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Information about the model used by this application.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>Performance metrics have not been added</h3>
            <p>
                The supplied project_app.py contains the prediction code and model
                loading step, but it does not include verified test accuracy,
                precision, recall, F1 score, ROC-AUC, or a confusion matrix.
            </p>
            <p>
                Add the evaluation results from your model-training notebook here
                when they are available. No performance values are displayed
                so that the dashboard does not show invented results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# ABOUT PROJECT PAGE
# ---------------------------------------------------------
elif page == "About Project":
    st.markdown(
        '<div class="main-title">About the Project</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">An overview of the Bankruptcy Prevention System.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Project Objective</div>', unsafe_allow_html=True)
    st.write(
        "The project uses six company-related risk indicators as input to a "
        "trained machine learning classification model. The application predicts "
        "whether the company is classified as bankrupt or non-bankrupt."
    )

    st.markdown('<div class="section-title">Input Features</div>', unsafe_allow_html=True)
    st.dataframe(
        {
            "Feature": [
                "Industrial Risk",
                "Management Risk",
                "Financial Flexibility",
                "Credibility",
                "Competitiveness",
                "Operating Risk",
            ],
            "Available Values": ["0, 0.5, 1"] * 6,
        },
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Technologies Used</div>', unsafe_allow_html=True)
    st.write("Python · NumPy · Joblib · Streamlit · Trained machine learning model")

    st.caption("Academic Machine Learning Project | Bankruptcy Prevention System")
