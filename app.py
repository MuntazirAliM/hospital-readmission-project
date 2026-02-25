import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Set up the browser tab title and layout
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS to tighten up the sidebar section headers and card styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 5px 0;
    }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        margin-top: 20px;
    }
    .sidebar-divider {
        border-top: 1px solid #333;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained XGBoost model and the list of feature names we trained it on
@st.cache_resource
def load_model():
    with open(r'C:\Users\munta\Documents\hospital-readmission-project\models\xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(r'C:\Users\munta\Documents\hospital-readmission-project\models\feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model()

# Human-readable labels for the SHAP chart so it doesn't show raw column names
feature_display_names = {
    'age': 'Age',
    'time_in_hospital': 'Days in Hospital',
    'num_lab_procedures': 'Lab Procedures',
    'num_procedures': 'Procedures',
    'num_medications': 'Medications',
    'number_outpatient': 'Prior Outpatient Visits',
    'number_emergency': 'Prior Emergency Visits',
    'number_inpatient': 'Prior Inpatient Visits',
    'number_diagnoses': 'Number of Diagnoses',
    'meds_per_day': 'Medications per Day',
    'procedures_per_day': 'Procedures per Day',
    'total_prior_visits': 'Total Prior Visits',
    'lab_per_day': 'Lab Procedures per Day',
    'gender': 'Gender',
    'change': 'Medication Change',
    'diabetesMed': 'On Diabetes Medication',
    'insulin_No': 'Not on Insulin',
    'insulin_Steady': 'Insulin — Steady',
    'insulin_Up': 'Insulin — Increased',
    'metformin_No': 'Not on Metformin',
    'metformin_Steady': 'Metformin — Steady',
    'metformin_Up': 'Metformin — Increased',
    'race_Asian': 'Race: Asian',
    'race_Caucasian': 'Race: Caucasian',
    'race_Hispanic': 'Race: Hispanic',
    'race_Other': 'Race: Other',
}

# Main page title and description
st.title("🏥 30-Day Readmission Risk — Diabetes Patients")
st.markdown("Use the sliders on the left to enter a patient's details. The model will estimate their risk of being readmitted within 30 days of discharge.")
st.markdown("---")

# Quick stats row at the top so the page feels like a real dashboard
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Model", "XGBoost")
col_b.metric("Training Records", "101,766")
col_c.metric("Model AUC", "0.63")
col_d.metric("Dataset", "1999–2008 US Hospitals")
st.markdown("---")

# Sidebar inputs split into logical sections so it's easy to fill in
st.sidebar.header("🧑‍⚕️ Patient Information")

st.sidebar.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
age = st.sidebar.slider("Age", 5, 95, 55, step=10)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">Hospital Stay</p>', unsafe_allow_html=True)
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 132, 45)
num_procedures = st.sidebar.slider("Number of Procedures", 0, 6, 1)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 15)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">Prior Visit History</p>', unsafe_allow_html=True)
number_outpatient = st.sidebar.slider("Prior Outpatient Visits", 0, 42, 0)
number_emergency = st.sidebar.slider("Prior Emergency Visits", 0, 76, 0)
number_inpatient = st.sidebar.slider("Prior Inpatient Visits", 0, 21, 0)

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">Medication Details</p>', unsafe_allow_html=True)
change = st.sidebar.selectbox("Medication Change During Visit?", ["No", "Yes"])
diabetesMed = st.sidebar.selectbox("On Diabetes Medication?", ["Yes", "No"])
insulin = st.sidebar.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"])
metformin = st.sidebar.selectbox("Metformin Status", ["No", "Steady", "Up", "Down"])

# Build a single-row dataframe from the sidebar inputs that exactly matches
# the column structure the model was trained on — every column must be present
def build_input():
    # Start everything at zero, then fill in what the user provided
    input_dict = {col: 0 for col in feature_names}

    # Numerical values straight from the sliders
    input_dict['age'] = age
    input_dict['time_in_hospital'] = time_in_hospital
    input_dict['num_lab_procedures'] = num_lab_procedures
    input_dict['num_procedures'] = num_procedures
    input_dict['num_medications'] = num_medications
    input_dict['number_outpatient'] = number_outpatient
    input_dict['number_emergency'] = number_emergency
    input_dict['number_inpatient'] = number_inpatient
    input_dict['number_diagnoses'] = number_diagnoses

    # Recreate the engineered features — the model was trained with these so we need them here too
    input_dict['meds_per_day'] = num_medications / time_in_hospital
    input_dict['procedures_per_day'] = num_procedures / time_in_hospital
    input_dict['total_prior_visits'] = number_outpatient + number_emergency + number_inpatient
    input_dict['lab_per_day'] = num_lab_procedures / time_in_hospital

    # Binary encode the yes/no fields to match how they were encoded during training
    input_dict['gender'] = 1 if gender == "Male" else 0
    input_dict['change'] = 1 if change == "Yes" else 0
    input_dict['diabetesMed'] = 1 if diabetesMed == "Yes" else 0

    # Flip the right one-hot insulin column based on what the user picked
    if insulin != "No":
        col = f'insulin_{insulin}'
        if col in input_dict:
            input_dict[col] = 1
    else:
        if 'insulin_No' in input_dict:
            input_dict['insulin_No'] = 1

    # Same for metformin
    if metformin != "No":
        col = f'metformin_{metformin}'
        if col in input_dict:
            input_dict[col] = 1
    else:
        if 'metformin_No' in input_dict:
            input_dict['metformin_No'] = 1

    return pd.DataFrame([input_dict])

input_df = build_input()

# Two column layout — prediction result on the left, patient summary on the right
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")

    prob = model.predict_proba(input_df)[0][1]

    # Show a different coloured banner depending on the risk level
    if prob >= 0.5:
        st.error("⚠️ This patient is likely to be readmitted within 30 days.")
    elif prob >= 0.3:
        st.warning("⚡ This patient has a moderate chance of readmission — worth monitoring.")
    else:
        st.success("✅ This patient is unlikely to be readmitted within 30 days.")

    st.metric(label="Readmission probability", value=f"{prob:.1%}")

    # Pick a bar colour that matches the risk level
    if prob >= 0.5:
        bar_color = "#e74c3c"
    elif prob >= 0.3:
        bar_color = "#f39c12"
    else:
        bar_color = "#2ecc71"

    # Custom progress bar since Streamlit's built-in one can't change colour
    st.markdown(f"""
        <div style="background-color:#2a2a2a; border-radius:8px; padding:4px;">
            <div style="background-color:{bar_color}; width:{prob*100:.1f}%; 
                        height:18px; border-radius:6px; transition: width 0.3s;">
            </div>
        </div>
        <p style="color:#888; font-size:12px; margin-top:4px;">{prob*100:.1f}% risk</p>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Patient Summary")
    summary_data = {
        "Detail": ["Age", "Days in Hospital", "Total Prior Visits",
                    "Medications", "Procedures per Day", "On Diabetes Med",
                    "Medication Changed"],
        "Value": [f"{age} years", f"{time_in_hospital} days",
                  number_outpatient + number_emergency + number_inpatient,
                  num_medications,
                  round(num_procedures / time_in_hospital, 2),
                  diabetesMed, change]
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

# SHAP waterfall chart showing which features drove this individual prediction
st.markdown("---")
st.subheader("What's driving this result?")
st.markdown("The chart below breaks down which factors pushed the risk up or down for this specific patient. **Red bars increase risk, blue bars reduce it.**")

with st.spinner("Calculating explanation..."):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)

    # Use the clean display names, with a fallback that tidies up any remaining raw column names
    display_names = [feature_display_names.get(f, f.replace('_', ' ').title()) for f in feature_names]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Match the chart background to the dark app theme
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=display_names
        ),
        show=False,
        max_display=10
    )

    # Make all axis text white so it shows up on the dark background
    ax.tick_params(colors='white', labelsize=11)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    # Subtle border lines
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    # Catch any remaining text elements and make them white too
    for text in fig.findobj(plt.Text):
        text.set_color('white')

    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()

# Footer
st.markdown("---")
st.caption("Personal Data Science Project | XGBoost model trained on 101,766 patient records | Muntazir Ali Mughal")