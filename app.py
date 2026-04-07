import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Portable paths ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODEL_PATH    = BASE_DIR / "models" / "xgboost_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalRisk — Readmission Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #F0F4F8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top header bar ── */
.header-bar {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 2.5rem;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.header-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -0.3px;
}
.header-subtitle {
    font-size: 12px;
    color: #64748B;
    font-weight: 400;
    margin-top: 1px;
}
.header-badge {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

/* ── Main content wrapper ── */
.main-wrapper {
    padding: 2rem 2.5rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.75rem;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.kpi-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.kpi-icon.blue  { background: #EFF6FF; }
.kpi-icon.green { background: #F0FDF4; }
.kpi-icon.amber { background: #FFFBEB; }
.kpi-icon.slate { background: #F8FAFC; }
.kpi-value {
    font-size: 20px;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.kpi-label {
    font-size: 12px;
    color: #64748B;
    font-weight: 400;
    margin-top: 2px;
}

/* ── Section label ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.75rem;
}

/* ── Result card ── */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    height: 100%;
}
.result-card-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 1.25rem;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 1.25rem;
}
.risk-badge.high   { background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; }
.risk-badge.medium { background:#FFFBEB; color:#D97706; border:1px solid #FDE68A; }
.risk-badge.low    { background:#F0FDF4; color:#16A34A; border:1px solid #BBF7D0; }

/* ── Probability display ── */
.prob-display {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 1rem;
}
.prob-number {
    font-family: 'DM Mono', monospace;
    font-size: 52px;
    font-weight: 500;
    letter-spacing: -2px;
    line-height: 1;
}
.prob-number.high   { color: #DC2626; }
.prob-number.medium { color: #D97706; }
.prob-number.low    { color: #16A34A; }
.prob-unit {
    font-size: 22px;
    color: #94A3B8;
    font-weight: 400;
}

/* ── Risk track ── */
.risk-track-wrap { margin-bottom: 1.25rem; }
.risk-track {
    height: 8px;
    background: #F1F5F9;
    border-radius: 99px;
    overflow: hidden;
    position: relative;
}
.risk-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.4s ease;
}
.risk-fill.high   { background: linear-gradient(90deg, #FCA5A5, #DC2626); }
.risk-fill.medium { background: linear-gradient(90deg, #FCD34D, #D97706); }
.risk-fill.low    { background: linear-gradient(90deg, #86EFAC, #16A34A); }
.risk-track-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 5px;
    font-size: 10px;
    color: #CBD5E1;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── Risk message ── */
.risk-message {
    background: #F8FAFC;
    border-left: 3px solid #E2E8F0;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #475569;
    line-height: 1.5;
}
.risk-message.high   { border-left-color: #DC2626; background: #FEF2F2; color: #7F1D1D; }
.risk-message.medium { border-left-color: #D97706; background: #FFFBEB; color: #78350F; }
.risk-message.low    { border-left-color: #16A34A; background: #F0FDF4; color: #14532D; }

/* ── Summary grid ── */
.summary-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    height: 100%;
}
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}
.summary-item {
    padding: 0.85rem 0;
    border-bottom: 1px solid #F1F5F9;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.summary-item:last-child, .summary-item:nth-last-child(2) { border-bottom: none; }
.summary-key {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
    padding-right: 1rem;
}
.summary-val {
    font-size: 13px;
    color: #0F172A;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

/* ── SHAP card ── */
.shap-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-top: 1.25rem;
}
.shap-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}
.shap-title {
    font-size: 16px;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -0.3px;
}
.shap-desc {
    font-size: 12px;
    color: #64748B;
    margin-top: 3px;
    line-height: 1.5;
}
.legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #64748B;
    font-weight: 500;
    margin-left: 8px;
}
.legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] > div {
    padding: 0 !important;
}
.sidebar-header {
    background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%);
    padding: 1.5rem 1.25rem 1.25rem;
    margin-bottom: 0;
}
.sidebar-header-title {
    font-size: 15px;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: -0.2px;
}
.sidebar-header-sub {
    font-size: 11px;
    color: rgba(255,255,255,0.7);
    margin-top: 3px;
}
.sidebar-section {
    padding: 0.9rem 1.25rem 0;
}
.sidebar-section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-divider {
    height: 1px;
    background: #F1F5F9;
    margin: 0.5rem 1.25rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSidebar"] label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
[data-testid="stSidebar"] .stSlider > div > div {
    background: #EFF6FF !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    border-color: #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #3B82F6 !important; }

/* Remove default Streamlit padding inside main */
[data-testid="stVerticalBlock"] > div { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model()

# ── Feature display names ─────────────────────────────────────────────────────
feature_display_names = {
    'age': 'Age', 'time_in_hospital': 'Days in Hospital',
    'num_lab_procedures': 'Lab Procedures', 'num_procedures': 'Procedures',
    'num_medications': 'Medications', 'number_outpatient': 'Prior Outpatient',
    'number_emergency': 'Prior Emergency', 'number_inpatient': 'Prior Inpatient',
    'number_diagnoses': 'Diagnoses', 'meds_per_day': 'Meds / Day',
    'procedures_per_day': 'Procedures / Day', 'total_prior_visits': 'Total Prior Visits',
    'lab_per_day': 'Lab / Day', 'gender': 'Gender', 'change': 'Med. Change',
    'diabetesMed': 'Diabetes Medication', 'admission_type_id': 'Admission Type',
    'discharge_disposition_id': 'Discharge Type', 'admission_source_id': 'Admission Source',
    'insulin_No': 'No Insulin', 'insulin_Steady': 'Insulin Steady',
    'insulin_Up': 'Insulin ↑', 'metformin_No': 'No Metformin',
    'metformin_Steady': 'Metformin Steady', 'metformin_Up': 'Metformin ↑',
    'race_Asian': 'Asian', 'race_Caucasian': 'Caucasian',
    'race_Hispanic': 'Hispanic', 'race_Other': 'Other Race',
    'repaglinide_No': 'No Repaglinide', 'repaglinide_Steady': 'Repaglinide Steady',
    'repaglinide_Up': 'Repaglinide ↑', 'nateglinide_No': 'No Nateglinide',
    'nateglinide_Steady': 'Nateglinide Steady', 'nateglinide_Up': 'Nateglinide ↑',
    'glimepiride_No': 'No Glimepiride', 'glimepiride_Steady': 'Glimepiride Steady',
    'glimepiride_Up': 'Glimepiride ↑', 'glipizide_No': 'No Glipizide',
    'glipizide_Steady': 'Glipizide Steady', 'glipizide_Up': 'Glipizide ↑',
    'glyburide_No': 'No Glyburide', 'glyburide_Steady': 'Glyburide Steady',
    'glyburide_Up': 'Glyburide ↑', 'pioglitazone_No': 'No Pioglitazone',
    'pioglitazone_Steady': 'Pioglitazone Steady', 'pioglitazone_Up': 'Pioglitazone ↑',
    'rosiglitazone_No': 'No Rosiglitazone', 'rosiglitazone_Steady': 'Rosiglitazone Steady',
    'rosiglitazone_Up': 'Rosiglitazone ↑', 'acarbose_No': 'No Acarbose',
    'acarbose_Steady': 'Acarbose Steady', 'acarbose_Up': 'Acarbose ↑',
}

# ── Top header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div class="header-left">
        <div class="header-logo">🏥</div>
        <div>
            <div class="header-title">ClinicalRisk</div>
            <div class="header-subtitle">Readmission Intelligence Platform</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <span class="legend-pill"><span class="legend-dot" style="background:#16A34A"></span>Low Risk</span>
        <span class="legend-pill"><span class="legend-dot" style="background:#D97706"></span>Moderate</span>
        <span class="legend-pill"><span class="legend-dot" style="background:#DC2626"></span>High Risk</span>
        <div class="header-badge">XGBoost · AUC 0.63</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-header-title">👤 Patient Profile</div>
        <div class="sidebar-header-sub">Enter patient details to generate risk assessment</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🧬 Demographics</div></div>', unsafe_allow_html=True)
    age    = st.slider("Age", 5, 95, 55, step=10)
    gender = st.selectbox("Gender", ["Female", "Male"])
    race   = st.selectbox("Race / Ethnicity", ["Caucasian", "African American", "Asian", "Hispanic", "Other"])

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🏨 Admission Details</div></div>', unsafe_allow_html=True)
    admission_type_id = st.selectbox("Admission Type", [1,2,3,4,5,6,7,8],
        format_func=lambda x: {1:"Emergency",2:"Urgent",3:"Elective",4:"Newborn",
        5:"Not Available",6:"NULL",7:"Trauma Center",8:"Not Mapped"}.get(x,str(x)))
    discharge_disposition_id = st.selectbox("Discharge Disposition", [1,2,3,4,6,11,13,14,18,19,20,22,23,25],
        format_func=lambda x: {1:"Home",2:"Care/Nursing Facility",3:"SNF",4:"ICF",
        6:"Home w/ Health Service",11:"Expired",13:"Hospice/Home",14:"Hospice/Facility",
        18:"Not Available",19:"Expired at Home",20:"Expired in Hospital",
        22:"Rehab Facility",23:"Long-term Care",25:"Not Mapped"}.get(x,str(x)))
    admission_source_id = st.selectbox("Admission Source", [1,2,3,4,5,6,7,8,9,17,20,22],
        format_func=lambda x: {1:"Physician Referral",2:"Clinic Referral",3:"HMO Referral",
        4:"Transfer — Hospital",5:"Transfer — SNF",6:"Transfer — Other",7:"Emergency Room",
        8:"Court/Law Enforcement",9:"Not Available",17:"NULL",20:"Not Mapped",
        22:"Transfer — Critical Access"}.get(x,str(x)))

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🩺 Hospital Stay</div></div>', unsafe_allow_html=True)
    time_in_hospital    = st.slider("Days in Hospital", 1, 14, 4)
    num_lab_procedures  = st.slider("Lab Procedures", 1, 132, 45)
    num_procedures      = st.slider("Clinical Procedures", 0, 6, 1)
    num_medications     = st.slider("Medications Prescribed", 1, 81, 15)
    number_diagnoses    = st.slider("Number of Diagnoses", 1, 16, 7)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">📋 Prior Visit History</div></div>', unsafe_allow_html=True)
    number_outpatient = st.slider("Outpatient Visits", 0, 42, 0)
    number_emergency  = st.slider("Emergency Visits", 0, 76, 0)
    number_inpatient  = st.slider("Inpatient Visits", 0, 21, 0)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">💊 Medication Regime</div></div>', unsafe_allow_html=True)
    change      = st.selectbox("Medication Change During Visit", ["No", "Yes"])
    diabetesMed = st.selectbox("On Diabetes Medication", ["Yes", "No"])
    insulin     = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    metformin   = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])

    with st.expander("Additional medications"):
        glipizide    = st.selectbox("Glipizide",    ["No","Steady","Up","Down"])
        glyburide    = st.selectbox("Glyburide",    ["No","Steady","Up","Down"])
        glimepiride  = st.selectbox("Glimepiride",  ["No","Steady","Up","Down"])
        pioglitazone = st.selectbox("Pioglitazone", ["No","Steady","Up","Down"])
        rosiglitazone= st.selectbox("Rosiglitazone",["No","Steady","Up","Down"])
        repaglinide  = st.selectbox("Repaglinide",  ["No","Steady","Up","Down"])
        nateglinide  = st.selectbox("Nateglinide",  ["No","Steady","Up","Down"])
        acarbose     = st.selectbox("Acarbose",     ["No","Steady","Up","Down"])

# ── Input builder ─────────────────────────────────────────────────────────────
def set_onehot(d, drug, val):
    for suffix in ['Steady', 'Up']:
        col = f'{drug}_{suffix}'
        if col in d:
            d[col] = 1 if val == suffix else 0
    no_col = f'{drug}_No'
    if no_col in d:
        d[no_col] = 1 if val in ['No', 'Down'] else 0

def build_input():
    d = {col: 0 for col in feature_names}
    d.update({
        'age': age, 'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures, 'num_procedures': num_procedures,
        'num_medications': num_medications, 'number_outpatient': number_outpatient,
        'number_emergency': number_emergency, 'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses, 'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'meds_per_day': num_medications / time_in_hospital,
        'procedures_per_day': num_procedures / time_in_hospital,
        'total_prior_visits': number_outpatient + number_emergency + number_inpatient,
        'lab_per_day': num_lab_procedures / time_in_hospital,
        'gender': 1 if gender == "Male" else 0,
        'change': 1 if change == "Yes" else 0,
        'diabetesMed': 1 if diabetesMed == "Yes" else 0,
    })
    race_map = {'Asian':'race_Asian','Caucasian':'race_Caucasian','Hispanic':'race_Hispanic','Other':'race_Other'}
    col = race_map.get(race)
    if col and col in d:
        d[col] = 1
    for drug, val in [
        ('insulin',insulin),('metformin',metformin),('glipizide',glipizide),
        ('glyburide',glyburide),('glimepiride',glimepiride),('pioglitazone',pioglitazone),
        ('rosiglitazone',rosiglitazone),('repaglinide',repaglinide),
        ('nateglinide',nateglinide),('acarbose',acarbose),
    ]:
        set_onehot(d, drug, val)
    return pd.DataFrame([d])

input_df = build_input()
prob     = model.predict_proba(input_df)[0][1]
pct      = prob * 100

if prob >= 0.5:
    risk_level, risk_icon, risk_label = "high", "⚠️", "High Risk"
    risk_msg = "This patient has a high probability of readmission. Recommend enhanced discharge planning and early follow-up within 7 days."
elif prob >= 0.3:
    risk_level, risk_icon, risk_label = "medium", "⚡", "Moderate Risk"
    risk_msg = "This patient shows elevated readmission risk. Consider structured follow-up and medication reconciliation at discharge."
else:
    risk_level, risk_icon, risk_label = "low", "✓", "Low Risk"
    risk_msg = "This patient has a low probability of readmission under current clinical parameters. Standard discharge protocol is appropriate."

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# KPI strip
total_prior = number_outpatient + number_emergency + number_inpatient
st.markdown(f"""
<div class="kpi-strip">
    <div class="kpi-card">
        <div class="kpi-icon blue">📊</div>
        <div>
            <div class="kpi-value">101,766</div>
            <div class="kpi-label">Training Records</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon green">🎯</div>
        <div>
            <div class="kpi-value">AUC 0.63</div>
            <div class="kpi-label">Model Performance</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon amber">🏥</div>
        <div>
            <div class="kpi-value">{time_in_hospital}d</div>
            <div class="kpi-label">Current Stay Length</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon slate">📋</div>
        <div>
            <div class="kpi-value">{total_prior}</div>
            <div class="kpi-label">Total Prior Visits</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Two-column row: result + summary
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-title">Risk Assessment</div>
        <div class="risk-badge {risk_level}">{risk_icon} {risk_label}</div>
        <div class="prob-display">
            <span class="prob-number {risk_level}">{pct:.1f}</span>
            <span class="prob-unit">%</span>
        </div>
        <div class="risk-track-wrap">
            <div class="risk-track">
                <div class="risk-fill {risk_level}" style="width:{pct:.1f}%"></div>
            </div>
            <div class="risk-track-labels">
                <span>0%</span><span>Low · &lt;30%</span><span>Moderate · 30–50%</span><span>High · &gt;50%</span><span>100%</span>
            </div>
        </div>
        <div class="risk-message {risk_level}">{risk_msg}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    procs_per_day = round(num_procedures / time_in_hospital, 2)
    meds_per_day  = round(num_medications / time_in_hospital, 1)
    lab_per_day   = round(num_lab_procedures / time_in_hospital, 1)
    st.markdown(f"""
    <div class="summary-card">
        <div class="result-card-title">Patient Summary</div>
        <div class="summary-grid">
            <div class="summary-item">
                <span class="summary-key">Age</span>
                <span class="summary-val">{age} yrs</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Gender</span>
                <span class="summary-val">{gender}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Days in Hospital</span>
                <span class="summary-val">{time_in_hospital}d</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Diagnoses</span>
                <span class="summary-val">{number_diagnoses}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Medications</span>
                <span class="summary-val">{num_medications}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Meds / Day</span>
                <span class="summary-val">{meds_per_day}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Procedures</span>
                <span class="summary-val">{num_procedures}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Procedures / Day</span>
                <span class="summary-val">{procs_per_day}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Lab Procedures</span>
                <span class="summary-val">{num_lab_procedures}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Lab / Day</span>
                <span class="summary-val">{lab_per_day}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Prior Visits</span>
                <span class="summary-val">{total_prior}</span>
            </div>
            <div class="summary-item">
                <span class="summary-key">Diabetes Med</span>
                <span class="summary-val">{diabetesMed}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# SHAP chart
st.markdown("""
<div class="shap-card">
    <div class="shap-header">
        <div>
            <div class="shap-title">Feature Contribution Analysis</div>
            <div class="shap-desc">SHAP values showing which clinical factors increased or decreased this patient's readmission risk</div>
        </div>
        <div>
            <span class="legend-pill"><span class="legend-dot" style="background:#DC2626"></span>Increases risk</span>
            <span class="legend-pill"><span class="legend-dot" style="background:#3B82F6"></span>Reduces risk</span>
        </div>
    </div>
""", unsafe_allow_html=True)

with st.spinner("Computing SHAP explanation..."):
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(input_df)
    display_names = [feature_display_names.get(f, f.replace('_',' ').title()) for f in feature_names]

    # Custom horizontal bar chart — cleaner than default SHAP waterfall
    vals  = shap_vals[0]
    pairs = sorted(zip(vals, display_names, input_df.iloc[0].values), key=lambda x: abs(x[0]), reverse=True)[:10]
    pairs = list(reversed(pairs))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    colors = ['#DC2626' if v > 0 else '#3B82F6' for v, _, _ in pairs]
    ypos   = range(len(pairs))
    bars   = ax.barh(list(ypos), [v for v,_,_ in pairs], color=colors,
                     alpha=0.85, height=0.55, zorder=3)

    # Value labels
    for bar, (v, _, fv) in zip(bars, pairs):
        x = bar.get_width()
        ax.text(x + (0.01 if x >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{v:+.3f}', va='center', ha='left' if x >= 0 else 'right',
                fontsize=9, color='#374151', fontweight='500',
                fontfamily='monospace')

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([n for _,n,_ in pairs], fontsize=11, color='#374151', fontfamily='DejaVu Sans')
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=10, color='#64748B', labelpad=8)
    ax.axvline(0, color='#E2E8F0', linewidth=1.5, zorder=2)
    ax.tick_params(axis='x', colors='#94A3B8', labelsize=9)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#F1F5F9', linewidth=1, zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, transparent=True)
    plt.close()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-size:11px;color:#CBD5E1;font-family:'DM Sans',sans-serif;">
    ClinicalRisk · XGBoost model trained on 101,766 diabetic patient records (1999–2008) ·
    For clinical decision support only — not a substitute for physician judgement ·
    Muntazir Ali Mughal · MSc Data Science, King's College London
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)