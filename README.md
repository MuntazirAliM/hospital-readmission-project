# Hospital Readmission Prediction

A machine learning project that predicts whether a diabetic patient will be readmitted to hospital within 30 days of discharge. Built as a personal data science portfolio project to demonstrate an end-to-end ML pipeline from raw data to deployed interactive application.

---

## Live Demo

Run the app locally with:
```bash
streamlit run app.py
```

---

## Project Structure
```
hospital-readmission-project/
├── data/
│   ├── diabetic_data.csv          # Raw dataset
│   ├── IDs_mapping.csv            # ID reference file
│   └── cleaned_data.csv           # Processed dataset used for modelling
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Cleaning and feature creation
│   ├── 03_modelling.ipynb         # Model training and evaluation
│   └── 04_shap_analysis.ipynb     # Model interpretability
├── images/
│   ├── readmission_distribution.png
│   ├── feature_distributions.png
│   ├── roc_curves.png
│   ├── shap_bar.png
│   └── shap_dot.png
├── models/
│   ├── xgboost_model.pkl          # Saved XGBoost model
│   └── feature_names.pkl          # Feature list used during training
├── app.py                         # Streamlit web application
└── README.md
```

---

## Dataset

**Diabetes 130-US Hospitals (1999–2008)**
- 101,766 patient records
- 50 features including demographics, diagnoses, medications, and lab results
- Source: UCI Machine Learning Repository via Kaggle
- Target variable: readmitted within 30 days (binary classification)

---

## Approach

**Exploratory Data Analysis**
Investigated missing values, class imbalance, and feature distributions. Found that only 11.2% of patients were readmitted within 30 days, confirming a significant class imbalance problem.

**Feature Engineering**
Created four derived features not present in the raw data:
- `meds_per_day` — medication count relative to length of stay
- `procedures_per_day` — procedural burden normalised by stay duration
- `total_prior_visits` — combined outpatient, emergency, and inpatient history
- `lab_per_day` — diagnostic intensity relative to stay length

Handled missing values by dropping columns with over 40% missing data and imputing the remainder with mode values. Converted age brackets to numeric midpoints and one-hot encoded categorical medication columns.

**Modelling**
Trained three models on SMOTE-balanced data to address class imbalance:

| Model | AUC-ROC |
|---|---|
| XGBoost | 0.6325 |
| Random Forest | 0.6135 |
| Logistic Regression | 0.5472 |

XGBoost was selected as the final model. Results are consistent with published benchmarks on this dataset (0.62–0.68 AUC), reflecting the genuine difficulty of predicting readmission from clinical records alone.

**Interpretability**
Used SHAP (SHapley Additive exPlanations) to explain individual predictions. Key findings:
- `procedures_per_day` and `total_prior_visits` (both engineered features) were among the strongest predictors
- Patients not on insulin showed higher readmission risk — a counterintuitive finding suggesting undertreated diabetes
- Medication changes during a visit were associated with increased risk, indicating treatment instability

---

## Streamlit App

The app allows users to enter patient details via interactive sliders and dropdowns and receive:
- A readmission risk probability
- A colour-coded risk banner (low / moderate / high)
- A SHAP waterfall chart explaining the individual prediction

---
### App Screenshots

![ROC Curves](images/roc_curves.png)
![SHAP Analysis](images/shap_dot.png)

---
## Tech Stack

- **Python 3.13**
- **pandas, NumPy** — data manipulation
- **scikit-learn** — modelling and evaluation
- **XGBoost** — gradient boosted classifier
- **imbalanced-learn** — SMOTE oversampling
- **SHAP** — model interpretability
- **Streamlit** — interactive web application
- **matplotlib, seaborn** — visualisation

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/hospital-readmission-project.git
cd hospital-readmission-project
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap streamlit matplotlib seaborn
```

3. Run the app
```bash
streamlit run app.py
```

---

## Author

**Muntazir Ali Mughal**  
MSc Data Science — King's College London  
muntaziralimughal6@gmail.com