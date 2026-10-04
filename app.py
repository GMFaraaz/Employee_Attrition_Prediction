import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊", layout="wide")

MODEL_PATH = "models/employee_attrition_model.joblib"
FEATURES = [
    "JobRole", "EducationField", "BusinessTravel", "OverTime", "MaritalStatus",
    "Department", "TotalWorkingYears", "JobLevel", "YearsSinceLastPromotion",
    "NumCompaniesWorked", "YearsWithCurrManager", "EnvironmentSatisfaction",
    "JobSatisfaction", "Gender", "DistanceFromHome", "JobInvolvement",
    "MonthlyIncome", "Age", "YearsAtCompany", "RelationshipSatisfaction",
    "PercentSalaryHike", "YearsInCurrentRole", "WorkLifeBalance", "DailyRate",
    "StockOptionLevel"
]

PROFILES = {
    "Custom Employee": {
        "Age": 30, "JobRole": "Sales Executive", "BusinessTravel": "Travel_Rarely",
        "Department": "Sales", "DistanceFromHome": 10, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3, "Gender": "Male", "JobInvolvement": 3, "JobLevel": 2,
        "JobSatisfaction": 3, "MaritalStatus": "Single", "MonthlyIncome": 5000,
        "NumCompaniesWorked": 2, "OverTime": "Yes", "PercentSalaryHike": 15,
        "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 8,
        "YearsAtCompany": 4, "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2, "WorkLifeBalance": 3, "DailyRate": 800,
    },
    "High-Risk Example": {
        "Age": 24, "JobRole": "Sales Representative", "BusinessTravel": "Travel_Frequently",
        "Department": "Sales", "DistanceFromHome": 25, "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1, "Gender": "Male", "JobInvolvement": 1, "JobLevel": 1,
        "JobSatisfaction": 1, "MaritalStatus": "Single", "MonthlyIncome": 2500,
        "NumCompaniesWorked": 5, "OverTime": "Yes", "PercentSalaryHike": 11,
        "RelationshipSatisfaction": 1, "StockOptionLevel": 0, "TotalWorkingYears": 2,
        "YearsAtCompany": 1, "YearsInCurrentRole": 0, "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0, "WorkLifeBalance": 1, "DailyRate": 500,
    },
    "Low-Risk Example": {
        "Age": 45, "JobRole": "Research Director", "BusinessTravel": "Travel_Rarely",
        "Department": "Research & Development", "DistanceFromHome": 3, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 4, "Gender": "Male", "JobInvolvement": 4, "JobLevel": 5,
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 15000,
        "NumCompaniesWorked": 1, "OverTime": "No", "PercentSalaryHike": 20,
        "RelationshipSatisfaction": 4, "StockOptionLevel": 3, "TotalWorkingYears": 22,
        "YearsAtCompany": 20, "YearsInCurrentRole": 8, "YearsSinceLastPromotion": 3,
        "YearsWithCurrManager": 8, "WorkLifeBalance": 4, "DailyRate": 1200,
    },
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
actual_schema = list(getattr(model, "feature_names_in_", []))
if actual_schema != FEATURES:
    st.error("The saved model schema does not match the application's expected 25 features.")
    st.write("Expected:", FEATURES)
    st.write("Found:", actual_schema)
    st.stop()

def predict(df):
    x = df[FEATURES]
    return model.predict(x), model.predict_proba(x)[:, 1]

def check_columns(df):
    return [c for c in FEATURES if c not in df.columns]

def label(v):
    return "Yes" if int(v) == 1 else "No"

st.sidebar.title("Employee Attrition")
st.sidebar.caption("Predictive Analytics Project")
page = st.sidebar.radio("Navigation", [
    "🏠 Overview", "👤 Individual Prediction", "📂 Batch Prediction",
    "📊 Batch Evaluation", "ℹ️ About the Model"
])
st.sidebar.divider()
st.sidebar.caption("Model: Logistic Regression")
st.sidebar.caption("Selected features: 25")

if page == "🏠 Overview":
    st.title("Employee Attrition Prediction")
    st.subheader("Predictive Analytics Project")
    st.write("Estimate the likelihood that an employee will leave the organization using the trained machine-learning model.")
    st.info("The model uses the IBM HR Analytics Employee Attrition dataset. This is a synthetic analytics dataset and does not represent real IBM employee records.")

    a, b = st.columns(2)
    with a:
        st.markdown("### 👤 Individual Prediction\nEnter one employee's information and receive a Yes/No prediction and attrition probability. Built-in hypothetical high- and low-risk profiles are included.")
    with b:
        st.markdown("### 📂 Batch Prediction\nUpload a CSV containing the model's 25 required features, generate predictions for every row, and download the results.")

    st.header("Model Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Records", "1,470")
    c2.metric("Selected Features", "25")
    c3.metric("CV ROC-AUC", "0.835")
    c4.metric("Test ROC-AUC", "0.791")

    st.header("Workflow")
    st.markdown("**Employee information → preprocessing → Logistic Regression → attrition prediction + probability**")

elif page == "👤 Individual Prediction":
    st.title("Individual Employee Prediction")
    st.write("Enter employee information or choose a built-in hypothetical profile.")
    profile_name = st.selectbox("Employee Profile", list(PROFILES))
    p = PROFILES[profile_name]
    if profile_name == "High-Risk Example":
        st.warning("Hypothetical demonstration profile designed to illustrate a higher-risk prediction.")
    elif profile_name == "Low-Risk Example":
        st.success("Hypothetical demonstration profile designed to illustrate a lower-risk prediction.")

    k = profile_name.replace(" ", "_").replace("-", "_").lower()
    with st.form("employee_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 100, p["Age"], key=f"age_{k}")
            bt = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
            business_travel = st.selectbox("Business Travel", bt, index=bt.index(p["BusinessTravel"]), key=f"bt_{k}")
            dept = ["Sales", "Research & Development", "Human Resources"]
            department = st.selectbox("Department", dept, index=dept.index(p["Department"]), key=f"dept_{k}")
            distance = st.number_input("Distance From Home", 1, 100, p["DistanceFromHome"], key=f"distance_{k}")
            edu = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
            education = st.selectbox("Education Field", edu, index=edu.index(p["EducationField"]), key=f"edu_{k}")
            env = st.slider("Environment Satisfaction", 1, 4, p["EnvironmentSatisfaction"], key=f"env_{k}")
            genders = ["Male", "Female"]
            gender = st.selectbox("Gender", genders, index=genders.index(p["Gender"]), key=f"gender_{k}")
            involvement = st.slider("Job Involvement", 1, 4, p["JobInvolvement"], key=f"involvement_{k}")
            job_level_options = {
                "Level 1 — Entry / Lowest": 1,
                "Level 2 — Junior / Intermediate": 2,
                "Level 3 — Mid-Level": 3,
                "Level 4 — Senior": 4,
                "Level 5 — Highest": 5,
            }
            job_level_label = st.selectbox(
                "Job Level",
                list(job_level_options.keys()),
                index=list(job_level_options.values()).index(p["JobLevel"]),
                key=f"level_{k}",
            )
            job_level = job_level_options[job_level_label]
        with c2:
            roles = ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"]
            role = st.selectbox("Job Role", roles, index=roles.index(p["JobRole"]), key=f"role_{k}")
            satisfaction = st.slider("Job Satisfaction", 1, 4, p["JobSatisfaction"], key=f"satisfaction_{k}")
            marital = ["Single", "Married", "Divorced"]
            marital_status = st.selectbox("Marital Status", marital, index=marital.index(p["MaritalStatus"]), key=f"marital_{k}")
            income = st.number_input("Monthly Income", 100, 100000, p["MonthlyIncome"], key=f"income_{k}")
            companies = st.number_input("Number of Companies Worked", 0, 20, p["NumCompaniesWorked"], key=f"companies_{k}")
            overtime_options = ["Yes", "No"]
            overtime = st.selectbox("Overtime", overtime_options, index=overtime_options.index(p["OverTime"]), key=f"overtime_{k}")
            hike = st.number_input("Percent Salary Hike", 0, 100, p["PercentSalaryHike"], key=f"hike_{k}")
            relationship = st.slider("Relationship Satisfaction", 1, 4, p["RelationshipSatisfaction"], key=f"relationship_{k}")
            stock_options = {
                "No stock options": 0,
                "Low stock-option benefit": 1,
                "Medium stock-option benefit": 2,
                "High stock-option benefit": 3,
            }
            stock_label = st.selectbox(
                "Stock Options",
                list(stock_options.keys()),
                index=list(stock_options.values()).index(p["StockOptionLevel"]),
                key=f"stock_{k}",
                help="A relative category for the employee's stock-option benefit. The dataset does not provide a specific monetary value for these categories.",
            )
            stock = stock_options[stock_label]
        with c3:
            total_years = st.number_input("Total Working Years", 0, 60, p["TotalWorkingYears"], key=f"total_{k}")
            company_years = st.number_input("Years At Company", 0, 50, p["YearsAtCompany"], key=f"company_{k}")
            role_years = st.number_input("Years In Current Role", 0, 30, p["YearsInCurrentRole"], key=f"roleyears_{k}")
            promotion = st.number_input("Years Since Last Promotion", 0, 30, p["YearsSinceLastPromotion"], key=f"promotion_{k}")
            manager_years = st.number_input("Years With Current Manager", 0, 30, p["YearsWithCurrManager"], key=f"manager_{k}")
            worklife = st.slider("Work Life Balance", 1, 4, p["WorkLifeBalance"], key=f"worklife_{k}")
            daily_rate = st.number_input("Daily Rate", 0, 2000, p["DailyRate"], key=f"daily_{k}")

        st.divider()
        submitted = st.form_submit_button("Predict Attrition", use_container_width=True)

    if submitted:
        employee = pd.DataFrame([{
            "JobRole": role, "EducationField": education, "BusinessTravel": business_travel,
            "OverTime": overtime, "MaritalStatus": marital_status, "Department": department,
            "TotalWorkingYears": total_years, "JobLevel": job_level,
            "YearsSinceLastPromotion": promotion, "NumCompaniesWorked": companies,
            "YearsWithCurrManager": manager_years, "EnvironmentSatisfaction": env,
            "JobSatisfaction": satisfaction, "Gender": gender, "DistanceFromHome": distance,
            "JobInvolvement": involvement, "MonthlyIncome": income, "Age": age,
            "YearsAtCompany": company_years, "RelationshipSatisfaction": relationship,
            "PercentSalaryHike": hike, "YearsInCurrentRole": role_years,
            "WorkLifeBalance": worklife, "DailyRate": daily_rate, "StockOptionLevel": stock,
        }])
        pred, prob = predict(employee)
        r1, r2 = st.columns(2)
        with r1:
            if int(pred[0]) == 1:
                st.error("⚠️ Predicted Attrition: YES")
            else:
                st.success("✅ Predicted Attrition: NO")
        with r2:
            st.metric("Probability of Attrition", f"{prob[0]:.1%}")

elif page == "📂 Batch Prediction":
    st.title("Batch CSV Prediction")
    st.write("Upload a CSV containing all 25 required model features. Additional columns are allowed and will be preserved.")
    uploaded = st.file_uploader("Upload employee CSV", type=["csv"], key="batch")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()
        missing = check_columns(df)
        if missing:
            st.error("The CSV is missing required model features.")
            st.write(missing)
        elif df.empty:
            st.warning("The CSV contains no rows.")
        else:
            pred, prob = predict(df)
            results = df.copy()
            results["PredictedAttrition"] = [label(x) for x in pred]
            results["AttritionProbability"] = prob
            st.success(f"Predictions generated for {len(results):,} employees.")
            st.dataframe(results, use_container_width=True)
            st.download_button("⬇️ Download Predictions CSV", results.to_csv(index=False).encode("utf-8"), "employee_attrition_predictions.csv", "text/csv", use_container_width=True)

elif page == "📊 Batch Evaluation":
    st.title("Batch Evaluation")
    st.write("Upload a CSV containing the 25 model features plus an actual `Attrition` column to evaluate predictions.")
    uploaded = st.file_uploader("Upload labelled employee CSV", type=["csv"], key="evaluation")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()
        missing = check_columns(df)
        if missing:
            st.error("The CSV is missing required model features.")
            st.write(missing)
            st.stop()
        if "Attrition" not in df.columns:
            st.warning("No `Attrition` column was found, so performance metrics cannot be calculated. Use Batch Prediction for prediction-only CSVs.")
            st.stop()
        if df.empty:
            st.warning("The CSV contains no rows.")
            st.stop()

        actual_raw = df["Attrition"]
        if actual_raw.dtype == object:
            actual_raw = actual_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
        actual = pd.to_numeric(actual_raw, errors="coerce")
        if actual.isna().any() or not set(actual.astype(int).unique()).issubset({0, 1}):
            st.error("`Attrition` must contain only Yes/No or 1/0 values.")
            st.stop()
        actual = actual.astype(int)
        pred, prob = predict(df)
        metrics = [accuracy_score(actual, pred), precision_score(actual, pred, zero_division=0), recall_score(actual, pred, zero_division=0), f1_score(actual, pred, zero_division=0)]
        auc = roc_auc_score(actual, prob) if actual.nunique() == 2 else None
        cols = st.columns(5)
        for col, name, value in zip(cols[:4], ["Accuracy", "Precision", "Recall", "F1 Score"], metrics):
            col.metric(name, f"{value:.1%}")
        cols[4].metric("ROC-AUC", f"{auc:.1%}" if auc is not None else "N/A")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(actual, pred, labels=[0, 1])
        st.dataframe(pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"]), use_container_width=True)
        results = df.copy()
        results["PredictedAttrition"] = [label(x) for x in pred]
        results["AttritionProbability"] = prob
        st.subheader("Prediction Details")
        st.dataframe(results, use_container_width=True)
        st.download_button("⬇️ Download Evaluation Results CSV", results.to_csv(index=False).encode("utf-8"), "employee_attrition_evaluation_results.csv", "text/csv", use_container_width=True)

else:
    st.title("About the Model")
    st.header("Dataset")
    st.write("The project uses the IBM HR Analytics Employee Attrition dataset with 1,470 records and 35 original columns. It is a synthetic analytics dataset and should not be interpreted as real IBM employee records.")
    st.header("Selected Features")
    st.dataframe(pd.DataFrame({"Feature": FEATURES}), use_container_width=True, hide_index=True)
    st.header("Final Model")
    st.write("Logistic Regression inside the project's preprocessing pipeline.")
    st.header("Reported Performance")
    st.table(pd.DataFrame({
        "Metric": ["Mean 5-Fold Cross-Validated ROC-AUC", "Held-Out Test Accuracy", "Held-Out Test Precision", "Held-Out Test Recall", "Held-Out Test F1 Score", "Held-Out Test ROC-AUC"],
        "Value": ["0.835", "0.748", "0.345", "0.638", "0.448", "0.791"],
    }))
    st.warning("Predicted probability is a model estimate, not a guarantee. The application is intended as an analytical decision-support tool, not an automatic employment decision system.")
    st.header("Project Workflow")
    st.markdown("**Dataset → Data Quality → EDA → Train/Test Split → Feature Selection → Feature-Set Experiment → Model Comparison → Cross-Validation → Final Evaluation → Retraining → Deployment**")
