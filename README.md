# Employee Attrition Prediction

A predictive analytics project that estimates the likelihood of employee attrition using machine learning.

## Overview

The project uses the IBM HR Analytics Employee Attrition dataset to:
- explore factors related to employee attrition
- select a useful set of features
- compare classification models
- evaluate the final model
- provide an interactive Streamlit application for predictions

> The dataset is synthetic and is intended for analytics practice. It does not represent real IBM employee records.

## Model

**Final model:** Logistic Regression  
**Selected features:** 25

| Metric | Result |
|---|---:|
| 5-Fold CV ROC-AUC | 0.835 |
| Test Accuracy | 0.748 |
| Test Precision | 0.345 |
| Test Recall | 0.638 |
| Test F1 Score | 0.448 |
| Test ROC-AUC | 0.791 |

The trained model is saved in `models/employee_attrition_model.joblib`.

## Streamlit App

The application provides:

- **Individual Prediction** — enter employee details and get an attrition prediction with probability.
- **Batch Prediction** — upload a CSV and generate predictions for multiple employees.
- **Batch Evaluation** — evaluate predictions when an `Attrition` column is available.
- **About the Model** — view the selected features and reported performance.

A sample prediction file is included at:

`data/sample/employee_attrition_sample.csv`

## Project Structure

```text
Employee_Attrition_Prediction/
├── data/
│   └── sample/
├── models/
│   └── employee_attrition_model.joblib
├── notebooks/
│   └── Employee_Attrition_Analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

The original dataset is kept outside the repository and is ignored by Git.

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Start the application with:

```bash
streamlit run app.py
```

The app will open in your browser.

## Workflow

```text
Data → EDA → Feature Selection → Model Comparison
    → Cross-Validation → Final Evaluation → Deployment
```

## Note

Predictions are model estimates and should not be treated as guaranteed outcomes or used as the sole basis for employment decisions.
