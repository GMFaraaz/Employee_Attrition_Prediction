# Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave the company based on workplace, demographic, compensation and satisfaction-related features.

The project includes a Streamlit web app, a trained scikit-learn model, sample CSV files, and the original analysis notebook.

## Live App

**Streamlit:** https://employee-attrition-prediction-ccai.streamlit.app/

**GitHub:** https://github.com/GMFaraaz/Employee_Attrition_Prediction

No installation is required if you only want to use the live version.

## What the app can do

### 1. Single Employee Prediction
Enter an employee's details through the form and get:

- Predicted attrition: **YES / NO**
- Probability of attrition
- A simple risk interpretation

Some fields use readable dropdowns rather than raw numeric codes. For example:

- **Job Level:** Entry Level, Junior, Mid-Level, Senior, Executive
- **Stock Option Level:** No Stock Options, Low, Moderate, High

### 2. Batch Prediction
Upload a CSV containing the 25 model input features.

The app generates predictions and attrition probabilities and provides the results as a downloadable CSV.

Use:

`data/sample/employee_attrition_sample.csv`

as a quick example.

### 3. Batch Evaluation
Upload a CSV containing the same 25 model features **plus an `Attrition` column** containing only `Yes/No` or `1/0`.

The app calculates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Use:

`data/sample/employee_attrition_evaluation_sample.csv`

for a small ready-to-test example.

> The evaluation sample is provided for demonstrating the app's evaluation feature. It should not be treated as an independent test set for judging the final model.

## Model

The project uses a scikit-learn classification pipeline with preprocessing for categorical and numerical features.

The model was evaluated using a held-out test set and cross-validation.

| Metric | Result |
|---|---:|
| 5-Fold CV ROC-AUC | 0.835 |
| Held-Out Test ROC-AUC | 0.791 |
| Held-Out Test Accuracy | 74.8% |
| Held-Out Test Precision | 34.5% |
| Held-Out Test Recall | 63.8% |
| Held-Out Test F1 | 44.8% |

The deployed model is retrained using the available dataset before being saved for the application. Therefore, metrics shown when evaluating the full training dataset in the app should not be confused with the held-out test results above.

## Project Structure

```text
Employee_Attrition_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sample/
│       ├── employee_attrition_sample.csv
│       └── employee_attrition_evaluation_sample.csv
│
├── models/
│   └── employee_attrition_model.joblib
│
└── notebooks/
    └── Employee_Attrition_Analysis.ipynb
```

## Run it locally

### 1. Get the project

**Option A — Clone with Git**

From the GitHub repository page, click **Code → HTTPS**, copy the URL, then run:

```bash
git clone https://github.com/GMFaraaz/Employee_Attrition_Prediction.git
cd Employee_Attrition_Prediction
```

**Option B — Download ZIP**

On GitHub:

1. Click the green **Code** button.
2. Click **Download ZIP**.
3. Extract the ZIP.
4. Open the extracted `Employee_Attrition_Prediction` folder in VS Code.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

Streamlit will open the application in your browser. If it does not open automatically, use the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Using the sample files

### For Batch Prediction

Upload:

```text
data/sample/employee_attrition_sample.csv
```

This file contains the 25 model features and is intended for prediction-only testing.

### For Batch Evaluation

Upload:

```text
data/sample/employee_attrition_evaluation_sample.csv
```

This file contains the 25 model features plus:

```text
Attrition
```

The `Attrition` column must contain only:

```text
Yes
No
```

or:

```text
1
0
```

## Running the notebook

The analysis notebook is located at:

```text
notebooks/Employee_Attrition_Analysis.ipynb
```

To open it with Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

The notebook covers the data analysis, preprocessing, model development and evaluation used for the project.

## Deploying your own copy

The easiest free deployment option used for this project is **Streamlit Community Cloud**.

1. Push the project to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Choose **Deploy a public app from GitHub**.
4. Select the repository.
5. Select the `main` branch.
6. Set the main file path to:

```text
app.py
```

7. Choose an available app URL.
8. Deploy.

The repository must contain at least:

```text
app.py
requirements.txt
models/employee_attrition_model.joblib
```

and any data files required by the application.

After deployment, future pushes to the selected GitHub branch can update the deployed app.

## Other ways to use the project

You do not have to use the hosted website.

- **Live Streamlit app** — quickest option; no setup.
- **GitHub + Streamlit Community Cloud** — deploy your own public copy.
- **Clone the repository** — run and modify the project locally.
- **Download ZIP** — use the project without Git.
- **Jupyter Notebook** — inspect the analysis and model-development work.
- **Batch CSV mode** — test multiple employees at once.

## Requirements

The main dependencies are listed in `requirements.txt`. The project uses Python, Streamlit, pandas, NumPy, scikit-learn, joblib and the supporting packages required by the application and notebook.

A virtual environment is recommended so the project's packages do not interfere with other Python projects.

## Notes

This project is intended as an academic machine-learning application and demonstration. Attrition predictions are estimates produced by a trained model and should not be treated as certain outcomes or as the sole basis for employment decisions.
