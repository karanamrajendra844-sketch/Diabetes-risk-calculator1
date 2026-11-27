import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Keep diabetes.csv in the same folder as this file.
df = pd.read_csv("diabetes.csv")

FEATURES = ["Glucose", "BMI", "Age", "BloodPressure"]
TARGET = "Outcome"

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# In this dataset, zero is an invalid/missing measurement for these fields.
for col in ["Glucose", "BMI", "BloodPressure"]:
    X[col] = X[col].replace(0, np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ))
])

logistic_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test)) * 100
forest_accuracy = accuracy_score(y_test, forest_model.predict(X_test)) * 100

def predict_diabetes(glucose, bmi, age, blood_pressure, model_choice):
    values = [glucose, bmi, age, blood_pressure]
    if any(v is None for v in values):
        raise gr.Error("Please enter all four values.")
    if not (0 < glucose <= 500 and 0 < bmi <= 100 and
            1 <= age <= 120 and 0 < blood_pressure <= 250):
        raise gr.Error("Please check the entered values.")
    row = pd.DataFrame([{
        "Glucose": glucose, "BMI": bmi, "Age": age,
        "BloodPressure": blood_pressure
    }])
    if model_choice == "Logistic Regression":
        model, accuracy = logistic_model, logistic_accuracy
    else:
        model, accuracy = forest_model, forest_accuracy
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1] * 100)
    label = ("Positive model prediction (Outcome = 1)" if prediction == 1
             else "Negative model prediction (Outcome = 0)")
    return label, round(probability, 2), round(float(accuracy), 2)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Diabetes Risk Screening Assistant")
    gr.Markdown("Educational screening demonstration using four Pima dataset features.")
    with gr.Row():
        glucose = gr.Number(label="Glucose (mg/dL)", minimum=1, maximum=500)
        bmi = gr.Number(label="BMI", minimum=1, maximum=100)
    with gr.Row():
        age = gr.Number(label="Age (years)", minimum=1, maximum=120, precision=0)
        bp = gr.Number(label="Blood pressure (mmHg)", minimum=1, maximum=250)
    model_choice = gr.Dropdown(
        ["Logistic Regression", "Random Forest"],
        value="Random Forest", label="Choose model"
    )
    submit = gr.Button("Run screening demonstration", variant="primary")
    prediction_out = gr.Textbox(label="Model prediction")
    probability_out = gr.Number(label="Estimated model probability for Outcome=1 (%)")
    accuracy_out = gr.Number(label="Test-set accuracy (%)")
    submit.click(
        predict_diabetes,
        [glucose, bmi, age, bp, model_choice],
        [prediction_out, probability_out, accuracy_out]
    )
    gr.Markdown(
        "**Important:** Educational demonstration only, not a medical diagnostic tool. "
        "The model probability is not a clinically validated personal risk estimate. "
        "Consult a qualified healthcare professional for medical evaluation."
    )

demo.launch()
