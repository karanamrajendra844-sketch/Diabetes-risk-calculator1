# Diabetes Risk Screening Assistant

A learning/demo project based on the Pima Indians Diabetes Database.

## Included files
- `Diabetes_Risk_Screening_Assistant_Colab.ipynb`: Recommended Google Colab notebook.
- `app.py`: Gradio app for running locally or on a Python hosting service.
- `requirements.txt`: Python dependencies.

## Dataset
Download `diabetes.csv` from:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

The CSV should contain:
`Glucose`, `BMI`, `Age`, `BloodPressure`, and `Outcome`.

## Run in Google Colab
1. Upload the `.ipynb` file to Google Colab or open it from Google Drive.
2. Run cells from top to bottom.
3. When prompted, upload `diabetes.csv`.
4. Run the final cell and open the Gradio share link.

## Run app.py locally
1. Put `diabetes.csv` in the same folder as `app.py`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`

## Model notes
- Exactly four features are used: Glucose, BMI, Age, BloodPressure.
- The target is `Outcome` (0 or 1).
- Logistic Regression and Random Forest Classifier are trained separately.
- Zero values in Glucose, BMI, and BloodPressure are treated as missing and imputed using the training-set median through the pipeline.
- A stratified 80/20 train/test split is used.

## Important
This is an educational machine-learning demonstration, not a medical diagnostic tool.
The model probability is not a clinically validated estimate of an individual's diabetes risk.
The Pima dataset has a limited population and should not be assumed to generalize to everyone.
Do not enter real personal health information into a public demo.
