# Questionnaire-Based Obesity Prediction App

This project now uses an individual-level obesity dataset and a questionnaire-style interface.

## What it predicts

The web app does not ask for true body weight or obesity label as input. It predicts:

- obesity probability
- overweight or obesity probability
- most likely obesity class
- estimated body weight
- estimated BMI and BMI category

## Dataset

- `obesity_questionnaire_dataset.csv`
- Source: UCI "Estimation of Obesity Levels Based On Eating Habits and Physical Condition"

## Files

- `generate_eda.py`: generates EDA figures and summary statistics
- `train_personal_obesity_models.py`: trains the classifier and regressor
- `predict_personal_metrics.py`: runs questionnaire-based inference from the command line
- `web_app.py`: starts the local questionnaire web app
- `personal_obesity_bundle.joblib`: saved model bundle after training
- `personal_model_report.json`: validation metrics report
- `GA2_REPORT_DRAFT.md`: report draft aligned to the assignment rubric
- `GA2_SLIDES_OUTLINE.md`: suggested slide deck structure
- `requirements.txt`: Python dependencies

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Generate EDA outputs

```bash
python generate_eda.py
```

EDA figures and summary files will be saved to `eda_outputs/`.

## Train

```bash
python train_personal_obesity_models.py
```

## Run the web app

```bash
python web_app.py
```

Then open `http://127.0.0.1:8000` in your browser.

## Command-line inference

```bash
python predict_personal_metrics.py --gender Female --age 23 --height 1.7 --family-history yes --favc no --fcvc 2.4 --ncp 3 --caec Sometimes --smoke no --ch2o 2 --scc no --faf 1 --tue 0.6 --calc Sometimes --mtrans Walking
```

## Model design

- Classifier: `RandomForestClassifier`
- Regressor: `RandomForestRegressor`
- Inputs: age, height, gender, family history, eating habits, smoking, water intake, physical activity, technology use, alcohol consumption, and transportation
- Excluded from input: `Weight` and `NObeyesdad`

## Notes

- Height is entered in meters.
- Several numeric questionnaire fields use the original dataset scoring scale.
- Predicted weight and BMI are model estimates rather than medical measurements.
