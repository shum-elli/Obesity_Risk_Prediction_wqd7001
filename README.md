# Obesity Risk Prediction

This repository contains the WQD7001 group assignment project for questionnaire-based obesity risk prediction. The project trains and compares multiple machine learning models, generates exploratory data analysis outputs, and provides a simple local web app for making predictions from lifestyle questionnaire inputs.

## Project Overview

The model predicts obesity-related weight categories using demographic and lifestyle features such as age, gender, family history, eating habits, water intake, physical activity, alcohol consumption, and technology usage.

The best model recorded in the included report is `HistGradientBoosting`.

## Repository Contents

- `web_app.py` - local web interface for entering questionnaire values and viewing predictions.
- `train_personal_obesity_models.py` - trains and evaluates the machine learning models.
- `predict_personal_metrics.py` - command-line prediction script.
- `generate_eda.py` - generates exploratory data analysis charts and summaries.
- `obesity_questionnaire_dataset.csv` - dataset used for training and analysis.
- `personal_model_report.json` - model metrics and training summary.
- `personal_obesity_bundle.joblib` - trained model bundle managed with Git LFS.
- `eda_outputs/` - generated charts and EDA summary files.
- `WQD7001_G6_Obesity_Risk_Prediction_Shum.ipynb` - project notebook.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you clone this repository, make sure Git LFS is installed so the trained model file is downloaded correctly:

```bash
git lfs install
git lfs pull
```

## Run the Web App

Start the local prediction web app:

```bash
python web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Train Models

To retrain the models:

```bash
python train_personal_obesity_models.py
```

## Generate EDA Outputs

To regenerate charts and summary files:

```bash
python generate_eda.py
```

## Notes

This project is intended for academic coursework and demonstration purposes. The predictions should not be treated as medical advice.
