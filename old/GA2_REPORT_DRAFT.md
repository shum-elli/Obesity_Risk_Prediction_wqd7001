# WQD7001 Group Assignment 2 Report Draft

Replace the cover details below before submission:

- Group name:
- Topic: Questionnaire-Based Obesity Risk Prediction and Web Data Product
- Group members and roles:

## 1. Project Background

Obesity is a major public health issue linked to cardiovascular disease, diabetes, hypertension, and reduced quality of life. Health organisations, wellness platforms, educational institutions, and community healthcare teams increasingly need simple tools that can identify obesity risk early and guide preventive intervention. However, many people do not have direct access to clinical screening tools or structured assessment support.

This project develops a questionnaire-based data product that estimates obesity-related outcomes without asking for the user's true body weight as an input. The final system predicts:

- probability of obesity
- probability of overweight or obesity
- most likely obesity class
- estimated body weight
- estimated BMI and BMI category

The project is suitable for preventive health awareness settings, wellness screening, education, and research demonstrations. It is not intended to replace medical diagnosis.

## 2. Problem Statement

Many people are unaware of how their lifestyle and behavioural factors relate to obesity risk. Existing health checks often require direct measurements or clinical support, which may not be convenient in early self-screening settings. Therefore, there is a need for a data-driven tool that can analyse questionnaire responses and provide interpretable obesity-related predictions in a simple web interface.

## 3. Project Objectives

1. To identify important behavioural, demographic, and physical factors associated with obesity using exploratory data analysis.
2. To build predictive machine learning models that estimate obesity class probability and body weight from questionnaire-style inputs.
3. To evaluate the predictive performance of the models using appropriate classification and regression metrics.
4. To deploy the trained models as a simple web-based data product for non-technical users.

## 4. Project Scope and Domain with Justification

This project lies in the public health and preventive analytics domain, which aligns well with the Sustainable Development Goals, especially SDG 3: Good Health and Well-Being. The focus is limited to obesity-related risk estimation using one publicly available individual-level dataset. The scope includes:

- data understanding and EDA
- model building for classification and regression
- interpretation of results
- reproducible workflow
- deployment of a local web application

The scope excludes clinical diagnosis, treatment recommendation, and use of personally identifiable information.

## 5. Data Description and Collection

The dataset used is the UCI Machine Learning Repository dataset titled "Estimation of Obesity Levels Based On Eating Habits and Physical Condition". It contains 2,111 individual-level records and 17 variables.

Dataset file used in this project:

- [obesity_questionnaire_dataset.csv](/I:/Assignments/WQD7001/project/obesity_questionnaire_dataset.csv)

The dataset contains the following types of variables:

- Demographic and physical: `Gender`, `Age`, `Height`, `Weight`
- Family and lifestyle: `family_history_with_overweight`, `FAVC`, `FCVC`, `NCP`, `CAEC`, `SMOKE`, `CH2O`, `SCC`, `FAF`, `TUE`, `CALC`, `MTRANS`
- Target label: `NObeyesdad`

The dataset is reliable for coursework because it is sourced from a recognised public repository and widely used in obesity prediction examples. However, it includes synthetic elements and should be interpreted as an educational dataset rather than a perfect representation of real-world clinical populations.

## 6. Data Cleaning

Data cleaning and preparation steps performed in this project include:

1. Verified the dataset structure and schema using `pandas`.
2. Checked missing values across all columns. No missing values were found.
3. Preserved categorical fields as categorical questionnaire inputs.
4. Preserved numeric fields such as age, height, meal frequency, water intake, physical activity, and technology usage scores.
5. Excluded `Weight` and `NObeyesdad` from the web app questionnaire inputs to avoid target leakage.
6. Used preprocessing pipelines in model training:
   - median imputation for numeric variables
   - most-frequent imputation for categorical variables
   - standard scaling for numeric features
   - one-hot encoding for categorical features

Although the dataset had no missing values, the preprocessing pipeline still includes imputation to support reproducibility and future extensibility.

## 7. Exploratory Data Analysis

EDA outputs are stored in [eda_outputs](/I:/Assignments/WQD7001/project/eda_outputs).

### 7.1 Dataset overview

- Total rows: 2,111
- Total columns: 17
- Missing values: 0 across all fields
- Obesity prevalence in the dataset, defined as `Obesity_Type_I`, `Obesity_Type_II`, or `Obesity_Type_III`: 46.04%

### 7.2 Key findings

1. The class distribution is relatively balanced across the seven obesity categories, which is helpful for classification modelling. The largest class is `Obesity_Type_I` with 351 records, while the smallest is `Insufficient_Weight` with 272 records.
2. Family history appears strongly associated with obesity risk. Respondents with family history of overweight show an obesity prevalence of 55.85%, compared with only 2.08% for those without family history.
3. Average BMI is slightly higher for females than males in this dataset: 30.13 versus 29.28.
4. Transportation mode shows a visible relationship with BMI. The highest average BMI is seen in `Public_Transportation` at 30.11, while the lowest is seen in `Walking` at 23.66.
5. Physical activity is negatively associated with BMI. The correlation between BMI and `FAF` is -0.1775.
6. Age has a weak positive relationship with BMI with correlation 0.2442.

### 7.3 EDA visualisations produced

- [class_distribution.png](/I:/Assignments/WQD7001/project/eda_outputs/class_distribution.png)
- [age_distribution.png](/I:/Assignments/WQD7001/project/eda_outputs/age_distribution.png)
- [bmi_distribution.png](/I:/Assignments/WQD7001/project/eda_outputs/bmi_distribution.png)
- [family_history_obesity_rate.png](/I:/Assignments/WQD7001/project/eda_outputs/family_history_obesity_rate.png)
- [gender_bmi.png](/I:/Assignments/WQD7001/project/eda_outputs/gender_bmi.png)
- [transport_bmi.png](/I:/Assignments/WQD7001/project/eda_outputs/transport_bmi.png)
- [snacking_bmi_boxplot.png](/I:/Assignments/WQD7001/project/eda_outputs/snacking_bmi_boxplot.png)
- [alcohol_bmi_boxplot.png](/I:/Assignments/WQD7001/project/eda_outputs/alcohol_bmi_boxplot.png)
- [age_bmi_scatter.png](/I:/Assignments/WQD7001/project/eda_outputs/age_bmi_scatter.png)

## 8. Data Modelling

Two separate models were built:

1. Classification model to predict obesity class probabilities.
2. Regression model to estimate body weight, which is later used with height to compute BMI.

### 8.1 Model design

- Classification algorithm: `RandomForestClassifier`
- Regression algorithm: `RandomForestRegressor`
- Training script: [train_personal_obesity_models.py](/I:/Assignments/WQD7001/project/train_personal_obesity_models.py)
- Questionnaire inference script: [predict_personal_metrics.py](/I:/Assignments/WQD7001/project/predict_personal_metrics.py)
- Web deployment: [web_app.py](/I:/Assignments/WQD7001/project/web_app.py)

### 8.2 Feature selection

Model inputs:

- `Gender`
- `Age`
- `Height`
- `family_history_with_overweight`
- `FAVC`
- `FCVC`
- `NCP`
- `CAEC`
- `SMOKE`
- `CH2O`
- `SCC`
- `FAF`
- `TUE`
- `CALC`
- `MTRANS`

Excluded from input:

- `Weight`
- `NObeyesdad`

This exclusion is important because the system is meant to simulate a questionnaire-based screening workflow.

### 8.3 Training and evaluation

The dataset was split into training and test sets with a fixed random seed for reproducibility. Classification performance was measured using accuracy, weighted F1, and AUC. Regression performance was measured using MAE, RMSE, and R-squared.

## 9. Model Performance and Interpretation

The evaluation results are stored in [personal_model_report.json](/I:/Assignments/WQD7001/project/personal_model_report.json).

### 9.1 Classification results

- Accuracy: 0.8818
- Weighted F1-score: 0.8843
- Obesity AUC: 0.9862
- Overweight-or-obesity AUC: 0.9763

Interpretation:

- The classifier performs strongly across the seven obesity categories.
- Severe obesity classes are predicted especially well, with very high precision and recall.
- `Normal_Weight` is the weakest class, suggesting some overlap with overweight classes, which is reasonable in real health behaviour data.

### 9.2 Regression results

- Weight MAE: 5.3207 kg
- Weight RMSE: 9.8438 kg
- Weight R-squared: 0.8644
- BMI MAE: 1.7992
- BMI RMSE: 3.2013

Interpretation:

- The regression model estimates weight reasonably well for a questionnaire-only setting.
- Prediction error is acceptable for educational screening and awareness applications, but not precise enough for clinical use.

## 10. Data Interpretation for Non-Technical Users

The deployed web application allows a user to answer a set of lifestyle and physical questions and receive:

- the estimated body weight
- the estimated BMI
- the BMI category
- the probability of obesity
- the probability of overweight or obesity
- the most likely class among the seven obesity levels

This turns the modelling results into a usable data product. A non-technical user does not need to understand machine learning details; instead, they receive structured risk-oriented outputs in a simple interface.

## 11. Plan for Reproducible Research

The project has been organised so that another user can reproduce the workflow from the project folder.

### 11.1 Files

- Dataset: [obesity_questionnaire_dataset.csv](/I:/Assignments/WQD7001/project/obesity_questionnaire_dataset.csv)
- Training script: [train_personal_obesity_models.py](/I:/Assignments/WQD7001/project/train_personal_obesity_models.py)
- EDA script: [generate_eda.py](/I:/Assignments/WQD7001/project/generate_eda.py)
- Inference script: [predict_personal_metrics.py](/I:/Assignments/WQD7001/project/predict_personal_metrics.py)
- Web app: [web_app.py](/I:/Assignments/WQD7001/project/web_app.py)
- Dependencies: [requirements.txt](/I:/Assignments/WQD7001/project/requirements.txt)

### 11.2 Reproduction steps

1. Install Python packages from `requirements.txt`.
2. Run `python generate_eda.py` to regenerate EDA figures and summaries.
3. Run `python train_personal_obesity_models.py` to retrain the models.
4. Run `python web_app.py` to start the local data product.

### 11.3 Reproducibility controls

- Fixed random seed in model training
- Local scripts for training and inference
- Saved model bundle for deterministic deployment
- Documented project structure and command-line usage in [README.md](/I:/Assignments/WQD7001/project/README.md)

## 12. Deployment of Data Product

The final data product is a local browser-based web application implemented in [web_app.py](/I:/Assignments/WQD7001/project/web_app.py). It exposes a questionnaire form, collects user answers, applies the trained classification and regression models, and returns interpretable predictions. This satisfies the assignment requirement that a data product must be presented.

The deployment is intentionally lightweight:

- no external database
- no cloud requirement
- easy local execution for presentation

This makes it suitable for classroom demonstration and reproducible testing.

## 13. Ethical Considerations

Several ethical issues should be acknowledged:

1. Health-related predictions can be misinterpreted as diagnosis. This system should only be used for educational and screening support purposes.
2. The dataset may not represent all demographic groups equally, so prediction bias may exist.
3. The source data contains synthetic elements, which limits real-world generalisability.
4. Sensitive personal factors such as family history and lifestyle behaviours should be handled with care in real deployments.
5. A high predicted risk may affect users emotionally, so results should be presented responsibly with proper disclaimers.

## 14. Insights and Conclusion

This project demonstrates that questionnaire-based variables can be used to predict obesity-related outcomes with strong performance in an educational machine learning setting. EDA shows meaningful relationships between obesity risk and family history, transportation behaviour, physical activity, and age. The modelling results show that:

- obesity class prediction is strong
- estimated weight and BMI are useful but approximate
- a lightweight web data product can translate modelling output into a practical user-facing tool

The project successfully combines data science workflow elements from problem understanding to deployment. For future work, the project can be improved by using a larger real-world dataset, performing hyperparameter tuning, adding SHAP-based explainability, and deploying the system to a cloud environment.

## 15. References

1. UCI Machine Learning Repository. Estimation of Obesity Levels Based On Eating Habits and Physical Condition.
2. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.
3. McKinney, W. Data Structures for Statistical Computing in Python.
4. Hunter, J. D. Matplotlib: A 2D Graphics Environment.
