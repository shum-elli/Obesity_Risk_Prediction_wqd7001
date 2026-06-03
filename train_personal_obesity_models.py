import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


TARGET_CLASS = "NObeyesdad"
RISK_CLASSES = {"Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"}
AT_RISK_CLASSES = {
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
}
FIELD_LABELS = {
    "Gender": "Gender",
    "Age": "Age",
    "Height": "Height (m)",
    "family_history_with_overweight": "Family history with overweight",
    "FAVC": "Frequent high-calorie food",
    "FCVC": "Vegetable consumption frequency",
    "NCP": "Main meals per day",
    "CAEC": "Food between meals",
    "SMOKE": "Smoking",
    "CH2O": "Daily water intake",
    "SCC": "Calories monitoring",
    "FAF": "Physical activity frequency",
    "TUE": "Technology usage time",
    "CALC": "Alcohol consumption",
    "MTRANS": "Transportation used",
}
FEATURE_DESCRIPTIONS = {
    "Age": {"type": "number", "min": 14.0, "max": 61.0, "step": 1.0, "default": 23.0, "hint": "Age in years"},
    "FCVC": {"type": "number", "min": 1.0, "max": 3.0, "step": 0.1, "default": 2.4, "hint": "1 = low vegetables, 3 = high vegetables"},
    "NCP": {"type": "number", "min": 1.0, "max": 4.0, "step": 0.1, "default": 3.0, "hint": "Number of main meals per day"},
    "CH2O": {"type": "number", "min": 1.0, "max": 3.0, "step": 0.1, "default": 2.0, "hint": "1 = low water intake, 3 = high water intake"},
    "FAF": {"type": "number", "min": 0.0, "max": 3.0, "step": 0.1, "default": 1.0, "hint": "0 = none, 3 = very frequent activity"},
    "TUE": {"type": "number", "min": 0.0, "max": 2.0, "step": 0.1, "default": 0.6, "hint": "Technology use score from the dataset"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train questionnaire-style obesity models.")
    parser.add_argument("--data", default="obesity_questionnaire_dataset.csv")
    parser.add_argument("--model-out", default="personal_obesity_bundle.joblib")
    parser.add_argument("--report-out", default="personal_model_report.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [column for column in x.columns if is_numeric_dtype(x[column])]
    categorical_features = [column for column in x.columns if column not in numeric_features]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

    args = parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model_out)
    report_path = Path(args.report_out)

    df = pd.read_csv(data_path)
    excluded_columns = {TARGET_CLASS, "Height", "Weight", "BMI"}
    feature_columns = [column for column in df.columns if column not in excluded_columns]
    x = df[feature_columns].copy()
    y_class = df[TARGET_CLASS].copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_class)
    classes = label_encoder.classes_.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, random_state=args.seed, stratify=y_encoded
    )
    y_test_labels = pd.Series(label_encoder.inverse_transform(y_test))

    # Train multiple classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=args.seed, class_weight="balanced", n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, random_state=args.seed, class_weight="balanced", n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=args.seed),
        "XGBoost": XGBClassifier(n_estimators=400, random_state=args.seed, eval_metric="mlogloss", n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=400, random_state=args.seed, num_leaves=31, n_jobs=-1, verbose=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
        "Gaussian NB": GaussianNB(),
    }
    
    trained_models = {}
    best_accuracy = -1
    best_model_name = None
    
    for model_name, base_classifier in classifiers.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(x)),
                ("classifier", base_classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        
        class_predictions = pipeline.predict(x_test)
        class_probabilities = pipeline.predict_proba(x_test)
        
        obesity_idx = [classes.index(label) for label in classes if label in RISK_CLASSES]
        at_risk_idx = [classes.index(label) for label in classes if label in AT_RISK_CLASSES]
        
        obesity_target = y_test_labels.isin(RISK_CLASSES).astype(int)
        at_risk_target = y_test_labels.isin(AT_RISK_CLASSES).astype(int)
        
        accuracy = float(accuracy_score(y_test, class_predictions))
        
        metrics = {
            "accuracy": round(accuracy, 4),
            "weighted_f1": round(float(f1_score(y_test, class_predictions, average="weighted")), 4),
            "obesity_auc": round(float(roc_auc_score(obesity_target, class_probabilities[:, obesity_idx].sum(axis=1))), 4),
            "overweight_or_obesity_auc": round(float(roc_auc_score(at_risk_target, class_probabilities[:, at_risk_idx].sum(axis=1))), 4),
            "classification_report": classification_report(
                y_test,
                class_predictions,
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
        }
        
        trained_models[model_name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "classes": classes,
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
        
        print(f"{model_name}: Accuracy={metrics['accuracy']}, F1={metrics['weighted_f1']}")
    
    report = {
        "dataset_path": str(data_path.resolve()),
        "rows": int(len(df)),
        "features_used": feature_columns,
        "best_model": best_model_name,
        "model_metrics": {
            name: {
                "accuracy": model_data["metrics"]["accuracy"],
                "weighted_f1": model_data["metrics"]["weighted_f1"],
                "obesity_auc": model_data["metrics"]["obesity_auc"],
                "overweight_or_obesity_auc": model_data["metrics"]["overweight_or_obesity_auc"],
                "classification_report": model_data["metrics"]["classification_report"],
            }
            for name, model_data in trained_models.items()
        },
        "notes": [
            "Inputs intentionally exclude height, weight, and BMI to avoid direct anthropometric leakage.",
            "Obesity probability is the summed probability of Obesity_Type_I, II, and III.",
            "The model is designed for lifestyle-based risk screening rather than BMI reconstruction.",
            f"Best performing model: {best_model_name}",
        ],
    }

    categorical_options = {}
    for column in x.columns:
        if not is_numeric_dtype(x[column]):
            categorical_options[column] = sorted(x[column].dropna().unique().tolist())

    feature_schema = {}
    for column in feature_columns:
        if column in categorical_options:
            feature_schema[column] = {"type": "select", "label": FIELD_LABELS[column], "options": categorical_options[column]}
        else:
            spec = FEATURE_DESCRIPTIONS[column]
            feature_schema[column] = {
                "type": spec["type"],
                "label": FIELD_LABELS[column],
                "min": spec["min"],
                "max": spec["max"],
                "step": spec["step"],
                "default": spec["default"],
                "hint": spec["hint"],
            }

    bundle = {
        "models": {
            name: {
                "pipeline": model_data["pipeline"],
                "metrics": {
                    "accuracy": model_data["metrics"]["accuracy"],
                    "weighted_f1": model_data["metrics"]["weighted_f1"],
                    "obesity_auc": model_data["metrics"]["obesity_auc"],
                    "overweight_or_obesity_auc": model_data["metrics"]["overweight_or_obesity_auc"],
                },
            }
            for name, model_data in trained_models.items()
        },
        "best_model": best_model_name,
        "feature_columns": feature_columns,
        "classes": classes,
        "risk_classes": sorted(RISK_CLASSES),
        "at_risk_classes": sorted(AT_RISK_CLASSES),
        "feature_schema": feature_schema,
        "notes": report["notes"],
    }

    joblib.dump(bundle, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Training complete.")
    print(f"Best model: {best_model_name}")
    for name, model_data in trained_models.items():
        print(f"  {name}: Accuracy={model_data['metrics']['accuracy']}")


if __name__ == "__main__":
    main()
