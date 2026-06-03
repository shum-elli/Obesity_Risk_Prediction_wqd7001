import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("personal_obesity_bundle.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict obesity risk from lifestyle questionnaire inputs.")
    parser.add_argument("--gender", required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--family-history", required=True, choices=["yes", "no"])
    parser.add_argument("--favc", required=True, choices=["yes", "no"])
    parser.add_argument("--fcvc", type=float, required=True)
    parser.add_argument("--ncp", type=float, required=True)
    parser.add_argument("--caec", required=True, choices=["no", "Sometimes", "Frequently", "Always"])
    parser.add_argument("--smoke", required=True, choices=["yes", "no"])
    parser.add_argument("--ch2o", type=float, required=True)
    parser.add_argument("--scc", required=True, choices=["yes", "no"])
    parser.add_argument("--faf", type=float, required=True)
    parser.add_argument("--tue", type=float, required=True)
    parser.add_argument("--calc", required=True, choices=["no", "Sometimes", "Frequently", "Always"])
    parser.add_argument("--mtrans", required=True, choices=["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(MODEL_PATH)

    row = {
        "Gender": args.gender,
        "Age": args.age,
        "family_history_with_overweight": args.family_history,
        "FAVC": args.favc,
        "FCVC": args.fcvc,
        "NCP": args.ncp,
        "CAEC": args.caec,
        "SMOKE": args.smoke,
        "CH2O": args.ch2o,
        "SCC": args.scc,
        "FAF": args.faf,
        "TUE": args.tue,
        "CALC": args.calc,
        "MTRANS": args.mtrans,
    }

    sample = pd.DataFrame([row], columns=bundle["feature_columns"])
    probabilities = bundle["classifier"].predict_proba(sample)[0]

    rows = []
    for label, prob in zip(bundle["classes"], probabilities):
        rows.append({"label": label, "probability": float(prob), "probability_percent": round(float(prob * 100.0), 2)})
    rows.sort(key=lambda item: item["probability"], reverse=True)

    obesity_prob = sum(item["probability"] for item in rows if item["label"] in bundle["risk_classes"])
    at_risk_prob = sum(item["probability"] for item in rows if item["label"] in bundle["at_risk_classes"])

    result = {
        "obesity_probability_percent": round(obesity_prob * 100.0, 2),
        "overweight_or_obesity_probability_percent": round(at_risk_prob * 100.0, 2),
        "top_prediction": rows[0],
        "class_probabilities": rows,
        "notes": bundle["notes"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
