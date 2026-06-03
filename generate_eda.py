import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("obesity_questionnaire_dataset.csv")
OUTPUT_DIR = Path("eda_outputs")


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    if bmi < 35:
        return "Obesity I"
    if bmi < 40:
        return "Obesity II"
    return "Obesity III"


def save_bar(series: pd.Series, title: str, ylabel: str, filename: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", ax=ax, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def save_hist(series: pd.Series, title: str, xlabel: str, filename: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series, bins=25, color=color, edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def save_boxplot(df: pd.DataFrame, column: str, by: str, title: str, filename: str, color: str) -> None:
    categories = list(df[by].dropna().unique())
    data = [df.loc[df[by] == category, column] for category in categories]
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=categories)
    for patch in bp["boxes"]:
        patch.set(facecolor=color, alpha=0.75)
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    obesity_mask = df["NObeyesdad"].str.contains("Obesity")
    ax.scatter(df.loc[~obesity_mask, x], df.loc[~obesity_mask, y], alpha=0.45, label="Non-obesity", color="#5b8c85")
    ax.scatter(df.loc[obesity_mask, x], df.loc[obesity_mask, y], alpha=0.45, label="Obesity", color="#d76b48")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["BMI_Category"] = df["BMI"].apply(bmi_category)
    df["Obesity_Risk"] = df["NObeyesdad"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"])

    save_bar(
        df["NObeyesdad"].value_counts().sort_values(ascending=False),
        "Distribution of obesity classes",
        "Count",
        "class_distribution.png",
        "#d76b48",
    )
    save_hist(df["Age"], "Age distribution", "Age", "age_distribution.png", "#6c8ebf")
    save_hist(df["BMI"], "BMI distribution", "BMI", "bmi_distribution.png", "#7bb274")
    save_bar(
        df.groupby("family_history_with_overweight")["Obesity_Risk"].mean().mul(100).sort_values(ascending=False),
        "Obesity prevalence by family history",
        "Obesity prevalence (%)",
        "family_history_obesity_rate.png",
        "#c87d2d",
    )
    save_bar(
        df.groupby("Gender")["BMI"].mean().sort_values(ascending=False),
        "Average BMI by gender",
        "Average BMI",
        "gender_bmi.png",
        "#8b6fb3",
    )
    save_bar(
        df.groupby("MTRANS")["BMI"].mean().sort_values(ascending=False),
        "Average BMI by transportation mode",
        "Average BMI",
        "transport_bmi.png",
        "#478f9b",
    )
    save_boxplot(df, "BMI", "CAEC", "BMI by snacking frequency", "snacking_bmi_boxplot.png", "#e2a458")
    save_boxplot(df, "BMI", "CALC", "BMI by alcohol consumption", "alcohol_bmi_boxplot.png", "#9d6c97")
    save_scatter(df, "Age", "BMI", "Age vs BMI by obesity status", "age_bmi_scatter.png")

    corr = df[["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]].corr(numeric_only=True)
    corr.round(4).to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "class_counts": df["NObeyesdad"].value_counts().to_dict(),
        "obesity_prevalence_percent": round(float(df["Obesity_Risk"].mean() * 100), 2),
        "average_bmi_by_gender": df.groupby("Gender")["BMI"].mean().round(2).to_dict(),
        "obesity_prevalence_by_family_history_percent": (
            df.groupby("family_history_with_overweight")["Obesity_Risk"].mean().mul(100).round(2).to_dict()
        ),
        "average_bmi_by_transport": df.groupby("MTRANS")["BMI"].mean().round(2).to_dict(),
        "numeric_summary": df[["Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"]].describe().round(4).to_dict(),
    }
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EDA outputs written to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
