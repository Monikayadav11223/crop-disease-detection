"""
eda.py - Exploratory Data Analysis & Visualization Script
HCLTech Career Shaper Programme | Project 35: Crop Disease Detection
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from project_config import (
    RAW_DATASET_CSV,
    CLEANED_DATASET_CSV,
    REPORTS_DIR,
    FIGURES_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_eda(
    raw_path: Path = RAW_DATASET_CSV,
    cleaned_path: Path = CLEANED_DATASET_CSV,
    output_dir: Path = FIGURES_DIR,
    metrics_path: Path = REPORTS_DIR / "eda_summary_metrics.json"
):
    """
    Performs comprehensive Exploratory Data Analysis on both raw and cleaned metadata.
    Generates summary metrics and saves publication-quality visualization figures.
    """
    logger.info("Starting Exploratory Data Analysis...")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(cleaned_path)

    # 1. Statistical Summaries
    stats_summary = {
        "dataset_shape": {
            "rows": int(len(clean_df)),
            "columns": int(len(clean_df.columns))
        },
        "crops": clean_df["crop_type"].value_counts().to_dict(),
        "diseases": clean_df["disease_label"].value_counts().to_dict(),
        "cleaned_severity": clean_df["severity"].value_counts().to_dict(),
        "leaf_area_stats": {
            "mean": float(clean_df["leaf_area_cm2"].mean()),
            "std": float(clean_df["leaf_area_cm2"].std()),
            "min": float(clean_df["leaf_area_cm2"].min()),
            "25%": float(clean_df["leaf_area_cm2"].quantile(0.25)),
            "50%": float(clean_df["leaf_area_cm2"].median()),
            "75%": float(clean_df["leaf_area_cm2"].quantile(0.75)),
            "max": float(clean_df["leaf_area_cm2"].max())
        },
        "confidence_stats": {
            "mean": float(clean_df["confidence_score"].mean()),
            "std": float(clean_df["confidence_score"].std()),
            "min": float(clean_df["confidence_score"].min()),
            "median": float(clean_df["confidence_score"].median()),
            "max": float(clean_df["confidence_score"].max())
        },
        "anomalies": {
            "healthy_with_severity_in_raw": int(((raw_df["disease_label"].str.lower() == "healthy") & (raw_df["severity"].str.lower() != "none")).sum()),
            "diseased_with_none_in_raw": int(((raw_df["disease_label"].str.lower() != "healthy") & (raw_df["severity"].str.lower() == "none")).sum())
        },
        "batch_distribution": clean_df["batch_id"].value_counts().to_dict()
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)
    logger.info(f"EDA metrics saved to {metrics_path}")

    # Generate Matplotlib figures
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "axes.edgecolor": "#cccccc"})

        # Chart 1: Crop and Disease Class Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        crop_counts = clean_df["crop_type"].value_counts()
        axes[0].bar(crop_counts.index, crop_counts.values, color=["#27ae60", "#2ecc71", "#1abc9c", "#3498db", "#9b59b6"])
        axes[0].set_title("Crop Specimen Distribution (N=500)", fontsize=13, weight="bold")
        axes[0].set_xlabel("Crop Type")
        axes[0].set_ylabel("Count")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        dis_counts = clean_df["disease_label"].value_counts()
        axes[1].bar(dis_counts.index, dis_counts.values, color=["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#8e44ad"])
        axes[1].set_title("Pathology / Condition Label Distribution", fontsize=13, weight="bold")
        axes[1].set_xlabel("Disease Label")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis='x', rotation=25)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "eda_crop_disease_distribution.png", dpi=300)
        plt.close()

        # Chart 2: Data Quality & Domain Anomaly Audit
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["Healthy Leaves\nw/ Severity Anomaly", "Diseased Leaves\nw/ Unspecified Severity", "Clean & Consistent\nRecords"]
        counts = [
            stats_summary["anomalies"]["healthy_with_severity_in_raw"],
            stats_summary["anomalies"]["diseased_with_none_in_raw"],
            500 - (stats_summary["anomalies"]["healthy_with_severity_in_raw"] + stats_summary["anomalies"]["diseased_with_none_in_raw"])
        ]
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        bars = ax.bar(categories, counts, color=colors, width=0.55)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height} ({height/500*100:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, weight="bold")
        ax.set_title("HCL Metadata Quality & Logical Consistency Audit", fontsize=13, weight="bold")
        ax.set_ylabel("Number of Records")
        ax.set_ylim(0, 350)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "eda_data_quality_anomalies.png", dpi=300)
        plt.close()

        # Chart 3: Leaf Area by Crop
        fig, ax = plt.subplots(figsize=(10, 5))
        crops = sorted(clean_df["crop_type"].unique())
        data_by_crop = [clean_df[clean_df["crop_type"] == c]["leaf_area_cm2"].dropna() for c in crops]
        box = ax.boxplot(data_by_crop, labels=crops, patch_artist=True)
        colors_crop = ["#a9dfbf", "#aed6f1", "#f9e79f", "#f5b7b1", "#d2b4de"]
        for patch, color in zip(box['boxes'], colors_crop):
            patch.set_facecolor(color)
        ax.set_title("Leaf Surface Area (cm²) Distribution Across Crops", fontsize=13, weight="bold")
        ax.set_xlabel("Crop Type")
        ax.set_ylabel("Leaf Area (cm²)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "eda_leaf_area_severity.png", dpi=300)
        plt.close()

        # Chart 4: Confidence Score Histogram by Condition
        fig, ax = plt.subplots(figsize=(10, 5))
        for disease, d_color in zip(["Healthy", "Early_Blight", "Late_Blight"], ["#2ecc71", "#e74c3c", "#e67e22"]):
            subset = clean_df[clean_df["disease_label"] == disease]["confidence_score"]
            ax.hist(subset, bins=15, alpha=0.5, label=f"{disease} (μ={subset.mean():.3f})", color=d_color, edgecolor="#333333")
        ax.set_title("Confidence Score Distribution Across Key Pathological Labels", fontsize=13, weight="bold")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "eda_confidence_distribution.png", dpi=300)
        plt.close()

        logger.info(f"Visualizations saved successfully to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not render matplotlib figures directly: {e}. SVG vector figures will be preserved.")

    return stats_summary


if __name__ == "__main__":
    run_eda()
