import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

from constants import dempe_class_names


def evaluate_and_save_metrics(
    y_true, y_pred, label_names, output_dir, model_name="model"
):
    os.makedirs(output_dir, exist_ok=True)

    # Map technical label names to friendly ones
    display_names = [dempe_class_names.get(label, label) for label in label_names]

    # 1. Textual classification report
    report = classification_report(
        y_true, y_pred, target_names=display_names, zero_division=0
    )
    with open(
        os.path.join(output_dir, f"{model_name}_classification_report.txt"), "w"
    ) as f:
        f.write(report)

    # 2. JSON summary metrics
    summary = {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "jaccard_score_samples": jaccard_score(y_true, y_pred, average="samples"),
        "jaccard_score_macro": jaccard_score(y_true, y_pred, average="macro"),
        "jaccard_score_micro": jaccard_score(y_true, y_pred, average="micro"),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    with open(os.path.join(output_dir, f"{model_name}_summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 3. PRF bar chart
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(display_names))
    width = 0.25

    plt.bar(x - width, precision, width=width, label="Precision", color="#1f77b4")
    plt.bar(x, recall, width=width, label="Recall", color="#2ca02c")
    plt.bar(x + width, f1, width=width, label="F1-score", color="#d62728")

    plt.xticks(ticks=x, labels=display_names, rotation=30, ha="right", fontsize=10)
    plt.ylabel("Score", fontsize=12)
    plt.title(
        f"{model_name.capitalize()} - Precision / Recall / F1 per Class", fontsize=14
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_prf_scores.png"))
    plt.close()

    # 4. Confusion matrix heatmaps (Improved layout)
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

    for i, cm in enumerate(conf_matrices):
        fig, ax = plt.subplots(figsize=(6, 6))  # Smaller but more compact size

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",  # More neutral and professional than YlGnBu
            xticklabels=["Pred: No", "Pred: Yes"],
            yticklabels=["True: No", "True: Yes"],
            cbar=False,
            annot_kws={"fontsize": 16, "weight": "bold"},  # Larger, bold annotation
            linewidths=1,
            linecolor="black",
            ax=ax,
        )

        # Axis labels and title
        ax.set_title(
            f"{model_name.capitalize()} - Confusion Matrix\n({display_names[i]})",
            fontsize=16,
            pad=20,
            weight="semibold",
        )
        ax.set_xlabel("Predicted", fontsize=14, labelpad=10)
        ax.set_ylabel("Actual", fontsize=14, labelpad=10)
        ax.tick_params(axis="both", labelsize=12)

        # Improve tick label spacing
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # Save with higher resolution
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_confusion_{label_names[i]}.png"),
            dpi=300,  # High resolution for paper
            bbox_inches="tight",
        )
        plt.close()

    # 5. Jaccard similarity histogram
    jaccard_vals = [
        jaccard_score(y_true[i], y_pred[i], average="binary", zero_division=0)
        for i in range(len(y_true))
    ]

    plt.figure(figsize=(7, 5))
    sns.histplot(jaccard_vals, bins=20, kde=True, color="mediumpurple")
    plt.title(f"{model_name.capitalize()} - Jaccard Similarity Distribution", fontsize=13)
    plt.xlabel("Jaccard Similarity (per sample)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_jaccard_distribution.png"))
    plt.close()

    print(f"✅ Evaluation completed. All reports and plots saved to: {output_dir}")
