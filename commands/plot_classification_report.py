import json

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report


@click.command()
@click.option(
    "--report-file",
    default="data/reports/logreg_classification_report.txt",
    type=click.Path(exists=True),
    help="Path to the classification report text file.",
)
@click.option(
    "--output-image",
    default="data/plots/classification_report_heatmap.png",
    type=click.Path(),
    help="Path to save the classification report heatmap image.",
)
def plot_classification_report(report_file, output_image):
    """
    Parses a classification report text file and visualizes it as a heatmap.
    """
    with open(report_file, "r") as file:
        lines = file.readlines()

    # Extract class lines only (exclude average metrics)
    class_lines = [
        line.strip() for line in lines if line.strip().startswith("DEMPE_Class")
    ]

    data = []
    classes = []
    for line in class_lines:
        parts = line.split()
        if len(parts) >= 5:
            class_name = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1_score = float(parts[3])
            support = int(parts[4])
            data.append([precision, recall, f1_score])
            classes.append(class_name)

    df = pd.DataFrame(data, columns=["Precision", "Recall", "F1-Score"], index=classes)

    plt.figure(figsize=(8, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
    plt.title("Classification Report Heatmap")
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

    click.echo(f"📊 Classification heatmap saved to {output_image}")


if __name__ == "__main__":
    plot_classification_report()
