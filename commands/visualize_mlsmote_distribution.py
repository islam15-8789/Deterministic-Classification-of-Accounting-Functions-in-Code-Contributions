import os

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Friendly DEMPE class names
dempe_class_names = {
    "DEMPE_Class_0": "Development",
    "DEMPE_Class_1": "Enhancement",
    "DEMPE_Class_2": "Maintenance",
    "DEMPE_Class_3": "Protection",
    "DEMPE_Class_4": "Exploitation",
}


@click.command()
@click.option(
    "--resampled-file",
    default="data/csv_data/resampled_mlsmote.csv",
    type=click.Path(exists=True),
    help="Path to the dataset after applying per-label SMOTE.",
)
@click.option(
    "--output-image",
    default="data/plots/mlsmote_distribution.png",
    type=click.Path(),
    help="Path to save the resampled label distribution plot image.",
)
def visualize_mlsmote_distribution(resampled_file, output_image):
    """
    Visualizes the label distribution of the resampled dataset and saves it as an image.
    """
    click.echo(f"📥 Loading resampled data from: {resampled_file}")
    df = pd.read_csv(resampled_file)
    label_cols = [col for col in df.columns if col.startswith("DEMPE_Class_")]
    label_counts = df[label_cols].sum().rename(index=dempe_class_names)

    click.echo("📊 Generating and saving resampled label distribution plot...")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")

    # Add value labels
    for i, v in enumerate(label_counts.values):
        ax.text(
            i,
            v + max(label_counts.values) * 0.01,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.title("Resampled DEMPE Class Distribution", fontsize=14)
    plt.xlabel("DEMPE Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=300)
    plt.close()

    click.echo(f"✅ Resampled label distribution saved to: {output_image}")


if __name__ == "__main__":
    visualize_mlsmote_distribution()
