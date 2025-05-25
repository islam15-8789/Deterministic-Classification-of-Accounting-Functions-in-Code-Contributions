import os

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from constants import dempe_class_names


@click.command()
@click.option(
    "--input-file",
    default="data/csv_data/cleaned_commits.csv",
    type=click.Path(exists=True),
    help="Path to the cleaned commits CSV file.",
)
@click.option(
    "--output-dir",
    default="data/plots",
    type=click.Path(),
    help="Directory to save visualizations.",
)
def visualize_cleaned_commits(input_file, output_dir):
    """
    Generates enhanced visualizations:
    - Bar chart of DEMPE class distribution
    - Pie chart of DEMPE class proportions
    - Heatmap of DEMPE label co-occurrences
    - Top words per class
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    class_cols = [col for col in df.columns if col.startswith("DEMPE_Class_")]

    # Melt for class-wise analysis
    melted = df.melt(
        id_vars=["Raw Serial Number", "Commit Message"],
        value_vars=class_cols,
        var_name="DEMPE_Class",
        value_name="Label",
    )
    melted = melted[melted["Label"] == 1]
    melted["DEMPE_Class"] = melted["DEMPE_Class"].map(dempe_class_names)

    # Bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=melted,
        y="DEMPE_Class",
        order=melted["DEMPE_Class"].value_counts().index,
        palette="cubehelix",
    )

    # Add labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", label_type="edge", fontsize=11)

    plt.title("DEMPE Class Distribution", fontsize=14)
    plt.xlabel("Number of Samples")
    plt.ylabel("DEMPE Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_distribution.png"))
    plt.close()

    # Pie chart
    plt.figure(figsize=(8, 8))
    class_counts = melted["DEMPE_Class"].value_counts()
    plt.pie(
        class_counts,
        labels=class_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
        colors=sns.color_palette("Set2"),
    )
    plt.title("DEMPE Class Proportions", fontsize=14)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pie_distribution.png"))
    plt.close()

    # Co-occurrence heatmap
    heatmap_data = df[class_cols].T.dot(df[class_cols])
    heatmap_data.index = [dempe_class_names[col] for col in heatmap_data.index]
    heatmap_data.columns = [dempe_class_names[col] for col in heatmap_data.columns]

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("DEMPE Class Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cooccurrence_heatmap.png"))
    plt.close()

    # Common words per class
    vectorizer = CountVectorizer(stop_words="english", max_features=100)
    all_words = {}
    bar_color = "#4c72b0"
    for col in class_cols:
        friendly = dempe_class_names[col]
        relevant_msgs = df[df[col] == 1]["Commit Message"]
        vec = vectorizer.fit_transform(relevant_msgs)
        word_freq = vec.toarray().sum(axis=0)
        vocab = vectorizer.get_feature_names_out()
        word_counts = dict(zip(vocab, word_freq))
        top_words = dict(
            sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        all_words[friendly] = top_words

        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), color=bar_color)
        plt.title(f"Top Words for {friendly}")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_words_{friendly}.png"))
        plt.close()

    # Summary CSV
    summary_df = pd.DataFrame(all_words).fillna(0).astype(int)
    summary_df.to_csv(os.path.join(output_dir, "top_words_summary.csv"))

    click.echo(f"✅ Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    visualize_cleaned_commits()
