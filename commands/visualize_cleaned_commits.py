import os
import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


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
    Visualizes:
    - Bar chart of DEMPE class distribution
    - Pie chart of DEMPE class distribution
    - Heatmap of co-occurrence between DEMPE classes
    - Top common words per class
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    class_cols = [col for col in df.columns if col.startswith("DEMPE_Class_")]

    # Melted for long-format class analysis
    melted = df.melt(
        id_vars=["Raw Serial Number", "Commit Message"],
        value_vars=class_cols,
        var_name="DEMPE_Class",
        value_name="Label",
    )
    melted = melted[melted["Label"] == 1]

    # --- Bar Plot ---
    class_counts = melted["DEMPE_Class"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="Blues_d")
    plt.title("Distribution of DEMPE Classes")
    plt.xlabel("DEMPE Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_distribution.png"))
    plt.close()

    # --- Pie Chart ---
    plt.figure(figsize=(8, 8))  # Increased size
    plt.pie(
        class_counts.values,
        labels=class_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
    )
    plt.title("DEMPE Class Distribution (Pie Chart)", fontsize=14)
    plt.axis("equal")  # Ensures circular shape
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pie_distribution.png"), bbox_inches="tight")
    plt.close()

    # --- Multi-label Heatmap ---
    heatmap_data = df[class_cols].T.dot(df[class_cols])
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("DEMPE Class Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cooccurrence_heatmap.png"))
    plt.close()

    # --- Common Words per Class ---
    vectorizer = CountVectorizer(stop_words="english", max_features=100)
    all_words = {}

    for col in class_cols:
        relevant_msgs = df[df[col] == 1]["Commit Message"]
        vec = vectorizer.fit_transform(relevant_msgs)
        word_freq = vec.toarray().sum(axis=0)
        vocab = vectorizer.get_feature_names_out()
        word_counts = dict(zip(vocab, word_freq))
        top_words = dict(
            sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        all_words[col] = top_words

        # Barplot for each class
        plt.figure(figsize=(8, 4))
        sns.barplot(
            x=list(top_words.values()), y=list(top_words.keys()), palette="viridis"
        )
        plt.title(f"Top Words for {col}")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_words_{col}.png"))
        plt.close()

    # Save top word summary
    summary_df = pd.DataFrame(all_words).fillna(0).astype(int)
    summary_csv = os.path.join(output_dir, "top_words_summary.csv")
    summary_df.to_csv(summary_csv)
    click.echo(f"📝 Top words per DEMPE class saved to {summary_csv}")

    click.echo(f"📊 All visualizations saved in: {output_dir}")


if __name__ == "__main__":
    visualize_cleaned_commits()
