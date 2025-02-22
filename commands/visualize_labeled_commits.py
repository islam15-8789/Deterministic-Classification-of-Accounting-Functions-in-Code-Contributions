import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not present
nltk.download("stopwords")

# Define mapping for DEMPE classes
DEMPE_CLASS_MAPPING = {
    "0": "Development",
    "1": "Enhancement",
    "2": "Maintenance",
    "3": "Protection",
    "4": "Exploitation",
    "Non-conventional": "Non-conventional",
}

# Command to plot DEMPE Function distribution
@click.command()
@click.option("--input_file", default="data/csv_data/labeled_commits.csv", type=click.Path(exists=True))
@click.option("--output_file", default = "figures/dempe_dist.png", type=click.Path())
def dempe_distribution(input_file, output_file):
    """Plots the distribution of commit messages across DEMPE functions and saves to output file."""
    click.echo(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df.columns = [col.strip() for col in df.columns]
    df["DEMPE Function Category"] = df["DEMPE Function Class"].astype(str).map(DEMPE_CLASS_MAPPING)

    click.echo("Generating DEMPE Function Category distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.countplot(
        y=df["DEMPE Function Category"],
        order=df["DEMPE Function Category"].value_counts().index,
        palette="viridis"
    )
    plt.xlabel("Count")
    plt.ylabel("DEMPE Function Category")
    plt.title("Distribution of Commit Messages Across DEMPE Functions")
    plt.savefig(output_file)
    click.echo(f"DEMPE function distribution saved to {output_file}!")


# Command to generate word cloud
@click.command()
@click.option("--input_file", default="data/csv_data/labeled_commits.csv", type=click.Path(exists=True))
@click.option("--output_file", default = "figures/wordcloud.png", type=click.Path())
def wordcloud(input_file, output_file):
    """Generates a word cloud from commit messages and saves to output file."""
    click.echo(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df.columns = [col.strip() for col in df.columns]

    click.echo("Generating word cloud for commit messages...")
    text = " ".join(df["Commit Message"].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Commit Messages")
    plt.savefig(output_file)
    click.echo(f"Word cloud saved to {output_file}!")


# Command to visualize most common words in commit messages
@click.command()
@click.option("--input_file", default="data/csv_data/labeled_commits.csv", type=click.Path(exists=True))
@click.option("--output_file", default = "figures/common_words.png", type=click.Path())
def common_words(input_file, output_file):
    """Displays a bar chart of the most common words in commit messages and saves to output file."""
    click.echo(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df.columns = [col.strip() for col in df.columns]

    click.echo("Generating bar plot for the most common words in commit messages...")
    stop_words = set(stopwords.words("english"))
    words = [
        word.lower()
        for line in df["Commit Message"].dropna()
        for word in line.split()
        if word.lower() not in stop_words
    ]
    word_counts = Counter(words)
    common_words = word_counts.most_common(20)

    words, counts = zip(*common_words)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), palette="magma")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("Most Common Words in Commit Messages")
    plt.savefig(output_file)
    click.echo(f"Common words visualization saved to {output_file}!")
