import click
import pandas as pd
import re

# Cleaning function
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()

@click.command()
@click.option("--input-file", default="data/csv_data/labeled_commits.csv", type=click.Path(exists=True), help="Path to the labeled commit CSV file.")
@click.option("--output-file", default="data/csv_data/cleaned_commits.csv", type=click.Path(), help="Path to save the cleaned CSV file.")
@click.option("--nonconv-output", default="data/csv_data/non_conventional_commits.csv", type=click.Path(), help="Path to save the non-conventional commits.")
def clean_commits(input_file, output_file, nonconv_output):
    """Cleans commit messages, saves cleaned data and non-conventional commits separately (no vectorization)."""
    click.echo(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Drop rows with missing values
    df.dropna(subset=["Commit Message", "DEMPE Function Class"], inplace=True)

    # Separate non-conventional commits
    non_conventional = df[df["DEMPE Function Class"] == "Non-conventional"]
    df = df[df["DEMPE Function Class"] != "Non-conventional"]

    # Clean commit messages
    click.echo("Cleaning commit messages...")
    df["Commit Message"] = df["Commit Message"].astype(str).apply(clean_text)

    # Save cleaned data
    df[["Raw Serial Number", "Commit Message", "DEMPE Function Class"]].to_csv(output_file, index=False)
    click.echo(f"Cleaned commits saved to {output_file}")

    # Save non-conventional data
    non_conventional.to_csv(nonconv_output, index=False)
    click.echo(f"Non-conventional commits saved to {nonconv_output}")

if __name__ == "__main__":
    clean_commits()
