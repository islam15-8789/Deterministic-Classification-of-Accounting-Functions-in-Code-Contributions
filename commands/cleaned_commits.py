import re

import click
import pandas as pd


# Cleaning function
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()


@click.command()
@click.option(
    "--input-file",
    default="data/csv_data/multilabel_labeled_commits.csv",
    type=click.Path(exists=True),
    help="Path to the labeled commit CSV file.",
)
@click.option(
    "--output-file",
    default="data/csv_data/cleaned_multilabel_commits.csv",
    type=click.Path(),
    help="Path to save the cleaned CSV file.",
)
@click.option(
    "--nonconv-output",
    default="data/csv_data/non_conventional_commits.csv",
    type=click.Path(),
    help="Path to save the non-conventional commits (all labels 0).",
)
def clean_commits(input_file, output_file, nonconv_output):
    """
    Cleans commit messages, retains multi-label structure,
    and separates non-conventional commits (rows with no DEMPE class label).
    """
    click.echo(f"📂 Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Drop rows with missing messages
    df.dropna(subset=["Commit Message"], inplace=True)

    # Identify label columns
    label_cols = [col for col in df.columns if col.startswith("DEMPE_Class_")]

    # Separate non-conventional commits (all DEMPE labels = 0)
    is_non_conventional = df[label_cols].sum(axis=1) == 0
    non_conventional = df[is_non_conventional]
    df = df[~is_non_conventional]

    # Clean commit messages
    click.echo("🧼 Cleaning commit messages...")
    df["Commit Message"] = df["Commit Message"].astype(str).apply(clean_text)

    # Save cleaned data
    df.to_csv(output_file, index=False)
    click.echo(f"✅ Cleaned multi-label commits saved to {output_file}")

    # Save non-conventional data
    non_conventional.to_csv(nonconv_output, index=False)
    click.echo(f"⚠️ Non-conventional commits saved to {nonconv_output}")


if __name__ == "__main__":
    clean_commits()
