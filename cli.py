import click

from commands.cleaned_commits import clean_commits
from commands.extract_raw_commits import extract_raw_commit_messages
from commands.fetch_commits import fetch_commits
from commands.label_commits import label_commits
from commands.pipeline import run_pipeline
from commands.plot_classification_report import plot_classification_report
from commands.split_train_test import split_dataset
from commands.train_test_logreg_model import train_test_logreg_model
from commands.visualize_cleaned_commits import visualize_cleaned_commits


@click.group()
def cli():
    """
    A CLI tool to reproduce the thesis result.
    """


@cli.command(
    help="Run the entire data pipeline: fetch commits and extract commit messages."
)
def execute_pipeline():
    """Run the data pipeline."""
    run_pipeline()


# Add commands to the CLI group
cli.add_command(fetch_commits, name="fetch-commits")
cli.add_command(extract_raw_commit_messages, name="extract-raw-commit-messages")
cli.add_command(label_commits, name="label-commits")
cli.add_command(split_dataset, name="split-dataset")
cli.add_command(clean_commits, name="clean-commits")
cli.add_command(visualize_cleaned_commits, name="visualize-cleaned-commits")
cli.add_command(train_test_logreg_model, name="train-test-logreg-model")
cli.add_command(plot_classification_report, name="plot-classification-report")

if __name__ == "__main__":
    cli()
