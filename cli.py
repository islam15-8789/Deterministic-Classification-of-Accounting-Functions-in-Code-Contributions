import click

from commands.extract_raw_commits import extract_raw_commit_messages
from commands.fetch_commits import fetch_commits
from commands.label_commits import label_commits
from commands.pipeline import run_pipeline
from commands.visualize_labeled_commits import dempe_distribution, wordcloud, common_words 

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
cli.add_command(fetch_commits)
cli.add_command(extract_raw_commit_messages)
cli.add_command(label_commits)
cli.add_command(dempe_distribution)
cli.add_command(wordcloud)
cli.add_command(common_words)

if __name__ == "__main__":
    cli()
