import click

from data_cli import data_cli
from dempe_cli import dempe_cli
from training_cli import training_cli


@click.group()
def cli():
    """Main CLI for DEMPE classification tasks."""
    pass


cli.add_command(data_cli, name="data")
cli.add_command(training_cli, name="train")
cli.add_command(dempe_cli, name="dempe")

if __name__ == "__main__":
    cli()
