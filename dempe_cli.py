import click

from commands.predict_dempe import predict_dempe


@click.group()
def dempe_cli():
    """
    CLI for DEMPE prediction and related tasks.
    """


# Add commands to the CLI group
dempe_cli.add_command(predict_dempe, name="predict-dempe")
