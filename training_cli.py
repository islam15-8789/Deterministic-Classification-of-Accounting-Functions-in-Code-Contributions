import click

from commands.train_classification_chain import train_classifier_chain_model
from commands.train_gbm_ovr import train_gbm_model
from commands.train_nn import train_nn_model
from commands.train_one_vs_rest_lg import train_one_vs_rest_lg_model
from commands.train_one_vs_rest_random_forest import train_one_vs_rest_random_forest


@click.group()
def training_cli():
    """
    Training CLI for different model pipelines.
    """


# Register model training commands
training_cli.add_command(train_one_vs_rest_lg_model, name="train-one-vs-rest-ovr")
training_cli.add_command(train_one_vs_rest_random_forest, name="train-random-forest-ovr")
training_cli.add_command(train_gbm_model, name="train-gbm-ovr")
training_cli.add_command(train_nn_model, name="train-nn")
training_cli.add_command(train_classifier_chain_model, name="train-classifier-chain")
