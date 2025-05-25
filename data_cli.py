import click

from commands.apply_mlsmote import apply_mlsmote
from commands.cleaned_commits import clean_commits
from commands.extract_raw_commits import extract_raw_commit_messages
from commands.fetch_commits import fetch_commits
from commands.label_commits import label_commits
from commands.plot_classification_report import plot_classification_report
from commands.split_train_test import split_dataset
from commands.train_classification_chain import train_classifier_chain_model
from commands.train_gbm_ovr import train_gbm_model
from commands.train_nn import train_nn_model
from commands.train_one_vs_rest_lg import train_one_vs_rest_lg_model
from commands.train_one_vs_rest_random_forest import train_one_vs_rest_random_forest
from commands.visualize_cleaned_commits import visualize_cleaned_commits
from commands.visualize_mlsmote_distribution import visualize_mlsmote_distribution


@click.group()
def data_cli():
    """Main entry for data CLI commands."""


# Attach individual commands
data_cli.add_command(fetch_commits, name="fetch-commits")
data_cli.add_command(extract_raw_commit_messages, name="extract-raw-commit-messages")
data_cli.add_command(label_commits, name="label-commits")
data_cli.add_command(split_dataset, name="split-dataset")
data_cli.add_command(clean_commits, name="clean-commits")
data_cli.add_command(visualize_cleaned_commits, name="visualize-cleaned-commits")
data_cli.add_command(plot_classification_report, name="plot-classification-report")
data_cli.add_command(apply_mlsmote, name="apply-mlsmote")
data_cli.add_command(
    visualize_mlsmote_distribution, name="visualize-mlsmote-distribution"
)
data_cli.add_command(train_one_vs_rest_lg_model, name="train-one-vs-rest-ovr")
data_cli.add_command(train_one_vs_rest_random_forest, name="train-random-forest-ovr")
data_cli.add_command(train_gbm_model, name="train-gbm-ovr")
data_cli.add_command(train_nn_model, name="train-nn")
data_cli.add_command(train_classifier_chain_model, name="train-classifier-chain")
