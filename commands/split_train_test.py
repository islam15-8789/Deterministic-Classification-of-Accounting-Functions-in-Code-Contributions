import click
import pandas as pd
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--input-file", default="data/csv_data/cleaned_commits.csv", type=click.Path(exists=True), help="Path to the labeled CSV file.")
@click.option("--train-output", default="data/csv_data/train_set.csv", type=click.Path(), help="Path to save the training set.")
@click.option("--test-output", default="data/csv_data/test_set.csv", type=click.Path(), help="Path to save the test set.")
@click.option("--test-size", default=0.2, help="Proportion of the dataset to include in the test split.")
def split_dataset(input_file, train_output, test_output, test_size):
    """
    Splits the labeled dataset into training and test sets, excluding non-conventional commits.
    Saves non-conventional commits separately.
    """
    click.echo(f"Reading labeled data from {input_file}...")
    df = pd.read_csv(input_file)
   
    X = df["Commit Message"]
    y = df["DEMPE Function Class"]

    click.echo("Performing train-test split on conventional commits...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    train_df = pd.DataFrame({"Commit Message": X_train, "DEMPE Function Class": y_train})
    test_df = pd.DataFrame({"Commit Message": X_test, "DEMPE Function Class": y_test})

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    click.echo(f"Training set saved to {train_output}")
    click.echo(f"Test set saved to {test_output}")
