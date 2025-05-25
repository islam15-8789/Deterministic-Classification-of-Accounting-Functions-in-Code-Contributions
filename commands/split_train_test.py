import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    "--input-file",
    default="data/csv_data/resampled_mlsmote.csv",
    type=click.Path(exists=True),
    help="Path to the resampled_mlsmote CSV file.",
)
@click.option(
    "--train-output",
    default="data/csv_data/train_re_sampled_mlsmote.csv",
    type=click.Path(),
    help="Path to save the training set.",
)
@click.option(
    "--test-output",
    default="data/csv_data/test_re_sampled_mlsmote.csv",
    type=click.Path(),
    help="Path to save the test set.",
)
@click.option(
    "--test-size",
    default=0.2,
    help="Proportion of the dataset to include in the test split.",
)
def split_dataset(input_file, train_output, test_output, test_size):
    """
    Splits multi-label dataset into training and test sets.
    Keeps all DEMPE_Class_* columns as labels.
    """
    click.echo(f"📥 Reading multi-label data from {input_file}...")
    df = pd.read_csv(input_file)

    label_cols = [col for col in df.columns if col.startswith("DEMPE_Class_")]
    feature_cols = [col for col in df.columns if col.startswith("f_")]

    X = df[feature_cols]
    y = df[label_cols]

    click.echo("🔄 Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Combine features and labels
    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    click.echo(f"✅ Training set saved to {train_output}")
    click.echo(f"✅ Test set saved to {test_output}")


if __name__ == "__main__":
    split_dataset()
