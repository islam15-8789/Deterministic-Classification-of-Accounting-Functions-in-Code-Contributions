import click
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


@click.command()
@click.option(
    "--train-file",
    default="data/csv_data/train_set.csv",
    type=click.Path(exists=True),
    help="Path to the training dataset.",
)
@click.option(
    "--test-file",
    default="data/csv_data/test_set.csv",
    type=click.Path(exists=True),
    help="Path to the test dataset.",
)
@click.option(
    "--model-output",
    default="data/models/tfidf_logreg_model.pkl",
    type=click.Path(),
    help="Path to save the trained model.",
)
@click.option(
    "--report-output",
    default="data/reports/logreg_classification_report.txt",
    type=click.Path(),
    help="Path to save the classification report.",
)
def train_test_logreg_model(train_file, test_file, model_output, report_output):
    """
    Trains a multi-label classifier using TF-IDF + Logistic Regression (OneVsRest)
    with class_weight='balanced'. Evaluates on test data and saves model + report.
    """
    click.echo(f"📥 Loading train set from {train_file}...")
    train_df = pd.read_csv(train_file)
    click.echo(f"📥 Loading test set from {test_file}...")
    test_df = pd.read_csv(test_file)

    label_cols = [col for col in train_df.columns if col.startswith("DEMPE_Class_")]

    X_train = train_df["Commit Message"]
    y_train = train_df[label_cols]
    X_test = test_df["Commit Message"]
    y_test = test_df[label_cols]

    click.echo(
        "🔧 Building TF-IDF + Logistic Regression pipeline (with class_weight='balanced')..."
    )
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear", max_iter=1000, class_weight="balanced"
                    )
                ),
            ),
        ]
    )

    click.echo("🚀 Training model...")
    pipeline.fit(X_train, y_train)

    click.echo("🧪 Evaluating model...")
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=label_cols, zero_division=0
    )

    with open(report_output, "w") as f:
        f.write(report)

    click.echo(f"✅ Classification report saved to {report_output}")

    joblib.dump(pipeline, model_output)
    click.echo(f"✅ Trained model saved to {model_output}")


if __name__ == "__main__":
    train_test_logreg_model()
