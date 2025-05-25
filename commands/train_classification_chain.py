import json
import os

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.helper import evaluate_and_save_metrics


@click.command()
@click.option(
    "--train-file",
    default="data/csv_data/train_re_sampled_mlsmote.csv",
    type=click.Path(exists=True),
    help="Path to the training CSV file.",
)
@click.option(
    "--test-file",
    default="data/csv_data/test_re_sampled_mlsmote.csv",
    type=click.Path(exists=True),
    help="Path to the test CSV file.",
)
@click.option(
    "--model-file",
    default="data/models/classifier_chain_model.pkl",
    type=click.Path(),
    help="Path to store the trained model.",
)
@click.option(
    "--params-file",
    default="data/models/classifier_chain_params.json",
    type=click.Path(),
    help="Path to store model training parameters and best score.",
)
def train_classifier_chain_model(train_file, test_file, model_file, params_file):
    """
    Trains a ClassifierChain with LogisticRegression and evaluates on test set.
    """
    click.echo(f"📅 Loading training data from {train_file}...")
    df_train = pd.read_csv(train_file)

    click.echo(f"📅 Loading test data from {test_file}...")
    df_test = pd.read_csv(test_file)

    label_cols = [col for col in df_train.columns if col.startswith("DEMPE_Class_")]
    feature_cols = [col for col in df_train.columns if col.startswith("f_")]

    X_train = df_train[feature_cols].values
    y_train = df_train[label_cols].values
    X_test = df_test[feature_cols].values
    y_test = df_test[label_cols].values

    click.echo(f"🔢 Features: {len(feature_cols)} | Labels: {len(label_cols)}")

    base_model = LogisticRegression(solver="liblinear")
    chain_model = ClassifierChain(base_model)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", chain_model),
        ]
    )

    param_grid = {
        "clf__base_estimator__C": [0.01, 0.1, 1, 10],
        "clf__base_estimator__penalty": ["l1", "l2"],
    }

    click.echo("🔍 Performing GridSearchCV...")
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_micro",
        cv=3,
        verbose=1,
        n_jobs=1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Evaluate on test set
    click.echo("📊 Evaluating on test set...")
    y_pred = best_model.predict(X_test)

    eval_dir = "data/reports/classifier_chain"
    os.makedirs(eval_dir, exist_ok=True)

    evaluate_and_save_metrics(
        y_true=y_test,
        y_pred=y_pred,
        label_names=label_cols,
        output_dir=eval_dir,
        model_name="classifier Chain",
    )

    joblib.dump(best_model, model_file)
    click.echo(f"✅ Trained model saved to: {model_file}")

    with open(params_file, "w") as f:
        json.dump(
            {
                "best_params": grid.best_params_,
                "best_score": grid.best_score_,
                "scoring": "f1_micro",
                "label_columns": label_cols,
            },
            f,
            indent=2,
        )

    click.echo(f"✅ Training details saved to: {params_file}")


if __name__ == "__main__":
    train_classifier_chain_model()
