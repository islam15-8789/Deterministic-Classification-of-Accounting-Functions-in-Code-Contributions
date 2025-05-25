import json
import os

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.helper import evaluate_and_save_metrics


@click.command()
@click.option(
    "--train-file",
    default="data/csv_data/train_re_sampled_mlsmote.csv",
    type=click.Path(exists=True),
    help="Path to the training CSV file with Sentence-BERT features and DEMPE labels.",
)
@click.option(
    "--test-file",
    default="data/csv_data/test_re_sampled_mlsmote.csv",
    type=click.Path(exists=True),
    help="Path to the test CSV file.",
)
@click.option(
    "--model-file",
    default="data/models/rf_ovr_model.pkl",
    type=click.Path(),
    help="Path to store the trained model.",
)
@click.option(
    "--params-file",
    default="data/models/rf_ovr_params.json",
    type=click.Path(),
    help="Path to store model training parameters and best score.",
)
def train_one_vs_rest_random_forest(train_file, test_file, model_file, params_file):
    """
    Trains a RandomForestClassifier with OneVsRest strategy and evaluates it on the test set.
    Designed for multilabel classification using Sentence-BERT embeddings.
    """
    click.echo(f"📥 Loading training data from {train_file}...")
    df_train = pd.read_csv(train_file)

    click.echo(f"📥 Loading test data from {test_file}...")
    df_test = pd.read_csv(test_file)

    label_cols = [col for col in df_train.columns if col.startswith("DEMPE_Class_")]
    feature_cols = [col for col in df_train.columns if col.startswith("f_")]

    X_train = df_train[feature_cols].values
    y_train = df_train[label_cols].values
    X_test = df_test[feature_cols].values
    y_test = df_test[label_cols].values
    print(
        y_test.sum(axis=0), "Number of samples per class"
    )  # Number of samples per class
    click.echo(f"🔢 Features: {len(feature_cols)} | Labels: {len(label_cols)}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", OneVsRestClassifier(RandomForestClassifier(random_state=42))),
        ]
    )

    param_grid = {
        "clf__estimator__n_estimators": [100, 200],
        "clf__estimator__max_depth": [None, 10, 20],
        "clf__estimator__min_samples_split": [2, 5],
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
    click.echo("📊 Classification report on test set:")
    y_pred = best_model.predict(X_test)

    eval_dir = "data/reports/random_forest_ovr"
    os.makedirs(eval_dir, exist_ok=True)

    evaluate_and_save_metrics(
        y_true=y_test,
        y_pred=y_pred,
        label_names=label_cols,
        output_dir=eval_dir,
        model_name="Random forest",
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
    train_one_vs_rest_random_forest()
