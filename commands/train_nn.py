import json
import os

import click
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import RandomSearch
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
    default="data/models/nn_multilabel_model.h5",
    type=click.Path(),
    help="Path to store the trained Keras model.",
)
@click.option(
    "--params-file",
    default="data/models/nn_multilabel_params.json",
    type=click.Path(),
    help="Path to store training parameters and summary.",
)
def train_nn_model(train_file, test_file, model_file, params_file):
    """
    Trains a feedforward neural network for multilabel classification using Keras with Keras Tuner.
    """
    click.echo(f"📥 Loading training data from {train_file}...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    label_cols = [col for col in df_train.columns if col.startswith("DEMPE_Class_")]
    feature_cols = [col for col in df_train.columns if col.startswith("f_")]

    X_train = df_train[feature_cols].values
    y_train = df_train[label_cols].values
    X_test = df_test[feature_cols].values
    y_test = df_test[label_cols].values

    click.echo(f"🔢 Features: {len(feature_cols)} | Labels: {len(label_cols)}")

    def build_model(hp):
        model = Sequential()
        model.add(
            Dense(
                hp.Int("units1", 128, 512, step=64),
                activation="relu",
                input_shape=(X_train.shape[1],),
            )
        )
        model.add(Dropout(hp.Float("dropout1", 0.2, 0.5, step=0.1)))
        model.add(Dense(hp.Int("units2", 64, 256, step=64), activation="relu"))
        model.add(Dropout(hp.Float("dropout2", 0.2, 0.5, step=0.1)))
        model.add(Dense(len(label_cols), activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=hp.Choice("lr", [0.001, 0.0005])),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        overwrite=True,
        directory="tuner_logs",
        project_name="multilabel_nn",
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    click.echo("🔍 Searching for best hyperparameters...")
    tuner.search(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=1000,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]

    # Evaluate on test set
    click.echo("📊 Classification report on test set:")
    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    eval_dir = "data/reports/nn"
    os.makedirs(eval_dir, exist_ok=True)

    evaluate_and_save_metrics(
        y_true=y_test,
        y_pred=y_pred,
        label_names=label_cols,
        output_dir=eval_dir,
        model_name="Neural Network",
    )

    tf.keras.models.save_model(best_model, model_file)
    click.echo(f"✅ Keras model saved to: {model_file}")

    with open(params_file, "w") as f:
        json.dump(
            {
                "best_hyperparameters": best_hp.values,
                "label_columns": label_cols,
            },
            f,
            indent=2,
        )

    click.echo(f"✅ Training details saved to: {params_file}")


if __name__ == "__main__":
    train_nn_model()
