import os
import json
import click
import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# Friendly class names
DEMPE_CLASSES = [
    "Development",
    "Enhancement",
    "Maintenance",
    "Protection",
    "Exploitation"
]

# Model options
MODELS = {
    1: ("Logistic Regression (OvR)", "data/models/logreg_ovr_model.pkl"),
    2: ("Random Forest (OvR)", "data/models/rf_ovr_model.pkl"),
    3: ("XGBoost/LightGBM (OvR)", "data/models/gbm_ovr_model.pkl"),
    4: ("Neural Network", "data/models/nn_multilabel_model.h5"),
    5: ("Classifier Chain", "data/models/classifier_chain_model.pkl"),
}

@click.command()
@click.option("--model-choice", type=int, default=None, help="Optional model choice (1-5)")
def predict_dempe(model_choice):
    """Interactive tool to classify commit messages into DEMPE classes."""
    console = Console()
    console.rule("[bold green]DEMPE Class Predictor")

    # Model selection
    if model_choice is None:
        console.print("\n[bold yellow]Select a trained model to use:[/bold yellow]")
        for k, v in MODELS.items():
            console.print(f"[cyan]{k}.[/cyan] {v[0]}")
        model_choice = IntPrompt.ask("\nEnter model number", choices=[str(k) for k in MODELS])

    model_name, model_path = MODELS[int(model_choice)]
    console.print(f"\n📦 Loading model: [green]{model_name}[/green]")

    if "Neural Network" in model_name:
        model = load_model(model_path)
    else:
        model = joblib.load(model_path)

    # SentenceBERT for encoding
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    while True:
        console.print("\n📝 Enter a commit message to classify (or type 'exit' to quit):")
        commit = Prompt.ask("Commit Message")
        if commit.lower() == "exit":
            console.print("[bold red]Exiting...[/bold red]")
            break

        X = sbert_model.encode([commit])

        if "Neural Network" in model_name:
            y_pred = (model.predict(X) > 0.5).astype(int)
        else:
            y_pred = model.predict(X)

        result_labels = [label for pred, label in zip(y_pred[0], DEMPE_CLASSES) if pred == 1]
        if result_labels:
            console.print(f"[bold green]✅ Predicted DEMPE Classes:[/bold green] {', '.join(result_labels)}")
        else:
            console.print("[bold red]❌ No DEMPE class confidently predicted.[/bold red]")

        if not Prompt.ask("Want to classify another commit?", default="y").lower().startswith("y"):
            break


if __name__ == "__main__":
    predict_dempe()
