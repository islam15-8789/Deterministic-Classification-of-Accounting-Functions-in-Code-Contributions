import click
from joblib import load
from rich import print
from constants import dempe_prediction_mapping
from commands.cleaned_commits import clean_text


@click.command()
@click.option(
    "--model-path",
    default="models/svm/svm_bundle.joblib",
    type=click.Path(exists=True),
    help="Path to the saved model and vectorizer bundle."
)
def classify_commit(model_path):
    """Classify commit messages using a trained SVM model in a loop until the user exits."""
    print("[bold cyan]🔍 Loading model and vectorizer...[/bold cyan]")
    bundle = load(model_path)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    print("[bold green]✔ Model loaded successfully![/bold green]")

    while True:
        commit = click.prompt("Enter a commit message (or type 0 to exit)")

        if commit.strip() == "0":
            print("[bold yellow]👋 Exiting classification loop. Goodbye![/bold yellow]")
            break

        cleaned = clean_text(commit)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        pred_label = dempe_prediction_mapping.get(pred, "Unknown")

        print(f"🧼 [italic]Cleaned:[/italic] {cleaned}")
        print(f"🚀 [bold]Predicted DEMPE Function:[/bold] [green]{pred_label}[/green] (Class {pred})\n")

if __name__ == "__main__":
    classify_commit()
