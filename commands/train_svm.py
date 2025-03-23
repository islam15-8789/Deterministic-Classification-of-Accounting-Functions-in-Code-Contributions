import click
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

@click.command()
@click.option("--train-file", default="data/csv_data/train_set.csv", type=click.Path(exists=True), help="Path to training set.")
@click.option("--test-file", default="data/csv_data/test_set.csv", type=click.Path(exists=True), help="Path to test set.")
@click.option("--model-output", default="models/svm/svm_bundle.joblib", type=click.Path(), help="Path to save model bundle.")
@click.option("--figures-dir", default="figures/svm", type=click.Path(), help="Directory to save visualizations.")
def train_svm(train_file, test_file, model_output, figures_dir):
    """Train SVM model, save it, and visualize confusion matrix & classification report."""
    click.echo("📥 Loading datasets...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df["Commit Message"]
    y_train = train_df["DEMPE Function Class"]
    X_test = test_df["Commit Message"]
    y_test = test_df["DEMPE Function Class"]

    click.echo("🧠 Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    click.echo("🔍 Performing Grid Search for SVM...")
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }
    clf = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    clf.fit(X_train_vec, y_train)

    best_model = clf.best_estimator_
    click.echo(f"✅ Best Parameters: {clf.best_params_}")

    y_pred = best_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    click.echo(f"🎯 Accuracy: {accuracy:.4f}")
    report = classification_report(y_test, y_pred)
    click.echo(report)

    # Save model and vectorizer
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    dump({"model": best_model, "vectorizer": vectorizer, "params": clf.best_params_}, model_output)
    click.echo(f"💾 Model saved to {model_output}")

    # Save Confusion Matrix
    os.makedirs(figures_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    click.echo(f"🖼️ Confusion Matrix saved to {cm_path}")

    # Save Classification Report
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cr_path = os.path.join("models/svm/", "classification_report.csv")
    report_df.to_csv(cr_path)
    click.echo(f"📊 Classification Report saved to {cr_path}")

if __name__ == "__main__":
    train_svm()
