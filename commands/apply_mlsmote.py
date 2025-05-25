import json
import os

import click
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from skmultilearn.model_selection import iterative_train_test_split


@click.command()
@click.option(
    "--input-file",
    default="data/csv_data/cleaned_commits.csv",
    type=click.Path(exists=True),
    help="Path to the cleaned multi-label CSV file with commit messages and DEMPE classes.",
)
@click.option(
    "--output-file",
    default="data/csv_data/resampled_mlsmote_bert.csv",
    type=click.Path(),
    help="Path to save the resampled dataset after applying multilabel SMOTE.",
)
@click.option(
    "--vectorizer-file",
    default="data/models/sentence_bert_model_name.txt",
    type=click.Path(),
    help="Path to save the Sentence-BERT model name used for encoding.",
)
@click.option(
    "--model-name",
    default="all-MiniLM-L6-v2",
    help="Pretrained Sentence-BERT model to use (e.g., all-MiniLM-L6-v2).",
)
@click.option(
    "--k",
    default=5,
    help="Number of nearest neighbors for MLSMOTE approximation.",
)
@click.option(
    "--samples-per-class",
    default=200,
    help="Number of synthetic samples to generate per underrepresented class.",
)
def apply_mlsmote(
    input_file, output_file, vectorizer_file, model_name, k, samples_per_class
):
    """
    Applies approximated MLSMOTE to commit message dataset using Sentence-BERT embeddings,
    while excluding the majority class from oversampling and synthetic label assignment.
    """
    click.echo(f"📥 Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    label_cols = [col for col in df.columns if col.startswith("DEMPE_Class_")]
    click.echo(f"🧷 Identified label columns: {label_cols}")

    df = df[df[label_cols].sum(axis=1) > 0]

    click.echo(f"🤖 Loading Sentence-BERT model: {model_name}...")
    model = SentenceTransformer(model_name)

    click.echo("🔢 Encoding commit messages into dense vectors...")
    X = model.encode(df["Commit Message"].astype(str).tolist(), show_progress_bar=True)
    y = df[label_cols].values

    os.makedirs(os.path.dirname(vectorizer_file), exist_ok=True)
    with open(vectorizer_file, "w") as f:
        f.write(model_name)
    click.echo(f"📁 Saved Sentence-BERT model name reference to: {vectorizer_file}")

    click.echo("🔀 Splitting with iterative stratification...")
    X_train, y_train, _, _ = iterative_train_test_split(np.array(X), y, test_size=0.0)

    click.echo("🧪 Calculating per-label sample counts...")
    label_sums = np.sum(y_train, axis=0)
    majority_class_idx = int(np.argmax(label_sums))

    click.echo(
        f"🚫 Excluding majority class: {label_cols[majority_class_idx]} from oversampling"
    )

    label_gap = {
        i: max(0, int(label_sums[majority_class_idx] - label_sums[i]))
        for i in range(len(label_cols))
        if i != majority_class_idx
    }
    total_needed = {
        i: min(samples_per_class, gap) for i, gap in label_gap.items() if gap > 0
    }

    click.echo("🧪 Generating balanced synthetic samples...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X_train)
    neighbors = nn.kneighbors(X_train, return_distance=False)

    synthetic_X = []
    synthetic_y = []

    np.random.seed(42)
    for class_idx, count in total_needed.items():
        class_indices = np.where(y_train[:, class_idx] == 1)[0]
        for _ in range(count):
            if len(class_indices) == 0:
                continue
            idx = np.random.choice(class_indices)
            ref_vec = X_train[idx]
            ref_label = y_train[idx]

            neighbor_idx = np.random.choice(neighbors[idx][1:])
            neighbor_vec = X_train[neighbor_idx]
            neighbor_label = y_train[neighbor_idx]

            lam = np.random.rand()
            synth_vec = ref_vec + lam * (neighbor_vec - ref_vec)
            synth_label = np.maximum(ref_label, neighbor_label)

            # Exclude majority class from synthetic label
            synth_label[majority_class_idx] = 0

            synthetic_X.append(synth_vec)
            synthetic_y.append(synth_label)

    click.echo("🧬 Synthetic samples created. Combining with original data...")
    X_final = np.vstack([X_train, np.array(synthetic_X)])
    y_final = np.vstack([y_train, np.array(synthetic_y)])

    final_df = pd.DataFrame(X_final, columns=[f"f_{i}" for i in range(X_final.shape[1])])
    for i, col in enumerate(label_cols):
        final_df[col] = y_final[:, i].astype(int)

    click.echo("📊 Final label distribution:")
    click.echo(final_df[label_cols].sum().to_string())

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    click.echo(f"✅ Resampled multilabel dataset saved to: {output_file}")


if __name__ == "__main__":
    apply_mlsmote()
