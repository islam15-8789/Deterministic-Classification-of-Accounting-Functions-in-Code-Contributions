import re

import click
import pandas as pd
from rich.console import Console
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from constants import dempe_conv_commit_mapping

console = Console()


class MultiLabelCommitClassifier:
    def __init__(self, file_path, output_file, commit_column="Commit Message"):
        self.file_path = file_path
        self.output_file = output_file
        self.commit_column = commit_column
        self.mapping = dempe_conv_commit_mapping

    def extract_commit_tags(self, commit_msg):
        tags = []
        commit_msg = commit_msg.lower()
        for tag in self.mapping:
            if (
                re.search(rf"\b{re.escape(tag)}(?:\(.+?\))?!?:", commit_msg)
                or f"{tag}:" in commit_msg
            ):
                tags.append(self.mapping[tag])
        return sorted(set(tags))

    def process_commits(self):
        try:
            console.print(
                f"📂 Loading commit data from: [bold green]{self.file_path}[/bold green]"
            )
            df = pd.read_csv(self.file_path)

            if self.commit_column not in df.columns:
                raise ValueError(f"Column '{self.commit_column}' not found in CSV file.")

            expanded_commits = []
            with tqdm(total=len(df), desc="Processing Commits", unit="row") as pbar:
                for _, row in df.iterrows():
                    commit_messages = str(row[self.commit_column]).split("\n")
                    for commit in commit_messages:
                        commit = commit.strip()
                        if commit:
                            labels = self.extract_commit_tags(commit)
                            expanded_commits.append(
                                {
                                    "Raw Serial Number": row.get("Serial Number", None),
                                    "Commit Message": commit,
                                    "DEMPE_Labels": labels,
                                }
                            )
                    pbar.update(1)

            expanded_df = pd.DataFrame(expanded_commits)

            # Multi-label binarization
            mlb = MultiLabelBinarizer(classes=sorted(set(self.mapping.values())))
            label_matrix = mlb.fit_transform(expanded_df["DEMPE_Labels"])
            label_df = pd.DataFrame(
                label_matrix, columns=[f"DEMPE_Class_{i}" for i in mlb.classes_]
            )
            result_df = pd.concat(
                [expanded_df.drop(columns=["DEMPE_Labels"]), label_df], axis=1
            )

            result_df.to_csv(self.output_file, index=False)

            console.print(
                f"\n✅ [bold cyan]Multi-label classification complete![/bold cyan] Results saved to: [bold green]{self.output_file}[/bold green]"
            )
            console.print("\n📊 [bold magenta]Label distribution:[/bold magenta]")
            console.print(label_df.sum().to_string())

        except Exception as e:
            console.print(f"❌ [bold red]Error processing the file:[/bold red] {e}")


@click.command()
@click.option(
    "--input-file",
    default="data/csv_data/raw_commit_messages.csv",
    help="Path to the CSV file containing raw commit messages.",
)
@click.option(
    "--output-file",
    default="data/csv_data/multilabel_labeled_commits.csv",
    help="Path to save the multi-labeled commit messages.",
)
def label_commits_multi(input_file, output_file):
    """
    CLI command to classify commit messages into multiple DEMPE classes.

    Example Usage:
    $ python label_commits_multi.py --input-file data/csv_data/raw_commit_messages.csv --output-file data/csv_data/labeled_commits.csv
    """
    classifier = MultiLabelCommitClassifier(input_file, output_file)
    classifier.process_commits()


@click.command()
@click.option(
    "--input-file",
    default="data/csv_data/raw_commit_messages.csv",
    help="Path to the CSV file containing raw commit messages.",
)
@click.option(
    "--output-file",
    default="data/csv_data/labeled_commits.csv",
    help="Path to save the labeled commit messages.",
)
def label_commits(input_file, output_file):
    """
    CLI command to classify commit messages based on the DEMPE framework.

    Example Usage:
    $ python label_commits.py --input-file data/csv_data/raw_commit_messages.csv --output-file data/csv_data/labeled_commits.csv
    """
    classifier = MultiLabelCommitClassifier(input_file, output_file)
    classifier.process_commits()


if __name__ == "__main__":
    label_commits()
