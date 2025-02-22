import pandas as pd
import re
import click
from tqdm import tqdm
from rich.console import Console
from constants import dempe_conv_commit_mapping

console = Console()

class CommitClassifier:
    def __init__(self, file_path, output_file, commit_column="Commit Message"):
        """
        Initializes the CommitClassifier with file paths and column name.

        Args:
            file_path (str): Path to the raw commits CSV file.
            output_file (str): Path to save the classified commits.
            commit_column (str): Column name containing commit messages.
        """
        self.file_path = file_path
        self.output_file = output_file
        self.commit_column = commit_column
        self.dempe_conv_commit_mapping = dempe_conv_commit_mapping


    def extract_commit_type(self, commit_message):
        """
        Extracts the conventional commit type from a commit message.

        Args:
            commit_message (str): The commit message text.

        Returns:
            str: The extracted commit type or None if not found.
        """
        commit_types = ["feat", "fix", "chore", "ci", "docs", "style", "refactor", "perf", "test", "build", "breaking change"]
        commit_message = commit_message.lower()

        for commit_type in commit_types:
            pattern = rf"\b{commit_type}(?:\(.+?\))?!?:"
            match = re.search(pattern, commit_message, re.IGNORECASE)
            if match:
                return commit_type  # Return the first matching commit type

        return None  # Return None if no match is found

    def classify_commit(self, commit_message):
        """
        Classifies a commit message based on its extracted type.

        Args:
            commit_message (str): The commit message text.

        Returns:
            int/str: The corresponding DEMPE function class or "Non-conventional commit".
        """
        commit_type = self.extract_commit_type(commit_message)
        return dempe_conv_commit_mapping.get(commit_type, "Non-conventional")

    def process_commits(self):
        """
        Reads the raw commits, splits multiple commits into separate rows, classifies them, 
        and saves the results. Displays progress using tqdm.
        """
        try:
            console.print(f"📂 Loading commit data from: [bold green]{self.file_path}[/bold green]")

            df = pd.read_csv(self.file_path)

            if self.commit_column not in df.columns:
                raise ValueError(f"Column '{self.commit_column}' not found in CSV file.")

            expanded_commits = []
            with tqdm(total=len(df), desc="Processing Commits", unit="commit") as pbar:
                for _, row in df.iterrows():
                    commit_messages = str(row[self.commit_column]).split("\n")
                    for commit in commit_messages:
                        commit = commit.strip()
                        if commit:
                            expanded_commits.append([row["Serial Number"], commit])
                    pbar.update(1)

            expanded_df = pd.DataFrame(expanded_commits, columns=["Raw Serial Number", self.commit_column])

            # Apply classification with progress bar
            tqdm.pandas(desc="Classifying Commits")
            expanded_df["DEMPE Function Class"] = expanded_df[self.commit_column].progress_apply(self.classify_commit)

            expanded_df.to_csv(self.output_file, index=False)

            console.print(f"\n✅ [bold cyan]Classification complete![/bold cyan] Results saved to: [bold green]{self.output_file}[/bold green]")

            # Print summary of classification
            class_counts = expanded_df["DEMPE Function Class"].value_counts()
            console.print("\n📊 [bold magenta]Summary of Commit Classification:[/bold magenta]")
            console.print(class_counts.to_string())

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
    default="data/csv_data/labeled_commits.csv",
    help="Path to save the labeled commit messages.",
)
def label_commits(input_file, output_file):
    """
    CLI command to classify commit messages based on the DEMPE framework.

    Example Usage:
    $ python label_commits.py --input-file data/csv_data/raw_commit_messages.csv --output-file data/csv_data/labeled_commits.csv
    """
    classifier = CommitClassifier(input_file, output_file)
    classifier.process_commits()

if __name__ == "__main__":
    label_commits()
