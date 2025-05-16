import click
import pandas as pd
import os

@click.command()
@click.option("--input-file", default="data/csv_data/cleaned_commits.csv", required=True, type=click.Path(exists=True), help="Path to the cleaned commits CSV file.")
@click.option("--output-file", default="data/csv_data/semantic_relabelled_commits.csv", type=click.Path(), help="Path to save the relabeled CSV file.")
@click.option("--unmatched-file", default="data/csv_data/semantic_unmatched_commits.csv", type=click.Path(), help="Path to save the unmatched commits CSV file.")
def semantic_relabel(input_file, output_file, unmatched_file):
    """Relabel commits based on semantic context keywords and existing DEMPE class. Save unmatched commits separately."""

    click.echo(f"Loading cleaned commits from {input_file}...")
    df = pd.read_csv(input_file)

    # Define semantic context keyword mapping
    semantic_rules = {
        "0": ["add feature", "implement", "introduce", "create", "initial version", "scaffold", "setup", "prototype", "first commit"],
        "1": ["improve performance", "optimize", "enhance", "clean code", "refactor", "restructure", "speed up", "reduce latency", "memory efficient"],
        "2": ["fix", "resolve issue", "bug", "patch", "typo", "dependency bump", "update packages", "align", "ci config", "update tests", "code style", "minor fix", "formatting"],
        "3": ["license", "copyright", "legal", "vulnerability", "auth", "security", "protect", "privacy", "compliance", "sanitization", "policy", "secure", "access control"],
        "4": ["deploy", "release", "production", "revenue", "monetize", "go live", "API monetization", "market", "analytics", "track usage", "build pipeline", "publish", "distribution"],
    }

    matched_rows = []
    unmatched_rows = []

    click.echo("Relabeling commit messages based on semantic keywords...")
    for _, row in df.iterrows():
        message = str(row["Commit Message"]).lower()
        existing_class = str(row["DEMPE Function Class"])

        # Skip invalid DEMPE classes
        if pd.isnull(existing_class) or existing_class not in semantic_rules:
            unmatched_rows.append(row)
            continue

        # Check for semantic keyword match
        semantic_match = any(keyword in message for keyword in semantic_rules[existing_class])

        if semantic_match:
            row["Relabeled DEMPE Class"] = existing_class
            matched_rows.append(row)
        else:
            unmatched_rows.append(row)

    # Convert to DataFrame
    df_matched = pd.DataFrame(matched_rows)
    df_unmatched = pd.DataFrame(unmatched_rows)

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(unmatched_file), exist_ok=True)

    df_matched.to_csv(output_file, index=False)
    df_unmatched.to_csv(unmatched_file, index=False)

    click.echo(f"✔ Relabeling complete! Matched commits saved to {output_file}")
    click.echo(f"✔ Unmatched commits saved to {unmatched_file}")

if __name__ == "__main__":
    semantic_relabel()
