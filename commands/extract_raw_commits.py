import csv
import json
import os

import click
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn

console = Console()

@click.command(
    help="""
Extract commit messages from JSON files in a specified directory and save them to a CSV file.

This command reads all JSON files in the specified input folder (default: data/raw_data),
extracts the commit.message field from each commit, and saves the messages into a single
CSV file at the specified output path (default: data/csv_data/raw_commit_messages.csv).

The CSV will have the following columns:
1. Serial Number
2. Commit Message
3. Label (empty)
"""
)
@click.option(
    "--input-folder",
    default="data/raw_data",
    help="Path to the input folder containing JSON files.",
)
@click.option(
    "--output-file",
    default="data/csv_data/raw_commit_messages.csv",
    help="Path to the output CSV file.",
)
def extract_raw_commit_messages(input_folder, output_file):
    """
    Extract commit messages from JSON files in the input folder
    and save them to the output file.
    """
    if not os.path.exists(input_folder):
        console.print(f"[bold red]Error:[/bold red] Input folder '{input_folder}' does not exist.")
        return

    # List all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    if not json_files:
        console.print(f"[bold red]Error:[/bold red] No JSON files found in the '{input_folder}' directory.")
        return

    console.print(f"[bold green]Found {len(json_files)} JSON files to process.[/bold green]\n")

    commit_messages = []
    errors = []
    successes = []

    # Process JSON files with a rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing JSON files", total=len(json_files))

        for json_file in json_files:
            json_path = os.path.join(input_folder, json_file)
            try:
                with open(json_path, "r") as file:
                    commits = json.load(file)
                    for commit in commits:
                        if "commit" in commit and "message" in commit["commit"]:
                            commit_messages.append(commit["commit"]["message"])
                successes.append(f"[bold green]Processed file:[/bold green] {json_file}")
            except json.JSONDecodeError:
                error_message = f"[bold red]Error decoding JSON:[/bold red] {json_file}"
                errors.append(error_message)
                console.log(error_message)
            except Exception as e:
                error_message = f"[bold red]Error processing file:[/bold red] {json_file} - {str(e)}"
                errors.append(error_message)
                console.log(error_message)
            progress.update(task, advance=1)

    if not commit_messages:
        console.print("[bold yellow]No commit messages found in the JSON files.[/bold yellow]")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write commit messages to the CSV file with a rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Writing to CSV", total=len(commit_messages))
        try:
            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(["Serial Number", "Commit Message", "Label"])  # Write header
                for i, message in enumerate(commit_messages, start=1):
                    writer.writerow([i, message, ""])  # Serial number, message, empty label
                    progress.update(task, advance=1)
            successes.append(f"[bold green]Commit messages successfully saved to:[/bold green] {output_file}")
        except Exception as e:
            errors.append(f"[bold red]Error writing to CSV file:[/bold red] {str(e)}")

    # Display results
    console.rule("[bold blue]Process Results")
    if successes:
        for success in successes:
            console.print(success)
    if errors:
        for error in errors:
            console.print(error)
