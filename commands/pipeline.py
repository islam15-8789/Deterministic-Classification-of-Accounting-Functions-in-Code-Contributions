import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize Rich console for fancy output
console = Console()

def run_pipeline():
    console.rule("[bold blue]Pipeline Execution")

    # Step 1: Fetch commits
    console.print("[bold yellow]Step 1: Fetching commits...[/bold yellow]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Fetching commits...", start=False)
        progress.start_task(task)
        result = subprocess.run(["pdm", "run", "c", "fetch-commits", "--input-file", "repos.json"])
        progress.stop()
    if result.returncode != 0:
        console.print("[bold red]❌ Error in fetch-commits. Exiting pipeline.[/bold red]")
        return
    console.print("[bold green]✔ Step 1 completed successfully![/bold green]")

    # Step 2: Extract commit messages
    console.print("\n[bold yellow]Step 2: Extracting commit messages...[/bold yellow]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Extracting commit messages...", start=False)
        progress.start_task(task)
        result = subprocess.run([
            "pdm", "run", "c", "extract-raw-commit-messages",
            "--input-folder", "data/raw_data",
            "--output-file", "data/csv_data/raw_commit_messages.csv"
        ])
        progress.stop()
    if result.returncode != 0:
        console.print("[bold red]❌ Error in extract-raw-commit-messages. Exiting pipeline.[/bold red]")
        return
    console.print("[bold green]✔ Step 2 completed successfully![/bold green]")

    # Step 3: Label raw commit messages
    console.print("\n[bold yellow]Step 3: Labeling commit messages...[/bold yellow]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Labeling commit messages...", start=False)
        progress.start_task(task)
        result = subprocess.run([
            "pdm", "run", "c", "label-commits",
            "--input-file", "data/csv_data/raw_commit_messages.csv",
            "--output-file", "data/csv_data/labeled_commits.csv"
        ])
        progress.stop()
    if result.returncode != 0:
        console.print("[bold red]❌ Error in label-commits. Exiting pipeline.[/bold red]")
        return
    console.print("[bold green]✔ Step 3 completed successfully![/bold green]")

    # Step 4: Clean labeled commit messages
    console.print("\n[bold yellow]Step 4: Cleaning labeled commit messages...[/bold yellow]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Cleaning labeled commits...", start=False)
        progress.start_task(task)
        result = subprocess.run([
            "pdm", "run", "c", "clean-commits",
            "--input-file", "data/csv_data/labeled_commits.csv",
            "--output-file", "data/csv_data/cleaned_commits.csv",
            "--nonconv-output", "data/csv_data/non_conventional_commits.csv"
        ])
        progress.stop()
    if result.returncode != 0:
        console.print("[bold red]❌ Error in clean-commits. Exiting pipeline.[/bold red]")
        return
    console.print("[bold green]✔ Step 4 completed successfully![/bold green]")

    # Step 5: Split cleaned commits into train and test sets
    console.print("\n[bold yellow]Step 5: Splitting train/test dataset...[/bold yellow]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Splitting dataset...", start=False)
        progress.start_task(task)
        result = subprocess.run([
            "pdm", "run", "c", "split-dataset",
            "--input-file", "data/csv_data/cleaned_commits.csv",
            "--train-output", "data/csv_data/train_set.csv",
            "--test-output", "data/csv_data/test_set.csv"
        ])
        progress.stop()
    if result.returncode != 0:
        console.print("[bold red]❌ Error in split-dataset. Exiting pipeline.[/bold red]")
        return
    console.print("[bold green]✔ Step 5 completed successfully![/bold green]")

    console.rule("[bold green]Pipeline executed successfully![/bold green]")
    console.print("[bold magenta]:rocket: All steps completed! Your data is ready for modeling.[/bold magenta]")

if __name__ == "__main__":
    run_pipeline()
