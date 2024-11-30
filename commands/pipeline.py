import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize Rich console for fancy output
console = Console()

def run_pipeline():
    console.rule("[bold blue]Pipeline Execution")
    
    # Step 1: Fetch commits
    console.print("[bold yellow]Step 1: Fetching commits...[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
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
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Extracting commit messages...", start=False)
        progress.start_task(task)
        result = subprocess.run(
            [
                "pdm", "run", "c", "extract-raw-commit-messages",
                "--input-folder", "data/raw_data",
                "--output-file", "data/csv_data/raw_commit_messages.csv"
            ]
        )
        progress.stop()

    if result.returncode != 0:
        console.print("[bold red]❌ Error in extract-raw-commit-messages. Exiting pipeline.[/bold red]")
        return
    console.print("[bold green]✔ Step 2 completed successfully![/bold green]")

    console.rule("[bold green]Pipeline executed successfully![/bold green]")
    console.print("[bold magenta]:rocket: All steps completed! Your data is ready.[/bold magenta]")

if __name__ == "__main__":
    run_pipeline()
