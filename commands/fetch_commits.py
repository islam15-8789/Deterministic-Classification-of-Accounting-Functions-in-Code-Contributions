import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import requests
from tqdm import tqdm


@click.command(
    help="""
Fetch commits from GitHub repositories listed in a JSON file and save them to individual JSON files.

This command reads a JSON file containing repository details, fetches the commit history for each repository
using the GitHub API, and saves the results into separate JSON files under the specified output folder
(default: ./data/raw_data).
"""
)
@click.option(
    "--input-file",
    default="repos.json",
    show_default=True,
    help="""
Path to the JSON file containing GitHub repository details.

The input file should be a JSON array where each object contains the following keys:
  - "repo_name": The URL of the repository
  - "owner": The owner or organization of the repository
  - "token": A personal GitHub access token.
""",
)
@click.option(
    "--output-folder",
    default="data/raw_data",
    show_default=True,
    help="Path to the folder where fetched commits will be saved.",
)
def fetch_commits(input_file, output_folder):
    """
    Fetch commits from GitHub repositories listed in a JSON file and
    save them to JSON files.
    """
    import os

    print("📦 Inside fetch_commits")
    print("🧭 CWD:", os.getcwd())
    print("📁 LS:", os.listdir("."))
    print("📁 Target output folder:", output_folder)
    try:
        # Read repository details from input file
        with open(input_file, "r") as file:
            repos = json.load(file)

        if not repos or not isinstance(repos, list):
            click.echo("The input file is empty or does not contain a valid JSON array.")
            return

        total_repos = len(repos)
        click.echo(f"Found {total_repos} repos to process.")

        # Create progress bars for each repo with unique positions
        progress_bars = []
        for i, repo in enumerate(repos):
            bar = tqdm(
                total=100,
                desc=repo["repo_name"].split("/")[-1],
                unit="%",
                position=i,
                ascii=" >=",
                leave=True,  # Keep progress bars after completion
            )
            progress_bars.append((repo, bar))

        results = []  # List to store results

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            future_to_repo = {
                executor.submit(fetch_commits_for_repo, repo, bar, output_folder): (
                    repo,
                    bar,
                )
                for repo, bar in progress_bars
            }

            for future in as_completed(future_to_repo):
                result, is_success = future.result()
                results.append((result, is_success))

        # Close all progress bars
        for _, bar in progress_bars:
            bar.close()

        # Display all results at the end
        click.echo("Process Results:")
        for message, is_success in results:
            if is_success:
                click.echo(f"SUCCESS: {message}")
            else:
                click.echo(f"ERROR: {message}")
        click.echo("\n")
    except FileNotFoundError:
        click.echo("Input file not found. Please provide a valid file path.")
    except json.JSONDecodeError:
        click.echo(
            "Failed to parse the JSON file. Please ensure it is properly formatted."
        )
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}")


def fetch_commits_for_repo(repo_info, progress_bar, output_folder):
    """
    Fetch commits for a single repository and update its progress bar.
    """
    repo_name = repo_info.get("repo_name")
    owner = repo_info.get("owner")
    token = repo_info.get("token")

    if not repo_name or not owner or not token:
        return f"Invalid entry found: {repo_info}", False

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        # Extract the actual repo name from the URL
        repo_name_short = repo_name.split("/")[-1]
        url = f"https://api.github.com/repos/{owner}/{repo_name_short}/commits"

        # Simulate progress stages
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Simulated progress update
        for _ in range(10):  # Simulating 10 progress steps
            time.sleep(0.2)  # Simulate a delay
            progress_bar.update(10)  # Update 10% of the progress

        commits = response.json()

        # Save commits to a JSON file in the output folder
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{repo_name_short}.json")
        with open(output_file, "w") as outfile:
            json.dump(commits, outfile, indent=4)

        progress_bar.set_description(f"Completed {repo_name_short}")
        progress_bar.n = progress_bar.total  # Mark as complete
        progress_bar.refresh()
        return f"Commits for {repo_name} saved to {output_file}", True

    except requests.exceptions.RequestException as e:
        progress_bar.set_description(f"Failed {repo_name_short}")
        progress_bar.n = progress_bar.total  # Mark as complete
        progress_bar.refresh()
        return f"Failed to fetch commits for {repo_name}: {e}", False

    except Exception as e:  # Catch any unexpected exceptions
        progress_bar.set_description(f"Error {repo_name_short}")
        progress_bar.n = progress_bar.total  # Mark as complete
        progress_bar.refresh()
        return f"Unexpected error for {repo_name}: {e}", False
