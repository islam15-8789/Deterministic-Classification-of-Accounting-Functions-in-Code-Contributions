import os
import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from commands.fetch_commits import fetch_commits, fetch_commits_for_repo


@pytest.fixture
def runner():
    """Fixture to create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def setup_test_environment(tmp_path):
    """
    Set up a temporary test environment with dummy JSON files and directories.

    Returns:
        input_file: Path to the input JSON file containing repository details.
        output_folder: Path to the folder where fetched commits will be saved.
    """
    input_file = tmp_path / "repos.json"
    output_folder = tmp_path / "test_data" / "raw_data"
    output_folder.mkdir(parents=True)
    return {"input_file": input_file, "output_folder": output_folder}


def test_missing_input_file(runner):
    """Test behavior when the input file does not exist."""
    result = runner.invoke(fetch_commits, ["--input-file", "non_existent.json"])
    assert result.exit_code == 0
    assert "Input file not found" in result.output


def test_empty_input_file(runner, setup_test_environment):
    """Test behavior when the input file is empty."""
    input_file = setup_test_environment["input_file"]

    # Create an empty JSON file
    with open(input_file, "w") as file:
        file.write("[]")

    result = runner.invoke(fetch_commits, ["--input-file", str(input_file)])
    assert result.exit_code == 0
    assert "The input file is empty or does not contain a valid JSON array." in result.output


def test_invalid_json_format(runner, setup_test_environment):
    """Test behavior when the input file contains invalid JSON."""
    input_file = setup_test_environment["input_file"]

    # Create an invalid JSON file
    with open(input_file, "w") as file:
        file.write("{invalid_json: true}")

    result = runner.invoke(fetch_commits, ["--input-file", str(input_file)])
    assert result.exit_code == 0
    assert "Failed to parse the JSON file" in result.output


@patch("commands.fetch_commits.requests.get")
def test_fetch_commits_for_repo_success(mock_get, setup_test_environment):
    """Test the fetch_commits_for_repo function with successful fetch."""
    repo_info = {"repo_name": "https://github.com/owner/repo", "owner": "owner", "token": "fake_token"}
    progress_bar = MagicMock()

    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = [{"commit": {"message": "Initial commit"}}]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Use the output folder from the fixture
    output_folder = setup_test_environment["output_folder"]

    # Pass the output_folder argument to the function
    result, success = fetch_commits_for_repo(repo_info, progress_bar, output_folder)

    # Verify the results
    assert success
    assert "Commits for https://github.com/owner/repo saved" in result

    # Check if the output file was created and contains the correct content
    repo_name_short = repo_info["repo_name"].split("/")[-1]
    output_file = os.path.join(output_folder, f"{repo_name_short}.json")
    assert os.path.exists(output_file)

    with open(output_file, "r") as file:
        commits = json.load(file)
        assert commits == [{"commit": {"message": "Initial commit"}}]




@patch("commands.fetch_commits.requests.get")
def test_fetch_commits_failure(mock_get, runner, setup_test_environment):
    """Test behavior when fetching commits fails for a repository."""
    input_file = setup_test_environment["input_file"]

    # Create a mock input file with repository details
    repos_data = [
        {"repo_name": "https://github.com/owner/repo1", "owner": "owner", "token": "fake_token"},
    ]
    with open(input_file, "w") as file:
        json.dump(repos_data, file)

    # Mock a failed response for requests.get
    mock_get.side_effect = Exception("Mocked request failure")

    result = runner.invoke(fetch_commits, ["--input-file", str(input_file)])
    assert result.exit_code == 0
    assert "Unexpected error for https://github.com/owner/repo1" in result.output


@patch("commands.fetch_commits.requests.get")
def test_fetch_commits_for_repo_failure(mock_get, setup_test_environment):
    """Test the fetch_commits_for_repo function with a failure."""
    repo_info = {"repo_name": "https://github.com/owner/repo", "owner": "owner", "token": "fake_token"}
    progress_bar = MagicMock()

    # Mock failed response
    mock_get.side_effect = Exception("Mocked request failure")

    # Retrieve output folder from the fixture
    output_folder = setup_test_environment["output_folder"]

    # Pass the output folder to the function
    result, success = fetch_commits_for_repo(repo_info, progress_bar, output_folder)

    # Verify results
    assert not success
    assert "Unexpected error for https://github.com/owner/repo" in result

