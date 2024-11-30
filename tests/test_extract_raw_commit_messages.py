import os
import json
import pytest
from click.testing import CliRunner
from commands.extract_raw_commits import extract_raw_commit_messages


@pytest.fixture
def runner():
    """Fixture to create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def setup_test_environment(tmp_path):
    """
    Set up a temporary test environment with dummy JSON files and directories.

    Returns:
        input_folder: Path to the input folder containing dummy JSON files.
        output_file: Path to the expected output CSV file.
    """
    input_folder = tmp_path / "test_data" / "raw_data"
    output_folder = tmp_path / "test_data" / "csv_data"
    input_folder.mkdir(parents=True)
    output_folder.mkdir(parents=True)

    return {
        "input_folder": input_folder,
        "output_file": output_folder / "raw_commit_messages.csv",
    }


def test_no_input_folder(runner, tmp_path):
    """Test when the input folder does not exist."""
    missing_folder = tmp_path / "missing_raw_data"
    result = runner.invoke(
        extract_raw_commit_messages,
        ["--input-folder", str(missing_folder), "--output-file", "dummy.csv"],
    )
    assert result.exit_code == 0

    # Check for the presence of key parts of the error message
    assert "Input folder" in result.output
    assert "does not exist." in result.output


def test_no_json_files(runner, setup_test_environment):
    """Test when no JSON files are present in the input folder."""
    input_folder = setup_test_environment["input_folder"]
    result = runner.invoke(
        extract_raw_commit_messages,
        ["--input-folder", str(input_folder), "--output-file", "dummy.csv"],
    )
    assert result.exit_code == 0

    # Extract raw content from Rich output (remove decorations)
    raw_output = " ".join(result.output.splitlines())

    # Assert essential phrases in the raw output
    assert "No JSON files found" in raw_output



def test_valid_json_files(runner, setup_test_environment):
    """Test with valid JSON files containing commit messages."""
    input_folder = setup_test_environment["input_folder"]
    output_file = setup_test_environment["output_file"]

    # Create dummy JSON files with commit messages
    commits_data = [{"commit": {"message": "Initial commit"}}, {"commit": {"message": "Add new feature"}}]
    json_file_1 = input_folder / "repo1.json"
    json_file_2 = input_folder / "repo2.json"

    with open(json_file_1, "w") as file:
        json.dump(commits_data, file)

    with open(json_file_2, "w") as file:
        json.dump(commits_data, file)

    result = runner.invoke(
        extract_raw_commit_messages,
        ["--input-folder", str(input_folder), "--output-file", str(output_file)],
    )
    assert result.exit_code == 0

    # Check for essential phrases in the output (ignoring formatting)
    assert "Found 2 JSON files to process." in result.output
    assert "raw_commit_messages.csv" in result.output  # Simplified check for output file

    # Verify the CSV file was created and contains the correct content
    assert output_file.exists()

    # Validate CSV contents
    with open(output_file, "r", encoding="utf-8") as csvfile:
        lines = csvfile.readlines()
        assert len(lines) == 5  # 1 header + 4 rows
        assert '"1","Initial commit",""' in lines[1].strip()
        assert '"2","Add new feature",""' in lines[2].strip()



def test_invalid_json_file(runner, setup_test_environment):
    """Test with an invalid JSON file in the input folder."""
    input_folder = setup_test_environment["input_folder"]
    output_file = setup_test_environment["output_file"]

    # Create an invalid JSON file
    invalid_json_file = input_folder / "invalid.json"
    with open(invalid_json_file, "w") as file:
        file.write("{invalid_json: true,}")  # Invalid JSON

    result = runner.invoke(
        extract_raw_commit_messages,
        ["--input-folder", str(input_folder), "--output-file", str(output_file)],
    )
    assert result.exit_code == 0
    assert "Found 1 JSON files to process." in result.output
    assert "Error decoding JSON:" in result.output
    assert "No commit messages found in the JSON files." in result.output


def test_mixed_json_files(runner, setup_test_environment):
    """Test with a mix of valid and invalid JSON files."""
    input_folder = setup_test_environment["input_folder"]
    output_file = setup_test_environment["output_file"]

    # Create valid and invalid JSON files
    valid_commits_data = [{"commit": {"message": "Fix bug"}}, {"commit": {"message": "Update docs"}}]
    valid_json_file = input_folder / "valid.json"
    invalid_json_file = input_folder / "invalid.json"

    with open(valid_json_file, "w") as file:
        json.dump(valid_commits_data, file)

    with open(invalid_json_file, "w") as file:
        file.write("{invalid_json: true,}")  # Invalid JSON

    result = runner.invoke(
        extract_raw_commit_messages,
        ["--input-folder", str(input_folder), "--output-file", str(output_file)],
    )
    assert result.exit_code == 0

    # Check for key parts of the output
    assert "Found 2 JSON files to process." in result.output
    assert "Error decoding JSON:" in result.output

    # Verify the commit messages were saved
    assert os.path.exists(output_file)

    # Verify the CSV contents
    with open(output_file, "r", encoding="utf-8") as csvfile:
        lines = csvfile.readlines()
        assert len(lines) == 3  # 1 header + 2 valid rows
        assert '"1","Fix bug",""' in lines[1].strip()
        assert '"2","Update docs",""' in lines[2].strip()

