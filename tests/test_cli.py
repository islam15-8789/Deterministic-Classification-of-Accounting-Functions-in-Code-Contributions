import pytest
from click.testing import CliRunner
from cli import cli


@pytest.fixture
def runner():
    """Fixture to create a Click CLI runner."""
    return CliRunner()


def test_cli_help(runner):
    """Test that the CLI help command works."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "A CLI tool to reproduce the thesis result." in result.output


def test_fetch_commits_command_registered(runner):
    """Test that the `fetch-commits` command is registered."""
    result = runner.invoke(cli, ["fetch-commits", "--help"])
    assert result.exit_code == 0
    assert "Fetch commits from GitHub repositories" in result.output


def test_extract_raw_commit_messages_command_registered(runner):
    """Test that the `extract-raw-commit-messages` command is registered."""
    result = runner.invoke(cli, ["extract-raw-commit-messages", "--help"])
    assert result.exit_code == 0
    assert "Extract commit messages from JSON files" in result.output
