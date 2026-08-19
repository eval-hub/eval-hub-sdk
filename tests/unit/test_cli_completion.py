"""Unit tests for EvalHub CLI completion commands."""

from __future__ import annotations

import pytest
from click.shell_completion import CompletionItem
from click.testing import CliRunner
from evalhub.cli.completion import PowerShellComplete
from evalhub.cli.main import main

pytestmark = pytest.mark.unit


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestCompletionHelp:
    def test_completion_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "--help"])
        assert result.exit_code == 0
        assert "bash" in result.output
        assert "zsh" in result.output
        assert "fish" in result.output
        assert "powershell" in result.output

    def test_completion_no_args_shows_usage(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion"])
        assert "Usage" in result.output

    def test_completion_listed_in_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "completion" in result.output


class TestBashCompletion:
    def test_generates_script(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "bash"])
        assert result.exit_code == 0
        assert "_evalhub_completion" in result.output
        assert "COMP_WORDS" in result.output
        assert "_EVALHUB_COMPLETE" in result.output

    def test_contains_complete_function(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "bash"])
        assert result.exit_code == 0
        assert "complete -o nosort -F _evalhub_completion evalhub" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "bash", "--help"])
        assert result.exit_code == 0
        assert "bash" in result.output.lower()


class TestZshCompletion:
    def test_generates_script(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "zsh"])
        assert result.exit_code == 0
        assert "_evalhub_completion" in result.output
        assert "compdef" in result.output
        assert "_EVALHUB_COMPLETE" in result.output

    def test_contains_compdef_registration(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "zsh"])
        assert result.exit_code == 0
        assert "compdef _evalhub_completion evalhub" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "zsh", "--help"])
        assert result.exit_code == 0
        assert "zsh" in result.output.lower()


class TestFishCompletion:
    def test_generates_script(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "fish"])
        assert result.exit_code == 0
        assert "_evalhub_completion" in result.output
        assert "_EVALHUB_COMPLETE" in result.output

    def test_contains_complete_registration(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "fish"])
        assert result.exit_code == 0
        assert "complete --no-files --command evalhub" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "fish", "--help"])
        assert result.exit_code == 0
        assert "fish" in result.output.lower()


class TestPowerShellCompletion:
    def test_generates_script(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "powershell"])
        assert result.exit_code == 0
        assert "Register-ArgumentCompleter" in result.output
        assert "evalhub" in result.output

    def test_contains_completion_result(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "powershell"])
        assert result.exit_code == 0
        assert "CompletionResult" in result.output

    def test_sets_evalhub_complete_env(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "powershell"])
        assert result.exit_code == 0
        assert "_EVALHUB_COMPLETE" in result.output
        assert "powershell_complete" in result.output

    def test_cleans_up_env_vars(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "powershell"])
        assert result.exit_code == 0
        assert "Remove-Item Env:_EVALHUB_COMPLETE" in result.output
        assert "Remove-Item Env:_EVALHUB_WORDS" in result.output
        assert "Remove-Item Env:_EVALHUB_WORD_TO_COMPLETE" in result.output

    def test_uses_comma_separated_format(self, runner: CliRunner) -> None:
        """PowerShell script should parse comma-separated completions (type,value,help)."""
        result = runner.invoke(main, ["completion", "powershell"])
        assert result.exit_code == 0
        assert "-split ','" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["completion", "powershell", "--help"])
        assert result.exit_code == 0
        assert "powershell" in result.output.lower()


class TestCompletionScriptsAreNonEmpty:
    """Ensure each shell produces a meaningful (non-trivial) script."""

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish", "powershell"])
    def test_output_is_substantial(self, runner: CliRunner, shell: str) -> None:
        result = runner.invoke(main, ["completion", shell])
        assert result.exit_code == 0
        # All scripts should be at least a few hundred chars
        assert len(result.output) > 100

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish", "powershell"])
    def test_output_references_evalhub(self, runner: CliRunner, shell: str) -> None:
        result = runner.invoke(main, ["completion", shell])
        assert result.exit_code == 0
        assert "evalhub" in result.output


class TestPowerShellCompleteClass:
    """Test the PowerShellComplete ShellComplete subclass directly."""

    def test_get_completion_args_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_EVALHUB_WORDS", "evalhub completion ba")
        monkeypatch.setenv("_EVALHUB_WORD_TO_COMPLETE", "ba")
        comp = PowerShellComplete(main, {}, "evalhub", "_EVALHUB_COMPLETE")
        args, incomplete = comp.get_completion_args()
        assert args == ["completion"]
        assert incomplete == "ba"

    def test_get_completion_args_empty_incomplete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("_EVALHUB_WORDS", "evalhub completion")
        monkeypatch.setenv("_EVALHUB_WORD_TO_COMPLETE", "")
        comp = PowerShellComplete(main, {}, "evalhub", "_EVALHUB_COMPLETE")
        args, incomplete = comp.get_completion_args()
        assert args == ["completion"]
        assert incomplete == ""

    def test_format_completion_with_help(self) -> None:
        comp = PowerShellComplete(main, {}, "evalhub", "_EVALHUB_COMPLETE")
        item = CompletionItem("bash", help="Generate bash script")
        result = comp.format_completion(item)
        assert result == "plain,bash,Generate bash script"

    def test_format_completion_without_help(self) -> None:
        comp = PowerShellComplete(main, {}, "evalhub", "_EVALHUB_COMPLETE")
        item = CompletionItem("bash")
        result = comp.format_completion(item)
        assert result == "plain,bash,"

    def test_registered_with_click(self) -> None:
        from click.shell_completion import get_completion_class

        cls = get_completion_class("powershell")
        assert cls is PowerShellComplete
