"""Shell completion script generation for the evalhub CLI."""

from __future__ import annotations

import os

import click
from click.shell_completion import (
    CompletionItem,
    ShellComplete,
    add_completion_class,
    split_arg_string,
)

_SOURCE_POWERSHELL = """\
Register-ArgumentCompleter -Native -CommandName %(prog_name)s -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $env:%(complete_var)s = 'powershell_complete'
    $env:_EVALHUB_WORDS = $commandAst.ToString()
    $env:_EVALHUB_WORD_TO_COMPLETE = $wordToComplete
    %(prog_name)s | ForEach-Object {
        $type, $value, $help = $_ -split ',', 3
        [System.Management.Automation.CompletionResult]::new(
            $value, $value,
            $(if ($type -eq 'dir') { 'ProviderContainer' }
              elseif ($type -eq 'file') { 'ProviderItem' }
              else { 'ParameterValue' }),
            $(if ($help) { $help } else { ' ' })
        )
    }
    Remove-Item Env:%(complete_var)s
    Remove-Item Env:_EVALHUB_WORDS
    Remove-Item Env:_EVALHUB_WORD_TO_COMPLETE
}
"""


class PowerShellComplete(ShellComplete):
    """Shell completion for PowerShell."""

    name = "powershell"
    source_template = _SOURCE_POWERSHELL

    def get_completion_args(self) -> tuple[list[str], str]:
        cwords = split_arg_string(os.environ["_EVALHUB_WORDS"])
        incomplete = os.environ.get("_EVALHUB_WORD_TO_COMPLETE", "")
        # All args after the program name, excluding the incomplete word
        args = cwords[1:]
        if args and args[-1] == incomplete:
            args = args[:-1]
        return args, incomplete

    def format_completion(self, item: CompletionItem) -> str:
        return f"{item.type},{item.value},{item.help or ''}"


add_completion_class(PowerShellComplete)


def _get_completion_script(shell: str) -> str:
    """Generate a completion script for the given shell using Click internals."""
    from click.shell_completion import get_completion_class

    from evalhub.cli.main import main as cli

    cls = get_completion_class(shell)
    if cls is None:
        raise click.ClickException(f"Unsupported shell for Click completion: {shell}")
    comp = cls(cli, {}, "evalhub", "_EVALHUB_COMPLETE")
    return comp.source()


@click.group("completion")
def completion() -> None:
    """Generate shell completion scripts.

    \b
    Supported shells: bash, zsh, fish, powershell.

    \b
    Quick setup:
      eval "$(evalhub completion bash)"     # bash (current session)
      eval "$(evalhub completion zsh)"      # zsh  (current session)
      evalhub completion fish | source      # fish (current session)

    \b
    Persistent setup:
      # bash
      evalhub completion bash > ~/.local/share/bash-completion/completions/evalhub

      # zsh  (add the directory to fpath before compinit in ~/.zshrc)
      evalhub completion zsh > ~/.zfunc/_evalhub

      # fish
      evalhub completion fish > ~/.config/fish/completions/evalhub.fish

      # powershell (source from $PROFILE)
      evalhub completion powershell > ~/.config/evalhub/completion.ps1
      Add-Content $PROFILE '. ~/.config/evalhub/completion.ps1'
    """


@completion.command("bash")
def completion_bash() -> None:
    """Generate bash completion script."""
    click.echo(_get_completion_script("bash"))


@completion.command("zsh")
def completion_zsh() -> None:
    """Generate zsh completion script."""
    click.echo(_get_completion_script("zsh"))


@completion.command("fish")
def completion_fish() -> None:
    """Generate fish completion script."""
    click.echo(_get_completion_script("fish"))


@completion.command("powershell")
def completion_powershell() -> None:
    """Generate PowerShell completion script."""
    click.echo(_get_completion_script("powershell"))
