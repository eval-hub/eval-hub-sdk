"""Shell completion script generation for the evalhub CLI."""

from __future__ import annotations

import click


_POWERSHELL_SCRIPT = """\
Register-ArgumentCompleter -Native -CommandName evalhub -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $env:_EVALHUB_COMPLETE = 'powershell_complete'
    $env:COMP_WORDS = $commandAst.ToString()
    $env:COMP_CWORD = $cursorPosition
    evalhub | ForEach-Object {
        $type, $value, $help = $_ -split '\\t', 3
        [System.Management.Automation.CompletionResult]::new(
            $value, $value,
            $(if ($type -eq 'dir') { 'ProviderContainer' }
              elseif ($type -eq 'file') { 'ProviderItem' }
              else { 'ParameterValue' }),
            $(if ($help) { $help } else { $value })
        )
    }
    Remove-Item Env:_EVALHUB_COMPLETE
    Remove-Item Env:COMP_WORDS
    Remove-Item Env:COMP_CWORD
}
"""


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

      # powershell (add to $PROFILE)
      evalhub completion powershell >> $PROFILE
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
    click.echo(_POWERSHELL_SCRIPT)
