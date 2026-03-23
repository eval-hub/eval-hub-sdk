"""Bootstrap entry point for the evalhub CLI.

Provides a user-friendly error when click is not installed.
"""


def main() -> None:
    try:
        from evalhub.cli.main import main as click_main
    except ModuleNotFoundError as exc:
        if exc.name == "click":
            raise SystemExit(
                "Error: the evalhub CLI requires the 'cli' extra.\n"
                "Install it with: pip install 'eval-hub-sdk[cli]'"
            ) from None
        raise
    click_main()
