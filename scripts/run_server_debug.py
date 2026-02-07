#!/usr/bin/env python3
"""Simple script to run eval-hub server with local config for debugging."""

import os
import sys
from pathlib import Path


def main() -> None:
    # Get the config directory path
    config_dir = Path(__file__).parent.parent / "tests" / "e2e"

    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    # Change to config directory
    print(f"Changing directory to: {config_dir}")
    os.chdir(config_dir)
    print(f"Current working directory: {os.getcwd()}")
    print()

    # List config files for debugging
    print("Config files found:")
    for item in sorted(config_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(config_dir)
            print(f"  {rel}")
    print()

    # Import and run the server
    print("Starting eval-hub server...")
    print("=" * 60)

    try:
        from evalhub_server.main import main as server_main

        server_main(["--local"])
    except ImportError as e:
        print(f"Error: Could not import evalhub_server: {e}", file=sys.stderr)
        print("Make sure the evalhub-server package is installed.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user (Ctrl+C)")
        sys.exit(0)


if __name__ == "__main__":
    main()
