from __future__ import annotations

import sys
from pathlib import Path


def _add_repo_root_to_syspath() -> None:
    """
    Ensure the repository root is on sys.path so we can import sibling folders.
    """
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> int:
    """
    Delegate execution to HW2's CLI.
    """
    _add_repo_root_to_syspath()

    try:
        from hw2.app import main as hw2_main  # type: ignore
    except Exception as e:
        print("ERROR: Failed to import hw2.app. Make sure hw2/app.py exists.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    return int(hw2_main())


if __name__ == "__main__":
    raise SystemExit(main())
