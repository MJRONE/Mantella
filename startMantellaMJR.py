"""Standalone launcher for MantellaMJR (new version).

This script lets you start Mantella as a plain Python application without
going through the packaged/exe entry point. It mirrors the older launcher
(``OldMantella/startMantellaMJR.py``) so the same workflow keeps working.

Behavior:
  * Sets the working directory to this file's directory (the project root)
    so all relative paths used by ``main.py`` resolve correctly.
  * Ensures the project root is on ``sys.path`` so ``from main import ...``
    works regardless of where the script is invoked from.
  * Invokes ``main.main()`` and, on failure, logs a traceback and keeps the
    console window open so double-click launches are still readable.
"""

import os
import sys
import logging
import traceback


def _set_working_directory_to_project_root() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    return base_dir


def main():
    _set_working_directory_to_project_root()
    try:
        from main import main as mantella_main
        mantella_main()
    except Exception as e:
        logging.error("".join(traceback.format_exception(e)))
        print("An error occurred while starting MantellaMJR. Details were logged.")
        try:
            input("Press Enter to exit.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
