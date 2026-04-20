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
        # Import after setting working directory and sys.path so relative imports/files work
        from main import main as mantella_main
        mantella_main()
    except Exception as e:
        # Mirror main.py behavior and keep window open on error when double-clicked
        logging.error("".join(traceback.format_exception(e)))
        print("An error occurred while starting MantellaMJR. Details were logged.")
        try:
            input("Press Enter to exit.")
        except Exception:
            # In case stdin is not available
            pass


if __name__ == "__main__":
    main()


