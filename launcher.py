import os
import sys
import webbrowser
import threading
import time
import uvicorn
from server import app
import subprocess


def open_browser():
    """Open the browser after a short delay to ensure server is running."""
    time.sleep(2)
    try:
        webbrowser.open("http://127.0.0.1:8123")
    except Exception:
        # Fallback: use macOS open command
        try:
            subprocess.run(["open", "http://127.0.0.1:8123"], check=False)
        except Exception:
            pass


def main():
    try:
        # Ensure we're running in the correct directory for relative paths
        # When running as an executable, sys._MEIPASS is the temp folder
        if getattr(sys, 'frozen', False):
            if hasattr(sys, '_MEIPASS'):
                # Change to the temp directory where resources are extracted
                os.chdir(sys._MEIPASS)
            else:
                # Fallback for onedir mode
                os.chdir(os.path.dirname(sys.executable))

        # Start browser in a separate thread
        threading.Thread(target=open_browser, daemon=True).start()

        # Run server
        uvicorn.run(app, host="127.0.0.1", port=8123, log_level="error")
    except Exception as e:
        # Log errors to a file when running as executable
        if getattr(sys, 'frozen', False):
            error_log = os.path.expanduser("~/Documents/Reader3_error.log")
            with open(error_log, "a") as f:
                f.write(f"Error: {e}\n")
        raise


if __name__ == "__main__":
    main()
