from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import cinderfilter_app
import main


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # diagnostic only
        return f"unavailable: {exc}"


def main_diag() -> int:
    app = QApplication.instance() or QApplication([])
    window = cinderfilter_app.CinderFilterAppWindow()
    window.show()

    def report() -> None:
        window._sync_main_geometry()
        app.processEvents()

        print()
        print("CinderFilter layout diagnostic")
        print("GIT HEAD:", _git_head())
        print("MAIN FILE:", Path(main.__file__).resolve())
        print("WINDOW FILE:", Path(cinderfilter_app.__file__).resolve())
        print("WINDOW CLASS:", type(window).__name__)
        for key, value in window.layout_measurements().items():
            print(f"{key}: {value}")

        screenshot = ROOT / "layout-diagnostic.png"
        saved = window.grab().save(str(screenshot))
        print("SCREENSHOT:", screenshot if saved else "capture failed")
        print()

        window.close()
        app.quit()

    QTimer.singleShot(1500, report)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main_diag())
