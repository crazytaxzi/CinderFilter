from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import main
import stable_output_app


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # diagnostic only
        return f"unavailable: {exc}"


def _effect_name(widget) -> str:
    effect = widget.graphicsEffect()
    return "None" if effect is None else type(effect).__name__


def main_diag() -> int:
    app = QApplication.instance() or QApplication([])
    window = stable_output_app.CinderFilterWindow()
    window.show()
    attempts = {"count": 0}

    def report() -> None:
        window._sync_main_geometry()
        app.processEvents()

        print()
        print("CinderFilter layout diagnostic")
        print("GIT HEAD:", _git_head())
        print("MAIN FILE:", Path(main.__file__).resolve())
        print("WINDOW FILE:", Path(stable_output_app.__file__).resolve())
        print("WINDOW CLASS:", type(window).__name__)
        print("ENGINE CLASS:", type(window.engine).__name__)
        print("OUTPUT LEVELING:", window.engine.output_leveling_preset)
        print("STACK GRAPHICS EFFECT:", _effect_name(window.stack))
        print("PAGE GRAPHICS EFFECT:", _effect_name(window.stack.currentWidget()))
        print("GPU DETECTION COMPLETE:", window._gpu_status is not None)
        for key, value in window.layout_measurements().items():
            print(f"{key}: {value}")

        screenshot = ROOT / "layout-diagnostic.png"
        saved = window.grab().save(str(screenshot))
        print("SCREENSHOT:", screenshot if saved else "capture failed")
        print()

        window.close()
        app.quit()

    def wait_for_gpu() -> None:
        attempts["count"] += 1
        if window._gpu_status is None and attempts["count"] < 24:
            QTimer.singleShot(250, wait_for_gpu)
            return
        report()

    QTimer.singleShot(500, wait_for_gpu)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main_diag())
