import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from device_catalog import CinderFilterWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CinderFilter")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = CinderFilterWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
