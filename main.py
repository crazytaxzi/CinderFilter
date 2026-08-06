import sys
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from cinderfilter_window import CinderWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CinderFilter")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = CinderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
