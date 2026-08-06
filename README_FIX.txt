CinderFilter UI title-bar startup fix

1. Close the UI.
2. Copy main.py over C:\Projects\CinderFilter\main.py
3. Run: python main.py

Fixes:
- Passes the real QMainWindow into MainView and TitleBar during construction.
- Stops shadowing QWidget.window() with a self.window attribute.
- Repairs minimize, maximize/restore, close, drag, and double-click maximize behavior.
