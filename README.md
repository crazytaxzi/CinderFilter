# CinderFilter UI Overhaul

This is a frameless PySide6 UI shell for CinderFilter using the neon **Cinder stream theme**.

## Included
- Frameless window with custom title bar
- Dark-mode-safe custom combo boxes
- Smooth fade transitions between pages
- Sidebar navigation
- Neon purple / neon green theme
- Overview page matching the CinderFilter control deck concept
- Placeholder pages for Devices / Voice / Tuning / Advanced / Diagnostics / About

## Why PySide6
Tkinter can be forced into a dark theme, but native dropdowns and window chrome are still ugly and inconsistent. If the requirement is **no compromise**, PySide6 is the right move for the visual shell.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Integration notes
Wire your existing backend to:
- input/output device combo boxes
- start filter button
- meters and status pills
- voice lock / pitch lock controls
- diagnostics footer

The current file is a polished shell ready for backend integration.
