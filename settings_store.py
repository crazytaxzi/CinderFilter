from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def default_settings_path() -> Path:
    """Return a per-user settings path that survives repo/ZIP updates."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        root = Path(local_app_data)
    else:
        root = Path.home() / ".config"
    return root / "CinderFilter" / "settings.json"


class SettingsStore:
    """Small atomic JSON settings store."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_settings_path()
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
            except (OSError, json.JSONDecodeError, UnicodeError):
                return {}
        return raw if isinstance(raw, dict) else {}

    def save(self, values: dict[str, Any]) -> None:
        payload = json.dumps(values, indent=2, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.path)
