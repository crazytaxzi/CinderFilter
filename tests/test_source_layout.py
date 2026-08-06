from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_one_application_entrypoint() -> None:
    assert (ROOT / "main.py").exists()
    assert list(ROOT.glob("cinderfilter_voice_lock*.py")) == []
    assert list(ROOT.glob("START_CINDERFILTER*.bat")) == [ROOT / "START_CINDERFILTER.bat"]


def test_runtime_has_no_tkinter_dependency() -> None:
    for name in ("main.py", "cinderfilter_window.py", "cinderfilter_core.py", "voice_lock.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(value == "tkinter" or value.startswith("tkinter.") for value in imports)


def test_ui_is_connected_not_mocked() -> None:
    source = (ROOT / "cinderfilter_window.py").read_text(encoding="utf-8")
    assert "CinderFilterEngine(" in source
    assert "enumerate_devices()" in source
    assert "EngineMetrics" in source
    for fake in ("Shure SM7B", "Focusrite Scarlett", "NVIDIA RTX 4080"):
        assert fake not in source
