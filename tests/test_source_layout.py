from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_single_launcher_and_window() -> None:
    assert list(ROOT.glob("START_CINDERFILTER*.bat")) == [ROOT / "START_CINDERFILTER.bat"]
    assert (ROOT / "cinderfilter_app.py").exists()
    assert (ROOT / "device_catalog.py").exists()
    assert not (ROOT / "responsive_window.py").exists()
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from device_catalog import CinderFilterWindow" in main
    assert "window = CinderFilterWindow()" in main


def test_runtime_has_no_tkinter_dependency() -> None:
    for name in (
        "main.py",
        "device_catalog.py",
        "cinderfilter_app.py",
        "cinderfilter_window.py",
        "layout_components.py",
        "cinderfilter_core.py",
        "voice_lock.py",
    ):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        modules.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(item == "tkinter" or item.startswith("tkinter.") for item in modules)


def test_top_left_layout_contract() -> None:
    app = (ROOT / "cinderfilter_app.py").read_text(encoding="utf-8")
    layout = (ROOT / "layout_components.py").read_text(encoding="utf-8")
    assert "self.setWidgetResizable(False)" in layout
    assert "self.setAlignment(Qt.AlignLeft | Qt.AlignTop)" in layout
    assert "layout.setAlignment(Qt.AlignTop)" in app
    assert "layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)" not in app
    assert "self.stack.setMinimumWidth(available)" not in app
    assert "self.stack.setMaximumWidth(available)" not in app
    assert "routing.setObjectName(\"RoutingCard\")" in app
    assert "def layout_measurements" in app


def test_device_catalog_filters_and_deduplicates() -> None:
    catalog = (ROOT / "device_catalog.py").read_text(encoding="utf-8")
    assert 'INPUT_API_FILTERS = ("All", "MME", "WASAPI", "Kernel")' in catalog
    assert "def _host_family" in catalog
    assert "def _dedupe_devices" in catalog
    assert "def _filtered_inputs" in catalog
    assert "input_api_filter" in catalog
    assert "duplicate endpoint aliases are hidden" in catalog
    assert "Qt.ToolTipRole" in catalog


def test_real_engine_is_connected() -> None:
    source = (ROOT / "cinderfilter_window.py").read_text(encoding="utf-8")
    assert "CinderFilterEngine(" in source
    assert "enumerate_devices()" in source
    assert "EngineMetrics" in source
