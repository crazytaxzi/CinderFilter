from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cinderfilter_core import (
    CUDA_PRESETS,
    SEPARATOR_PRESETS,
    STRENGTHS,
    CinderFilterEngine,
    DeviceChoice,
    EngineConfig,
    EngineMetrics,
)
from gpu_detector import GpuRuntimeStatus, detect_gpu_runtime
from settings_store import SettingsStore
from ui_components import (
    BG,
    BORDER,
    DANGER,
    GREEN,
    MUTED,
    PANEL,
    PURPLE,
    PURPLE_SOFT,
    TEXT,
    BackdropFrame,
    DarkCombo,
    FadeStack,
    GlowCard,
    NavButton,
    RingGauge,
    SegmentMeter,
    SegmentedControl,
    TitleBar,
    ToggleSwitch,
    secondary_button_style,
    slider_style,
)

try:
    import psutil
except ImportError:
    psutil = None

APP_DIR = Path(__file__).resolve().parent


class AppSignals(QObject):
    status = Signal(str)
    metrics = Signal(object)
    engine_event = Signal(str, object)
    job_done = Signal(str, object)
    job_error = Signal(str, str)


class MeterPanel(GlowCard):
    def __init__(self, title: str, accent: str, minimum: float, maximum: float, parent=None) -> None:
        super().__init__(title, accent, parent)
        self.meter = SegmentMeter(accent, minimum, maximum)
        self.value = QLabel("--")
        self.value.setStyleSheet(f"color:{accent}; border:none; font-size:13px; font-weight:800;")
        self.body.addWidget(self.meter)
        self.body.addWidget(self.value)


class DiagnosticItem(QFrame):
    def __init__(self, label: str, value: str = "--", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background:transparent; border:none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        name = QLabel(label)
        name.setStyleSheet(f"color:{MUTED}; border:none; font-size:10px;")
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color:{TEXT}; border:none; font-size:12px; font-weight:800;")
        layout.addWidget(name)
        layout.addWidget(self.value_label)


class CinderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CinderFilter")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1680, 960)
        self.setMinimumSize(1180, 760)

        self.signals = AppSignals()
        self.settings_store = SettingsStore()
        self.saved = self._migrate_settings(self.settings_store.load())
        self.engine = CinderFilterEngine(
            self.signals.status.emit,
            self.signals.metrics.emit,
            self.signals.engine_event.emit,
            app_dir=APP_DIR,
        )
        self.inputs: list[DeviceChoice] = []
        self.outputs: list[DeviceChoice] = []
        self._log_lines: list[str] = []
        self._jobs: set[str] = set()
        self._gpu_status: GpuRuntimeStatus | None = None

        self._build_shell()
        self._connect_signals()
        self._restore_ui()
        self.refresh_devices()
        self._refresh_profiles()
        self.detect_gpu()

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(450)
        self.save_timer.timeout.connect(self.save_settings)
        self.cpu_timer = QTimer(self)
        self.cpu_timer.timeout.connect(self._update_cpu)
        self.cpu_timer.start(1200)

    @staticmethod
    def _migrate_settings(raw: dict) -> dict:
        values = dict(raw) if isinstance(raw, dict) else {}
        for current, old in {
            "strength": "ai_strength",
            "target_extraction_enabled": "v2_enabled",
            "separator_preset": "v2_preset",
            "separator_device": "v2_device",
        }.items():
            if current not in values and old in values:
                values[current] = values[old]
        reduction = values.get("voice_reduction_db", values.get("voice_reduction", 24))
        if isinstance(reduction, str):
            try:
                reduction = float(reduction.split()[0])
            except (ValueError, IndexError):
                reduction = 24.0
        values["voice_reduction_db"] = reduction
        geometry = values.get("geometry", values.get("window_geometry"))
        if isinstance(geometry, str):
            match = re.fullmatch(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)", geometry.strip())
            if match:
                width, height, x, y = map(int, match.groups())
                values["geometry"] = [x, y, width, height]
        values["version"] = 3
        return values

    def _build_shell(self) -> None:
        outer = QWidget()
        outer.setStyleSheet("background:transparent;")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(9, 9, 9, 9)
        shell = QFrame()
        shell.setObjectName("Shell")
        shell.setStyleSheet(f"QFrame#Shell {{ background:{BG}; border:1px solid #303848; border-radius:22px; }}")
        shadow = QGraphicsDropShadowEffect(shell)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 7)
        shell.setGraphicsEffect(shadow)
        outer_layout.addWidget(shell)
        self.setCentralWidget(outer)

        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        self.title_bar = TitleBar(self)
        shell_layout.addWidget(self.title_bar)

        content = BackdropFrame()
        content.setObjectName("Content")
        content.setStyleSheet("QFrame#Content { background:transparent; border:none; }")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(10, 0, 10, 10)
        content_layout.setSpacing(12)
        shell_layout.addWidget(content, 1)

        sidebar = QFrame()
        sidebar.setFixedWidth(176)
        sidebar.setStyleSheet("background:rgba(8,11,18,225); border:none; border-radius:16px;")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(8, 12, 8, 12)
        side.setSpacing(5)
        self.nav_buttons: list[NavButton] = []
        for index, (label, icon) in enumerate((
            ("Overview", "⌂"), ("Devices", "▣"), ("Voice", "♬"),
            ("Tuning", "☷"), ("Advanced", "⚙"), ("Diagnostics", "⌁"), ("About", "ⓘ"),
        )):
            button = NavButton(label, icon)
            button.clicked.connect(lambda _checked=False, page=index: self._show_page(page))
            self.nav_buttons.append(button)
            side.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        side.addStretch(1)
        brand = QLabel("✦  CINDER STREAM\n     CONTROL DECK")
        brand.setStyleSheet(f"color:{PURPLE}; border:none; padding:8px; font-size:10px; font-weight:900;")
        side.addWidget(brand)
        version = QLabel("v3 unified")
        version.setStyleSheet(f"color:{MUTED}; border:none; padding-left:8px; font-size:10px;")
        side.addWidget(version)
        content_layout.addWidget(sidebar)

        self.stack = FadeStack()
        self.stack.setStyleSheet("background:transparent; border:none;")
        content_layout.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_overview())
        self.stack.addWidget(self._build_devices_page())
        self.stack.addWidget(self._build_voice_page())
        self.stack.addWidget(self._build_tuning_page())
        self.stack.addWidget(self._build_advanced_page())
        self.stack.addWidget(self._build_diagnostics_page())
        self.stack.addWidget(self._build_about_page())
        self.title_bar.start_button.clicked.connect(self.start_or_stop)

    def _page_scroll(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background:transparent; border:none; }"
            "QScrollBar:vertical { background:#090D14; width:9px; border-radius:4px; }"
            "QScrollBar::handle:vertical { background:#313A4B; border-radius:4px; min-height:30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }"
        )
        page = QWidget()
        page.setStyleSheet("background:transparent;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 8, 8)
        layout.setSpacing(12)
        scroll.setWidget(page)
        return scroll, layout

    def _build_overview(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        routing = GlowCard("Routing", PURPLE)
        route_grid = QGridLayout()
        self.input_combo = DarkCombo()
        self.output_combo = DarkCombo()
        self.input_route_meter = SegmentMeter(GREEN)
        self.output_route_meter = SegmentMeter(GREEN)
        self.input_route_db = QLabel("-60.0 dB")
        self.output_route_db = QLabel("-60.0 dB")
        for label in (self.input_route_db, self.output_route_db):
            label.setStyleSheet(f"color:{PURPLE}; border:none; font-size:11px; font-weight:800;")
        route_grid.addWidget(self._route_block("Input Device", self.input_combo, self.input_route_meter, self.input_route_db), 0, 0)
        flow = QLabel("━━━  ✦  ━━━")
        flow.setAlignment(Qt.AlignCenter)
        flow.setStyleSheet(f"color:{PURPLE}; border:none; font-size:22px; font-weight:900;")
        route_grid.addWidget(flow, 0, 1)
        route_grid.addWidget(self._route_block("Processed Output", self.output_combo, self.output_route_meter, self.output_route_db), 0, 2)
        route_grid.setColumnStretch(0, 1)
        route_grid.setColumnStretch(2, 1)
        routing.body.addLayout(route_grid)
        layout.addWidget(routing)

        controls = QHBoxLayout()
        controls.setSpacing(12)
        controls.addWidget(self._noise_card(), 1)
        controls.addWidget(self._voice_card(), 1)
        controls.addWidget(self._pitch_card(), 1)
        layout.addLayout(controls)

        meters = QHBoxLayout()
        meters.setSpacing(12)
        self.input_panel = MeterPanel("Input Level", GREEN, -60, 0)
        self.output_panel = MeterPanel("Output Level", GREEN, -60, 0)
        self.voice_panel = MeterPanel("Voice Match", PURPLE, 0, 100)
        self.reduction_panel = MeterPanel("Noise Reduction", GREEN, 0, 60)
        for panel in (self.input_panel, self.output_panel, self.voice_panel, self.reduction_panel):
            meters.addWidget(panel, 1)
        layout.addLayout(meters)

        diagnostics = GlowCard("Diagnostics", PURPLE)
        row = QHBoxLayout()
        self.diag_gpu = DiagnosticItem("GPU Detected")
        self.diag_backend = DiagnosticItem("Backend")
        self.diag_rtf = DiagnosticItem("RTF")
        self.diag_queue = DiagnosticItem("Queue Depth")
        self.diag_rate = DiagnosticItem("Sample Rate", "48 kHz")
        self.diag_block = DiagnosticItem("CUDA Preset")
        for item in (self.diag_gpu, self.diag_backend, self.diag_rtf, self.diag_queue, self.diag_rate, self.diag_block):
            row.addWidget(item, 1)
        open_diag = QPushButton("Open Diagnostics")
        open_diag.setStyleSheet(secondary_button_style())
        open_diag.clicked.connect(lambda: self._show_page(5))
        row.addWidget(open_diag)
        diagnostics.body.addLayout(row)
        layout.addWidget(diagnostics)
        self.status_banner = QLabel("Stopped")
        self.status_banner.setWordWrap(True)
        self.status_banner.setStyleSheet(f"color:{TEXT}; background:rgba(12,17,26,225); border:1px solid {BORDER}; border-radius:11px; padding:10px;")
        layout.addWidget(self.status_banner)
        layout.addStretch(1)
        return scroll

    def _route_block(self, title: str, combo: DarkCombo, meter: SegmentMeter, db: QLabel) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"background:#0A0F17; border:1px solid {BORDER}; border-radius:12px;")
        box = QVBoxLayout(frame)
        name = QLabel(title)
        name.setStyleSheet(f"color:{TEXT}; border:none; font-size:12px; font-weight:700;")
        box.addWidget(name)
        box.addWidget(combo)
        line = QHBoxLayout()
        line.addWidget(meter, 1)
        line.addWidget(db)
        box.addLayout(line)
        return frame

    def _noise_card(self) -> GlowCard:
        card = GlowCard("Main Noise Reducer", GREEN)
        card.body.addWidget(self._small("Noise Engine"))
        self.noise_backend_combo = DarkCombo(("CUDA", "Auto", "CPU"))
        card.body.addWidget(self.noise_backend_combo)
        card.body.addWidget(self._small("CUDA Latency Preset"))
        self.noise_preset_combo = DarkCombo(tuple(CUDA_PRESETS))
        card.body.addWidget(self.noise_preset_combo)
        card.body.addWidget(self._small("AI Strength"))
        self.strength_combo = DarkCombo(tuple(STRENGTHS))
        card.body.addWidget(self.strength_combo)
        self.noise_status = QLabel("CUDA engine not loaded")
        self.noise_status.setWordWrap(True)
        self.noise_status.setStyleSheet(f"color:{GREEN}; background:#0B1710; border:1px solid #26442A; border-radius:10px; padding:9px;")
        card.body.addWidget(self.noise_status)
        preload = QPushButton("Preload CUDA Denoiser")
        preload.setStyleSheet(secondary_button_style(GREEN))
        preload.clicked.connect(self.preload_cuda)
        card.body.addWidget(preload)
        return card

    def _voice_card(self) -> GlowCard:
        card = GlowCard("Voice Lock", PURPLE)
        profile = QHBoxLayout()
        profile.addWidget(self._small("Voice Profile"))
        profile.addStretch(1)
        self.voice_profile_label = QLabel("MISSING")
        self.voice_profile_label.setStyleSheet(f"color:{DANGER}; border:none; font-weight:900;")
        profile.addWidget(self.voice_profile_label)
        card.body.addLayout(profile)
        card.body.addLayout(self._toggle_row("Enable Voice Lock", self._make_toggle("voice_lock_toggle", True)))
        card.body.addLayout(self._toggle_row("Target-Speaker Extraction", self._make_toggle("target_toggle", False)))
        card.body.addWidget(self._small("Other-Voice Reduction"))
        reduction = QHBoxLayout()
        self.voice_reduction_slider = QSlider(Qt.Horizontal)
        self.voice_reduction_slider.setRange(0, 48)
        self.voice_reduction_slider.setStyleSheet(slider_style(PURPLE))
        self.voice_reduction_value = QLabel("24 dB")
        self.voice_reduction_value.setStyleSheet(f"color:{TEXT}; border:none;")
        reduction.addWidget(self.voice_reduction_slider, 1)
        reduction.addWidget(self.voice_reduction_value)
        card.body.addLayout(reduction)
        card.body.addWidget(self._small("Strictness"))
        self.voice_strictness = SegmentedControl(("Conservative", "Balanced", "Aggressive"), "Balanced")
        card.body.addWidget(self.voice_strictness)
        confidence = QHBoxLayout()
        self.confidence_ring = RingGauge(PURPLE)
        self.voice_status = QLabel("Enroll your voice to activate identity filtering")
        self.voice_status.setWordWrap(True)
        self.voice_status.setStyleSheet(f"color:{MUTED}; border:none;")
        confidence.addWidget(self.confidence_ring)
        confidence.addWidget(self.voice_status, 1)
        card.body.addLayout(confidence)
        return card

    def _pitch_card(self) -> GlowCard:
        card = GlowCard("Pitch Lock", GREEN)
        card.body.addLayout(self._toggle_row("Pitch Distinction", self._make_toggle("pitch_toggle", True)))
        card.body.addLayout(self._toggle_row("Fail-Closed on Uncertainty", self._make_toggle("fail_closed_toggle", True)))
        card.body.addWidget(self._small("Pitch Margin"))
        line = QHBoxLayout()
        self.pitch_margin_slider = QSlider(Qt.Horizontal)
        self.pitch_margin_slider.setRange(-15, 50)
        self.pitch_margin_slider.setStyleSheet(slider_style(GREEN))
        self.pitch_margin_value = QLabel("0 Hz")
        self.pitch_margin_value.setStyleSheet(f"color:{TEXT}; border:none;")
        line.addWidget(self.pitch_margin_slider, 1)
        line.addWidget(self.pitch_margin_value)
        card.body.addLayout(line)
        self.pitch_profile_label = QLabel("No pitch profile calibrated")
        self.pitch_profile_label.setWordWrap(True)
        self.pitch_profile_label.setStyleSheet(f"color:{MUTED}; background:#0B1510; border:1px solid #203A25; border-radius:10px; padding:9px;")
        card.body.addWidget(self.pitch_profile_label)
        calibrate = QPushButton("Calibrate My Pitch")
        calibrate.setStyleSheet(secondary_button_style(GREEN))
        calibrate.clicked.connect(self.calibrate_pitch)
        card.body.addWidget(calibrate)
        return card

    def _build_devices_page(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        card = GlowCard("Audio Devices", PURPLE)
        grid = QGridLayout()
        grid.addWidget(self._small("Physical Input"), 0, 0)
        grid.addWidget(self._small("Processed Output / Virtual Cable"), 0, 1)
        self.device_input_combo = DarkCombo()
        self.device_output_combo = DarkCombo()
        grid.addWidget(self.device_input_combo, 1, 0)
        grid.addWidget(self.device_output_combo, 1, 1)
        card.body.addLayout(grid)
        refresh = QPushButton("Refresh Windows Audio Endpoints")
        refresh.setStyleSheet(secondary_button_style())
        refresh.clicked.connect(self.refresh_devices)
        card.body.addWidget(refresh)
        layout.addWidget(card)
        table_card = GlowCard("Detected Endpoints", GREEN)
        self.device_table = QTableWidget(0, 5)
        self.device_table.setHorizontalHeaderLabels(("Type", "Name", "Host API", "Channels", "Default Rate"))
        self.device_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.device_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.device_table.horizontalHeader().setStretchLastSection(True)
        self.device_table.setStyleSheet(self._table_style())
        table_card.body.addWidget(self.device_table)
        layout.addWidget(table_card)
        warning = QLabel("CinderFilter owns the physical microphone. Route only its processed output into VoiceMeeter; do not add the raw microphone beside it.")
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#FFD1B9; background:#25150F; border:1px solid #5B3120; border-radius:11px; padding:11px;")
        layout.addWidget(warning)
        layout.addStretch(1)
        return scroll

    def _build_voice_page(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        voice = GlowCard("Voice Identity", PURPLE)
        self.voice_detail_label = QLabel()
        self.voice_detail_label.setWordWrap(True)
        self.voice_detail_label.setStyleSheet(f"color:{TEXT}; border:none; font-size:13px;")
        voice.body.addWidget(self.voice_detail_label)
        actions = QHBoxLayout()
        for text, callback, color in (
            ("Enroll / Replace Voice Profile", self.enroll_voice, PURPLE),
            ("Delete Voice Profile", self.delete_voice, DANGER),
            ("Preload Target Extraction", self.preload_separator, GREEN),
        ):
            button = QPushButton(text)
            button.setStyleSheet(secondary_button_style(color))
            button.clicked.connect(callback)
            actions.addWidget(button)
        voice.body.addLayout(actions)
        layout.addWidget(voice)
        pitch = GlowCard("Pitch Identity", GREEN)
        self.pitch_detail_label = QLabel()
        self.pitch_detail_label.setWordWrap(True)
        self.pitch_detail_label.setStyleSheet(f"color:{TEXT}; border:none; font-size:13px;")
        pitch.body.addWidget(self.pitch_detail_label)
        row = QHBoxLayout()
        for text, callback, color in (("Calibrate Pitch", self.calibrate_pitch, GREEN), ("Delete Pitch Profile", self.delete_pitch, DANGER)):
            button = QPushButton(text)
            button.setStyleSheet(secondary_button_style(color))
            button.clicked.connect(callback)
            row.addWidget(button)
        pitch.body.addLayout(row)
        layout.addWidget(pitch)
        privacy = GlowCard("Privacy Behavior", PURPLE)
        text = QLabel("Fail-closed target extraction requires voiceprint, pitch range, and source confidence. Uncertainty becomes silence. Explicit CUDA mode also mutes during worker restart instead of creating the CPU denoiser.")
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{MUTED}; border:none; font-size:13px;")
        privacy.body.addWidget(text)
        layout.addWidget(privacy)
        layout.addStretch(1)
        return scroll

    def _build_tuning_page(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        card = GlowCard("Noise and Identity Tuning", PURPLE)
        grid = QGridLayout()
        grid.addWidget(self._small("Noise Strength"), 0, 0)
        self.tuning_strength = DarkCombo(tuple(STRENGTHS))
        grid.addWidget(self.tuning_strength, 1, 0)
        grid.addWidget(self._small("Voice Strictness"), 0, 1)
        self.tuning_strictness = DarkCombo(("Conservative", "Balanced", "Aggressive"))
        grid.addWidget(self.tuning_strictness, 1, 1)
        grid.addWidget(self._small("Target Extraction Preset"), 2, 0)
        self.separator_preset_combo = DarkCombo(tuple(SEPARATOR_PRESETS))
        grid.addWidget(self.separator_preset_combo, 3, 0)
        grid.addWidget(self._small("Target Extraction Device"), 2, 1)
        self.separator_device_combo = DarkCombo(("Auto", "CUDA", "CPU"))
        grid.addWidget(self.separator_device_combo, 3, 1)
        card.body.addLayout(grid)
        layout.addWidget(card)
        bypass = GlowCard("Routing Test", DANGER)
        bypass.body.addLayout(self._toggle_row("Bypass All Filtering (raw microphone)", self._make_toggle("bypass_toggle", False)))
        note = QLabel("Bypass is intentionally blocked while fail-closed target extraction is enabled, preventing an accidental raw-mic leak.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#FFD0D2; border:none;")
        bypass.body.addWidget(note)
        layout.addWidget(bypass)
        layout.addStretch(1)
        return scroll

    def _build_advanced_page(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        runtime = GlowCard("Runtime Management", GREEN)
        self.runtime_detail = QLabel("Detecting GPU and PyTorch runtime…")
        self.runtime_detail.setWordWrap(True)
        self.runtime_detail.setStyleSheet(f"color:{TEXT}; border:none;")
        runtime.body.addWidget(self.runtime_detail)
        row = QHBoxLayout()
        cuda = QPushButton("Install / Repair Main CUDA Runtime")
        noise = QPushButton("Install / Repair CUDA Noise Engine")
        cuda.setStyleSheet(secondary_button_style(GREEN))
        noise.setStyleSheet(secondary_button_style(PURPLE))
        cuda.clicked.connect(lambda: self.launch_tool("install_torch_cuda.ps1"))
        noise.clicked.connect(lambda: self.launch_tool("install_cuda_noise_engine.ps1"))
        row.addWidget(cuda)
        row.addWidget(noise)
        runtime.body.addLayout(row)
        layout.addWidget(runtime)
        architecture = GlowCard("Unified Architecture", PURPLE)
        text = QLabel("One PySide application owns all routes and settings. Model services are modules, not alternate applications. Runtime installers live under tools/ and close the app before replacing DLLs.")
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{MUTED}; border:none;")
        architecture.body.addWidget(text)
        layout.addWidget(architecture)
        layout.addStretch(1)
        return scroll

    def _build_diagnostics_page(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        card = GlowCard("Live Runtime", PURPLE)
        self.diagnostic_summary = QLabel("Stopped")
        self.diagnostic_summary.setWordWrap(True)
        self.diagnostic_summary.setStyleSheet(f"color:{TEXT}; border:none; font-family:Consolas; font-size:12px;")
        card.body.addWidget(self.diagnostic_summary)
        buttons = QHBoxLayout()
        detect = QPushButton("Detect GPU Again")
        logs = QPushButton("Open CUDA Log Folder")
        detect.setStyleSheet(secondary_button_style(GREEN))
        logs.setStyleSheet(secondary_button_style(PURPLE))
        detect.clicked.connect(self.detect_gpu)
        logs.clicked.connect(self.open_log_folder)
        buttons.addWidget(detect)
        buttons.addWidget(logs)
        card.body.addLayout(buttons)
        layout.addWidget(card)
        log_card = GlowCard("Application Log", GREEN)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(f"background:#070B11; color:#C7D0DD; border:1px solid {BORDER}; border-radius:10px; padding:8px; font-family:Consolas;")
        log_card.body.addWidget(self.log_view)
        layout.addWidget(log_card)
        return scroll

    def _build_about_page(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        card = GlowCard("CinderFilter v3 Unified", PURPLE)
        title = QLabel("One app. One launcher. One settings model. One audio controller.")
        title.setStyleSheet(f"color:{TEXT}; border:none; font-size:18px; font-weight:900;")
        card.body.addWidget(title)
        body = QLabel("The patch-era Tkinter variants, alternate launchers, top-level repair batches, duplicate CUDA workers, and hardcoded UI mock were removed from the active Git tree. Git history preserves the audit base for forensic reference.")
        body.setWordWrap(True)
        body.setStyleSheet(f"color:{MUTED}; border:none; font-size:13px;")
        card.body.addWidget(body)
        layout.addWidget(card)
        layout.addStretch(1)
        return scroll

    def _make_toggle(self, name: str, value: bool) -> ToggleSwitch:
        toggle = ToggleSwitch(value)
        setattr(self, name, toggle)
        return toggle

    def _toggle_row(self, text: str, toggle: ToggleSwitch) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(text)
        label.setStyleSheet(f"color:{TEXT}; border:none; font-size:12px;")
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(toggle)
        return row

    @staticmethod
    def _small(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color:{MUTED}; border:none; font-size:11px; font-weight:700;")
        return label

    def _connect_signals(self) -> None:
        self.signals.status.connect(self._on_status)
        self.signals.metrics.connect(self._on_metrics)
        self.signals.engine_event.connect(self._on_engine_event)
        self.signals.job_done.connect(self._on_job_done)
        self.signals.job_error.connect(self._on_job_error)
        for combo in (self.noise_backend_combo, self.noise_preset_combo, self.strength_combo, self.separator_preset_combo, self.separator_device_combo):
            combo.currentTextChanged.connect(self._settings_changed)
        for toggle in (self.voice_lock_toggle, self.target_toggle, self.pitch_toggle, self.fail_closed_toggle, self.bypass_toggle):
            toggle.toggled.connect(self._settings_changed)
        self.voice_reduction_slider.valueChanged.connect(self._voice_reduction_changed)
        self.pitch_margin_slider.valueChanged.connect(self._pitch_margin_changed)
        self.voice_strictness.changed.connect(self._strictness_changed)
        self.input_combo.currentIndexChanged.connect(lambda index: self._mirror_combo(self.input_combo, self.device_input_combo, index))
        self.device_input_combo.currentIndexChanged.connect(lambda index: self._mirror_combo(self.device_input_combo, self.input_combo, index))
        self.output_combo.currentIndexChanged.connect(lambda index: self._mirror_combo(self.output_combo, self.device_output_combo, index))
        self.device_output_combo.currentIndexChanged.connect(lambda index: self._mirror_combo(self.device_output_combo, self.output_combo, index))
        self.strength_combo.currentTextChanged.connect(lambda text: self._set_combo(self.tuning_strength, text))
        self.tuning_strength.currentTextChanged.connect(lambda text: self._set_combo(self.strength_combo, text))
        self.tuning_strictness.currentTextChanged.connect(lambda text: self.voice_strictness.setValue(text))

    def _restore_ui(self) -> None:
        self.noise_backend_combo.setCurrentText(str(self.saved.get("noise_backend", "CUDA")))
        self.noise_preset_combo.setCurrentText(str(self.saved.get("noise_cuda_preset", "Balanced")))
        self.strength_combo.setCurrentText(str(self.saved.get("strength", "Balanced")))
        self.tuning_strength.setCurrentText(self.strength_combo.currentText())
        self.separator_preset_combo.setCurrentText(str(self.saved.get("separator_preset", "Fast")))
        self.separator_device_combo.setCurrentText(str(self.saved.get("separator_device", "Auto")))
        strictness = str(self.saved.get("voice_strictness", "Balanced"))
        self.voice_strictness.setValue(strictness, emit=False)
        self.tuning_strictness.setCurrentText(strictness)
        self.voice_reduction_slider.setValue(int(float(self.saved.get("voice_reduction_db", 24))))
        self.pitch_margin_slider.setValue(int(float(self.saved.get("pitch_margin_hz", 0))))
        self.voice_lock_toggle.setChecked(bool(self.saved.get("voice_lock_enabled", True)))
        self.target_toggle.setChecked(bool(self.saved.get("target_extraction_enabled", False)))
        self.pitch_toggle.setChecked(bool(self.saved.get("pitch_enabled", True)))
        self.fail_closed_toggle.setChecked(bool(self.saved.get("pitch_fail_closed", True)))
        self.bypass_toggle.setChecked(bool(self.saved.get("bypass", False)))
        geometry = self.saved.get("geometry")
        if isinstance(geometry, list) and len(geometry) == 4:
            try:
                self.setGeometry(*map(int, geometry))
            except Exception:
                pass

    def refresh_devices(self) -> None:
        try:
            self.inputs, self.outputs = self.engine.enumerate_devices()
        except BaseException as exc:
            self._show_error("Audio device scan failed", str(exc))
            return
        self._fill_combo(self.input_combo, self.inputs, str(self.saved.get("input_device_key", "")))
        self._fill_combo(self.device_input_combo, self.inputs, self._device_key(self.input_combo))
        self._fill_combo(self.output_combo, self.outputs, str(self.saved.get("output_device_key", "")))
        self._fill_combo(self.device_output_combo, self.outputs, self._device_key(self.output_combo))
        rows = [("Input", item) for item in self.inputs] + [("Output", item) for item in self.outputs]
        self.device_table.setRowCount(len(rows))
        for row, (kind, device) in enumerate(rows):
            values = (kind, device.name, device.host_api, str(device.input_channels if kind == "Input" else device.output_channels), f"{device.default_sample_rate:.0f} Hz")
            for column, value in enumerate(values):
                self.device_table.setItem(row, column, QTableWidgetItem(value))
        self.device_table.resizeColumnsToContents()
        self._on_status(f"Detected {len(self.inputs)} input and {len(self.outputs)} output endpoints")

    @staticmethod
    def _fill_combo(combo: DarkCombo, choices: list[DeviceChoice], preferred: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        selected = -1
        for index, choice in enumerate(choices):
            combo.addItem(choice.label, choice)
            if choice.key.casefold() == preferred.casefold():
                selected = index
        combo.setCurrentIndex(selected if selected >= 0 else (0 if choices else -1))
        combo.blockSignals(False)

    def current_config(self) -> EngineConfig:
        return EngineConfig(
            strength=self.strength_combo.currentText(), bypass=self.bypass_toggle.isChecked(),
            voice_lock_enabled=self.voice_lock_toggle.isChecked(), voice_reduction_db=float(self.voice_reduction_slider.value()),
            voice_strictness=self.voice_strictness.value(), target_extraction_enabled=self.target_toggle.isChecked(),
            separator_preset=self.separator_preset_combo.currentText(), separator_device=self.separator_device_combo.currentText(),
            pitch_enabled=self.pitch_toggle.isChecked(), pitch_fail_closed=self.fail_closed_toggle.isChecked(),
            pitch_margin_hz=float(self.pitch_margin_slider.value()), noise_backend=self.noise_backend_combo.currentText(),
            noise_cuda_preset=self.noise_preset_combo.currentText(),
        ).normalized()

    def start_or_stop(self) -> None:
        if self.engine.running:
            self.title_bar.start_button.setEnabled(False)
            self.run_job("stop", self.engine.stop)
            return
        input_choice, output_choice = self.input_combo.currentData(), self.output_combo.currentData()
        if not isinstance(input_choice, DeviceChoice) or not isinstance(output_choice, DeviceChoice):
            self._show_error("Choose devices", "Select a physical input and processed output.")
            return
        config = self.current_config()
        self.save_settings()
        self.title_bar.start_button.setEnabled(False)
        self._on_status("Starting models and audio streams…")
        self.run_job("start", lambda: self.engine.start(input_choice.index, output_choice.index, config))

    def preload_cuda(self) -> None:
        self.run_job("preload_cuda", lambda: self.engine.preload_cuda(self.strength_combo.currentText(), self.noise_preset_combo.currentText()))

    def preload_separator(self) -> None:
        self.run_job("preload_separator", lambda: self.engine.preload_separator(self.current_config()))

    def enroll_voice(self) -> None:
        choice = self.input_combo.currentData()
        if not isinstance(choice, DeviceChoice):
            self._show_error("Choose an input", "Select the physical microphone first.")
            return
        self.run_job("enroll_voice", lambda: self.engine.record_voice_profile(choice.index))

    def calibrate_pitch(self) -> None:
        choice = self.input_combo.currentData()
        if not isinstance(choice, DeviceChoice):
            self._show_error("Choose an input", "Select the physical microphone first.")
            return
        self.run_job("calibrate_pitch", lambda: self.engine.record_pitch_profile(choice.index))

    def delete_voice(self) -> None:
        if QMessageBox.question(self, "Delete voice profile", "Delete the enrolled voice embedding?") == QMessageBox.StandardButton.Yes:
            self.engine.delete_voice_profile()
            self._refresh_profiles()

    def delete_pitch(self) -> None:
        if QMessageBox.question(self, "Delete pitch profile", "Delete the calibrated pitch range?") == QMessageBox.StandardButton.Yes:
            self.engine.delete_pitch_profile()
            self._refresh_profiles()

    def launch_tool(self, name: str) -> None:
        script = APP_DIR / "tools" / name
        if not script.exists():
            self._show_error("Tool missing", str(script))
            return
        if self.engine.running:
            self.engine.stop()
        try:
            subprocess.Popen(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-AppRoot", str(APP_DIR), "-ParentPid", str(os.getpid())],
                cwd=str(APP_DIR), creationflags=0x00000010 if os.name == "nt" else 0,
            )
            self._on_status(f"Closing while {name} updates runtime files…")
            QTimer.singleShot(250, self.close)
        except BaseException as exc:
            self._show_error("Could not launch runtime tool", str(exc))

    def detect_gpu(self) -> None:
        self.run_job("gpu_detect", detect_gpu_runtime)

    def run_job(self, name: str, function) -> None:
        if name in self._jobs:
            return
        self._jobs.add(name)
        def worker() -> None:
            try:
                self.signals.job_done.emit(name, function())
            except BaseException as exc:
                self.signals.job_error.emit(name, f"{type(exc).__name__}: {exc}")
        threading.Thread(target=worker, name=f"CinderFilter-{name}", daemon=True).start()

    def _on_job_done(self, name: str, result: object) -> None:
        self._jobs.discard(name)
        if name in {"start", "stop"}:
            self.title_bar.start_button.setEnabled(True)
            self.title_bar.setRunning(self.engine.running)
        if name == "preload_cuda":
            device = result.get("device", "CUDA") if isinstance(result, dict) else "CUDA"
            self._on_status(f"CUDA denoiser preloaded on {device}")
        elif name == "preload_separator":
            self._on_status("Target separator preloaded and ready")
        elif name == "calibrate_pitch":
            self._refresh_profiles()
        elif name == "gpu_detect" and isinstance(result, GpuRuntimeStatus):
            self._apply_gpu_status(result)

    def _on_job_error(self, name: str, message: str) -> None:
        self._jobs.discard(name)
        self.title_bar.start_button.setEnabled(True)
        self.title_bar.setRunning(self.engine.running)
        self._append_log(f"ERROR [{name}] {message}")
        self._show_error(name.replace("_", " ").title(), message)
        self._on_status(message)

    def _on_status(self, text: str) -> None:
        self.status_banner.setText(text)
        self._append_log(text)
        if "CUDA" in text:
            self.noise_status.setText(text)
        if "Voice" in text or "voice" in text:
            self.voice_status.setText(text)

    def _on_metrics(self, metrics: EngineMetrics) -> None:
        for meter in (self.input_route_meter, self.input_panel.meter): meter.setValue(metrics.input_db)
        for meter in (self.output_route_meter, self.output_panel.meter): meter.setValue(metrics.output_db)
        self.input_route_db.setText(f"{metrics.input_db:.1f} dB")
        self.output_route_db.setText(f"{metrics.output_db:.1f} dB")
        self.input_panel.value.setText(f"{metrics.input_db:.1f} dB")
        self.output_panel.value.setText(f"{metrics.output_db:.1f} dB")
        similarity = metrics.voice_similarity
        confidence = 0.0 if similarity is None else max(0.0, min(1.0, (similarity + 0.05) / 0.50))
        self.confidence_ring.setValue(confidence)
        self.voice_panel.meter.setValue(confidence * 100)
        self.voice_panel.value.setText("--" if similarity is None else f"{similarity:.3f}")
        self.reduction_panel.meter.setValue(metrics.noise_reduction_db)
        self.reduction_panel.value.setText(f"{metrics.noise_reduction_db:.1f} dB")
        self.diag_backend.value_label.setText(metrics.active_backend)
        self.diag_rtf.value_label.setText(f"{metrics.cuda_rtf:.2f}x" if metrics.cuda_rtf else "--")
        self.diag_queue.value_label.setText(str(metrics.cuda_queue))
        self.diag_block.value_label.setText(self.noise_preset_combo.currentText())
        self.diagnostic_summary.setText(
            f"Backend: {metrics.active_backend}    CUDA RTF: {metrics.cuda_rtf:.2f}    CUDA queue: {metrics.cuda_queue}\n"
            f"Separator RTF: {metrics.separator_rtf:.2f}    Separator queue: {metrics.separator_queue}    Voice match: {metrics.voice_similarity}\n"
            f"Dropped input: {metrics.dropped_input}    Output underruns: {metrics.output_underruns}"
        )

    def _on_engine_event(self, kind: str, payload: object) -> None:
        if kind == "running":
            self.title_bar.setRunning(bool(payload))
            self.title_bar.start_button.setEnabled(True)
        elif kind in {"voice_profile", "pitch_profile"}:
            self._refresh_profiles()

    def _apply_gpu_status(self, status: GpuRuntimeStatus) -> None:
        self._gpu_status = status
        adapter = next((item for item in status.adapters if item.vendor == "NVIDIA"), None)
        name = adapter.name if adapter else (status.adapters[0].name if status.adapters else "No GPU")
        self.title_bar.gpu_pill.setStatus(name, "CUDA Ready" if status.torch_cuda_available else "CPU Runtime")
        self.diag_gpu.value_label.setText(name)
        self.runtime_detail.setText(status.summary() + "\n" + status.detail())

    def _update_cpu(self) -> None:
        if psutil is None:
            self.title_bar.cpu_pill.setStatus("System", "--")
            return
        try:
            name = os.environ.get("PROCESSOR_IDENTIFIER", "Windows CPU")
            self.title_bar.cpu_pill.setStatus(name[:24] + ("…" if len(name) > 24 else ""), f"{psutil.cpu_percent():.0f}%")
        except Exception:
            self.title_bar.cpu_pill.setStatus("System", "--")

    def _refresh_profiles(self) -> None:
        if self.engine.voice_profile_ready:
            self.voice_profile_label.setText("ACTIVE")
            self.voice_profile_label.setStyleSheet(f"color:{GREEN}; border:none; font-weight:900;")
            self.voice_detail_label.setText("Voice profile: READY. The embedding is local and target extraction can be preloaded.")
        else:
            self.voice_profile_label.setText("MISSING")
            self.voice_profile_label.setStyleSheet(f"color:{DANGER}; border:none; font-weight:900;")
            self.voice_detail_label.setText("No voice profile. Select the physical microphone and record a 12-second profile.")
        profile = self.engine.pitch_guard.profile
        if profile is None:
            text = "No pitch profile calibrated"
            detail = "No pitch profile. Pitch distinction cannot fail closed until calibration is complete."
        else:
            text = f"Median {profile.median_hz:.0f} Hz · range {profile.low_hz:.0f}–{profile.high_hz:.0f} Hz · cutoff {profile.upper_limit_hz:.0f} Hz"
            detail = "Pitch profile: " + text
        self.pitch_profile_label.setText(text)
        self.pitch_detail_label.setText(detail)

    def _voice_reduction_changed(self, value: int) -> None:
        self.voice_reduction_value.setText(f"{value} dB")
        self._settings_changed()

    def _pitch_margin_changed(self, value: int) -> None:
        self.pitch_margin_value.setText(f"{'+' if value > 0 else ''}{value} Hz")
        self._settings_changed()

    def _strictness_changed(self, value: str) -> None:
        self._set_combo(self.tuning_strictness, value)
        self._settings_changed()

    def _settings_changed(self, *_args) -> None:
        if hasattr(self, "save_timer"):
            self.save_timer.start()
        if self.engine.running:
            self.status_banner.setText("Settings changed — Stop and Start to apply routing or model changes")
        self.engine.set_bypass(self.bypass_toggle.isChecked())

    def save_settings(self) -> None:
        config = self.current_config()
        geometry = self.geometry()
        values = {"version": 3, "input_device_key": self._device_key(self.input_combo), "output_device_key": self._device_key(self.output_combo), **config.__dict__, "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()]}
        try:
            self.settings_store.save(values)
            self.saved = values
        except OSError as exc:
            self._append_log(f"Settings save failed: {exc}")

    @staticmethod
    def _device_key(combo: DarkCombo) -> str:
        choice = combo.currentData()
        return choice.key if isinstance(choice, DeviceChoice) else ""

    def _mirror_combo(self, source: DarkCombo, target: DarkCombo, index: int) -> None:
        if target.currentIndex() == index:
            return
        target.blockSignals(True); target.setCurrentIndex(index); target.blockSignals(False)
        self._settings_changed()

    @staticmethod
    def _set_combo(combo: DarkCombo, text: str) -> None:
        if combo.currentText() != text:
            combo.blockSignals(True); combo.setCurrentText(text); combo.blockSignals(False)

    def _show_page(self, index: int) -> None:
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)
        self.stack.fadeTo(index)

    def _append_log(self, text: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        self._log_lines = (self._log_lines + [line])[-300:]
        self.log_view.setPlainText("\n".join(self._log_lines))
        cursor = self.log_view.textCursor(); cursor.movePosition(QTextCursor.End); self.log_view.setTextCursor(cursor)

    def open_log_folder(self) -> None:
        path = self.engine.cuda_noise.log_path.parent
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:
            self._show_error("Could not open log folder", str(exc))

    def _show_error(self, title: str, message: str) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Critical)
        box.setText(message)
        box.setStyleSheet(f"QMessageBox {{ background:{PANEL}; color:{TEXT}; }} QMessageBox QLabel {{ color:{TEXT}; min-width:360px; }} QMessageBox QPushButton {{ color:{TEXT}; background:#101722; border:1px solid {PURPLE_SOFT}; border-radius:9px; padding:8px 16px; min-width:88px; }} QMessageBox QPushButton:hover {{ background:#1B1028; border-color:{PURPLE}; }}")
        box.exec()

    @staticmethod
    def _table_style() -> str:
        return f"QTableWidget {{ background:#080C13; color:{TEXT}; border:1px solid {BORDER}; border-radius:10px; gridline-color:#1C2330; }} QHeaderView::section {{ background:#111824; color:{MUTED}; border:none; border-bottom:1px solid {BORDER}; padding:7px; }} QTableWidget::item {{ padding:6px; }} QTableWidget::item:selected {{ background:#2D1341; }}"

    def closeEvent(self, event) -> None:
        self.save_settings()
        self.engine.stop()
        event.accept()
