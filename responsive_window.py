from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cinderfilter_core import DeviceChoice
from cinderfilter_window import CinderWindow, DiagnosticItem, MeterPanel
from gpu_detector import GpuRuntimeStatus
from ui_components import (
    BORDER,
    GREEN,
    PURPLE,
    TEXT,
    DarkCombo,
    GlowCard,
    SegmentMeter,
    secondary_button_style,
)


class ReflowGrid(QWidget):
    """A grid that changes column count instead of overflowing its viewport."""

    def __init__(
        self,
        widgets: list[QWidget],
        breakpoints: tuple[tuple[int, int], ...],
        spacing: int = 12,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._widgets = widgets
        self._breakpoints = tuple(sorted(breakpoints, reverse=True))
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(spacing)
        self._grid.setVerticalSpacing(spacing)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        QTimer.singleShot(0, self._reflow)

    def _column_count(self) -> int:
        width = max(0, self.width())
        for minimum_width, columns in self._breakpoints:
            if width >= minimum_width:
                return columns
        return 1

    def _reflow(self) -> None:
        columns = self._column_count()
        if columns == self._columns:
            return
        self._columns = columns

        while self._grid.count():
            self._grid.takeAt(0)

        for column in range(max(columns, 1)):
            self._grid.setColumnStretch(column, 1)

        for index, widget in enumerate(self._widgets):
            row, column = divmod(index, columns)
            self._grid.addWidget(widget, row, column)

        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._reflow()


class ResponsiveCinderWindow(CinderWindow):
    """Unified CinderFilter window with DPI-safe, width-responsive overview."""

    SETTINGS_VERSION = 4

    @staticmethod
    def _migrate_settings(raw: dict) -> dict:
        original_version = 0
        if isinstance(raw, dict):
            try:
                original_version = int(raw.get("version", 0))
            except (TypeError, ValueError):
                original_version = 0

        values = CinderWindow._migrate_settings(raw)

        # Earlier patch-era recovery saved CPU after CUDA installation failures.
        # This user explicitly asked for a CUDA-focused main reducer, so migrate
        # that legacy value once. A later explicit CPU choice is preserved.
        if original_version < ResponsiveCinderWindow.SETTINGS_VERSION:
            if str(values.get("noise_backend", "CUDA")) == "CPU":
                values["noise_backend"] = "CUDA"
            if str(values.get("noise_cuda_preset", "Balanced")) == "Low Latency":
                values["noise_cuda_preset"] = "Balanced"

        values["version"] = ResponsiveCinderWindow.SETTINGS_VERSION
        return values

    def __init__(self) -> None:
        self._responsive_sections: list[ReflowGrid] = []
        super().__init__()
        self.setMinimumSize(980, 680)

        for combo in (
            self.input_combo,
            self.output_combo,
            self.device_input_combo,
            self.device_output_combo,
        ):
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )

        QTimer.singleShot(0, self._fit_window_to_screen)

    def _fit_window_to_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry().adjusted(8, 8, -8, -8)
        width = min(self.width(), available.width())
        height = min(self.height(), available.height())
        width = max(self.minimumWidth(), width)
        height = max(self.minimumHeight(), height)

        x = min(max(self.x(), available.left()), available.right() - width + 1)
        y = min(max(self.y(), available.top()), available.bottom() - height + 1)
        self.setGeometry(x, y, width, height)

    def _page_scroll(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { background:transparent; border:none; }"
            "QScrollArea > QWidget > QWidget { background:transparent; }"
            "QScrollBar:vertical { background:#090D14; width:9px; border-radius:4px; }"
            "QScrollBar::handle:vertical { background:#313A4B; border-radius:4px; min-height:30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }"
        )

        page = QWidget()
        page.setStyleSheet("background:transparent;")
        page.setMinimumWidth(0)
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 8, 8)
        layout.setSpacing(12)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        scroll.setWidget(page)
        return scroll, layout

    def _build_overview(self) -> QScrollArea:
        scroll, layout = self._page_scroll()

        routing = GlowCard("Routing", PURPLE)
        self.input_combo = DarkCombo()
        self.output_combo = DarkCombo()
        self.input_route_meter = SegmentMeter(GREEN)
        self.output_route_meter = SegmentMeter(GREEN)
        self.input_route_db = QLabel("-60.0 dB")
        self.output_route_db = QLabel("-60.0 dB")
        for label in (self.input_route_db, self.output_route_db):
            label.setStyleSheet(
                f"color:{PURPLE}; border:none; font-size:11px; font-weight:800;"
            )

        input_block = self._route_block(
            "Input Device", self.input_combo, self.input_route_meter, self.input_route_db
        )
        output_block = self._route_block(
            "Processed Output",
            self.output_combo,
            self.output_route_meter,
            self.output_route_db,
        )
        flow = QLabel("━━━  ✦  ━━━")
        flow.setAlignment(Qt.AlignCenter)
        flow.setStyleSheet(
            f"color:{PURPLE}; border:none; font-size:20px; font-weight:900;"
        )
        flow.setMaximumHeight(44)

        route_reflow = ReflowGrid(
            [input_block, flow, output_block],
            ((1050, 3), (0, 1)),
            spacing=10,
        )
        self._responsive_sections.append(route_reflow)
        routing.body.addWidget(route_reflow)
        layout.addWidget(routing)

        controls = ReflowGrid(
            [self._noise_card(), self._voice_card(), self._pitch_card()],
            ((1600, 3), (900, 2), (0, 1)),
        )
        self._responsive_sections.append(controls)
        layout.addWidget(controls)

        self.input_panel = MeterPanel("Input Level", GREEN, -60, 0)
        self.output_panel = MeterPanel("Output Level", GREEN, -60, 0)
        self.voice_panel = MeterPanel("Voice Match", PURPLE, 0, 100)
        self.reduction_panel = MeterPanel("Noise Reduction", GREEN, 0, 60)
        meters = ReflowGrid(
            [
                self.input_panel,
                self.output_panel,
                self.voice_panel,
                self.reduction_panel,
            ],
            ((1500, 4), (650, 2), (0, 1)),
        )
        self._responsive_sections.append(meters)
        layout.addWidget(meters)

        diagnostics = GlowCard("Diagnostics", PURPLE)
        self.diag_gpu = DiagnosticItem("GPU Detected")
        self.diag_backend = DiagnosticItem("Backend")
        self.diag_rtf = DiagnosticItem("RTF")
        self.diag_queue = DiagnosticItem("Queue Depth")
        self.diag_rate = DiagnosticItem("Sample Rate", "48 kHz")
        self.diag_block = DiagnosticItem("CUDA Preset")
        open_diag = QPushButton("Open Diagnostics")
        open_diag.setStyleSheet(secondary_button_style())
        open_diag.clicked.connect(lambda: self._show_page(5))

        diagnostic_grid = ReflowGrid(
            [
                self.diag_gpu,
                self.diag_backend,
                self.diag_rtf,
                self.diag_queue,
                self.diag_rate,
                self.diag_block,
                open_diag,
            ],
            ((1450, 7), (850, 4), (0, 2)),
        )
        self._responsive_sections.append(diagnostic_grid)
        diagnostics.body.addWidget(diagnostic_grid)
        layout.addWidget(diagnostics)

        self.status_banner = QLabel("Stopped")
        self.status_banner.setWordWrap(True)
        self.status_banner.setStyleSheet(
            f"color:{TEXT}; background:rgba(12,17,26,225); "
            f"border:1px solid {BORDER}; border-radius:11px; padding:10px;"
        )
        layout.addWidget(self.status_banner)
        layout.addStretch(1)
        return scroll

    @staticmethod
    def _clean_device_name(choice: DeviceChoice) -> str:
        raw = str(choice.name or "Unknown device")
        text = raw.replace("\r", " ").replace("\n", " ")
        text = text.replace("%0", " ").replace("%1", " ")
        text = re.sub(
            r"@?System32\\drivers\\[^,;]+,[^;]+;?",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip(" ,·-")

        groups = [value.strip() for value in re.findall(r"\(([^()]*)\)", text)]
        product = groups[-1] if groups else ""
        folded = text.casefold()
        if product and "hands-free" in folded:
            text = f"{product} — Hands-Free"
        elif product and ("stereo" in folded or "headphones" in folded):
            text = f"{product} — Stereo"

        # Keep the current value compact. The complete endpoint identity remains
        # available in the tooltip and Device page.
        if len(text) > 78:
            text = text[:75].rstrip() + "…"
        return text or "Unknown device"

    @staticmethod
    def _fill_combo(combo: DarkCombo, choices: list[DeviceChoice], preferred: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        selected = -1
        for index, choice in enumerate(choices):
            clean = ResponsiveCinderWindow._clean_device_name(choice)
            display = f"{clean}  ·  {choice.host_api}"
            combo.addItem(display, choice)
            combo.setItemData(index, choice.label, Qt.ToolTipRole)
            if choice.key.casefold() == preferred.casefold():
                selected = index
        combo.setCurrentIndex(selected if selected >= 0 else (0 if choices else -1))
        combo.blockSignals(False)

    def _apply_gpu_status(self, status: GpuRuntimeStatus) -> None:
        super()._apply_gpu_status(status)
        if status.torch_cuda_available and self.noise_backend_combo.currentText() == "CUDA":
            self.noise_status.setText(
                "CUDA runtime ready — preload the denoiser or press Start Filter"
            )

    def save_settings(self) -> None:
        config = self.current_config()
        geometry = self.geometry()
        values = {
            "version": self.SETTINGS_VERSION,
            "input_device_key": self._device_key(self.input_combo),
            "output_device_key": self._device_key(self.output_combo),
            **config.__dict__,
            "geometry": [
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            ],
        }
        try:
            self.settings_store.save(values)
            self.saved = values
        except OSError as exc:
            self._append_log(f"Settings save failed: {exc}")
