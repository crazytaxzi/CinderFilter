from __future__ import annotations

import re

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
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


class ViewportLockedScrollArea(QScrollArea):
    """A vertical scroll area whose page can never exceed the viewport width."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.verticalScrollBar().rangeChanged.connect(self._schedule_width_sync)

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt API
        widget.setMinimumWidth(0)
        widget.setMaximumWidth(16_777_215)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.MinimumExpanding)
        super().setWidget(widget)
        self._schedule_width_sync()

    def _schedule_width_sync(self, *_args) -> None:
        QTimer.singleShot(0, self._sync_page_width)

    def _sync_page_width(self) -> None:
        page = self.widget()
        if page is None:
            return
        target = max(1, self.viewport().width())
        page.setMinimumWidth(target)
        page.setMaximumWidth(target)
        if page.width() != target:
            page.resize(target, max(page.height(), page.sizeHint().height()))
        self.horizontalScrollBar().setValue(0)
        page.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._schedule_width_sync()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._schedule_width_sync()


class ReflowGrid(QWidget):
    """A width-flexible grid that changes columns instead of overflowing."""

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
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)

        for widget in self._widgets:
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16_777_215)
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            for combo in widget.findChildren(QComboBox):
                combo.setMinimumWidth(0)
                combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            for button in widget.findChildren(QPushButton):
                button.setMinimumWidth(0)

        QTimer.singleShot(0, self._reflow)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def _column_count(self) -> int:
        width = max(0, self.contentsRect().width())
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

        for column in range(8):
            self._grid.setColumnStretch(column, 0)
        for column in range(max(columns, 1)):
            self._grid.setColumnStretch(column, 1)

        for index, widget in enumerate(self._widgets):
            row, column = divmod(index, columns)
            self._grid.addWidget(widget, row, column)

        self._grid.invalidate()
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._reflow()


class RouteReflow(QWidget):
    """Two flexible route cards with a compact center signal indicator."""

    def __init__(
        self,
        input_block: QWidget,
        flow: QWidget,
        output_block: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._input = input_block
        self._flow = flow
        self._output = output_block
        self._wide: bool | None = None
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(8)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)

        for block in (self._input, self._output):
            block.setMinimumWidth(0)
            block.setMaximumWidth(16_777_215)
            block.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._flow.setMinimumWidth(70)
        self._flow.setMaximumWidth(150)
        self._flow.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        QTimer.singleShot(0, self._reflow)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def _reflow(self) -> None:
        wide = self.contentsRect().width() >= 780
        if wide == self._wide:
            return
        self._wide = wide
        while self._grid.count():
            self._grid.takeAt(0)
        for column in range(3):
            self._grid.setColumnStretch(column, 0)

        if wide:
            self._grid.addWidget(self._input, 0, 0)
            self._grid.addWidget(self._flow, 0, 1)
            self._grid.addWidget(self._output, 0, 2)
            self._grid.setColumnStretch(0, 1)
            self._grid.setColumnStretch(2, 1)
        else:
            self._grid.addWidget(self._input, 0, 0)
            self._grid.addWidget(self._flow, 1, 0, alignment=Qt.AlignCenter)
            self._grid.addWidget(self._output, 2, 0)
            self._grid.setColumnStretch(0, 1)

        self._grid.invalidate()
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._reflow()


class ResponsiveCinderWindow(CinderWindow):
    """Unified CinderFilter window with viewport-locked responsive pages."""

    SETTINGS_VERSION = 5

    @staticmethod
    def _migrate_settings(raw: dict) -> dict:
        original_version = 0
        if isinstance(raw, dict):
            try:
                original_version = int(raw.get("version", 0))
            except (TypeError, ValueError):
                original_version = 0

        values = CinderWindow._migrate_settings(raw)
        if original_version < 4:
            if str(values.get("noise_backend", "CUDA")) == "CPU":
                values["noise_backend"] = "CUDA"
            if str(values.get("noise_cuda_preset", "Balanced")) == "Low Latency":
                values["noise_cuda_preset"] = "Balanced"

        values["version"] = ResponsiveCinderWindow.SETTINGS_VERSION
        return values

    def __init__(self) -> None:
        self._responsive_sections: list[QWidget] = []
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
            combo.setMinimumContentsLength(10)
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

        for scroll in self.findChildren(ViewportLockedScrollArea):
            scroll._schedule_width_sync()

    def _page_scroll(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = ViewportLockedScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
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
        page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 8, 8)
        layout.setSpacing(12)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        scroll.setWidget(page)
        return scroll, layout

    def _build_overview(self) -> QScrollArea:
        scroll, layout = self._page_scroll()

        routing = GlowCard("Routing", PURPLE)
        routing.setMinimumWidth(0)
        routing.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
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
        flow = QLabel("━━  ✦  ━━")
        flow.setAlignment(Qt.AlignCenter)
        flow.setStyleSheet(
            f"color:{PURPLE}; border:none; font-size:18px; font-weight:900;"
        )
        flow.setMaximumHeight(40)

        route_reflow = RouteReflow(input_block, flow, output_block)
        self._responsive_sections.append(route_reflow)
        routing.body.addWidget(route_reflow)
        layout.addWidget(routing)

        controls = ReflowGrid(
            [self._noise_card(), self._voice_card(), self._pitch_card()],
            ((1380, 3), (760, 2), (0, 1)),
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
            ((1320, 4), (620, 2), (0, 1)),
        )
        self._responsive_sections.append(meters)
        layout.addWidget(meters)

        diagnostics = GlowCard("Diagnostics", PURPLE)
        diagnostics.setMinimumWidth(0)
        diagnostics.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
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
            ((1320, 7), (720, 4), (0, 2)),
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

        if len(text) > 64:
            text = text[:61].rstrip() + "…"
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