from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
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
from layout_components import FillFadeStack, ReflowGrid, RouteReflow, ViewportLockedScrollArea
from ui_components import (
    BG,
    BORDER,
    GREEN,
    PURPLE,
    TEXT,
    BackdropFrame,
    DarkCombo,
    GlowCard,
    NavButton,
    SegmentMeter,
    TitleBar,
    secondary_button_style,
)


class ResponsiveCinderWindow(CinderWindow):
    """Unified CinderFilter window with an exact-width main application column."""

    SETTINGS_VERSION = 6

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
        QTimer.singleShot(0, self._sync_main_geometry)

    def _build_shell(self) -> None:
        """Build the only shell, with a stack that cannot inherit hidden-page width."""
        outer = QWidget()
        outer.setStyleSheet("background:transparent;")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(9, 9, 9, 9)

        shell = QFrame()
        shell.setObjectName("Shell")
        shell.setStyleSheet(
            f"QFrame#Shell {{ background:{BG}; border:1px solid #303848; border-radius:22px; }}"
        )
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
        content_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        shell_layout.addWidget(content, 1)
        self._content_frame = content
        self._content_layout = content_layout

        sidebar = QFrame()
        sidebar.setFixedWidth(176)
        sidebar.setStyleSheet(
            "background:rgba(8,11,18,225); border:none; border-radius:16px;"
        )
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(8, 12, 8, 12)
        side.setSpacing(5)
        self.sidebar = sidebar
        self.nav_buttons: list[NavButton] = []
        for index, (label, icon) in enumerate(
            (
                ("Overview", "⌂"),
                ("Devices", "▣"),
                ("Voice", "♬"),
                ("Tuning", "☷"),
                ("Advanced", "⚙"),
                ("Diagnostics", "⌁"),
                ("About", "ⓘ"),
            )
        ):
            button = NavButton(label, icon)
            button.clicked.connect(
                lambda _checked=False, page=index: self._show_page(page)
            )
            self.nav_buttons.append(button)
            side.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        side.addStretch(1)
        brand = QLabel("✦  CINDER STREAM\n     CONTROL DECK")
        brand.setStyleSheet(
            f"color:{PURPLE}; border:none; padding:8px; font-size:10px; font-weight:900;"
        )
        side.addWidget(brand)
        version = QLabel("v3 unified")
        version.setStyleSheet(
            "color:#939CAD; border:none; padding-left:8px; font-size:10px;"
        )
        side.addWidget(version)
        content_layout.addWidget(sidebar, 0)

        self.stack = FillFadeStack()
        self.stack.setStyleSheet("background:transparent; border:none;")
        self.stack.setMinimumSize(0, 0)
        content_layout.addWidget(self.stack, 1)
        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 1)

        self.stack.addWidget(self._build_overview())
        self.stack.addWidget(self._build_devices_page())
        self.stack.addWidget(self._build_voice_page())
        self.stack.addWidget(self._build_tuning_page())
        self.stack.addWidget(self._build_advanced_page())
        self.stack.addWidget(self._build_diagnostics_page())
        self.stack.addWidget(self._build_about_page())
        self.stack.setCurrentIndex(0)
        self.title_bar.start_button.clicked.connect(self.start_or_stop)

    def _available_stack_width(self) -> int:
        if not hasattr(self, "_content_frame"):
            return 0
        margins = self._content_layout.contentsMargins()
        return max(
            1,
            self._content_frame.contentsRect().width()
            - margins.left()
            - margins.right()
            - self.sidebar.width()
            - self._content_layout.spacing(),
        )

    def _sync_main_geometry(self) -> None:
        if not hasattr(self, "stack"):
            return
        available = self._available_stack_width()
        if available <= 1:
            return

        self.stack.setMinimumWidth(available)
        self.stack.setMaximumWidth(available)
        self.stack.resize(available, self.stack.height())
        self.stack.fit_current()

        current = self.stack.currentWidget()
        if current is not None:
            current.setMinimumWidth(available)
            current.setMaximumWidth(available)
            current.setGeometry(0, 0, available, self.stack.height())

        for scroll in self.stack.findChildren(ViewportLockedScrollArea):
            scroll.sync_page_width()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_main_geometry)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_main_geometry)

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
        self._sync_main_geometry()

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
        page.setMinimumSize(0, 0)
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
            [self.input_panel, self.output_panel, self.voice_panel, self.reduction_panel],
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
