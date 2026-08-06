from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cinderfilter_app import CinderFilterAppWindow as BaseCinderFilterAppWindow
from cinderfilter_core import DeviceChoice
from layout_components import ReflowGrid, RouteReflow, ViewportLockedScrollArea
from ui_components import (
    BORDER,
    GREEN,
    MUTED,
    PURPLE,
    TEXT,
    DarkCombo,
    GlowCard,
    SegmentMeter,
    SegmentedControl,
    secondary_button_style,
)


MAX_WIDGET = 16_777_215
INPUT_API_FILTERS = ("All", "MME", "WASAPI", "Kernel")


class CinderFilterWindow(BaseCinderFilterAppWindow):
    """The single app window with a cleaned, filterable audio-device catalog."""

    SETTINGS_VERSION = 8

    @staticmethod
    def _migrate_settings(raw: dict) -> dict:
        values = BaseCinderFilterAppWindow._migrate_settings(raw)
        selected = str(values.get("input_api_filter", "All"))
        values["input_api_filter"] = selected if selected in INPUT_API_FILTERS else "All"
        values["version"] = CinderFilterWindow.SETTINGS_VERSION
        return values

    def __init__(self) -> None:
        self._raw_input_count = 0
        self._raw_output_count = 0
        self._input_aliases_hidden = 0
        self._output_aliases_hidden = 0
        self._syncing_input_filter = False
        super().__init__()

    def _page_scroll(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = ViewportLockedScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background:transparent; border:none; }"
            "QScrollArea > QWidget > QWidget { background:transparent; }"
            "QScrollBar:vertical { background:transparent; width:7px; margin:2px 0; }"
            "QScrollBar::handle:vertical { background:#2B3445; border-radius:3px; min-height:34px; }"
            f"QScrollBar::handle:vertical:hover {{ background:{PURPLE}; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }"
        )
        page = QWidget()
        page.setObjectName("ScrollPage")
        page.setStyleSheet("QWidget#ScrollPage { background:transparent; }")
        page.setMinimumSize(0, 0)
        page.setMaximumSize(MAX_WIDGET, MAX_WIDGET)
        page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 8, 8)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignTop)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        scroll.setWidget(page)
        return scroll, layout

    def _input_route_block(
        self,
        combo: DarkCombo,
        meter: SegmentMeter,
        db: QLabel,
    ) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"background:#0A0F17; border:1px solid {BORDER}; border-radius:12px;"
        )
        box = QVBoxLayout(frame)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(7)

        heading = QHBoxLayout()
        title = QLabel("Input Device")
        title.setStyleSheet(
            f"color:{TEXT}; border:none; font-size:12px; font-weight:700;"
        )
        self.input_filter_count = QLabel("")
        self.input_filter_count.setStyleSheet(
            f"color:{MUTED}; border:none; font-size:10px;"
        )
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.input_filter_count)
        box.addLayout(heading)

        self.input_api_filter = SegmentedControl(INPUT_API_FILTERS, "All")
        self.input_api_filter.setToolTip(
            "Filter microphones by Windows host API. Kernel means WDM-KS / Kernel Streaming."
        )
        box.addWidget(self.input_api_filter)
        box.addWidget(combo)

        line = QHBoxLayout()
        line.addWidget(meter, 1)
        line.addWidget(db)
        box.addLayout(line)
        return frame

    def _build_overview(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        scroll.setObjectName("OverviewScroll")
        self.overview_scroll = scroll
        self.overview_page = scroll.widget()

        routing = GlowCard("Routing", PURPLE)
        routing.setObjectName("RoutingCard")
        routing.setMinimumWidth(0)
        routing.setMaximumWidth(MAX_WIDGET)
        routing.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.routing_card = routing
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

        input_block = self._input_route_block(
            self.input_combo, self.input_route_meter, self.input_route_db
        )
        output_block = self._route_block(
            "Processed Output", self.output_combo, self.output_route_meter, self.output_route_db
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

        self.input_panel = self._make_meter_panel("Input Level", GREEN, -60, 0)
        self.output_panel = self._make_meter_panel("Output Level", GREEN, -60, 0)
        self.voice_panel = self._make_meter_panel("Voice Match", PURPLE, 0, 100)
        self.reduction_panel = self._make_meter_panel("Noise Reduction", GREEN, 0, 60)
        meters = ReflowGrid(
            [self.input_panel, self.output_panel, self.voice_panel, self.reduction_panel],
            ((1320, 4), (620, 2), (0, 1)),
        )
        self._responsive_sections.append(meters)
        layout.addWidget(meters)

        diagnostics = GlowCard("Diagnostics", PURPLE)
        diagnostics.setMinimumWidth(0)
        diagnostics.setMaximumWidth(MAX_WIDGET)
        diagnostics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.diag_gpu = self._make_diagnostic_item("GPU Detected")
        self.diag_backend = self._make_diagnostic_item("Backend")
        self.diag_rtf = self._make_diagnostic_item("RTF")
        self.diag_queue = self._make_diagnostic_item("Queue Depth")
        self.diag_rate = self._make_diagnostic_item("Sample Rate", "48 kHz")
        self.diag_block = self._make_diagnostic_item("CUDA Preset")
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
        return scroll

    def _make_meter_panel(self, title: str, accent: str, minimum: float, maximum: float):
        from cinderfilter_window import MeterPanel

        return MeterPanel(title, accent, minimum, maximum)

    def _make_diagnostic_item(self, title: str, value: str = "--"):
        from cinderfilter_window import DiagnosticItem

        return DiagnosticItem(title, value)

    def _build_devices_page(self) -> QScrollArea:
        scroll, layout = self._page_scroll()
        card = GlowCard("Audio Devices", PURPLE)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        grid.addWidget(self._small("Physical Input"), 0, 0)
        grid.addWidget(self._small("Processed Output / Virtual Cable"), 0, 1)

        self.device_input_api_filter = SegmentedControl(INPUT_API_FILTERS, "All")
        self.device_input_api_filter.setToolTip(
            "Filter microphones by Windows host API. Kernel means WDM-KS / Kernel Streaming."
        )
        grid.addWidget(self.device_input_api_filter, 1, 0)
        output_note = QLabel("All output APIs")
        output_note.setStyleSheet(
            f"color:{MUTED}; border:none; font-size:11px; padding:8px 0;"
        )
        grid.addWidget(output_note, 1, 1)

        self.device_input_combo = DarkCombo()
        self.device_output_combo = DarkCombo()
        grid.addWidget(self.device_input_combo, 2, 0)
        grid.addWidget(self.device_output_combo, 2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        card.body.addLayout(grid)

        refresh = QPushButton("Refresh Windows Audio Endpoints")
        refresh.setStyleSheet(secondary_button_style())
        refresh.clicked.connect(self.refresh_devices)
        card.body.addWidget(refresh)
        self.endpoint_summary = QLabel("Scanning Windows audio endpoints…")
        self.endpoint_summary.setWordWrap(True)
        self.endpoint_summary.setStyleSheet(
            f"color:{MUTED}; border:none; font-size:11px;"
        )
        card.body.addWidget(self.endpoint_summary)
        layout.addWidget(card)

        table_card = GlowCard("Detected Endpoints", GREEN)
        self.device_table = QTableWidget(0, 5)
        self.device_table.setHorizontalHeaderLabels(
            ("Type", "Name", "Host API", "Channels", "Default Rate")
        )
        self.device_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.device_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.device_table.horizontalHeader().setStretchLastSection(True)
        self.device_table.setStyleSheet(self._table_style())
        table_card.body.addWidget(self.device_table)
        layout.addWidget(table_card)

        warning = QLabel(
            "CinderFilter owns the physical microphone. Route only its processed output into "
            "VoiceMeeter; do not add the raw microphone beside it."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "color:#FFD1B9; background:#25150F; border:1px solid #5B3120; "
            "border-radius:11px; padding:11px;"
        )
        layout.addWidget(warning)
        return scroll

    def _connect_signals(self) -> None:
        super()._connect_signals()
        self.input_api_filter.changed.connect(self._input_filter_changed)
        self.device_input_api_filter.changed.connect(self._input_filter_changed)

    def _restore_ui(self) -> None:
        super()._restore_ui()
        value = str(self.saved.get("input_api_filter", "All"))
        if value not in INPUT_API_FILTERS:
            value = "All"
        self.input_api_filter.setValue(value, emit=False)
        self.device_input_api_filter.setValue(value, emit=False)

    @staticmethod
    def _host_family(choice: DeviceChoice) -> str:
        host = choice.host_api.casefold()
        if "wasapi" in host:
            return "WASAPI"
        if "mme" in host:
            return "MME"
        if "wdm-ks" in host or "wdm ks" in host or "kernel" in host:
            return "Kernel"
        if "directsound" in host:
            return "DirectSound"
        return "Other"

    @classmethod
    def _dedupe_devices(
        cls,
        choices: list[DeviceChoice],
        direction: str,
    ) -> list[DeviceChoice]:
        seen: set[tuple[object, ...]] = set()
        kept: list[DeviceChoice] = []
        for choice in choices:
            channels = choice.input_channels if direction == "input" else choice.output_channels
            key = (
                cls._clean_device_name(choice).casefold(),
                choice.host_api.casefold(),
                channels,
                round(choice.default_sample_rate),
            )
            if key in seen:
                continue
            seen.add(key)
            kept.append(choice)

        family_order = {"WASAPI": 0, "Kernel": 1, "MME": 2, "DirectSound": 3, "Other": 4}
        kept.sort(
            key=lambda choice: (
                family_order.get(cls._host_family(choice), 9),
                cls._clean_device_name(choice).casefold(),
                choice.name.casefold(),
            )
        )
        return kept

    def _filtered_inputs(self) -> list[DeviceChoice]:
        selected = self.input_api_filter.value()
        if selected == "All":
            return list(self.inputs)
        return [choice for choice in self.inputs if self._host_family(choice) == selected]

    def _input_filter_changed(self, value: str) -> None:
        if self._syncing_input_filter:
            return
        value = value if value in INPUT_API_FILTERS else "All"
        self._syncing_input_filter = True
        try:
            self.input_api_filter.setValue(value, emit=False)
            self.device_input_api_filter.setValue(value, emit=False)
        finally:
            self._syncing_input_filter = False

        preferred = self._device_key(self.input_combo)
        self.saved["input_api_filter"] = value
        self._populate_input_combos(preferred)
        self._settings_changed()

    def _populate_input_combos(self, preferred: str) -> None:
        choices = self._filtered_inputs()
        self._fill_combo(self.input_combo, choices, preferred)
        self._fill_combo(
            self.device_input_combo,
            choices,
            self._device_key(self.input_combo),
        )
        self.input_filter_count.setText(
            f"{len(choices)} shown · {self.input_api_filter.value()}"
        )
        self._update_endpoint_summary()

    def _update_endpoint_summary(self) -> None:
        if not hasattr(self, "endpoint_summary"):
            return
        visible_inputs = len(self._filtered_inputs()) if self.inputs else 0
        hidden = self._input_aliases_hidden + self._output_aliases_hidden
        text = (
            f"{len(self.inputs)} unique inputs and {len(self.outputs)} unique outputs. "
            f"Input filter shows {visible_inputs}."
        )
        if hidden:
            text += f" {hidden} duplicate endpoint aliases are hidden."
        self.endpoint_summary.setText(text)

    def refresh_devices(self) -> None:
        try:
            raw_inputs, raw_outputs = self.engine.enumerate_devices()
        except BaseException as exc:
            self._show_error("Audio device scan failed", str(exc))
            return

        self._raw_input_count = len(raw_inputs)
        self._raw_output_count = len(raw_outputs)
        self.inputs = self._dedupe_devices(raw_inputs, "input")
        self.outputs = self._dedupe_devices(raw_outputs, "output")
        self._input_aliases_hidden = self._raw_input_count - len(self.inputs)
        self._output_aliases_hidden = self._raw_output_count - len(self.outputs)

        preferred_input = str(self.saved.get("input_device_key", ""))
        if isinstance(self.input_combo.currentData(), DeviceChoice):
            preferred_input = self._device_key(self.input_combo)
        self._populate_input_combos(preferred_input)

        preferred_output = str(self.saved.get("output_device_key", ""))
        if isinstance(self.output_combo.currentData(), DeviceChoice):
            preferred_output = self._device_key(self.output_combo)
        self._fill_combo(self.output_combo, self.outputs, preferred_output)
        self._fill_combo(
            self.device_output_combo,
            self.outputs,
            self._device_key(self.output_combo),
        )

        rows = [("Input", item) for item in self.inputs] + [
            ("Output", item) for item in self.outputs
        ]
        self.device_table.setRowCount(len(rows))
        for row, (kind, device) in enumerate(rows):
            channels = device.input_channels if kind == "Input" else device.output_channels
            values = (
                kind,
                self._clean_device_name(device),
                device.host_api,
                str(channels),
                f"{device.default_sample_rate:.0f} Hz",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setToolTip(device.label)
                self.device_table.setItem(row, column, item)
        self.device_table.resizeColumnsToContents()
        self._update_endpoint_summary()

        hidden = self._input_aliases_hidden + self._output_aliases_hidden
        suffix = f"; {hidden} duplicate aliases hidden" if hidden else ""
        self._on_status(
            f"Detected {len(self.inputs)} unique input and {len(self.outputs)} unique output endpoints{suffix}"
        )

    @staticmethod
    def _clean_device_name(choice: DeviceChoice) -> str:
        raw = str(choice.name or "Unknown device")
        text = raw.replace("\r", " ").replace("\n", " ")
        text = text.replace("%0", " ").replace("%1", " ")
        text = re.sub(
            r"@?System32\\drivers\\[^,;]+,[^;]+;?", " ", text, flags=re.IGNORECASE
        )
        text = re.sub(r"\s+", " ", text).strip(" ,·-")
        groups = [value.strip() for value in re.findall(r"\(([^()]*)\)" , text)]
        product = groups[-1] if groups else ""
        folded = text.casefold()
        if product and "hands-free" in folded:
            text = f"{product} — Hands-Free"
        elif product and ("stereo" in folded or "headphones" in folded):
            text = f"{product} — Stereo"
        if len(text) > 52:
            text = text[:49].rstrip() + "…"
        return text or "Unknown device"

    @classmethod
    def _display_host(cls, choice: DeviceChoice) -> str:
        family = cls._host_family(choice)
        if family in {"MME", "WASAPI", "Kernel", "DirectSound"}:
            return family
        return choice.host_api.removeprefix("Windows ")

    @classmethod
    def _fill_combo(cls, combo: DarkCombo, choices: list[DeviceChoice], preferred: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        selected = -1
        for index, choice in enumerate(choices):
            clean = cls._clean_device_name(choice)
            display = f"{clean}  ·  {cls._display_host(choice)}"
            combo.addItem(display, choice)
            combo.setItemData(index, choice.label, Qt.ToolTipRole)
            if choice.key.casefold() == preferred.casefold():
                selected = index
        combo.setCurrentIndex(selected if selected >= 0 else (0 if choices else -1))
        combo.blockSignals(False)

    def save_settings(self) -> None:
        config = self.current_config()
        geometry = self.geometry()
        values = {
            "version": self.SETTINGS_VERSION,
            "input_device_key": self._device_key(self.input_combo),
            "output_device_key": self._device_key(self.output_combo),
            "input_api_filter": self.input_api_filter.value(),
            **config.__dict__,
            "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()],
        }
        try:
            self.settings_store.save(values)
            self.saved = values
        except OSError as exc:
            self._append_log(f"Settings save failed: {exc}")
