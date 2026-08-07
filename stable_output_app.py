from __future__ import annotations

import cinderfilter_window as window_module
from device_catalog import CinderFilterWindow as DeviceCatalogWindow
from output_leveler import OUTPUT_LEVEL_PRESETS, OutputLevelMetrics
from stable_audio_core import StabilizedCinderFilterEngine
from ui_components import GREEN, MUTED, DarkCombo


class CinderFilterWindow(DeviceCatalogWindow):
    """Single CinderFilter app with final output stabilization."""

    SETTINGS_VERSION = 9

    @staticmethod
    def _migrate_settings(raw: dict) -> dict:
        values = DeviceCatalogWindow._migrate_settings(raw)
        preset = str(values.get("output_leveling", "Stable"))
        values["output_leveling"] = (
            preset if preset in OUTPUT_LEVEL_PRESETS else "Stable"
        )
        values["version"] = CinderFilterWindow.SETTINGS_VERSION
        return values

    def __init__(self) -> None:
        # CinderWindow resolves this module global when its constructor creates
        # the engine. Replace it before the base constructor runs.
        window_module.CinderFilterEngine = StabilizedCinderFilterEngine
        super().__init__()

    def _noise_card(self):
        card = super()._noise_card()
        card.body.addWidget(self._small("Final Output Leveling"))
        self.output_leveling_combo = DarkCombo(tuple(OUTPUT_LEVEL_PRESETS))
        self.output_leveling_combo.setToolTip(
            "Downward-only speech stabilization and a final peak limiter. "
            "It never boosts silence or room noise."
        )
        card.body.addWidget(self.output_leveling_combo)
        self.output_leveling_status = self._small(
            "Stable keeps loud phrases controlled without raising quiet noise."
        )
        self.output_leveling_status.setWordWrap(True)
        self.output_leveling_status.setStyleSheet(
            f"color:{MUTED}; border:none; font-size:11px; font-weight:700;"
        )
        card.body.addWidget(self.output_leveling_status)
        return card

    def _connect_signals(self) -> None:
        super()._connect_signals()
        self.output_leveling_combo.currentTextChanged.connect(
            self._output_leveling_changed
        )

    def _restore_ui(self) -> None:
        super()._restore_ui()
        preset = str(self.saved.get("output_leveling", "Stable"))
        if preset not in OUTPUT_LEVEL_PRESETS:
            preset = "Stable"
        self.output_leveling_combo.setCurrentText(preset)
        self.engine.set_output_leveling(preset)
        self._set_leveler_status(preset, None)

    def _output_leveling_changed(self, preset: str) -> None:
        selected = preset if preset in OUTPUT_LEVEL_PRESETS else "Stable"
        self.engine.set_output_leveling(selected)
        self.saved["output_leveling"] = selected
        self._set_leveler_status(selected, None)
        if hasattr(self, "save_timer"):
            self.save_timer.start()

    def _set_leveler_status(
        self,
        preset: str,
        metrics: OutputLevelMetrics | None,
    ) -> None:
        if metrics is None:
            descriptions = {
                "Off": "No final leveling; only safety clipping remains.",
                "Gentle": "Light control that preserves most natural dynamics.",
                "Stable": "Smooths loud phrases and catches peaks without boosting noise.",
                "Broadcast": "Tighter, more consistent stream-style output level.",
            }
            self.output_leveling_status.setText(descriptions.get(preset, ""))
            return
        reduction = metrics.total_reduction_db
        limiter = metrics.limiter_reduction_db
        self.output_leveling_status.setText(
            f"{metrics.preset} · {reduction:.1f} dB total control"
            + (f" · limiter {limiter:.1f} dB" if limiter >= 0.1 else "")
        )
        self.output_leveling_status.setStyleSheet(
            f"color:{GREEN}; border:none; font-size:11px; font-weight:700;"
        )

    def _on_engine_event(self, kind: str, payload: object) -> None:
        super()._on_engine_event(kind, payload)
        if kind == "leveler_metrics" and isinstance(payload, OutputLevelMetrics):
            self._set_leveler_status(payload.preset, payload)
        elif kind == "leveler_preset":
            self._set_leveler_status(str(payload), None)

    def save_settings(self) -> None:
        super().save_settings()
        self.saved["version"] = self.SETTINGS_VERSION
        self.saved["output_leveling"] = self.output_leveling_combo.currentText()
        try:
            self.settings_store.save(self.saved)
        except OSError as exc:
            self._append_log(f"Settings save failed: {exc}")
