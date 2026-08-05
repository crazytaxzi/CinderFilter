from __future__ import annotations

import re
from typing import Any

import cinderfilter_voice_lock_v2_pitch as pitch
from settings_store import SettingsStore


_DEVICE_PREFIX = re.compile(r"^\s*\d+\s*:\s*")
_ALLOWED_AI_STRENGTHS = {"Natural", "Balanced", "Strong", "Maximum"}
_ALLOWED_REDUCTIONS = {"12 dB", "18 dB", "24 dB", "36 dB", "48 dB"}
_ALLOWED_STRICTNESS = {"Conservative", "Balanced", "Aggressive"}
_ALLOWED_V2_PRESETS = {"Fast", "Balanced", "Quality"}
_ALLOWED_COMPUTE = {"Auto", "CUDA", "CPU"}
_ALLOWED_PITCH_MARGINS = {"-15 Hz", "0 Hz", "15 Hz", "30 Hz", "50 Hz"}


def _device_key(label: str) -> str:
    """Strip the unstable PortAudio index while preserving name + host API."""
    return _DEVICE_PREFIX.sub("", str(label), count=1).strip()


class PersistentPitchLockApp(pitch.PitchLockV2App):
    """Pitch-Locked v2 with per-user persistent settings."""

    SETTINGS_VERSION = 1

    def __init__(self) -> None:
        self._settings_store = SettingsStore()
        self._loaded_settings: dict[str, Any] = self._settings_store.load()
        self._save_job: str | None = None
        self._restoring_settings = True
        self._refreshing_devices = False
        self._desired_input_key = str(self._loaded_settings.get("input_device_key", ""))
        self._desired_output_key = str(self._loaded_settings.get("output_device_key", ""))

        super().__init__()

        self._restore_settings()
        self._restoring_settings = False
        self._bind_persistent_variables()
        self._apply_restored_runtime_settings()

    def refresh_devices(self) -> None:
        """Refresh endpoints without forgetting a temporarily missing preferred route."""
        self._refreshing_devices = True
        try:
            super().refresh_devices()
            self._restore_device(self.input_var, self.input_choices, self._desired_input_key)
            self._restore_device(self.output_var, self.output_choices, self._desired_output_key)
        finally:
            self._refreshing_devices = False

    @staticmethod
    def _restore_device(var, choices, desired_key: str) -> None:
        if not desired_key:
            return

        desired_folded = desired_key.casefold()
        for label in choices:
            if _device_key(label).casefold() == desired_folded:
                var.set(label)
                return

        desired_name = desired_key.split("[", 1)[0].strip().casefold()
        if desired_name:
            for label in choices:
                candidate_name = _device_key(label).split("[", 1)[0].strip().casefold()
                if candidate_name == desired_name:
                    var.set(label)
                    return

    def _restore_settings(self) -> None:
        values = self._loaded_settings

        self._set_allowed(self.strength_var, values.get("ai_strength"), _ALLOWED_AI_STRENGTHS)
        self.bypass_var.set(bool(values.get("bypass", False)))
        self._set_allowed(self.voice_reduction_var, values.get("voice_reduction"), _ALLOWED_REDUCTIONS)
        self._set_allowed(self.voice_strictness_var, values.get("voice_strictness"), _ALLOWED_STRICTNESS)
        self._set_allowed(self.v2_preset_var, values.get("v2_preset"), _ALLOWED_V2_PRESETS)
        self._set_allowed(self.v2_device_var, values.get("v2_device"), _ALLOWED_COMPUTE)
        self._set_allowed(self.pitch_margin_var, values.get("pitch_margin"), _ALLOWED_PITCH_MARGINS)

        has_voice = self.voice_service.has_profile
        has_pitch = self.pitch_guard.has_profile
        self.voice_lock_var.set(bool(values.get("voice_lock_enabled", False)) and has_voice)
        self.v2_enabled_var.set(bool(values.get("v2_enabled", False)) and has_voice)
        self.pitch_enabled_var.set(bool(values.get("pitch_enabled", True)) and has_pitch)
        self.pitch_fail_closed_var.set(bool(values.get("pitch_fail_closed", True)))

        geometry = values.get("window_geometry")
        if isinstance(geometry, str) and re.fullmatch(
            r"\d{3,5}x\d{3,5}(?:[+-]\d{1,6}[+-]\d{1,6})?",
            geometry,
        ):
            try:
                self.geometry(geometry)
            except Exception:
                pass

        self._restore_device(self.input_var, self.input_choices, self._desired_input_key)
        self._restore_device(self.output_var, self.output_choices, self._desired_output_key)

    @staticmethod
    def _set_allowed(variable, candidate: Any, allowed: set[str]) -> None:
        if isinstance(candidate, str) and candidate in allowed:
            variable.set(candidate)

    def _apply_restored_runtime_settings(self) -> None:
        self.engine.set_bypass(self.bypass_var.get())
        self._apply_voice_settings()
        self._apply_pitch_settings()
        self.engine.configure_v2(
            self.v2_enabled_var.get(),
            self.v2_preset_var.get(),
            self.v2_device_var.get(),
            self.voice_strictness_var.get(),
        )
        if self._loaded_settings:
            self.status_var.set("Settings restored — ready")

    def _bind_persistent_variables(self) -> None:
        variables = (
            self.input_var,
            self.output_var,
            self.strength_var,
            self.bypass_var,
            self.voice_lock_var,
            self.voice_reduction_var,
            self.voice_strictness_var,
            self.v2_enabled_var,
            self.v2_preset_var,
            self.v2_device_var,
            self.pitch_enabled_var,
            self.pitch_fail_closed_var,
            self.pitch_margin_var,
        )
        for variable in variables:
            variable.trace_add("write", self._setting_changed)

    def _setting_changed(self, *_args) -> None:
        if self._restoring_settings or self._refreshing_devices:
            return
        self._schedule_save()

    def _schedule_save(self) -> None:
        if self._save_job is not None:
            try:
                self.after_cancel(self._save_job)
            except Exception:
                pass
        self._save_job = self.after(350, self._save_settings_now)

    def _settings_snapshot(self) -> dict[str, Any]:
        input_key = _device_key(self.input_var.get())
        output_key = _device_key(self.output_var.get())
        if input_key:
            self._desired_input_key = input_key
        if output_key:
            self._desired_output_key = output_key

        return {
            "version": self.SETTINGS_VERSION,
            "input_device_key": self._desired_input_key,
            "output_device_key": self._desired_output_key,
            "ai_strength": self.strength_var.get(),
            "bypass": bool(self.bypass_var.get()),
            "voice_lock_enabled": bool(self.voice_lock_var.get()),
            "voice_reduction": self.voice_reduction_var.get(),
            "voice_strictness": self.voice_strictness_var.get(),
            "v2_enabled": bool(self.v2_enabled_var.get()),
            "v2_preset": self.v2_preset_var.get(),
            "v2_device": self.v2_device_var.get(),
            "pitch_enabled": bool(self.pitch_enabled_var.get()),
            "pitch_fail_closed": bool(self.pitch_fail_closed_var.get()),
            "pitch_margin": self.pitch_margin_var.get(),
            "window_geometry": self.geometry(),
        }

    def _save_settings_now(self) -> None:
        self._save_job = None
        try:
            self._settings_store.save(self._settings_snapshot())
        except OSError as exc:
            self.status_var.set(f"Could not save settings: {exc}")

    def start_filtering(self) -> None:
        self._save_settings_now()
        super().start_filtering()

    def preload_v2(self) -> None:
        self._save_settings_now()
        super().preload_v2()

    def _on_close(self) -> None:
        if self._save_job is not None:
            try:
                self.after_cancel(self._save_job)
            except Exception:
                pass
            self._save_job = None
        self._save_settings_now()
        super()._on_close()


def main() -> None:
    app = PersistentPitchLockApp()
    app.mainloop()


if __name__ == "__main__":
    main()
