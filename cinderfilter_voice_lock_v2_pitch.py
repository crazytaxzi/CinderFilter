from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk

import cinderfilter_voice_lock_v2 as v2
from deepfilternet_rs import DeepFilterNetRealtime
from pitch_lock_v2 import PitchGuard, PitchLockedTargetSeparator, PitchSeparationMetrics
from voice_lock import SAMPLE_RATE, VoiceLockedAudioEngine


class PitchLockedAudioEngine(v2.TargetSpeakerAudioEngine):
    """v2 engine that can remain silent instead of falling back to v1."""

    def __init__(self, voice_service, separator, on_status, on_metrics) -> None:
        super().__init__(voice_service, separator, on_status, on_metrics)
        self._strict_fail_closed = True
        self._v2_failed_closed = False

    def configure_fail_closed(self, enabled: bool) -> None:
        self._strict_fail_closed = bool(enabled)

    def start(self, input_device: int, output_device: int, strength: str) -> None:
        self._v2_active = False
        self._v2_failed_closed = False
        if self._v2_requested:
            try:
                self.separator.start(
                    self._v2_preset,
                    self._v2_device,
                    self._v2_strictness,
                )
                self._v2_active = True
            except BaseException as exc:
                if self._strict_fail_closed:
                    raise RuntimeError(
                        f"Strict Pitch Lock refused to start without v2: {exc}"
                    ) from exc
                self.on_status(f"Pitch Lock v2 unavailable; using v1 fallback: {exc}")
        # Call the v1 engine directly so v2 is not started twice by the parent.
        VoiceLockedAudioEngine.start(self, input_device, output_device, strength)

    def _process_loop(self) -> None:
        processor: DeepFilterNetRealtime | None = None
        last_metrics = 0.0
        try:
            processor = DeepFilterNetRealtime(
                model_path=None,
                atten_lim=self._atten_lim,
                log_level="warn",
                compensate_delay=False,
                post_filter_beta=0.0,
            )
            if int(processor.sample_rate) != SAMPLE_RATE:
                raise RuntimeError(
                    f"Model requires {processor.sample_rate} Hz, expected {SAMPLE_RATE} Hz."
                )
            self._model_ready.set()

            while self._running.is_set():
                try:
                    raw_block = self._input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                using_v2 = False
                source = raw_block
                if not self._bypass.is_set():
                    if self._v2_active:
                        extracted = self.separator.process_block(raw_block)
                        if extracted is None:
                            self._v2_active = False
                            if self._strict_fail_closed:
                                self._v2_failed_closed = True
                                source = np.zeros_like(raw_block, dtype=np.float32)
                                using_v2 = True
                                self.on_status(
                                    "STRICT PITCH LOCK: v2 failed or overloaded — output muted"
                                )
                            else:
                                self.on_status("Pitch Lock v2 fell back to v1")
                        else:
                            source = extracted
                            using_v2 = True
                    elif self._v2_requested and self._strict_fail_closed and self._v2_failed_closed:
                        source = np.zeros_like(raw_block, dtype=np.float32)
                        using_v2 = True

                if self._bypass.is_set():
                    processed = raw_block
                else:
                    processed = np.asarray(
                        processor.process_chunk(source), dtype=np.float32
                    ).reshape(-1)

                if processed.size == 0:
                    continue
                processed = np.clip(processed, -1.0, 1.0)

                if using_v2:
                    similarity = self.separator.last_similarity
                    with self._voice_guard:
                        self._voice_similarity = similarity
                        self._voice_target_gain = 1.0
                        self._voice_current_gain = 1.0
                    gain = 1.0
                else:
                    self._feed_voice_lock(processed)
                    processed = self._apply_voice_gain(processed)
                    with self._voice_guard:
                        gain = self._voice_current_gain
                        similarity = self._voice_similarity

                self._output_level = self._peak_db(processed)
                try:
                    self._output_queue.put_nowait(processed)
                except queue.Full:
                    try:
                        self._output_queue.get_nowait()
                        self._output_queue.put_nowait(processed)
                    except queue.Empty:
                        pass

                now = v2.time.monotonic()
                if now - last_metrics >= 0.10:
                    self.on_metrics(
                        self._input_level,
                        self._output_level,
                        self._dropped_input,
                        self._output_underruns,
                        gain,
                        similarity,
                    )
                    last_metrics = now
        except BaseException as exc:
            self._worker_error = exc
            self.on_status(f"AI processing error: {exc}")
            self._running.clear()
        finally:
            if processor is not None:
                try:
                    processor.close()
                except BaseException:
                    pass
            self._model_ready.set()


class PitchLockV2App(v2.VoiceLockV2App):
    def __init__(self) -> None:
        super().__init__()
        self.title("CinderFilter — Pitch-Locked Voice Lock v2")
        self.geometry("900x970")
        self.minsize(840, 860)

        # Replace the ordinary separator with the pitch-aware one while keeping
        # the existing v2 engine and UI contract.
        self.separator.stop()
        self.pitch_guard = PitchGuard()
        self.separator = PitchLockedTargetSeparator(
            self.pitch_guard,
            self._thread_v2_status,
            self._thread_v2_metrics,
        )
        self.engine = PitchLockedAudioEngine(
            self.voice_service,
            self.separator,
            self._thread_status,
            self._thread_metrics,
        )

        self.pitch_enabled_var = tk.BooleanVar(value=True)
        self.pitch_fail_closed_var = tk.BooleanVar(value=True)
        self.pitch_margin_var = tk.StringVar(value="0 Hz")
        self.pitch_profile_var = tk.StringVar()
        self._add_pitch_ui()
        self._refresh_pitch_profile_text()

    def _add_pitch_ui(self) -> None:
        frame = ttk.LabelFrame(
            self,
            text="Pitch Lock — reject voices above my enrolled range",
            padding=12,
        )
        frame.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            frame,
            text="Enable pitch distinction",
            variable=self.pitch_enabled_var,
            command=self._apply_pitch_settings,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            frame,
            text="Fail closed on uncertain pitch",
            variable=self.pitch_fail_closed_var,
            command=self._apply_pitch_settings,
        ).grid(row=0, column=1, sticky="w", padx=(14, 0))
        ttk.Button(frame, text="Calibrate My Pitch", command=self.calibrate_pitch).grid(
            row=0, column=2, padx=(8, 4)
        )
        ttk.Button(frame, text="Delete Pitch Profile", command=self.delete_pitch_profile).grid(
            row=0, column=3
        )

        ttk.Label(frame, text="Extra upper margin").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        margin = ttk.Combobox(
            frame,
            textvariable=self.pitch_margin_var,
            values=("-15 Hz", "0 Hz", "15 Hz", "30 Hz", "50 Hz"),
            state="readonly",
            width=12,
        )
        margin.grid(row=1, column=1, sticky="w", pady=(10, 0))
        margin.bind("<<ComboboxSelected>>", lambda _event: self._apply_pitch_settings())

        ttk.Label(frame, textvariable=self.pitch_profile_var, wraplength=840).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )
        ttk.Label(
            frame,
            text=(
                "Strict mode requires both a voiceprint match and pitch inside your range. "
                "A higher voice, an uncertain match, a pitchless chunk, or a v2 failure becomes silence."
            ),
            wraplength=840,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

    def calibrate_pitch(self) -> None:
        choice = self.input_choices.get(self.input_var.get())
        if choice is None:
            messagebox.showwarning("Choose an input", "Select the physical microphone first.")
            return
        threading.Thread(target=self._calibrate_pitch_worker, args=(choice.index,), daemon=True).start()

    def _calibrate_pitch_worker(self, input_index: int) -> None:
        try:
            if self.engine.running:
                self.engine.stop()
                self._ui_events.put(("running", False))
            seconds = 12
            self._ui_events.put(
                (
                    "v2_status",
                    f"Recording {seconds} seconds for pitch calibration — use your normal stream voice...",
                )
            )
            recording = sd.rec(
                int(seconds * 48_000),
                samplerate=48_000,
                channels=1,
                dtype="float32",
                device=input_index,
                blocking=True,
            )
            profile = self.pitch_guard.calibrate(
                np.asarray(recording[:, 0], dtype=np.float32)
            )
            self._ui_events.put(("pitch_profile", profile))
            self._ui_events.put(
                (
                    "v2_status",
                    f"Pitch profile ready — upper cutoff {profile.upper_limit_hz:.0f} Hz",
                )
            )
        except BaseException as exc:
            self._ui_events.put(("v2_status", f"Pitch calibration failed: {exc}"))

    def delete_pitch_profile(self) -> None:
        self.pitch_guard.delete_profile()
        self.pitch_enabled_var.set(False)
        self._apply_pitch_settings()
        self._refresh_pitch_profile_text()
        self.v2_status_var.set("Pitch profile deleted")

    def _apply_pitch_settings(self) -> None:
        margin = float(self.pitch_margin_var.get().split()[0])
        self.separator.configure_pitch_lock(
            self.pitch_enabled_var.get(),
            self.pitch_fail_closed_var.get(),
            margin,
        )
        self.engine.configure_fail_closed(self.pitch_fail_closed_var.get())
        if self.engine.running:
            self.v2_status_var.set("Restart filtering to apply pitch-lock changes")

    def start_filtering(self) -> None:
        if self.v2_enabled_var.get() and self.pitch_enabled_var.get() and not self.pitch_guard.has_profile:
            messagebox.showwarning(
                "Pitch profile missing",
                "Calibrate your pitch before enabling Pitch-Locked v2.",
            )
            return
        self._apply_pitch_settings()
        super().start_filtering()

    def preload_v2(self) -> None:
        if self.pitch_enabled_var.get() and not self.pitch_guard.has_profile:
            messagebox.showwarning(
                "Calibrate pitch first",
                "Pitch-Locked v2 needs your pitch profile before preloading.",
            )
            return
        self._apply_pitch_settings()
        super().preload_v2()

    def _drain_ui_events(self) -> None:
        try:
            while True:
                event = self._ui_events.get_nowait()
                kind = event[0]
                if kind == "status":
                    self.status_var.set(event[1])
                elif kind == "voice_status":
                    self.voice_status_var.set(event[1])
                elif kind == "voice_result":
                    self._show_voice_result(event[1])
                elif kind == "profile":
                    self._apply_profile_state(bool(event[1]))
                elif kind == "metrics":
                    self._update_voice_metrics(*event[1:])
                elif kind == "v2_status":
                    self.v2_status_var.set(str(event[1]))
                elif kind == "v2_metrics":
                    self._show_v2_metrics(event[1])
                elif kind == "pitch_profile":
                    self._refresh_pitch_profile_text()
                elif kind == "running":
                    self._set_running_ui(bool(event[1]))
                elif kind == "error":
                    self._show_start_error(str(event[1]))
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self._drain_ui_events)

    def _show_v2_metrics(self, metrics) -> None:
        if isinstance(metrics, PitchSeparationMetrics):
            selected = (
                "--" if metrics.selected_similarity is None else f"{metrics.selected_similarity:.3f}"
            )
            pa = "--" if metrics.pitch_a_hz is None else f"{metrics.pitch_a_hz:.0f}"
            pb = "--" if metrics.pitch_b_hz is None else f"{metrics.pitch_b_hz:.0f}"
            state_a = "PASS" if metrics.pitch_a_allowed else "REJECT"
            state_b = "PASS" if metrics.pitch_b_allowed else "REJECT"
            fallback = "    FALLBACK" if metrics.fallback else ""
            self.v2_metrics_var.set(
                f"RTF: {metrics.realtime_factor:.2f}    Match: {selected}    "
                f"Pitch A/B: {pa}/{pb} Hz ({state_a}/{state_b})    "
                f"Limit: {metrics.pitch_limit_hz:.0f} Hz{fallback}"
            )
        else:
            super()._show_v2_metrics(metrics)

    def _refresh_pitch_profile_text(self) -> None:
        profile = self.pitch_guard.profile
        if profile is None:
            self.pitch_profile_var.set("No pitch profile calibrated")
        else:
            self.pitch_profile_var.set(
                f"Pitch profile: median {profile.median_hz:.0f} Hz, enrolled range "
                f"{profile.low_hz:.0f}–{profile.high_hz:.0f} Hz, strict upper cutoff "
                f"{profile.upper_limit_hz:.0f} Hz"
            )


def main() -> None:
    app = PitchLockV2App()
    app.mainloop()


if __name__ == "__main__":
    main()
