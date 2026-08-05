from __future__ import annotations

import queue
import threading
import time

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from deepfilternet_rs import DeepFilterNetRealtime

import cinderfilter_voice_lock as v1
from target_separator_v2 import SeparationMetrics, TargetSpeakerSeparator
from voice_lock import SAMPLE_RATE, VoiceLockedAudioEngine


class TargetSpeakerAudioEngine(VoiceLockedAudioEngine):
    """Voice Lock v1 plus an optional overlapping-speaker extraction path."""

    def __init__(self, voice_service, separator: TargetSpeakerSeparator, on_status, on_metrics) -> None:
        super().__init__(voice_service, on_status, on_metrics)
        self.separator = separator
        self._v2_requested = False
        self._v2_preset = "Fast"
        self._v2_device = "Auto"
        self._v2_strictness = "Balanced"
        self._v2_active = False

    def configure_v2(
        self,
        enabled: bool,
        preset: str,
        device: str,
        strictness: str,
    ) -> None:
        self._v2_requested = bool(enabled)
        self._v2_preset = preset
        self._v2_device = device
        self._v2_strictness = strictness

    def start(self, input_device: int, output_device: int, strength: str) -> None:
        self._v2_active = False
        if self._v2_requested:
            try:
                self.separator.start(
                    self._v2_preset,
                    self._v2_device,
                    self._v2_strictness,
                )
                self._v2_active = True
            except BaseException as exc:
                self.on_status(f"Voice Lock v2 unavailable; using v1 fallback: {exc}")
                self._v2_active = False
        super().start(input_device, output_device, strength)

    def stop(self) -> None:
        super().stop()
        self.separator.stop()
        self._v2_active = False

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
                if not self._bypass.is_set() and self._v2_active:
                    extracted = self.separator.process_block(raw_block)
                    if extracted is None:
                        self._v2_active = False
                        self.on_status("Voice Lock v2 fell back to v1")
                    else:
                        source = extracted
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

                now = time.monotonic()
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


class VoiceLockV2App(v1.VoiceLockApp):
    def __init__(self) -> None:
        super().__init__()
        self.title("CinderFilter — Voice Lock v2")
        self.geometry("860x835")
        self.minsize(800, 760)

        self.v2_enabled_var = tk.BooleanVar(value=False)
        self.v2_preset_var = tk.StringVar(value="Fast")
        self.v2_device_var = tk.StringVar(value="Auto")
        self.v2_status_var = tk.StringVar(
            value="v2 not loaded — enroll a voice profile, then preload or start"
        )
        self.v2_metrics_var = tk.StringVar(
            value="RTF: --    Target match: --    Nominal chunk delay: 1.6 s"
        )

        self.separator = TargetSpeakerSeparator(
            self._thread_v2_status,
            self._thread_v2_metrics,
        )
        self.engine = TargetSpeakerAudioEngine(
            self.voice_service,
            self.separator,
            self._thread_status,
            self._thread_metrics,
        )
        self._add_v2_ui()

    def _add_v2_ui(self) -> None:
        frame = ttk.LabelFrame(
            self,
            text="Voice Lock v2 — extract my voice during overlapping speech",
            padding=12,
        )
        frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            frame,
            text="Enable target-speaker extraction",
            variable=self.v2_enabled_var,
            command=self._v2_toggle,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(frame, text="Preload v2 Models", command=self.preload_v2).grid(
            row=0, column=2, padx=(8, 0)
        )

        ttk.Label(frame, text="Latency / quality").grid(row=1, column=0, sticky="w", pady=(10, 0))
        preset = ttk.Combobox(
            frame,
            textvariable=self.v2_preset_var,
            values=("Fast", "Balanced", "Quality"),
            state="readonly",
            width=14,
        )
        preset.grid(row=1, column=1, sticky="w", pady=(10, 0))
        preset.bind("<<ComboboxSelected>>", lambda _event: self._v2_preset_changed())

        ttk.Label(frame, text="Compute device").grid(
            row=1, column=2, sticky="e", padx=(16, 6), pady=(10, 0)
        )
        compute = ttk.Combobox(
            frame,
            textvariable=self.v2_device_var,
            values=("Auto", "CUDA", "CPU"),
            state="readonly",
            width=10,
        )
        compute.grid(row=1, column=3, sticky="e", pady=(10, 0))

        ttk.Label(frame, textvariable=self.v2_status_var, wraplength=790).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )
        ttk.Label(frame, textvariable=self.v2_metrics_var).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            frame,
            text=(
                "v2 separates two speech streams, chooses the one matching your enrolled voice, "
                "then sends it through DeepFilterNet. It intentionally delays audio and falls back "
                "to v1 if inference cannot keep up."
            ),
            wraplength=790,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def _v2_toggle(self) -> None:
        if self.v2_enabled_var.get() and not self.voice_service.has_profile:
            self.v2_enabled_var.set(False)
            messagebox.showwarning("Enroll first", "Voice Lock v2 needs your enrolled voice profile.")
            return
        if self.engine.running:
            self.v2_status_var.set("Restart filtering to apply the v2 change")

    def _v2_preset_changed(self) -> None:
        delays = {"Fast": 1.6, "Balanced": 2.4, "Quality": 3.2}
        delay = delays.get(self.v2_preset_var.get(), 1.6)
        self.v2_metrics_var.set(
            f"RTF: --    Target match: --    Nominal chunk delay: {delay:.1f} s"
        )
        if self.engine.running:
            self.v2_status_var.set("Restart filtering to apply the latency preset")

    def preload_v2(self) -> None:
        if not self.voice_service.has_profile:
            messagebox.showwarning("Enroll first", "Enroll your voice before loading v2.")
            return
        threading.Thread(target=self._preload_v2_worker, daemon=True).start()

    def _preload_v2_worker(self) -> None:
        try:
            self._ui_events.put(("v2_status", "Loading v2 models..."))
            self.separator.start(
                self.v2_preset_var.get(),
                self.v2_device_var.get(),
                self.voice_strictness_var.get(),
            )
            self._ui_events.put(("v2_status", "v2 models loaded and ready"))
        except BaseException as exc:
            self._ui_events.put(("v2_status", f"v2 preload failed: {exc}"))

    def delete_voice_profile(self) -> None:
        self.v2_enabled_var.set(False)
        self.separator.stop()
        super().delete_voice_profile()
        self.v2_status_var.set("v2 stopped — voice profile deleted")

    def _apply_profile_state(self, exists: bool) -> None:
        super()._apply_profile_state(exists)
        separator = getattr(self, "separator", None)
        if exists and separator is not None and separator.running:
            separator.stop()
            status = getattr(self, "v2_status_var", None)
            if status is not None:
                status.set("Voice profile changed — preload v2 again")

    def start_filtering(self) -> None:
        if self.v2_enabled_var.get():
            if not self.voice_service.has_profile:
                messagebox.showwarning("Voice profile missing", "Enroll your voice before using v2.")
                return
            self.voice_lock_var.set(True)
            self._apply_voice_settings()
        self.engine.configure_v2(
            self.v2_enabled_var.get(),
            self.v2_preset_var.get(),
            self.v2_device_var.get(),
            self.voice_strictness_var.get(),
        )
        super().start_filtering()

    def _thread_v2_status(self, text: str) -> None:
        self._ui_events.put(("v2_status", text))

    def _thread_v2_metrics(self, metrics: SeparationMetrics) -> None:
        self._ui_events.put(("v2_metrics", metrics))

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
                elif kind == "running":
                    self._set_running_ui(bool(event[1]))
                elif kind == "error":
                    self._show_start_error(str(event[1]))
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self._drain_ui_events)

    def _show_v2_metrics(self, metrics: SeparationMetrics) -> None:
        selected = "--" if metrics.selected_similarity is None else f"{metrics.selected_similarity:.3f}"
        fallback = "    FALLBACK TO v1" if metrics.fallback else ""
        self.v2_metrics_var.set(
            f"RTF: {metrics.realtime_factor:.2f}    Target match: {selected}    "
            f"Sources: {metrics.similarity_a:.3f}/{metrics.similarity_b:.3f}    "
            f"Device: {metrics.device.upper()}{fallback}"
        )


def main() -> None:
    app = VoiceLockV2App()
    app.mainloop()


if __name__ == "__main__":
    main()
