from __future__ import annotations

import math
import queue
import threading
import traceback

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk

import cinderfilter_threadsafe  # applies the PyO3 thread-affinity hotfix
import cinderfilter as base
from voice_lock import SAMPLE_RATE, VoiceLockedAudioEngine, VoiceLockService


class VoiceLockApp(base.CinderFilterApp):
    def __init__(self) -> None:
        super().__init__()
        self.title("CinderFilter — Voice Lock")
        self.geometry("820x660")
        self.minsize(760, 620)

        self.voice_lock_var = tk.BooleanVar(value=False)
        self.voice_reduction_var = tk.StringVar(value="24 dB")
        self.voice_strictness_var = tk.StringVar(value="Balanced")
        self.voice_profile_var = tk.StringVar(value="No voice profile enrolled")
        self.voice_status_var = tk.StringVar(value="Enroll your voice to enable Voice Lock")
        self.voice_match_var = tk.StringVar(value="Match: --    Voice gain: 0.0 dB")

        self.voice_service = VoiceLockService(
            self._thread_voice_status,
            self._thread_voice_result,
            self._thread_profile,
        )
        # Replace the untouched base engine before any route starts.
        self.engine = VoiceLockedAudioEngine(
            self.voice_service, self._thread_status, self._thread_metrics
        )
        self._add_voice_ui()
        self._apply_profile_state(self.voice_service.has_profile)

    def _add_voice_ui(self) -> None:
        frame = ttk.LabelFrame(
            self,
            text="Voice Lock — keep my voice, suppress other speakers",
            padding=12,
        )
        frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 18))
        frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            frame,
            text="Enable Voice Lock",
            variable=self.voice_lock_var,
            command=self._toggle_voice_lock,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.voice_profile_var, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=12
        )
        ttk.Button(frame, text="Enroll My Voice", command=self.enroll_voice).grid(
            row=0, column=2, padx=(8, 4)
        )
        ttk.Button(frame, text="Delete Profile", command=self.delete_voice_profile).grid(
            row=0, column=3
        )

        ttk.Label(frame, text="Other voice reduction").grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )
        reduction = ttk.Combobox(
            frame,
            textvariable=self.voice_reduction_var,
            values=("12 dB", "18 dB", "24 dB", "36 dB", "48 dB"),
            state="readonly",
            width=12,
        )
        reduction.grid(row=1, column=1, sticky="w", pady=(12, 0))
        reduction.bind("<<ComboboxSelected>>", lambda _event: self._apply_voice_settings())

        ttk.Label(frame, text="Strictness").grid(
            row=1, column=2, sticky="e", padx=(12, 6), pady=(12, 0)
        )
        strictness = ttk.Combobox(
            frame,
            textvariable=self.voice_strictness_var,
            values=("Conservative", "Balanced", "Aggressive"),
            state="readonly",
            width=14,
        )
        strictness.grid(row=1, column=3, sticky="e", pady=(12, 0))
        strictness.bind("<<ComboboxSelected>>", lambda _event: self._apply_voice_settings())

        ttk.Label(frame, textvariable=self.voice_status_var, wraplength=740).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )
        ttk.Label(frame, textvariable=self.voice_match_var).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            frame,
            text=(
                "v1 suppresses a different dominant speaker. Uncertain overlapping speech is "
                "protected so your voice is less likely to be cut."
            ),
            wraplength=740,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def enroll_voice(self) -> None:
        choice = self.input_choices.get(self.input_var.get())
        if choice is None:
            messagebox.showwarning("Choose an input", "Select the physical microphone first.")
            return
        threading.Thread(target=self._enroll_worker, args=(choice.index,), daemon=True).start()

    def _enroll_worker(self, input_index: int) -> None:
        try:
            if self.engine.running:
                self.engine.stop()
                self._ui_events.put(("running", False))
            seconds = 12
            self._ui_events.put(
                ("voice_status", f"Recording {seconds} seconds — speak naturally and continuously...")
            )
            recording = sd.rec(
                int(seconds * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=input_index,
                blocking=True,
            )
            self.voice_service.enroll(np.asarray(recording[:, 0], dtype=np.float32))
        except BaseException as exc:
            self._ui_events.put(("voice_status", f"Enrollment recording failed: {exc}"))

    def delete_voice_profile(self) -> None:
        self.voice_lock_var.set(False)
        self.engine.configure_voice_lock(False, 24.0, "Balanced")
        self.voice_service.delete_profile()

    def _toggle_voice_lock(self) -> None:
        if self.voice_lock_var.get() and not self.voice_service.has_profile:
            self.voice_lock_var.set(False)
            messagebox.showwarning("Enroll first", "Record your voice profile before enabling Voice Lock.")
            return
        self._apply_voice_settings()

    def _apply_voice_settings(self) -> None:
        reduction = float(self.voice_reduction_var.get().split()[0])
        self.engine.configure_voice_lock(
            self.voice_lock_var.get(), reduction, self.voice_strictness_var.get()
        )

    def start_filtering(self) -> None:
        if self.voice_lock_var.get() and not self.voice_service.has_profile:
            messagebox.showwarning("Voice profile missing", "Enroll your voice before using Voice Lock.")
            return
        self._apply_voice_settings()
        super().start_filtering()

    def _thread_voice_status(self, text: str) -> None:
        self._ui_events.put(("voice_status", text))

    def _thread_voice_result(self, similarity: float | None) -> None:
        self.engine.update_voice_similarity(similarity)
        self._ui_events.put(("voice_result", similarity))

    def _thread_profile(self, exists: bool) -> None:
        self._ui_events.put(("profile", exists))

    def _thread_metrics(
        self,
        input_db: float,
        output_db: float,
        dropped: int,
        underruns: int,
        gain: float,
        similarity: float | None,
    ) -> None:
        self._ui_events.put(
            ("metrics", input_db, output_db, dropped, underruns, gain, similarity)
        )

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
                elif kind == "running":
                    self._set_running_ui(bool(event[1]))
                elif kind == "error":
                    self._show_start_error(str(event[1]))
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self._drain_ui_events)

    def _apply_profile_state(self, exists: bool) -> None:
        self.voice_profile_var.set("Voice profile: READY" if exists else "No voice profile enrolled")
        if not exists:
            self.voice_lock_var.set(False)

    def _show_voice_result(self, similarity: float | None) -> None:
        if similarity is not None:
            self.voice_match_var.set(f"Speaker match: {similarity:.3f}")

    def _update_voice_metrics(
        self,
        input_db: float,
        output_db: float,
        dropped: int,
        underruns: int,
        gain: float,
        similarity: float | None,
    ) -> None:
        # Preserve the base meters/stats, then add Voice Lock telemetry.
        super()._update_metrics(input_db, output_db, dropped, underruns)
        gain_db = 20.0 * math.log10(max(gain, 1e-4))
        match = "--" if similarity is None else f"{similarity:.3f}"
        self.voice_match_var.set(f"Match: {match}    Voice gain: {gain_db:.1f} dB")


def main() -> None:
    app = VoiceLockApp()
    app.mainloop()


if __name__ == "__main__":
    main()
