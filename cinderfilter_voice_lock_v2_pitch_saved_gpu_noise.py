from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from deepfilternet_rs import DeepFilterNetRealtime

import cinderfilter_voice_lock_v2_pitch as pitch
import cinderfilter_voice_lock_v2_pitch_saved_gpu as gpu
from cuda_noise_bridge import CudaNoiseBridge, CudaNoiseMetrics, CUDA_NOISE_PRESETS
from voice_lock import SAMPLE_RATE, VoiceLockedAudioEngine

_CREATE_NEW_CONSOLE = 0x00000010 if os.name == "nt" else 0


class CudaMainNoiseAudioEngine(pitch.PitchLockedAudioEngine):
    """Pitch-Locked engine with a selectable CPU or CUDA main denoiser."""

    def __init__(self, voice_service, separator, cuda_noise, on_status, on_metrics) -> None:
        super().__init__(voice_service, separator, on_status, on_metrics)
        self.cuda_noise = cuda_noise
        self._noise_backend = "CUDA"
        self._noise_preset = "Low Latency"
        self._cuda_noise_active = False

    def configure_noise_backend(self, backend: str, preset: str) -> None:
        self._noise_backend = backend if backend in {"CUDA", "Auto", "CPU"} else "Auto"
        self._noise_preset = preset if preset in CUDA_NOISE_PRESETS else "Low Latency"

    def start(self, input_device: int, output_device: int, strength: str) -> None:
        self._cuda_noise_active = False
        if self._noise_backend in {"CUDA", "Auto"}:
            try:
                atten = self.STRENGTHS.get(strength, self.STRENGTHS["Balanced"])
                self.cuda_noise.start(atten, self._noise_preset)
                self._cuda_noise_active = True
            except BaseException as exc:
                if self._noise_backend == "CUDA":
                    raise RuntimeError(f"CUDA main noise reducer could not start: {exc}") from exc
                self.on_status(f"CUDA main denoiser unavailable; using CPU: {exc}")
        super().start(input_device, output_device, strength)

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            self.cuda_noise.stop()
            self._cuda_noise_active = False

    def _new_cpu_processor(self) -> DeepFilterNetRealtime:
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
        return processor

    def _process_loop(self) -> None:
        processor: DeepFilterNetRealtime | None = None
        last_metrics = 0.0
        try:
            if not self._cuda_noise_active:
                processor = self._new_cpu_processor()
                self.on_status("Main noise reducer: CPU Rust DeepFilterNet")
            else:
                self.on_status(
                    f"Main noise reducer: CUDA DeepFilterNet3 on {self.cuda_noise.device}"
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
                elif self._cuda_noise_active:
                    cuda_output = self.cuda_noise.process_block(source)
                    if cuda_output is None:
                        self._cuda_noise_active = False
                        self.on_status(
                            "CUDA main noise reducer failed — loading CPU fallback to keep the mic alive"
                        )
                        processor = self._new_cpu_processor()
                        processed = np.asarray(
                            processor.process_chunk(source), dtype=np.float32
                        ).reshape(-1)
                    else:
                        processed = np.asarray(cuda_output, dtype=np.float32).reshape(-1)
                else:
                    if processor is None:
                        processor = self._new_cpu_processor()
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


class CudaMainNoiseApp(gpu.GpuAwarePitchLockApp):
    """Full CinderFilter UI with a GPU/CPU selector for the main denoiser."""

    def __init__(self) -> None:
        super().__init__()
        self.title("CinderFilter — CUDA Main Noise Reducer")
        self.geometry("980x1240")
        self.minsize(900, 940)

        loaded = getattr(self, "_loaded_settings", {})
        backend = str(loaded.get("noise_backend", "CUDA"))
        preset = str(loaded.get("noise_cuda_preset", "Low Latency"))
        self.noise_backend_var = tk.StringVar(
            value=backend if backend in {"CUDA", "Auto", "CPU"} else "CUDA"
        )
        self.noise_preset_var = tk.StringVar(
            value=preset if preset in CUDA_NOISE_PRESETS else "Low Latency"
        )
        self.noise_status_var = tk.StringVar(value="CUDA main noise reducer not loaded")
        self.noise_metrics_var = tk.StringVar(
            value="Main denoiser RTF: --    Queue: --    Backend: --"
        )
        self._noise_events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.cuda_noise = CudaNoiseBridge(
            self._thread_noise_status,
            self._thread_noise_metrics,
        )
        self.engine = CudaMainNoiseAudioEngine(
            self.voice_service,
            self.separator,
            self.cuda_noise,
            self._thread_status,
            self._thread_metrics,
        )

        self._add_noise_engine_ui()
        self.noise_backend_var.trace_add("write", self._noise_setting_changed)
        self.noise_preset_var.trace_add("write", self._noise_setting_changed)
        self.after(100, self._drain_noise_events)
        self._refresh_noise_install_state()

    def _add_noise_engine_ui(self) -> None:
        frame = ttk.LabelFrame(
            self,
            text="Main Noise Reducer — CPU Rust or CUDA DeepFilterNet3",
            padding=12,
        )
        frame.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 18))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Noise engine").grid(row=0, column=0, sticky="w")
        backend = ttk.Combobox(
            frame,
            textvariable=self.noise_backend_var,
            values=("CUDA", "Auto", "CPU"),
            state="readonly",
            width=14,
        )
        backend.grid(row=0, column=1, sticky="w", padx=(8, 16))

        ttk.Label(frame, text="CUDA latency").grid(row=0, column=2, sticky="e")
        preset = ttk.Combobox(
            frame,
            textvariable=self.noise_preset_var,
            values=tuple(CUDA_NOISE_PRESETS),
            state="readonly",
            width=14,
        )
        preset.grid(row=0, column=3, sticky="e", padx=(8, 0))

        ttk.Label(
            frame,
            textvariable=self.noise_status_var,
            font=("Segoe UI", 10, "bold"),
            wraplength=900,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(frame, textvariable=self.noise_metrics_var).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 8)
        )

        self.install_noise_button = ttk.Button(
            frame,
            text="Install / Repair CUDA Noise Engine",
            command=self.install_cuda_noise_engine,
        )
        self.install_noise_button.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Button(
            frame,
            text="Preload CUDA Denoiser",
            command=self.preload_cuda_noise,
        ).grid(row=3, column=2, columnspan=2, sticky="e")

        ttk.Label(
            frame,
            text=(
                "CUDA runs the actual DeepFilterNet3 noise model on the RTX GPU. CPU keeps the "
                "current Rust path. Auto prefers CUDA and falls back to CPU if the sidecar is unavailable."
            ),
            wraplength=900,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def _refresh_noise_install_state(self) -> None:
        if self.cuda_noise.installed:
            self.noise_status_var.set(
                "CUDA noise sidecar installed — select CUDA and preload or start filtering"
            )
            self.install_noise_button.configure(text="Repair CUDA Noise Engine")
        else:
            self.noise_status_var.set(
                "CUDA noise sidecar is not installed yet — CPU denoiser remains available"
            )
            self.install_noise_button.configure(text="Install CUDA Noise Engine")

    def _noise_setting_changed(self, *_args) -> None:
        if not getattr(self, "_restoring_settings", False):
            self._schedule_save()
        if self.engine.running:
            self.noise_status_var.set("Restart filtering to apply the main noise-engine change")

    def _settings_snapshot(self):
        values = super()._settings_snapshot()
        values["noise_backend"] = self.noise_backend_var.get()
        values["noise_cuda_preset"] = self.noise_preset_var.get()
        return values

    def _configure_noise_engine(self) -> None:
        self.engine.configure_noise_backend(
            self.noise_backend_var.get(),
            self.noise_preset_var.get(),
        )

    def start_filtering(self) -> None:
        self._configure_noise_engine()
        self._save_settings_now()
        super().start_filtering()

    def preload_cuda_noise(self) -> None:
        if self.engine.running:
            messagebox.showinfo("Already live", "Stop filtering before preloading another noise backend.")
            return
        threading.Thread(target=self._preload_cuda_noise_worker, daemon=True).start()

    def _preload_cuda_noise_worker(self) -> None:
        try:
            self._noise_events.put(("status", "Loading DeepFilterNet3 on CUDA..."))
            strength = self.engine.STRENGTHS.get(
                self.strength_var.get(), self.engine.STRENGTHS["Balanced"]
            )
            info = self.cuda_noise.start(strength, self.noise_preset_var.get())
            self._noise_events.put(
                (
                    "status",
                    f"CUDA main denoiser ready — {info.get('device', 'CUDA')} / {info.get('model', 'DeepFilterNet3')}",
                )
            )
        except BaseException as exc:
            self._noise_events.put(("status", f"CUDA main denoiser preload failed: {exc}"))

    def install_cuda_noise_engine(self) -> None:
        script = Path(__file__).resolve().parent / "install-cuda-noise-engine.ps1"
        if not script.exists():
            messagebox.showerror("Installer missing", f"Missing file:\n{script}")
            return
        confirmed = messagebox.askyesno(
            "Install CUDA main noise reducer",
            (
                "This creates a dedicated Python 3.11 CUDA environment for the official "
                "DeepFilterNet3 GPU backend. It is a large one-time download and keeps the "
                "working CPU engine untouched.\n\nContinue?"
            ),
        )
        if not confirmed:
            return
        self._save_settings_now()
        try:
            subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-AppRoot",
                    str(Path(__file__).resolve().parent),
                    "-ParentPid",
                    str(os.getpid()),
                ],
                cwd=str(Path(__file__).resolve().parent),
                creationflags=_CREATE_NEW_CONSOLE,
            )
        except BaseException as exc:
            messagebox.showerror("Could not launch installer", str(exc))
            return
        self.status_var.set("Closing for CUDA main noise-engine installation...")
        self.after(300, self._on_close)

    def _thread_noise_status(self, text: str) -> None:
        self._noise_events.put(("status", text))

    def _thread_noise_metrics(self, metrics: CudaNoiseMetrics) -> None:
        self._noise_events.put(("metrics", metrics))

    def _drain_noise_events(self) -> None:
        try:
            while True:
                kind, payload = self._noise_events.get_nowait()
                if kind == "status":
                    self.noise_status_var.set(str(payload))
                elif kind == "metrics" and isinstance(payload, CudaNoiseMetrics):
                    suffix = "    CPU FALLBACK" if payload.fallback else ""
                    self.noise_metrics_var.set(
                        f"Main denoiser RTF: {payload.realtime_factor:.2f}    "
                        f"Queue: {payload.queue_depth}    Backend: {payload.backend}    "
                        f"Device: {payload.device}{suffix}"
                    )
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._drain_noise_events)


def main() -> None:
    app = CudaMainNoiseApp()
    app.mainloop()


if __name__ == "__main__":
    main()
