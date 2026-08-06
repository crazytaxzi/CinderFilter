from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd
from deepfilternet_rs import DeepFilterNetRealtime

import tkinter as tk
from tkinter import messagebox, ttk

SAMPLE_RATE = 48_000
MODEL_FRAME_SIZE = 480  # DeepFilterNet internally consumes 10 ms frames at 48 kHz.


@dataclass(frozen=True)
class DeviceChoice:
    index: int
    label: str
    channels: int


class AudioEngine:
    """Real-time mic -> DeepFilterNet -> selected playback/virtual-cable output."""

    STRENGTHS = {
        "Natural": 20.0,
        "Balanced": 45.0,
        "Strong": 70.0,
        "Maximum": 100.0,
    }

    def __init__(
        self,
        on_status: Callable[[str], None],
        on_metrics: Callable[[float, float, int, int], None],
    ) -> None:
        self.on_status = on_status
        self.on_metrics = on_metrics
        self._running = threading.Event()
        self._bypass = threading.Event()
        self._input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self._output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self._worker: threading.Thread | None = None
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        # DeepFilterNetRealtime is a PyO3 unsendable object. It must be created,
        # used, and destroyed on the same worker thread. Never store it here.
        self._model_ready = threading.Event()
        self._worker_error: BaseException | None = None
        self._atten_lim = self.STRENGTHS["Balanced"]
        self._output_channels = 2
        self._dropped_input = 0
        self._output_underruns = 0
        self._input_level = 0.0
        self._output_level = 0.0
        self._pending_output = np.empty(0, dtype=np.float32)
        self._last_status = ""

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def set_bypass(self, enabled: bool) -> None:
        if enabled:
            self._bypass.set()
        else:
            self._bypass.clear()

    def start(self, input_device: int, output_device: int, strength: str) -> None:
        if self.running:
            self.stop()

        self._clear_queues()
        self._dropped_input = 0
        self._output_underruns = 0
        self._pending_output = np.empty(0, dtype=np.float32)

        out_info = sd.query_devices(output_device)
        max_out = int(out_info["max_output_channels"])
        if max_out < 1:
            raise RuntimeError("The selected output device has no playback channels.")
        self._output_channels = 2 if max_out >= 2 else 1

        self._atten_lim = self.STRENGTHS.get(strength, self.STRENGTHS["Balanced"])
        self._model_ready.clear()
        self._worker_error = None
        self.on_status("Loading DeepFilterNet model on AI thread...")

        self._running.set()
        self._worker = threading.Thread(
            target=self._process_loop,
            name="CinderFilter-AI",
            daemon=True,
        )
        self._worker.start()

        # PyO3 marks DeepFilterNetRealtime as unsendable. The worker constructs it
        # and signals us only after the model is ready on that same thread.
        if not self._model_ready.wait(timeout=30.0):
            self.stop()
            raise RuntimeError("DeepFilterNet model loading timed out.")
        if self._worker_error is not None:
            error = self._worker_error
            self.stop()
            raise RuntimeError(f"DeepFilterNet failed to initialize: {error}") from error
        if not self._running.is_set():
            self.stop()
            raise RuntimeError("The AI worker stopped before audio streams could start.")

        try:
            self._output_stream = sd.OutputStream(
                device=output_device,
                samplerate=SAMPLE_RATE,
                blocksize=0,
                channels=self._output_channels,
                dtype="float32",
                latency="low",
                callback=self._output_callback,
            )
            self._input_stream = sd.InputStream(
                device=input_device,
                samplerate=SAMPLE_RATE,
                blocksize=0,
                channels=1,
                dtype="float32",
                latency="low",
                callback=self._input_callback,
            )
            # Capture first so the AI worker can prime a couple of frames before
            # the output clock begins. This avoids a burst of silence/underruns.
            self._input_stream.start()
            prime_deadline = time.monotonic() + 0.15
            while self._output_queue.qsize() < 2 and time.monotonic() < prime_deadline:
                time.sleep(0.005)
            self._output_stream.start()
        except Exception:
            self.stop()
            raise

        self.on_status("LIVE — AI filtering active")

    def stop(self) -> None:
        self._running.clear()

        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass

        self._input_stream = None
        self._output_stream = None

        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.5)
        self._worker = None

        # The processor is closed by _process_loop on the worker thread that
        # created it. Closing it here would trigger the same PyO3 thread panic.
        self._clear_queues()
        self.on_status("Stopped")

    def _clear_queues(self) -> None:
        for q in (self._input_queue, self._output_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _input_callback(self, indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            self._last_status = f"Input: {status}"

        mono = np.ascontiguousarray(indata[:frames, 0], dtype=np.float32)
        self._input_level = self._peak_db(mono)

        try:
            self._input_queue.put_nowait(mono.copy())
        except queue.Full:
            self._dropped_input += 1
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(mono.copy())
            except queue.Empty:
                pass

    def _process_loop(self) -> None:
        processor: DeepFilterNetRealtime | None = None
        last_metrics = 0.0

        try:
            # DeepFilterNetRealtime is declared unsendable by PyO3. Construct,
            # process, and close it entirely inside this worker thread.
            processor = DeepFilterNetRealtime(
                model_path=None,
                atten_lim=self._atten_lim,
                log_level="warn",
                # Streaming must preserve a continuous timeline. Delay
                # compensation is intended for files and can shorten startup.
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
                    block = self._input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._bypass.is_set():
                    processed = block
                else:
                    processed = processor.process_chunk(block)
                    processed = np.asarray(processed, dtype=np.float32).reshape(-1)

                if processed.size == 0:
                    continue
                processed = np.clip(processed, -1.0, 1.0)
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
                    )
                    last_metrics = now
        except BaseException as exc:
            # PyO3 PanicException derives from BaseException, not necessarily
            # Exception, so catch it explicitly and report it to the UI.
            self._worker_error = exc
            self._last_status = f"AI processing error: {exc}"
            self.on_status(self._last_status)
            self._running.clear()
        finally:
            # Always release the Rust object on its creator thread.
            if processor is not None:
                try:
                    processor.close()
                except BaseException:
                    pass
            self._model_ready.set()

    def _output_callback(self, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            self._last_status = f"Output: {status}"

        while self._pending_output.size < frames:
            try:
                next_block = self._output_queue.get_nowait()
                self._pending_output = np.concatenate((self._pending_output, next_block))
            except queue.Empty:
                break

        available = min(frames, self._pending_output.size)
        mono = np.zeros(frames, dtype=np.float32)
        if available:
            mono[:available] = self._pending_output[:available]
            self._pending_output = self._pending_output[available:]
        if available < frames:
            self._output_underruns += 1

        if self._output_channels == 1:
            outdata[:, 0] = mono
        else:
            outdata[:, :] = mono[:, None]

    @staticmethod
    def _peak_db(audio: np.ndarray) -> float:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return max(-60.0, 20.0 * np.log10(max(peak, 1e-6)))


class CinderFilterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CinderFilter")
        self.geometry("760x470")
        self.minsize(690, 430)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.input_choices: dict[str, DeviceChoice] = {}
        self.output_choices: dict[str, DeviceChoice] = {}

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.strength_var = tk.StringVar(value="Balanced")
        self.bypass_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Stopped")
        self.input_meter_var = tk.DoubleVar(value=0.0)
        self.output_meter_var = tk.DoubleVar(value=0.0)
        self.stats_var = tk.StringVar(value="Dropped input: 0    Output underruns: 0")

        self._ui_events: queue.Queue[tuple] = queue.Queue()
        self.engine = AudioEngine(self._thread_status, self._thread_metrics)
        self._build_ui()
        self.refresh_devices()
        self.after(50, self._drain_ui_events)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        shell = ttk.Frame(self, padding=18)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(1, weight=1)

        title = ttk.Label(shell, text="CinderFilter", font=("Segoe UI", 24, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        subtitle = ttk.Label(
            shell,
            text="Physical microphone → weighted DeepFilterNet AI → virtual or hardware output",
        )
        subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 18))

        ttk.Label(shell, text="Input device").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.input_combo = ttk.Combobox(shell, textvariable=self.input_var, state="readonly")
        self.input_combo.grid(row=2, column=1, sticky="ew")

        ttk.Label(shell, text="Output device").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=9)
        self.output_combo = ttk.Combobox(shell, textvariable=self.output_var, state="readonly")
        self.output_combo.grid(row=3, column=1, sticky="ew", pady=9)

        refresh = ttk.Button(shell, text="Refresh devices", command=self.refresh_devices)
        refresh.grid(row=2, column=2, rowspan=2, sticky="ns", padx=(10, 0))

        ttk.Label(shell, text="AI strength").grid(row=4, column=0, sticky="w", padx=(0, 10))
        self.strength_combo = ttk.Combobox(
            shell,
            textvariable=self.strength_var,
            values=list(AudioEngine.STRENGTHS),
            state="readonly",
            width=18,
        )
        self.strength_combo.grid(row=4, column=1, sticky="w")

        bypass = ttk.Checkbutton(
            shell,
            text="Bypass AI (routing test)",
            variable=self.bypass_var,
            command=self._toggle_bypass,
        )
        bypass.grid(row=4, column=2, sticky="e")

        separator = ttk.Separator(shell)
        separator.grid(row=5, column=0, columnspan=3, sticky="ew", pady=18)

        ttk.Label(shell, text="Input peak").grid(row=6, column=0, sticky="w")
        self.input_meter = ttk.Progressbar(shell, maximum=60, variable=self.input_meter_var)
        self.input_meter.grid(row=6, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(shell, text="Output peak").grid(row=7, column=0, sticky="w")
        self.output_meter = ttk.Progressbar(shell, maximum=60, variable=self.output_meter_var)
        self.output_meter.grid(row=7, column=1, columnspan=2, sticky="ew", pady=4)

        status_box = ttk.LabelFrame(shell, text="Status", padding=12)
        status_box.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(16, 12))
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(status_box, textvariable=self.stats_var).grid(row=1, column=0, sticky="w", pady=(5, 0))

        buttons = ttk.Frame(shell)
        buttons.grid(row=9, column=0, columnspan=3, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(buttons, text="START FILTERING", command=self.start_filtering)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5), ipady=8)
        self.stop_button = ttk.Button(buttons, text="STOP", command=self.stop_filtering, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0), ipady=8)

        hint = ttk.Label(
            shell,
            text=(
                "To change routes while live, pick new devices and press START FILTERING again. "
                "CinderFilter safely restarts the audio streams."
            ),
            wraplength=700,
        )
        hint.grid(row=10, column=0, columnspan=3, sticky="w", pady=(13, 0))

    def refresh_devices(self) -> None:
        try:
            devices = sd.query_devices()
            self.input_choices.clear()
            self.output_choices.clear()

            for index, device in enumerate(devices):
                try:
                    host = sd.query_hostapis(int(device["hostapi"]))["name"]
                except Exception:
                    host = "Unknown API"
                name = str(device["name"])

                if int(device["max_input_channels"]) > 0:
                    label = f"{index}: {name}  [{host}]"
                    self.input_choices[label] = DeviceChoice(
                        index=index,
                        label=label,
                        channels=int(device["max_input_channels"]),
                    )
                if int(device["max_output_channels"]) > 0:
                    label = f"{index}: {name}  [{host}]"
                    self.output_choices[label] = DeviceChoice(
                        index=index,
                        label=label,
                        channels=int(device["max_output_channels"]),
                    )

            self.input_combo["values"] = list(self.input_choices)
            self.output_combo["values"] = list(self.output_choices)

            defaults = sd.default.device
            default_input = int(defaults[0]) if defaults and defaults[0] is not None else -1
            default_output = int(defaults[1]) if defaults and defaults[1] is not None else -1

            self._select_index(self.input_var, self.input_choices, default_input)
            self._select_index(self.output_var, self.output_choices, default_output)
            self.status_var.set(
                f"Found {len(self.input_choices)} inputs and {len(self.output_choices)} outputs"
            )
        except Exception as exc:
            messagebox.showerror("Device scan failed", str(exc))

    @staticmethod
    def _select_index(var: tk.StringVar, choices: dict[str, DeviceChoice], index: int) -> None:
        current = var.get()
        if current in choices:
            return
        for label, choice in choices.items():
            if choice.index == index:
                var.set(label)
                return
        if choices:
            var.set(next(iter(choices)))

    def start_filtering(self) -> None:
        input_choice = self.input_choices.get(self.input_var.get())
        output_choice = self.output_choices.get(self.output_var.get())
        if input_choice is None or output_choice is None:
            messagebox.showwarning("Choose devices", "Select both an input and an output device.")
            return
        if input_choice.index == output_choice.index:
            messagebox.showwarning(
                "Bad route",
                "Input and output resolve to the same endpoint. Choose a playback/virtual-cable output.",
            )
            return

        self.start_button.configure(state="disabled")
        self.status_var.set("Starting...")
        threading.Thread(
            target=self._start_worker,
            args=(
                input_choice.index,
                output_choice.index,
                self.strength_var.get(),
                self.bypass_var.get(),
            ),
            daemon=True,
        ).start()

    def _start_worker(
        self, input_index: int, output_index: int, strength: str, bypass: bool
    ) -> None:
        try:
            self.engine.set_bypass(bypass)
            self.engine.start(input_index, output_index, strength)
            self._ui_events.put(("running", True))
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self._ui_events.put(("error", details))

    def _show_start_error(self, details: str) -> None:
        self._set_running_ui(False)
        self.status_var.set("Failed to start")
        messagebox.showerror(
            "CinderFilter could not start",
            details
            + "\n\nTry another version of the same device under a different host API "
            + "(WASAPI, WDM-KS, DirectSound, or MME).",
        )

    def stop_filtering(self) -> None:
        self.stop_button.configure(state="disabled")
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        self.engine.stop()
        self._ui_events.put(("running", False))

    def _set_running_ui(self, running: bool) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _toggle_bypass(self) -> None:
        self.engine.set_bypass(self.bypass_var.get())

    def _thread_status(self, text: str) -> None:
        self._ui_events.put(("status", text))

    def _thread_metrics(self, input_db: float, output_db: float, dropped: int, underruns: int) -> None:
        self._ui_events.put(("metrics", input_db, output_db, dropped, underruns))

    def _drain_ui_events(self) -> None:
        try:
            while True:
                event = self._ui_events.get_nowait()
                kind = event[0]
                if kind == "status":
                    self.status_var.set(event[1])
                elif kind == "metrics":
                    self._update_metrics(*event[1:])
                elif kind == "running":
                    self._set_running_ui(bool(event[1]))
                elif kind == "error":
                    self._show_start_error(str(event[1]))
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self._drain_ui_events)

    def _update_metrics(self, input_db: float, output_db: float, dropped: int, underruns: int) -> None:
        self.input_meter_var.set(max(0.0, min(60.0, input_db + 60.0)))
        self.output_meter_var.set(max(0.0, min(60.0, output_db + 60.0)))
        self.stats_var.set(f"Dropped input: {dropped}    Output underruns: {underruns}")

    def _on_close(self) -> None:
        try:
            self.engine.stop()
        finally:
            self.destroy()


def main() -> None:
    app = CinderFilterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
