from __future__ import annotations

import json
import os
import queue
import secrets
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Callable

import numpy as np

AUDIO_RATE = 48_000
REQUEST_HEADER = struct.Struct("<II")
RESPONSE_HEADER = struct.Struct("<IIf")
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass(frozen=True)
class CudaNoisePreset:
    name: str
    chunk_seconds: float
    hop_seconds: float


CUDA_NOISE_PRESETS: dict[str, CudaNoisePreset] = {
    "Low Latency": CudaNoisePreset("Low Latency", 0.20, 0.10),
    "Balanced": CudaNoisePreset("Balanced", 0.40, 0.20),
    "Quality": CudaNoisePreset("Quality", 0.80, 0.40),
}


@dataclass(frozen=True)
class CudaNoiseMetrics:
    processing_seconds: float
    realtime_factor: float
    queue_depth: int
    device: str
    backend: str
    fallback: bool


class CudaNoiseBridge:
    """Nonblocking audio bridge to a Python 3.11 CUDA DeepFilterNet3 sidecar."""

    def __init__(
        self,
        on_status: Callable[[str], None],
        on_metrics: Callable[[CudaNoiseMetrics], None],
        app_dir: Path | None = None,
    ) -> None:
        self.app_dir = app_dir or Path(__file__).resolve().parent
        self.on_status = on_status
        self.on_metrics = on_metrics

        self._process: subprocess.Popen | None = None
        self._connection = None
        self._listener: Listener | None = None
        self._io_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._tasks: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=2)
        self._results: queue.Queue[tuple[int, np.ndarray, float]] = queue.Queue(maxsize=3)
        self._state_lock = threading.Lock()

        self._preset = CUDA_NOISE_PRESETS["Low Latency"]
        self._strength = 45.0
        self._device = "not loaded"
        self._running = False
        self._failed = False
        self._last_error = ""
        self._input_buffer = np.empty(0, np.float32)
        self._output_buffer = np.empty(0, np.float32)
        self._previous_tail: np.ndarray | None = None
        self._sequence = 0
        self._overload_strikes = 0

    @property
    def sidecar_python(self) -> Path:
        return self.app_dir / ".venv_cuda_noise" / "Scripts" / "python.exe"

    @property
    def worker_script(self) -> Path:
        return self.app_dir / "cuda_noise_worker.py"

    @property
    def installed(self) -> bool:
        return self.sidecar_python.exists() and self.worker_script.exists()

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running and not self._failed

    @property
    def failed(self) -> bool:
        with self._state_lock:
            return self._failed

    @property
    def device(self) -> str:
        with self._state_lock:
            return self._device

    @property
    def last_error(self) -> str:
        with self._state_lock:
            return self._last_error

    @property
    def nominal_latency_seconds(self) -> float:
        return self._preset.chunk_seconds

    def start(self, strength: float, preset_name: str, timeout: float = 240.0) -> dict:
        preset = CUDA_NOISE_PRESETS.get(preset_name, CUDA_NOISE_PRESETS["Low Latency"])
        if self.running and self._preset == preset and abs(self._strength - strength) < 0.01:
            return {"device": self.device, "model": "DeepFilterNet3"}

        self.stop()
        if not self.installed:
            raise RuntimeError(
                "CUDA Noise Engine is not installed. Run INSTALL_CUDA_NOISE_ENGINE.bat first."
            )

        self._preset = preset
        self._strength = float(strength)
        self._reset_buffers()
        self._stop_event.clear()
        with self._state_lock:
            self._failed = False
            self._last_error = ""
            self._running = False

        pipe_name = rf"\\.\pipe\CinderFilterNoise_{os.getpid()}_{uuid.uuid4().hex}"
        auth = secrets.token_bytes(24)
        listener = Listener(pipe_name, family="AF_PIPE", authkey=auth)
        self._listener = listener

        log_dir = Path(os.environ.get("LOCALAPPDATA", str(self.app_dir))) / "CinderFilter"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "cuda-noise-worker.log", "a", encoding="utf-8")
        command = [
            str(self.sidecar_python),
            str(self.worker_script),
            "--pipe",
            pipe_name,
            "--auth",
            auth.hex(),
            "--atten",
            str(self._strength),
        ]
        self.on_status("Loading DeepFilterNet3 on the CUDA noise sidecar...")
        self._process = subprocess.Popen(
            command,
            cwd=str(self.app_dir),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            creationflags=_CREATE_NO_WINDOW,
        )

        accepted: queue.Queue = queue.Queue(maxsize=1)

        def accept_connection() -> None:
            try:
                accepted.put((listener.accept(), None))
            except BaseException as exc:
                accepted.put((None, exc))

        threading.Thread(target=accept_connection, daemon=True).start()
        deadline = time.monotonic() + timeout
        conn = None
        error = None
        while time.monotonic() < deadline:
            try:
                conn, error = accepted.get(timeout=0.25)
                break
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(
                        f"CUDA noise worker exited during startup. Check {log_dir / 'cuda-noise-worker.log'}"
                    )
        if conn is None:
            self.stop()
            if error is not None:
                raise RuntimeError(f"CUDA noise worker connection failed: {error}") from error
            raise RuntimeError("CUDA noise worker model loading timed out")

        self._connection = conn
        try:
            info = json.loads(conn.recv_bytes().decode("utf-8"))
        except BaseException as exc:
            self.stop()
            raise RuntimeError(f"CUDA noise worker sent an invalid handshake: {exc}") from exc
        if info.get("status") != "ready":
            self.stop()
            raise RuntimeError(f"CUDA noise worker did not become ready: {info}")

        with self._state_lock:
            self._device = str(info.get("device", "CUDA"))
            self._running = True
            self._failed = False
        self._io_thread = threading.Thread(
            target=self._io_loop,
            name="CinderFilter-CUDA-Noise-IO",
            daemon=True,
        )
        self._io_thread.start()
        self.on_status(
            f"CUDA main noise reducer ready — {self.device}, DeepFilterNet3, {preset.name}"
        )
        return info

    def stop(self) -> None:
        self._stop_event.set()
        conn = self._connection
        if conn is not None:
            try:
                conn.send_bytes(REQUEST_HEADER.pack(0, 0))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        self._connection = None

        if self._io_thread and self._io_thread.is_alive():
            self._io_thread.join(timeout=2.0)
        self._io_thread = None

        if self._process is not None and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._process = None

        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass
        self._listener = None
        with self._state_lock:
            self._running = False
        self._clear_queue(self._tasks)
        self._clear_queue(self._results)
        self._reset_buffers()

    def process_block(self, audio: np.ndarray) -> np.ndarray | None:
        if not self.running:
            return None
        block = np.asarray(audio, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return block.copy()

        self._input_buffer = np.concatenate((self._input_buffer, block))
        chunk = int(round(self._preset.chunk_seconds * AUDIO_RATE))
        hop = int(round(self._preset.hop_seconds * AUDIO_RATE))

        while self._input_buffer.size >= chunk:
            item = np.ascontiguousarray(self._input_buffer[:chunk], dtype=np.float32)
            try:
                self._tasks.put_nowait((self._sequence, item))
                self._sequence += 1
                self._overload_strikes = max(0, self._overload_strikes - 1)
            except queue.Full:
                self._overload_strikes += 1
                if self._overload_strikes >= 3:
                    self._fail("CUDA noise engine cannot keep up with real-time audio")
                    return None
            self._input_buffer = self._input_buffer[hop:]
            if self._overload_strikes:
                break

        self._drain_results()
        if self.failed:
            return None
        needed = block.size
        if self._output_buffer.size < needed:
            return np.zeros(needed, np.float32)
        output = np.ascontiguousarray(self._output_buffer[:needed], dtype=np.float32)
        self._output_buffer = self._output_buffer[needed:]
        return output

    def _io_loop(self) -> None:
        conn = self._connection
        if conn is None:
            return
        while not self._stop_event.is_set():
            try:
                sequence, audio = self._tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                conn.send_bytes(
                    REQUEST_HEADER.pack(sequence, audio.size) + audio.astype("<f4", copy=False).tobytes()
                )
                payload = conn.recv_bytes()
                if len(payload) < RESPONSE_HEADER.size:
                    raise RuntimeError("Short response from CUDA noise worker")
                returned_sequence, count, elapsed = RESPONSE_HEADER.unpack_from(payload, 0)
                expected = RESPONSE_HEADER.size + count * 4
                if len(payload) != expected:
                    raise RuntimeError(
                        f"Bad CUDA noise response length: got {len(payload)}, expected {expected}"
                    )
                output = np.frombuffer(
                    payload,
                    dtype="<f4",
                    offset=RESPONSE_HEADER.size,
                    count=count,
                ).copy()
                try:
                    self._results.put_nowait((returned_sequence, output, float(elapsed)))
                except queue.Full:
                    try:
                        self._results.get_nowait()
                        self._results.put_nowait((returned_sequence, output, float(elapsed)))
                    except queue.Empty:
                        pass
            except BaseException as exc:
                if not self._stop_event.is_set():
                    self._fail(f"CUDA noise worker failed: {type(exc).__name__}: {exc}")
                return

    def _drain_results(self) -> None:
        while True:
            try:
                _sequence, audio, elapsed = self._results.get_nowait()
            except queue.Empty:
                return
            self._append_overlap(audio)
            rtf = elapsed / max(self._preset.hop_seconds, 1e-6)
            if rtf > 1.15:
                self._overload_strikes += 1
            elif rtf < 0.90:
                self._overload_strikes = max(0, self._overload_strikes - 1)
            if self._overload_strikes >= 3:
                self._fail("CUDA main noise reducer is slower than the selected hop window")
            self.on_metrics(
                CudaNoiseMetrics(
                    processing_seconds=elapsed,
                    realtime_factor=rtf,
                    queue_depth=self._tasks.qsize(),
                    device=self.device,
                    backend="CUDA DeepFilterNet3",
                    fallback=self.failed,
                )
            )

    def _append_overlap(self, target: np.ndarray) -> None:
        hop = int(round(self._preset.hop_seconds * AUDIO_RATE))
        needed = hop * 2
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if target.size < needed:
            target = np.pad(target, (0, needed - target.size))
        elif target.size > needed:
            target = target[:needed]
        first = target[:hop]
        second = target[hop:needed]
        if self._previous_tail is None:
            emitted = first
        else:
            fade = np.linspace(0.0, 1.0, hop, endpoint=False, dtype=np.float32)
            emitted = self._previous_tail * (1.0 - fade) + first * fade
        self._previous_tail = second.copy()
        self._output_buffer = np.concatenate((self._output_buffer, emitted.astype(np.float32)))

    def _fail(self, message: str) -> None:
        with self._state_lock:
            if self._failed:
                return
            self._failed = True
            self._last_error = message
        self.on_status(message)
        self.on_metrics(
            CudaNoiseMetrics(0.0, 0.0, self._tasks.qsize(), self.device, "CUDA", True)
        )

    def _reset_buffers(self) -> None:
        self._input_buffer = np.empty(0, np.float32)
        self._output_buffer = np.empty(0, np.float32)
        self._previous_tail = None
        self._sequence = 0
        self._overload_strikes = 0

    @staticmethod
    def _clear_queue(items: queue.Queue) -> None:
        while True:
            try:
                items.get_nowait()
            except queue.Empty:
                return
