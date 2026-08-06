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
from typing import Callable, IO

import numpy as np

AUDIO_RATE = 48_000
REQUEST_HEADER = struct.Struct("<II")
RESPONSE_HEADER = struct.Struct("<IIf")
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass(frozen=True)
class CudaPreset:
    name: str
    chunk_seconds: float


CUDA_PRESETS = {
    "Low Latency": CudaPreset("Low Latency", 0.25),
    "Balanced": CudaPreset("Balanced", 0.50),
    "Quality": CudaPreset("Quality", 1.00),
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
    """Supervised named-pipe bridge to the lookahead-safe CUDA worker."""

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
        self._tasks: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=16)
        self._results: queue.Queue[tuple[int, np.ndarray, float]] = queue.Queue(maxsize=16)
        self._state_lock = threading.Lock()
        self._log_handle: IO[str] | None = None
        self._preset = CUDA_PRESETS["Balanced"]
        self._strength = 45.0
        self._device = "not loaded"
        self._running = False
        self._failed = False
        self._last_error = ""
        self._input_buffer = np.empty(0, np.float32)
        self._output_buffer = np.empty(0, np.float32)
        self._sequence = 0
        self._ewma_rtf: float | None = None
        self._streaming_mode = "not loaded"
        self._latency_ms = 0.0
        self._future_samples = 0

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

    @property
    def log_path(self) -> Path:
        root = Path(os.environ.get("LOCALAPPDATA", str(self.app_dir))) / "CinderFilter"
        return root / "cuda-noise-worker.log"

    def start(self, strength: float, preset_name: str, timeout: float = 240.0) -> dict:
        preset = CUDA_PRESETS.get(preset_name, CUDA_PRESETS["Balanced"])
        if self.running and self._preset == preset and abs(self._strength - strength) < 0.01:
            return {"device": self.device, "model": "DeepFilterNet3", "streaming_mode": self._streaming_mode}
        self.stop()
        if not self.installed:
            raise RuntimeError("CUDA Noise Engine is not installed. Use Advanced -> Install / Repair CUDA Noise Engine.")

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
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_path, "a", encoding="utf-8", buffering=1)
        chunk_samples = int(round(preset.chunk_seconds * AUDIO_RATE))
        command = [
            str(self.sidecar_python),
            str(self.worker_script),
            "--pipe", pipe_name,
            "--auth", auth.hex(),
            "--atten", str(self._strength),
            "--chunk-samples", str(chunk_samples),
        ]
        self.on_status("Loading lookahead-safe DeepFilterNet3 CUDA worker...")
        self._process = subprocess.Popen(
            command,
            cwd=str(self.app_dir),
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=self._log_handle,
            creationflags=_CREATE_NO_WINDOW,
        )

        accepted: queue.Queue[tuple[object | None, BaseException | None]] = queue.Queue(maxsize=1)

        def accept() -> None:
            try:
                accepted.put((listener.accept(), None))
            except BaseException as exc:
                accepted.put((None, exc))

        threading.Thread(target=accept, daemon=True).start()
        deadline = time.monotonic() + timeout
        conn = None
        error = None
        while time.monotonic() < deadline:
            try:
                conn, error = accepted.get(timeout=0.25)
                break
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    code = self._process.returncode
                    self.stop()
                    raise RuntimeError(f"CUDA worker exited during startup (code {code}). Check {self.log_path}")
        if conn is None:
            self.stop()
            if error is not None:
                raise RuntimeError(f"CUDA worker connection failed: {error}") from error
            raise RuntimeError("CUDA worker model loading timed out")

        self._connection = conn
        try:
            if not conn.poll(30.0):
                raise TimeoutError("worker handshake timed out")
            info = json.loads(conn.recv_bytes().decode("utf-8"))
        except BaseException as exc:
            self.stop()
            raise RuntimeError(f"CUDA worker sent an invalid handshake: {exc}") from exc
        if info.get("status") != "ready" or not info.get("streaming"):
            self.stop()
            raise RuntimeError(f"CUDA worker did not become ready: {info}")

        with self._state_lock:
            self._device = str(info.get("device", "CUDA"))
            self._running = True
            self._streaming_mode = str(info.get("streaming_mode", "lookahead-context"))
            self._latency_ms = float(info.get("latency_ms", preset.chunk_seconds * 1000.0))
            self._future_samples = int(info.get("future_context_samples", 0))
        self._io_thread = threading.Thread(target=self._io_loop, name="CinderFilter-CUDA-IO", daemon=True)
        self._io_thread.start()
        self.on_status(f"CUDA main reducer ready — {self.device}, {preset.name}, {self._latency_ms:.0f} ms fixed delay")
        return info

    def stop(self) -> None:
        self._stop_event.set()
        self._clear_queue(self._tasks)
        try:
            self._tasks.put_nowait(None)
        except queue.Full:
            pass
        if self._io_thread and self._io_thread.is_alive():
            self._io_thread.join(3.0)
        self._io_thread = None
        conn, self._connection = self._connection, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(3.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        with self._state_lock:
            self._running = False
        self._clear_queue(self._tasks)
        self._clear_queue(self._results)
        self._reset_buffers()

    def process_block(self, audio: np.ndarray) -> np.ndarray | None:
        if not self.running:
            return None
        block = np.asarray(audio, np.float32).reshape(-1)
        if not block.size:
            return block.copy()
        self._input_buffer = np.concatenate((self._input_buffer, block))
        chunk = int(round(self._preset.chunk_seconds * AUDIO_RATE))
        request = chunk + self._future_samples
        while request > chunk and self._input_buffer.size >= request:
            item = np.ascontiguousarray(self._input_buffer[:request], np.float32)
            self._input_buffer = self._input_buffer[chunk:]
            task = (self._sequence, item)
            self._sequence += 1
            try:
                self._tasks.put_nowait(task)
            except queue.Full:
                self._fail("CUDA queue exceeded sixteen chunks; the worker is genuinely below real time")
                return None
        self._drain_results()
        if self.failed:
            return None
        if self._output_buffer.size < block.size:
            return np.zeros(block.size, np.float32)
        output = np.ascontiguousarray(self._output_buffer[: block.size], np.float32)
        self._output_buffer = self._output_buffer[block.size :]
        return output

    def _io_loop(self) -> None:
        conn = self._connection
        if conn is None:
            return
        timeout = max(8.0, self._preset.chunk_seconds * 16.0)
        while not self._stop_event.is_set():
            try:
                task = self._tasks.get(timeout=0.1)
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    self._fail(f"CUDA worker exited with code {self._process.returncode}")
                    return
                continue
            if task is None:
                try:
                    conn.send_bytes(REQUEST_HEADER.pack(0, 0))
                except Exception:
                    pass
                return
            sequence, audio = task
            try:
                conn.send_bytes(REQUEST_HEADER.pack(sequence, audio.size) + audio.astype("<f4", copy=False).tobytes())
                if not conn.poll(timeout):
                    raise TimeoutError(f"no response for {timeout:.1f}s")
                payload = conn.recv_bytes()
                if len(payload) < RESPONSE_HEADER.size:
                    raise RuntimeError("Short response from CUDA worker")
                returned, count, elapsed = RESPONSE_HEADER.unpack_from(payload, 0)
                expected = RESPONSE_HEADER.size + count * 4
                if len(payload) != expected:
                    raise RuntimeError(f"Bad response length: {len(payload)} != {expected}")
                output = np.frombuffer(payload, dtype="<f4", offset=RESPONSE_HEADER.size, count=count).copy()
                self._results.put_nowait((returned, output, float(elapsed)))
            except queue.Full:
                self._fail("CUDA result queue overflowed")
                return
            except BaseException as exc:
                if not self._stop_event.is_set():
                    self._fail(f"CUDA worker failed: {type(exc).__name__}: {exc}")
                return

    def _drain_results(self) -> None:
        while True:
            try:
                _sequence, audio, elapsed = self._results.get_nowait()
            except queue.Empty:
                return
            self._output_buffer = np.concatenate((self._output_buffer, np.asarray(audio, np.float32).reshape(-1)))
            maximum = int(AUDIO_RATE * max(3.0, self._preset.chunk_seconds * 6.0))
            if self._output_buffer.size > maximum:
                self._fail("CUDA output latency exceeded the safety bound")
                return
            rtf = elapsed / max(self._preset.chunk_seconds, 1e-6)
            self._ewma_rtf = rtf if self._ewma_rtf is None else 0.15 * rtf + 0.85 * self._ewma_rtf
            self.on_metrics(CudaNoiseMetrics(elapsed, float(self._ewma_rtf), self._tasks.qsize(), self.device, "CUDA DeepFilterNet3 Lookahead-Safe", False))

    def _fail(self, message: str) -> None:
        with self._state_lock:
            if self._failed:
                return
            self._failed = True
            self._running = False
            code = None if self._process is None else self._process.poll()
            self._last_error = message if code is None else f"{message} (process exit {code})"
        self.on_status(f"{self._last_error}. Log: {self.log_path}")
        self.on_metrics(CudaNoiseMetrics(0.0, float(self._ewma_rtf or 0.0), self._tasks.qsize(), self.device, "CUDA DeepFilterNet3 Lookahead-Safe", True))

    def _reset_buffers(self) -> None:
        self._input_buffer = np.empty(0, np.float32)
        self._output_buffer = np.empty(0, np.float32)
        self._sequence = 0
        self._ewma_rtf = None

    @staticmethod
    def _clear_queue(items: queue.Queue) -> None:
        while True:
            try:
                items.get_nowait()
            except queue.Empty:
                return
