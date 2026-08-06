from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
from deepfilternet_rs import DeepFilterNetRealtime

from cuda_noise_bridge import CUDA_PRESETS, CudaNoiseBridge, CudaNoiseMetrics
from pitch_lock_v2 import PitchGuard, PitchLockedTargetSeparator, PitchSeparationMetrics
from target_separator_v2 import PRESETS as SEPARATOR_PRESETS, SeparationMetrics
from voice_lock import VoiceLockService

SAMPLE_RATE = 48_000
STRENGTHS = {"Natural": 20.0, "Balanced": 45.0, "Strong": 70.0, "Maximum": 100.0}
VOICE_THRESHOLDS = {"Conservative": 0.24, "Balanced": 0.30, "Aggressive": 0.36}


@dataclass(frozen=True)
class DeviceChoice:
    index: int
    name: str
    host_api: str
    input_channels: int
    output_channels: int
    default_sample_rate: float

    @property
    def key(self) -> str:
        return f"{self.name} [{self.host_api}]"

    @property
    def label(self) -> str:
        return f"{self.name}  ·  {self.host_api}"


@dataclass(frozen=True)
class EngineConfig:
    strength: str = "Balanced"
    bypass: bool = False
    voice_lock_enabled: bool = True
    voice_reduction_db: float = 24.0
    voice_strictness: str = "Balanced"
    target_extraction_enabled: bool = False
    separator_preset: str = "Fast"
    separator_device: str = "Auto"
    pitch_enabled: bool = True
    pitch_fail_closed: bool = True
    pitch_margin_hz: float = 0.0
    noise_backend: str = "CUDA"
    noise_cuda_preset: str = "Balanced"

    def normalized(self) -> "EngineConfig":
        return replace(
            self,
            strength=self.strength if self.strength in STRENGTHS else "Balanced",
            voice_reduction_db=float(np.clip(self.voice_reduction_db, 0.0, 48.0)),
            voice_strictness=(self.voice_strictness if self.voice_strictness in VOICE_THRESHOLDS else "Balanced"),
            separator_preset=(self.separator_preset if self.separator_preset in SEPARATOR_PRESETS else "Fast"),
            separator_device=(self.separator_device if self.separator_device in {"Auto", "CUDA", "CPU"} else "Auto"),
            pitch_margin_hz=float(np.clip(self.pitch_margin_hz, -20.0, 80.0)),
            noise_backend=(self.noise_backend if self.noise_backend in {"CUDA", "Auto", "CPU"} else "Auto"),
            noise_cuda_preset=(self.noise_cuda_preset if self.noise_cuda_preset in CUDA_PRESETS else "Balanced"),
        )


@dataclass(frozen=True)
class EngineMetrics:
    input_db: float = -60.0
    output_db: float = -60.0
    dropped_input: int = 0
    output_underruns: int = 0
    voice_gain: float = 1.0
    voice_similarity: float | None = None
    noise_reduction_db: float = 0.0
    active_backend: str = "Stopped"
    cuda_rtf: float = 0.0
    cuda_queue: int = 0
    cuda_device: str = "--"
    separator_rtf: float = 0.0
    separator_queue: int = 0
    separator_device: str = "--"
    selected_similarity: float | None = None
    pitch_a_hz: float | None = None
    pitch_b_hz: float | None = None
    pitch_limit_hz: float | None = None


class CinderFilterEngine:
    """One real-time controller for the unified PySide application."""

    def __init__(
        self,
        on_status: Callable[[str], None],
        on_metrics: Callable[[EngineMetrics], None],
        on_event: Callable[[str, object], None],
        app_dir: Path | None = None,
    ) -> None:
        self.app_dir = app_dir or Path(__file__).resolve().parent
        self.on_status = on_status
        self.on_metrics = on_metrics
        self.on_event = on_event
        self.voice_service = VoiceLockService(self._voice_status, self._voice_result, self._voice_profile_changed)
        self.pitch_guard = PitchGuard()
        self.separator = PitchLockedTargetSeparator(self.pitch_guard, self._separator_status, self._separator_metrics)
        self.cuda_noise = CudaNoiseBridge(self._cuda_status, self._cuda_metrics, app_dir=self.app_dir)

        self._config = EngineConfig().normalized()
        self._running = threading.Event()
        self._bypass = threading.Event()
        self._input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=80)
        self._output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=80)
        self._worker: threading.Thread | None = None
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._model_ready = threading.Event()
        self._worker_error: BaseException | None = None
        self._output_channels = 2
        self._pending_output = np.empty(0, np.float32)
        self._dropped_input = 0
        self._output_underruns = 0
        self._input_level = -60.0
        self._output_level = -60.0

        self._voice_guard = threading.Lock()
        self._voice_buffer = np.empty(0, np.float32)
        self._voice_since_submit = 0
        self._voice_similarity: float | None = None
        self._voice_target_gain = 1.0
        self._voice_current_gain = 1.0

        self._cuda_active = False
        self._separator_active = False
        self._separator_failed_closed = False
        self._active_backend = "Stopped"
        self._recovery_thread: threading.Thread | None = None
        self._recovery_stop = threading.Event()
        self._recovery_lock = threading.Lock()
        self._metric_lock = threading.Lock()
        self._metrics = EngineMetrics()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def metrics(self) -> EngineMetrics:
        with self._metric_lock:
            return self._metrics

    @property
    def voice_profile_ready(self) -> bool:
        return self.voice_service.has_profile

    @property
    def pitch_profile_ready(self) -> bool:
        return self.pitch_guard.has_profile

    @staticmethod
    def enumerate_devices() -> tuple[list[DeviceChoice], list[DeviceChoice]]:
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        inputs: list[DeviceChoice] = []
        outputs: list[DeviceChoice] = []
        for index, raw in enumerate(devices):
            host_index = int(raw.get("hostapi", 0))
            host = str(host_apis[host_index]["name"]) if 0 <= host_index < len(host_apis) else "Unknown"
            choice = DeviceChoice(
                index=index,
                name=str(raw.get("name", f"Device {index}")),
                host_api=host,
                input_channels=int(raw.get("max_input_channels", 0)),
                output_channels=int(raw.get("max_output_channels", 0)),
                default_sample_rate=float(raw.get("default_samplerate", SAMPLE_RATE)),
            )
            if choice.input_channels:
                inputs.append(choice)
            if choice.output_channels:
                outputs.append(choice)
        inputs.sort(key=lambda x: (x.name.casefold(), x.host_api.casefold()))
        outputs.sort(key=lambda x: (x.name.casefold(), x.host_api.casefold()))
        return inputs, outputs

    def start(self, input_device: int, output_device: int, config: EngineConfig) -> None:
        config = config.normalized()
        if self.running:
            self.stop()
        if config.bypass and config.target_extraction_enabled and config.pitch_fail_closed:
            raise RuntimeError(
                "Bypass is disabled while fail-closed target extraction is enabled. "
                "Disable fail-closed extraction before deliberately routing the raw microphone."
            )

        input_info = sd.query_devices(input_device)
        output_info = sd.query_devices(output_device)
        if int(input_info["max_input_channels"]) < 1:
            raise RuntimeError("The selected input has no recording channels.")
        max_output = int(output_info["max_output_channels"])
        if max_output < 1:
            raise RuntimeError("The selected output has no playback channels.")

        self._config = config
        self._output_channels = 2 if max_output >= 2 else 1
        self._bypass.set() if config.bypass else self._bypass.clear()
        self._reset_stream_state()
        self._recovery_stop.clear()
        try:
            self._configure_separator(config)
            self._configure_noise_backend(config)
        except BaseException:
            self.separator.stop()
            self.cuda_noise.stop()
            self._separator_active = self._cuda_active = False
            self._active_backend = "Stopped"
            raise

        self._model_ready.clear()
        self._worker_error = None
        self._running.set()
        self._worker = threading.Thread(target=self._process_loop, name="CinderFilter-Audio-Core", daemon=True)
        self._worker.start()
        if not self._model_ready.wait(45.0):
            self.stop()
            raise RuntimeError("The main noise reducer timed out while loading.")
        if self._worker_error is not None:
            error = self._worker_error
            self.stop()
            raise RuntimeError(f"The audio engine failed to initialize: {error}") from error
        if not self._running.is_set():
            self.stop()
            raise RuntimeError("The audio worker stopped before the streams opened.")

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
            self._input_stream.start()
            deadline = time.monotonic() + 0.35
            while self._output_queue.qsize() < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self._output_stream.start()
        except BaseException:
            self.stop()
            raise
        self.on_status(f"LIVE — {self._active_backend}")
        self.on_event("running", True)

    def stop(self) -> None:
        self._recovery_stop.set()
        self._running.clear()
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._input_stream = self._output_stream = None
        if self._worker and self._worker.is_alive() and self._worker is not threading.current_thread():
            self._worker.join(2.5)
        self._worker = None
        recovery = self._recovery_thread
        if recovery and recovery.is_alive() and recovery is not threading.current_thread():
            recovery.join(1.5)
        self._recovery_thread = None
        self.separator.stop()
        self.cuda_noise.stop()
        self._separator_active = self._cuda_active = False
        self._active_backend = "Stopped"
        self._clear_queues()
        with self._metric_lock:
            self._metrics = replace(self._metrics, active_backend="Stopped", cuda_queue=0, separator_queue=0)
        self.on_status("Stopped")
        self.on_event("running", False)

    def set_bypass(self, enabled: bool) -> None:
        self._bypass.set() if enabled else self._bypass.clear()

    def preload_cuda(self, strength: str, preset: str) -> dict:
        return self.cuda_noise.start(STRENGTHS.get(strength, 45.0), preset)

    def preload_separator(self, config: EngineConfig) -> None:
        config = config.normalized()
        if not self.voice_profile_ready:
            raise RuntimeError("Enroll a voice profile before preloading target extraction.")
        if config.pitch_enabled and not self.pitch_profile_ready:
            raise RuntimeError("Calibrate a pitch profile before preloading Pitch Lock.")
        self.separator.configure_pitch_lock(config.pitch_enabled, config.pitch_fail_closed, config.pitch_margin_hz)
        self.separator.start(config.separator_preset, config.separator_device, config.voice_strictness)

    def record_voice_profile(self, input_device: int, seconds: int = 12) -> None:
        if self.running:
            raise RuntimeError("Stop filtering before voice enrollment.")
        self.on_status(f"Recording {seconds} seconds for Voice Lock enrollment...")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=input_device, blocking=True)
        self.voice_service.enroll(np.asarray(audio[:, 0], np.float32))

    def delete_voice_profile(self) -> None:
        self.separator.stop()
        self.voice_service.delete_profile()

    def record_pitch_profile(self, input_device: int, seconds: int = 12):
        if self.running:
            raise RuntimeError("Stop filtering before pitch calibration.")
        self.on_status(f"Recording {seconds} seconds for pitch calibration...")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=input_device, blocking=True)
        profile = self.pitch_guard.calibrate(np.asarray(audio[:, 0], np.float32))
        self.on_event("pitch_profile", profile)
        self.on_status(f"Pitch profile ready — upper cutoff {profile.upper_limit_hz:.0f} Hz")
        return profile

    def delete_pitch_profile(self) -> None:
        self.separator.stop()
        self.pitch_guard.delete_profile()
        self.on_event("pitch_profile", None)
        self.on_status("Pitch profile deleted")

    def _reset_stream_state(self) -> None:
        self._clear_queues()
        self._pending_output = np.empty(0, np.float32)
        self._dropped_input = self._output_underruns = 0
        self._voice_buffer = np.empty(0, np.float32)
        self._voice_since_submit = 0
        self._voice_current_gain = self._voice_target_gain = 1.0
        self._voice_similarity = None
        self._separator_failed_closed = False
        with self._metric_lock:
            self._metrics = EngineMetrics()

    def _configure_separator(self, config: EngineConfig) -> None:
        self._separator_active = False
        if not config.target_extraction_enabled:
            self.separator.stop()
            return
        if not self.voice_profile_ready:
            raise RuntimeError("Target extraction requires an enrolled voice profile.")
        if config.pitch_enabled and not self.pitch_profile_ready:
            raise RuntimeError("Pitch Lock requires a calibrated pitch profile.")
        self.separator.configure_pitch_lock(config.pitch_enabled, config.pitch_fail_closed, config.pitch_margin_hz)
        try:
            self.separator.start(config.separator_preset, config.separator_device, config.voice_strictness)
            self._separator_active = True
        except BaseException as exc:
            if config.pitch_fail_closed:
                raise RuntimeError(f"Fail-closed target extraction could not start: {exc}") from exc
            self.on_status(f"Target extraction unavailable; using Voice Lock gate: {exc}")

    def _configure_noise_backend(self, config: EngineConfig) -> None:
        self._cuda_active = False
        if config.noise_backend in {"CUDA", "Auto"}:
            try:
                self.cuda_noise.start(STRENGTHS.get(config.strength, 45.0), config.noise_cuda_preset)
                self._cuda_active = True
                self._active_backend = "CUDA DeepFilterNet3"
                return
            except BaseException as exc:
                if config.noise_backend == "CUDA":
                    raise RuntimeError(f"CUDA main noise reducer could not start: {exc}") from exc
                self.on_status(f"CUDA unavailable in Auto mode; using CPU: {exc}")
        self._active_backend = "CPU Rust DeepFilterNet"

    def _cpu_processor(self) -> DeepFilterNetRealtime:
        if self._config.noise_backend == "CUDA":
            raise RuntimeError("CPU denoiser creation is blocked in explicit CUDA mode.")
        processor = DeepFilterNetRealtime(
            model_path=None,
            atten_lim=STRENGTHS.get(self._config.strength, 45.0),
            log_level="warn",
            compensate_delay=False,
            post_filter_beta=0.0,
        )
        if int(processor.sample_rate) != SAMPLE_RATE:
            raise RuntimeError(f"DeepFilterNet requires {processor.sample_rate} Hz, expected {SAMPLE_RATE} Hz.")
        return processor

    def _process_loop(self) -> None:
        cpu: DeepFilterNetRealtime | None = None
        last_metrics = 0.0
        try:
            if not self._cuda_active:
                cpu = self._cpu_processor()
                self._active_backend = "CPU Rust DeepFilterNet"
            self._model_ready.set()
            while self._running.is_set():
                try:
                    raw = self._input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                source = raw
                using_target = False
                if not self._bypass.is_set() and self._config.target_extraction_enabled:
                    if self._separator_active:
                        extracted = self.separator.process_block(raw)
                        if extracted is None:
                            self._separator_active = False
                            if self._config.pitch_fail_closed:
                                self._separator_failed_closed = True
                                source = np.zeros_like(raw)
                                using_target = True
                                self.on_status("Target extractor stopped — fail-closed output muted")
                            else:
                                self.on_status("Target extractor stopped — Voice Lock gate remains active")
                        else:
                            source = extracted
                            using_target = True
                    elif self._separator_failed_closed and self._config.pitch_fail_closed:
                        source = np.zeros_like(raw)
                        using_target = True

                if self._bypass.is_set():
                    processed = raw
                elif self._cuda_active:
                    result = self.cuda_noise.process_block(source)
                    if result is None:
                        self._cuda_active = False
                        if self._config.noise_backend == "CUDA":
                            processed = np.zeros_like(source)
                            self.on_status("CUDA worker stopped — output muted while CUDA restarts")
                            self._begin_cuda_recovery()
                        else:
                            self.on_status("CUDA failed in Auto mode — switching to CPU")
                            cpu = self._cpu_processor()
                            self._active_backend = "CPU Rust DeepFilterNet"
                            processed = np.asarray(cpu.process_chunk(source), np.float32).reshape(-1)
                    else:
                        processed = np.asarray(result, np.float32).reshape(-1)
                elif self._config.noise_backend == "CUDA":
                    processed = np.zeros_like(source)
                    self._begin_cuda_recovery()
                else:
                    if cpu is None:
                        cpu = self._cpu_processor()
                    processed = np.asarray(cpu.process_chunk(source), np.float32).reshape(-1)

                if not processed.size:
                    continue
                processed = np.clip(processed, -1.0, 1.0)
                if using_target:
                    similarity = self.separator.last_similarity
                    with self._voice_guard:
                        self._voice_similarity = similarity
                        self._voice_target_gain = self._voice_current_gain = 1.0
                    gain = 1.0
                else:
                    self._feed_voice_lock(processed)
                    processed = self._apply_voice_gain(processed)
                    with self._voice_guard:
                        similarity = self._voice_similarity
                        gain = self._voice_current_gain

                self._output_level = self._peak_db(processed)
                reduction = max(0.0, self._rms_db(raw) - self._rms_db(processed))
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
                    self._publish_metrics(gain, similarity, reduction)
                    last_metrics = now
        except BaseException as exc:
            self._worker_error = exc
            self.on_status(f"Audio processing error: {type(exc).__name__}: {exc}")
            self._running.clear()
        finally:
            if cpu is not None:
                try:
                    cpu.close()
                except BaseException:
                    pass
            self._model_ready.set()

    def _begin_cuda_recovery(self) -> None:
        if self._config.noise_backend != "CUDA" or self._recovery_stop.is_set():
            return
        with self._recovery_lock:
            if self._recovery_thread and self._recovery_thread.is_alive():
                return
            self._recovery_thread = threading.Thread(target=self._cuda_recovery_loop, name="CinderFilter-CUDA-Recovery", daemon=True)
            self._recovery_thread.start()

    def _cuda_recovery_loop(self) -> None:
        delays = (0.5, 1.0, 2.0, 5.0, 10.0)
        attempt = 0
        while self._running.is_set() and not self._recovery_stop.is_set():
            if self._recovery_stop.wait(delays[min(attempt, len(delays) - 1)]):
                return
            try:
                self.on_status(f"Restarting CUDA denoiser — attempt {attempt + 1}")
                self.cuda_noise.start(STRENGTHS.get(self._config.strength, 45.0), self._config.noise_cuda_preset)
                if not self._running.is_set():
                    self.cuda_noise.stop()
                    return
                self._cuda_active = True
                self._active_backend = "CUDA DeepFilterNet3"
                self.on_status(f"CUDA denoiser recovered on {self.cuda_noise.device}")
                return
            except BaseException as exc:
                self._cuda_active = False
                self.on_status(f"CUDA restart attempt {attempt + 1} failed: {exc}")
                attempt += 1

    def _feed_voice_lock(self, audio: np.ndarray) -> None:
        if not self._config.voice_lock_enabled or not self.voice_profile_ready:
            with self._voice_guard:
                self._voice_target_gain = 1.0
            return
        self._voice_buffer = np.concatenate((self._voice_buffer, audio))
        limit = int(SAMPLE_RATE * 1.8)
        if self._voice_buffer.size > limit:
            self._voice_buffer = self._voice_buffer[-limit:]
        self._voice_since_submit += audio.size
        if (
            self._voice_buffer.size >= int(SAMPLE_RATE * 1.5)
            and self._voice_since_submit >= int(SAMPLE_RATE * 0.5)
            and self.voice_service.verify(self._voice_buffer)
        ):
            self._voice_since_submit = 0

    def _voice_result(self, similarity: float | None) -> None:
        with self._voice_guard:
            self._voice_similarity = similarity
            if not self._config.voice_lock_enabled or similarity is None:
                self._voice_target_gain = 1.0
            else:
                threshold = VOICE_THRESHOLDS.get(self._config.voice_strictness, 0.30)
                if similarity >= threshold:
                    gain = 1.0
                elif similarity >= threshold - 0.055:
                    gain = 0.72
                else:
                    gain = 10.0 ** (-self._config.voice_reduction_db / 20.0)
                self._voice_target_gain = float(np.clip(gain, 0.003, 1.0))
        self.on_event("voice_similarity", similarity)

    def _apply_voice_gain(self, audio: np.ndarray) -> np.ndarray:
        with self._voice_guard:
            current = self._voice_current_gain
            target = self._voice_target_gain if self._config.voice_lock_enabled else 1.0
        duration = audio.size / SAMPLE_RATE
        tau = 0.12 if target < current else 0.34
        new = current + (target - current) * (1.0 - math.exp(-duration / tau))
        ramp = np.linspace(current, new, audio.size, dtype=np.float32)
        with self._voice_guard:
            self._voice_current_gain = float(new)
        return np.asarray(audio * ramp, np.float32)

    def _input_callback(self, indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            self.on_status(f"Input stream: {status}")
        mono = np.ascontiguousarray(indata[:frames, 0], np.float32)
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

    def _output_callback(self, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            self.on_status(f"Output stream: {status}")
        while self._pending_output.size < frames:
            try:
                self._pending_output = np.concatenate((self._pending_output, self._output_queue.get_nowait()))
            except queue.Empty:
                break
        available = min(frames, self._pending_output.size)
        mono = np.zeros(frames, np.float32)
        if available:
            mono[:available] = self._pending_output[:available]
            self._pending_output = self._pending_output[available:]
        if available < frames:
            self._output_underruns += 1
        if self._output_channels == 1:
            outdata[:, 0] = mono
        else:
            outdata[:, :] = mono[:, None]

    def _cuda_metrics(self, metrics: CudaNoiseMetrics) -> None:
        with self._metric_lock:
            self._metrics = replace(
                self._metrics,
                cuda_rtf=metrics.realtime_factor,
                cuda_queue=metrics.queue_depth,
                cuda_device=metrics.device,
                active_backend=metrics.backend,
            )
        self.on_event("cuda_metrics", metrics)

    def _separator_metrics(self, metrics: SeparationMetrics) -> None:
        pitch_a = pitch_b = pitch_limit = None
        if isinstance(metrics, PitchSeparationMetrics):
            pitch_a, pitch_b, pitch_limit = metrics.pitch_a_hz, metrics.pitch_b_hz, metrics.pitch_limit_hz
        with self._metric_lock:
            self._metrics = replace(
                self._metrics,
                separator_rtf=metrics.realtime_factor,
                separator_queue=metrics.queue_depth,
                separator_device=metrics.device,
                selected_similarity=metrics.selected_similarity,
                pitch_a_hz=pitch_a,
                pitch_b_hz=pitch_b,
                pitch_limit_hz=pitch_limit,
            )
        self.on_event("separator_metrics", metrics)

    def _publish_metrics(self, gain: float, similarity: float | None, reduction: float) -> None:
        with self._metric_lock:
            self._metrics = replace(
                self._metrics,
                input_db=self._input_level,
                output_db=self._output_level,
                dropped_input=self._dropped_input,
                output_underruns=self._output_underruns,
                voice_gain=gain,
                voice_similarity=similarity,
                noise_reduction_db=reduction,
                active_backend=self._active_backend,
            )
            snapshot = self._metrics
        self.on_metrics(snapshot)

    def _voice_status(self, text: str) -> None:
        self.on_status(text)
        self.on_event("voice_status", text)

    def _voice_profile_changed(self, exists: bool) -> None:
        self.on_event("voice_profile", exists)

    def _separator_status(self, text: str) -> None:
        self.on_status(text)
        self.on_event("separator_status", text)

    def _cuda_status(self, text: str) -> None:
        self.on_status(text)
        self.on_event("cuda_status", text)

    def _clear_queues(self) -> None:
        for items in (self._input_queue, self._output_queue):
            while True:
                try:
                    items.get_nowait()
                except queue.Empty:
                    break

    @staticmethod
    def _peak_db(audio: np.ndarray) -> float:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return max(-60.0, 20.0 * math.log10(max(peak, 1e-6)))

    @staticmethod
    def _rms_db(audio: np.ndarray) -> float:
        if not audio.size:
            return -60.0
        rms = float(np.sqrt(np.mean(np.asarray(audio, np.float32) ** 2) + 1e-12))
        return max(-60.0, 20.0 * math.log10(max(rms, 1e-6)))
