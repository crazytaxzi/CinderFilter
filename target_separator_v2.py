from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from voice_lock import MODEL_DIR, PROFILE_PATH, VoiceLockService

AUDIO_RATE = 48_000
SEPARATION_RATE = 16_000
APP_DIR = Path(__file__).resolve().parent
SEPARATOR_MODEL_DIR = APP_DIR / "models" / "sepformer-whamr16k"


@dataclass(frozen=True)
class V2Preset:
    name: str
    chunk_seconds: float
    hop_seconds: float


PRESETS: dict[str, V2Preset] = {
    "Fast": V2Preset("Fast", 1.6, 0.8),
    "Balanced": V2Preset("Balanced", 2.4, 1.2),
    "Quality": V2Preset("Quality", 3.2, 1.6),
}


@dataclass(frozen=True)
class SeparationMetrics:
    processing_seconds: float
    realtime_factor: float
    similarity_a: float
    similarity_b: float
    selected_source: int | None
    selected_similarity: float | None
    queue_depth: int
    device: str
    fallback: bool


class TargetSpeakerSeparator:
    """Chunked two-speaker separation followed by target-speaker selection.

    The heavy SpeechBrain models live entirely on one worker thread. The live
    audio thread only submits overlapping chunks and consumes completed target
    audio, so it never blocks on neural inference.
    """

    MODEL_SOURCE = "speechbrain/sepformer-whamr16k"

    def __init__(
        self,
        on_status: Callable[[str], None],
        on_metrics: Callable[[SeparationMetrics], None],
    ) -> None:
        self.on_status = on_status
        self.on_metrics = on_metrics

        self._submit_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=2)
        self._result_queue: queue.Queue[tuple[int, np.ndarray, SeparationMetrics]] = queue.Queue(
            maxsize=3
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._state_lock = threading.Lock()

        self._preset = PRESETS["Fast"]
        self._device_preference = "Auto"
        self._strictness = "Balanced"
        self._device_name = "not loaded"
        self._startup_error: BaseException | None = None
        self._fallback = False
        self._overload_strikes = 0
        self._dropped_chunks = 0

        self._input_buffer = np.empty(0, dtype=np.float32)
        self._output_buffer = np.empty(0, dtype=np.float32)
        self._previous_tail: np.ndarray | None = None
        self._sequence = 0
        self._last_similarity: float | None = None
        self._last_metrics: SeparationMetrics | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    @property
    def fallback(self) -> bool:
        with self._state_lock:
            return self._fallback

    @property
    def last_similarity(self) -> float | None:
        with self._state_lock:
            return self._last_similarity

    @property
    def last_metrics(self) -> SeparationMetrics | None:
        with self._state_lock:
            return self._last_metrics

    @property
    def nominal_latency_seconds(self) -> float:
        return self._preset.chunk_seconds

    def start(
        self,
        preset_name: str,
        device_preference: str,
        strictness: str,
        timeout: float = 600.0,
    ) -> None:
        preset = PRESETS.get(preset_name, PRESETS["Fast"])
        requested = (preset.name, device_preference, strictness)
        current = (self._preset.name, self._device_preference, self._strictness)
        if self.running and requested == current and not self.fallback:
            if not self._ready_event.wait(timeout=timeout):
                raise RuntimeError("Voice Lock v2 model loading timed out.")
            if self._startup_error is not None:
                raise RuntimeError(f"Voice Lock v2 failed to initialize: {self._startup_error}")
            return

        self.stop()
        if self._thread and self._thread.is_alive():
            raise RuntimeError("The previous v2 worker is still shutting down. Try Start again shortly.")
        if not PROFILE_PATH.exists():
            raise RuntimeError("Voice Lock v2 requires an enrolled voice profile.")

        self._preset = preset
        self._device_preference = device_preference
        self._strictness = strictness
        self._startup_error = None
        self._fallback = False
        self._overload_strikes = 0
        self._dropped_chunks = 0
        self._reset_stream_buffers()
        self._stop_event.clear()
        self._ready_event.clear()

        self._thread = threading.Thread(
            target=self._worker,
            name="CinderFilter-TargetSeparator-v2",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self.stop()
            raise RuntimeError("Voice Lock v2 model loading timed out.")
        if self._startup_error is not None:
            error = self._startup_error
            self.stop()
            raise RuntimeError(f"Voice Lock v2 failed to initialize: {error}") from error

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._thread
        if worker and worker.is_alive():
            worker.join(timeout=15.0)
        if worker is None or not worker.is_alive():
            self._thread = None
        self._ready_event.clear()
        self._reset_stream_buffers()
        self._clear_queue(self._submit_queue)
        self._clear_queue(self._result_queue)

    def process_block(self, audio_48k: np.ndarray) -> np.ndarray | None:
        """Return delayed target audio, silence while priming, or None on fallback."""

        if not self.running or self.fallback:
            return None

        audio = np.asarray(audio_48k, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return audio.copy()

        self._input_buffer = np.concatenate((self._input_buffer, audio))
        chunk_samples = int(round(self._preset.chunk_seconds * AUDIO_RATE))
        hop_samples = int(round(self._preset.hop_seconds * AUDIO_RATE))

        while self._input_buffer.size >= chunk_samples:
            chunk = np.ascontiguousarray(self._input_buffer[:chunk_samples], dtype=np.float32)
            submitted = False
            try:
                self._submit_queue.put_nowait((self._sequence, chunk))
                submitted = True
                self._sequence += 1
            except queue.Full:
                self._dropped_chunks += 1
                self._overload_strikes += 1
                if self._overload_strikes >= 3:
                    self._set_fallback(
                        "Voice Lock v2 cannot keep up in real time; falling back to v1."
                    )
                    return None
            finally:
                # Advance even when overloaded so memory cannot grow without bound.
                self._input_buffer = self._input_buffer[hop_samples:]
            if not submitted:
                break

        self._drain_results()
        if self.fallback:
            return None

        needed = audio.size
        if self._output_buffer.size < needed:
            # Initial model/chunk latency is intentional. Fail closed so another
            # speaker is not leaked while the target extractor is priming.
            return np.zeros(needed, dtype=np.float32)

        output = np.ascontiguousarray(self._output_buffer[:needed], dtype=np.float32)
        self._output_buffer = self._output_buffer[needed:]
        return output

    def _drain_results(self) -> None:
        while True:
            try:
                _sequence, target, metrics = self._result_queue.get_nowait()
            except queue.Empty:
                return
            self._append_overlap(target)
            with self._state_lock:
                self._last_similarity = metrics.selected_similarity
                self._last_metrics = metrics
            if metrics.realtime_factor <= 0.95:
                self._overload_strikes = max(0, self._overload_strikes - 1)
            elif metrics.realtime_factor > 1.15:
                self._overload_strikes += 1
                if self._overload_strikes >= 3:
                    self._set_fallback(
                        "Voice Lock v2 inference is slower than its hop window; falling back to v1."
                    )
            self.on_metrics(metrics)

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

    def _worker(self) -> None:
        try:
            self.on_status("Loading Voice Lock v2 separation and speaker models...")
            import torch
            import torchaudio
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.inference.speaker import SpeakerRecognition
            from speechbrain.utils.fetching import LocalStrategy

            device = self._choose_device(torch)
            self._device_name = device
            SEPARATOR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            separator = SepformerSeparation.from_hparams(
                source=self.MODEL_SOURCE,
                savedir=str(SEPARATOR_MODEL_DIR),
                run_opts={"device": device},
                local_strategy=LocalStrategy.COPY,
            )
            speaker = SpeakerRecognition.from_hparams(
                source=VoiceLockService.MODEL_SOURCE,
                savedir=str(MODEL_DIR),
                run_opts={"device": device},
                local_strategy=LocalStrategy.COPY,
            )
            profile = VoiceLockService._normalize(np.load(PROFILE_PATH))

            self.on_status(
                f"Voice Lock v2 ready on {device.upper()} — {self._preset.name} mode"
            )
            self._ready_event.set()

            while not self._stop_event.is_set():
                try:
                    sequence, chunk_48k = self._submit_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                started = time.perf_counter()
                chunk_16k = self._resample(
                    torch, torchaudio, chunk_48k, AUDIO_RATE, SEPARATION_RATE, device
                )
                sources = self._separate(separator, torch, chunk_16k, device)
                similarities = [
                    self._speaker_similarity(speaker, torch, source, profile, device)
                    for source in sources[:2]
                ]
                while len(similarities) < 2:
                    similarities.append(-1.0)

                selected, gain = self._choose_target(similarities)
                if selected is None:
                    target_16k = np.zeros_like(chunk_16k, dtype=np.float32)
                    selected_similarity = None
                else:
                    target_16k = np.asarray(sources[selected] * gain, dtype=np.float32)
                    selected_similarity = float(similarities[selected])

                target_48k = self._resample(
                    torch, torchaudio, target_16k, SEPARATION_RATE, AUDIO_RATE, device
                )
                target_48k = self._fit_length(target_48k, chunk_48k.size)
                target_48k = self._safe_level(target_48k, chunk_48k)

                elapsed = time.perf_counter() - started
                rtf = elapsed / max(self._preset.hop_seconds, 1e-6)
                metrics = SeparationMetrics(
                    processing_seconds=elapsed,
                    realtime_factor=rtf,
                    similarity_a=float(similarities[0]),
                    similarity_b=float(similarities[1]),
                    selected_source=selected,
                    selected_similarity=selected_similarity,
                    queue_depth=self._submit_queue.qsize(),
                    device=device,
                    fallback=False,
                )
                try:
                    self._result_queue.put_nowait((sequence, target_48k, metrics))
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                        self._result_queue.put_nowait((sequence, target_48k, metrics))
                    except queue.Empty:
                        pass
        except BaseException as exc:
            self._startup_error = exc
            self._set_fallback(f"Voice Lock v2 error: {exc}")
        finally:
            self._ready_event.set()

    def _choose_device(self, torch) -> str:
        preference = self._device_preference.lower()
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
            return "cuda"
        if preference == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _choose_target(self, similarities: list[float]) -> tuple[int | None, float]:
        order = np.argsort(np.asarray(similarities, dtype=np.float32))[::-1]
        best = int(order[0])
        best_score = float(similarities[best])
        second_score = float(similarities[int(order[1])]) if len(order) > 1 else -1.0
        margin = best_score - second_score

        thresholds = {
            "Conservative": (0.12, 0.19),
            "Balanced": (0.16, 0.23),
            "Aggressive": (0.20, 0.28),
        }
        floor, full = thresholds.get(self._strictness, thresholds["Balanced"])
        if best_score < floor:
            return None, 0.0
        if best_score < full:
            gain = 0.30 + 0.55 * ((best_score - floor) / max(full - floor, 1e-6))
        else:
            gain = 1.0
        if margin < 0.025:
            # Ambiguous chunks are kept quieter, not deleted. This protects the
            # target when both separated streams contain some of the same voice.
            gain *= 0.72
        return best, float(np.clip(gain, 0.0, 1.0))

    @staticmethod
    def _separate(separator, torch, chunk_16k: np.ndarray, device: str) -> list[np.ndarray]:
        waveform = torch.from_numpy(chunk_16k).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            estimated = separator.separate_batch(waveform)
        array = estimated.detach().float().cpu().numpy()
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise RuntimeError(f"Unexpected separator output shape: {array.shape}")
        if array.shape[1] <= 4:
            return [np.ascontiguousarray(array[:, index], dtype=np.float32) for index in range(array.shape[1])]
        if array.shape[0] <= 4:
            return [np.ascontiguousarray(array[index], dtype=np.float32) for index in range(array.shape[0])]
        raise RuntimeError(f"Cannot identify speaker axis in separator output: {array.shape}")

    @staticmethod
    def _speaker_similarity(speaker, torch, source: np.ndarray, profile: np.ndarray, device: str) -> float:
        source = np.asarray(source, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(source * source) + 1e-12))
        if rms < 0.001:
            return -1.0
        waveform = torch.from_numpy(source).float().unsqueeze(0).to(device)
        lengths = torch.ones(1, device=device)
        with torch.inference_mode():
            embedding = speaker.encode_batch(waveform, lengths)
        normalized = VoiceLockService._normalize(
            embedding.detach().float().cpu().numpy().reshape(-1)
        )
        return float(np.dot(profile, normalized))

    @staticmethod
    def _resample(torch, torchaudio, audio: np.ndarray, source_rate: int, target_rate: int, device: str) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if source_rate == target_rate:
            return audio.copy()
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            converted = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return np.ascontiguousarray(converted.squeeze(0).detach().cpu().numpy(), dtype=np.float32)

    @staticmethod
    def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < length:
            return np.pad(audio, (0, length - audio.size)).astype(np.float32)
        return np.ascontiguousarray(audio[:length], dtype=np.float32)

    @staticmethod
    def _safe_level(target: np.ndarray, mixture: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        mixture = np.asarray(mixture, dtype=np.float32)
        target_rms = float(np.sqrt(np.mean(target * target) + 1e-12))
        mixture_rms = float(np.sqrt(np.mean(mixture * mixture) + 1e-12))
        ceiling = max(mixture_rms * 1.35, 0.01)
        if target_rms > ceiling:
            target = target * (ceiling / target_rms)
        return np.clip(target, -1.0, 1.0).astype(np.float32)

    def _set_fallback(self, message: str) -> None:
        with self._state_lock:
            if self._fallback:
                return
            self._fallback = True
            last = self._last_metrics
            fallback_metrics = SeparationMetrics(
                processing_seconds=0.0 if last is None else last.processing_seconds,
                realtime_factor=0.0 if last is None else last.realtime_factor,
                similarity_a=-1.0 if last is None else last.similarity_a,
                similarity_b=-1.0 if last is None else last.similarity_b,
                selected_source=None,
                selected_similarity=None,
                queue_depth=self._submit_queue.qsize(),
                device=self._device_name,
                fallback=True,
            )
            self._last_metrics = fallback_metrics
        self.on_status(message)
        self.on_metrics(fallback_metrics)

    def _reset_stream_buffers(self) -> None:
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._output_buffer = np.empty(0, dtype=np.float32)
        self._previous_tail = None
        self._sequence = 0
        with self._state_lock:
            self._last_similarity = None
            self._last_metrics = None

    @staticmethod
    def _clear_queue(items: queue.Queue) -> None:
        while True:
            try:
                items.get_nowait()
            except queue.Empty:
                return
