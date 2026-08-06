from __future__ import annotations

import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from deepfilternet_rs import DeepFilterNetRealtime

import cinderfilter_threadsafe  # applies the PyO3 thread-affinity hotfix
import cinderfilter as base

SAMPLE_RATE = 48_000
VOICE_RATE = 16_000
APP_DIR = Path(__file__).resolve().parent
PROFILE_DIR = APP_DIR / "profiles"
MODEL_DIR = APP_DIR / "models" / "spkrec-ecapa-voxceleb"
PROFILE_PATH = PROFILE_DIR / "voice_profile.npy"
PROFILE_META_PATH = PROFILE_DIR / "voice_profile.json"


class VoiceLockService:
    """Speaker embedding worker isolated from the live audio thread."""

    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(
        self,
        on_status: Callable[[str], None],
        on_result: Callable[[float | None], None],
        on_profile: Callable[[bool], None],
    ) -> None:
        self.on_status = on_status
        self.on_result = on_result
        self.on_profile = on_profile
        self._commands: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=3)
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._verify_pending = threading.Event()
        self._profile_lock = threading.Lock()
        self._profile: np.ndarray | None = self._read_profile()

    @property
    def has_profile(self) -> bool:
        with self._profile_lock:
            return self._profile is not None

    def enroll(self, audio: np.ndarray) -> None:
        self._ensure_thread()
        self._put("enroll", audio)

    def verify(self, audio: np.ndarray) -> bool:
        if not self.has_profile or self._verify_pending.is_set():
            return False
        self._ensure_thread()
        self._verify_pending.set()
        try:
            self._commands.put_nowait(("verify", self._mono(audio).copy()))
            return True
        except queue.Full:
            self._verify_pending.clear()
            return False

    def delete_profile(self) -> None:
        with self._profile_lock:
            self._profile = None
        for path in (PROFILE_PATH, PROFILE_META_PATH):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.on_result(None)
        self.on_profile(False)
        self.on_status("No voice profile enrolled")

    def _ensure_thread(self) -> None:
        with self._thread_lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker, name="CinderFilter-VoiceLock", daemon=True
            )
            self._thread.start()

    def _put(self, kind: str, audio: np.ndarray) -> None:
        item = (kind, self._mono(audio).copy())
        try:
            self._commands.put_nowait(item)
        except queue.Full:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                pass
            self._commands.put_nowait(item)

    def _worker(self) -> None:
        model = None
        torch = None
        while True:
            kind, audio = self._commands.get()
            try:
                if model is None:
                    self.on_status("Loading Voice Lock model (first use may download it)...")
                    import torch as torch_module
                    from speechbrain.inference.speaker import SpeakerRecognition
                    from speechbrain.utils.fetching import LocalStrategy

                    torch = torch_module
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    model = SpeakerRecognition.from_hparams(
                        source=self.MODEL_SOURCE,
                        savedir=str(MODEL_DIR),
                        run_opts={"device": "cpu"},
                        local_strategy=LocalStrategy.COPY,
                    )
                    self.on_status("Voice Lock model ready")
                if kind == "enroll":
                    self._enroll(model, torch, audio)
                else:
                    self._verify(model, torch, audio)
            except BaseException as exc:
                self.on_status(f"Voice Lock error: {exc}")
                self.on_result(None)
                self._verify_pending.clear()

    def _enroll(self, model, torch, audio_48k: np.ndarray) -> None:
        audio = self._downsample(audio_48k)
        if audio.size < VOICE_RATE * 6:
            raise RuntimeError("Enrollment needs at least 6 seconds of clear speech.")
        chunks: list[np.ndarray] = []
        size = VOICE_RATE * 3
        for start in range(0, audio.size - size + 1, size):
            chunk = audio[start : start + size]
            if self._rms_db(chunk) > -42.0:
                chunks.append(self._embed(model, torch, chunk))
        if len(chunks) < 2:
            raise RuntimeError("Not enough clear speech. Move closer and enroll again.")
        profile = self._normalize(np.mean(np.stack(chunks), axis=0))
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PROFILE_PATH, profile)
        PROFILE_META_PATH.write_text(
            json.dumps(
                {
                    "model": self.MODEL_SOURCE,
                    "sample_rate": VOICE_RATE,
                    "created_unix": time.time(),
                    "segments": len(chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with self._profile_lock:
            self._profile = profile
        self.on_profile(True)
        self.on_status(f"Voice profile ready ({len(chunks)} samples averaged)")

    def _verify(self, model, torch, audio_48k: np.ndarray) -> None:
        try:
            audio = self._downsample(audio_48k)
            if audio.size < VOICE_RATE or self._speech_fraction(audio) < 0.22:
                self.on_result(None)
                return
            embedding = self._embed(model, torch, audio)
            with self._profile_lock:
                profile = None if self._profile is None else self._profile.copy()
            self.on_result(None if profile is None else float(np.dot(profile, embedding)))
        finally:
            self._verify_pending.clear()

    @staticmethod
    def _mono(audio: np.ndarray) -> np.ndarray:
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    @staticmethod
    def _downsample(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        usable = audio.size - audio.size % 3
        if usable <= 0:
            return np.empty(0, np.float32)
        audio = audio[:usable].reshape(-1, 3).mean(axis=1)
        audio -= float(np.mean(audio))
        return np.ascontiguousarray(audio, dtype=np.float32)

    @classmethod
    def _embed(cls, model, torch, audio: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.inference_mode():
            value = model.encode_batch(waveform, torch.ones(1))
        return cls._normalize(value.detach().cpu().numpy().reshape(-1))

    @staticmethod
    def _normalize(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if norm <= 1e-6:
            raise RuntimeError("Speaker model returned an empty embedding.")
        return value / norm

    @staticmethod
    def _speech_fraction(audio: np.ndarray) -> float:
        frame = int(VOICE_RATE * 0.03)
        count = audio.size // frame
        if count <= 0:
            return 0.0
        frames = audio[: count * frame].reshape(count, frame)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        return float(np.mean(20.0 * np.log10(np.maximum(rms, 1e-6)) > -48.0))

    @staticmethod
    def _rms_db(audio: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        return 20.0 * math.log10(max(rms, 1e-6))

    @staticmethod
    def _read_profile() -> np.ndarray | None:
        try:
            if PROFILE_PATH.exists():
                return VoiceLockService._normalize(np.load(PROFILE_PATH))
        except Exception:
            pass
        return None


class VoiceLockedAudioEngine(base.AudioEngine):
    """Thread-safe DeepFilterNet engine plus side-channel speaker gating."""

    THRESHOLDS = {"Conservative": 0.24, "Balanced": 0.30, "Aggressive": 0.36}

    def __init__(self, voice_service: VoiceLockService, on_status, on_metrics) -> None:
        super().__init__(on_status, on_metrics)
        self.voice_service = voice_service
        self._voice_enabled = threading.Event()
        self._voice_guard = threading.Lock()
        self._voice_buffer = np.empty(0, np.float32)
        self._voice_since_submit = 0
        self._voice_similarity: float | None = None
        self._voice_target_gain = 1.0
        self._voice_current_gain = 1.0
        self._voice_reduction_db = 24.0
        self._voice_strictness = "Balanced"

    def configure_voice_lock(self, enabled: bool, reduction_db: float, strictness: str) -> None:
        with self._voice_guard:
            self._voice_reduction_db = max(0.0, min(48.0, float(reduction_db)))
            self._voice_strictness = strictness
            if not enabled:
                self._voice_target_gain = 1.0
                self._voice_similarity = None
        self._voice_enabled.set() if enabled else self._voice_enabled.clear()

    def update_voice_similarity(self, similarity: float | None) -> None:
        with self._voice_guard:
            self._voice_similarity = similarity
            if not self._voice_enabled.is_set() or similarity is None:
                self._voice_target_gain = 1.0
                return
            threshold = self.THRESHOLDS.get(self._voice_strictness, 0.30)
            if similarity >= threshold:
                gain = 1.0
            elif similarity >= threshold - 0.055:
                gain = 0.72  # protect uncertain/overlapping target speech
            else:
                gain = 10.0 ** (-self._voice_reduction_db / 20.0)
            self._voice_target_gain = float(np.clip(gain, 0.003, 1.0))

    def start(self, input_device: int, output_device: int, strength: str) -> None:
        self._voice_buffer = np.empty(0, np.float32)
        self._voice_since_submit = 0
        self._voice_current_gain = 1.0
        super().start(input_device, output_device, strength)

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
                    block = self._input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._bypass.is_set():
                    processed = block
                else:
                    processed = np.asarray(
                        processor.process_chunk(block), dtype=np.float32
                    ).reshape(-1)
                if processed.size == 0:
                    continue
                processed = np.clip(processed, -1.0, 1.0)
                self._feed_voice_lock(processed)
                processed = self._apply_voice_gain(processed)
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
                    with self._voice_guard:
                        gain = self._voice_current_gain
                        similarity = self._voice_similarity
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

    def _feed_voice_lock(self, audio: np.ndarray) -> None:
        if not self._voice_enabled.is_set() or not self.voice_service.has_profile:
            return
        self._voice_buffer = np.concatenate((self._voice_buffer, audio))
        limit = int(SAMPLE_RATE * 1.8)
        if self._voice_buffer.size > limit:
            self._voice_buffer = self._voice_buffer[-limit:]
        self._voice_since_submit += audio.size
        if (
            self._voice_buffer.size >= int(SAMPLE_RATE * 1.5)
            and self._voice_since_submit >= int(SAMPLE_RATE * 0.50)
            and self.voice_service.verify(self._voice_buffer)
        ):
            self._voice_since_submit = 0

    def _apply_voice_gain(self, audio: np.ndarray) -> np.ndarray:
        with self._voice_guard:
            current = self._voice_current_gain
            target = self._voice_target_gain if self._voice_enabled.is_set() else 1.0
        duration = audio.size / SAMPLE_RATE
        tau = 0.12 if target < current else 0.34
        new = current + (target - current) * (1.0 - math.exp(-duration / tau))
        ramp = np.linspace(current, new, audio.size, dtype=np.float32)
        with self._voice_guard:
            self._voice_current_gain = float(new)
        return np.asarray(audio * ramp, dtype=np.float32)
