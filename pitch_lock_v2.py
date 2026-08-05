from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import target_separator_v2 as base
from voice_lock import MODEL_DIR, PROFILE_PATH, VoiceLockService

APP_DIR = Path(__file__).resolve().parent
PITCH_PROFILE_PATH = APP_DIR / "profiles" / "pitch_profile.json"


@dataclass(frozen=True)
class PitchProfile:
    median_hz: float
    low_hz: float
    high_hz: float
    upper_limit_hz: float
    voiced_frames: int


@dataclass(frozen=True)
class PitchSeparationMetrics(base.SeparationMetrics):
    pitch_a_hz: float | None
    pitch_b_hz: float | None
    pitch_a_allowed: bool
    pitch_b_allowed: bool
    pitch_limit_hz: float
    fail_closed: bool


class PitchGuard:
    """Lightweight F0 estimator and persisted target pitch profile."""

    MIN_HZ = 65.0
    MAX_HZ = 500.0

    def __init__(self) -> None:
        self._profile = self.load_profile()

    @property
    def profile(self) -> PitchProfile | None:
        return self._profile

    @property
    def has_profile(self) -> bool:
        return self._profile is not None

    def calibrate(self, audio_48k: np.ndarray, margin_hz: float = 35.0) -> PitchProfile:
        audio = np.asarray(audio_48k, dtype=np.float32).reshape(-1)
        usable = audio.size - (audio.size % 3)
        if usable <= 0:
            raise RuntimeError("Pitch calibration received no usable audio.")
        audio_16k = audio[:usable].reshape(-1, 3).mean(axis=1)
        values = self.estimate_track(audio_16k)
        if values.size < 24:
            raise RuntimeError(
                "Not enough voiced speech for pitch calibration. Speak continuously in your normal register."
            )

        median = float(np.median(values))
        low = float(np.percentile(values, 10))
        high = float(np.percentile(values, 90))
        # The upper limit deliberately follows the enrolled high range plus a
        # safety margin. This keeps normal emphasis while rejecting distinctly
        # higher voices.
        upper = float(max(high + margin_hz, median * 1.30))
        upper = float(np.clip(upper, median + 18.0, 320.0))
        profile = PitchProfile(median, low, high, upper, int(values.size))

        PITCH_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PITCH_PROFILE_PATH.write_text(
            json.dumps(
                {
                    "median_hz": profile.median_hz,
                    "low_hz": profile.low_hz,
                    "high_hz": profile.high_hz,
                    "upper_limit_hz": profile.upper_limit_hz,
                    "voiced_frames": profile.voiced_frames,
                    "created_unix": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._profile = profile
        return profile

    def delete_profile(self) -> None:
        self._profile = None
        try:
            PITCH_PROFILE_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    def classify(self, source_16k: np.ndarray, margin_hz: float = 0.0) -> tuple[float | None, bool]:
        profile = self._profile
        if profile is None:
            return None, False
        values = self.estimate_track(source_16k)
        if values.size < 5:
            # Pitchless/mostly unvoiced chunks are not proof of identity. The
            # caller decides whether uncertainty should pass or fail closed.
            return None, False
        pitch = float(np.median(values))
        return pitch, pitch <= (profile.upper_limit_hz + margin_hz)

    @classmethod
    def estimate_track(cls, audio_16k: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if audio.size < 640:
            return np.empty(0, dtype=np.float32)
        audio = audio - float(np.mean(audio))

        frame_size = 640  # 40 ms
        hop = 320  # 20 ms
        min_lag = max(1, int(16_000 / cls.MAX_HZ))
        max_lag = min(frame_size - 2, int(16_000 / cls.MIN_HZ))
        window = np.hanning(frame_size).astype(np.float32)
        pitches: list[float] = []

        for start in range(0, audio.size - frame_size + 1, hop):
            frame = audio[start : start + frame_size]
            rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
            if rms < 0.006:
                continue
            frame = frame * window
            corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
            zero = float(corr[0])
            if zero <= 1e-8:
                continue
            search = corr[min_lag : max_lag + 1]
            if search.size == 0:
                continue
            rel = int(np.argmax(search))
            lag = min_lag + rel
            confidence = float(search[rel] / zero)
            if confidence < 0.28:
                continue

            # Parabolic interpolation around the autocorrelation peak.
            refined = float(lag)
            if 1 <= lag < corr.size - 1:
                left, center, right = map(float, corr[lag - 1 : lag + 2])
                denom = left - 2.0 * center + right
                if abs(denom) > 1e-12:
                    refined += 0.5 * (left - right) / denom
            hz = 16_000.0 / max(refined, 1.0)
            if cls.MIN_HZ <= hz <= cls.MAX_HZ:
                pitches.append(hz)

        if not pitches:
            return np.empty(0, dtype=np.float32)
        values = np.asarray(pitches, dtype=np.float32)
        # Remove octave slips and isolated outliers before taking medians.
        center = float(np.median(values))
        keep = (values >= center * 0.62) & (values <= center * 1.62)
        return values[keep]

    @staticmethod
    def load_profile() -> PitchProfile | None:
        try:
            data = json.loads(PITCH_PROFILE_PATH.read_text(encoding="utf-8"))
            return PitchProfile(
                median_hz=float(data["median_hz"]),
                low_hz=float(data["low_hz"]),
                high_hz=float(data["high_hz"]),
                upper_limit_hz=float(data["upper_limit_hz"]),
                voiced_frames=int(data.get("voiced_frames", 0)),
            )
        except Exception:
            return None


class PitchLockedTargetSeparator(base.TargetSpeakerSeparator):
    """SepFormer + voiceprint selection + lower-pitch fail-closed guard."""

    def __init__(
        self,
        pitch_guard: PitchGuard,
        on_status: Callable[[str], None],
        on_metrics: Callable[[base.SeparationMetrics], None],
    ) -> None:
        super().__init__(on_status, on_metrics)
        self.pitch_guard = pitch_guard
        self._pitch_enabled = True
        self._fail_closed = True
        self._pitch_margin_hz = 0.0

    def configure_pitch_lock(
        self,
        enabled: bool,
        fail_closed: bool,
        pitch_margin_hz: float,
    ) -> None:
        self._pitch_enabled = bool(enabled)
        self._fail_closed = bool(fail_closed)
        self._pitch_margin_hz = float(np.clip(pitch_margin_hz, -20.0, 80.0))

    def start(
        self,
        preset_name: str,
        device_preference: str,
        strictness: str,
        timeout: float = 600.0,
    ) -> None:
        if self._pitch_enabled and not self.pitch_guard.has_profile:
            raise RuntimeError("Pitch Lock requires a pitch calibration profile.")
        super().start(preset_name, device_preference, strictness, timeout)

    def _worker(self) -> None:
        try:
            self.on_status("Loading pitch-locked v2 separation and speaker models...")
            import torch
            import torchaudio
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.inference.speaker import SpeakerRecognition
            from speechbrain.utils.fetching import LocalStrategy

            device = self._choose_device(torch)
            self._device_name = device
            base.SEPARATOR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            separator = SepformerSeparation.from_hparams(
                source=self.MODEL_SOURCE,
                savedir=str(base.SEPARATOR_MODEL_DIR),
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

            pitch_profile = self.pitch_guard.profile
            if self._pitch_enabled and pitch_profile is None:
                raise RuntimeError("Pitch profile disappeared before v2 could start.")

            self.on_status(
                f"Pitch-Locked v2 ready on {device.upper()} — {self._preset.name} mode"
            )
            self._ready_event.set()

            while not self._stop_event.is_set():
                try:
                    sequence, chunk_48k = self._submit_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                started = time.perf_counter()
                chunk_16k = self._resample(
                    torch, torchaudio, chunk_48k, base.AUDIO_RATE, base.SEPARATION_RATE, device
                )
                sources = self._separate(separator, torch, chunk_16k, device)
                while len(sources) < 2:
                    sources.append(np.zeros_like(chunk_16k, dtype=np.float32))

                similarities = [
                    self._speaker_similarity(speaker, torch, source, profile, device)
                    for source in sources[:2]
                ]
                pitches: list[float | None] = []
                pitch_allowed: list[bool] = []
                for source in sources[:2]:
                    if self._pitch_enabled:
                        pitch, allowed = self.pitch_guard.classify(
                            source, margin_hz=self._pitch_margin_hz
                        )
                    else:
                        pitch, allowed = None, True
                    pitches.append(pitch)
                    pitch_allowed.append(allowed)

                selected, gain = self._choose_pitch_locked_target(
                    similarities, pitches, pitch_allowed
                )
                if selected is None:
                    target_16k = np.zeros_like(chunk_16k, dtype=np.float32)
                    selected_similarity = None
                else:
                    target_16k = np.asarray(sources[selected] * gain, dtype=np.float32)
                    selected_similarity = float(similarities[selected])

                target_48k = self._resample(
                    torch, torchaudio, target_16k, base.SEPARATION_RATE, base.AUDIO_RATE, device
                )
                target_48k = self._fit_length(target_48k, chunk_48k.size)
                target_48k = self._safe_level(target_48k, chunk_48k)

                elapsed = time.perf_counter() - started
                rtf = elapsed / max(self._preset.hop_seconds, 1e-6)
                metrics = PitchSeparationMetrics(
                    processing_seconds=elapsed,
                    realtime_factor=rtf,
                    similarity_a=float(similarities[0]),
                    similarity_b=float(similarities[1]),
                    selected_source=selected,
                    selected_similarity=selected_similarity,
                    queue_depth=self._submit_queue.qsize(),
                    device=device,
                    fallback=False,
                    pitch_a_hz=pitches[0],
                    pitch_b_hz=pitches[1],
                    pitch_a_allowed=pitch_allowed[0],
                    pitch_b_allowed=pitch_allowed[1],
                    pitch_limit_hz=(
                        0.0
                        if pitch_profile is None
                        else pitch_profile.upper_limit_hz + self._pitch_margin_hz
                    ),
                    fail_closed=self._fail_closed,
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
            self._set_fallback(f"Pitch-Locked v2 error: {exc}")
        finally:
            self._ready_event.set()

    def _choose_pitch_locked_target(
        self,
        similarities: list[float],
        pitches: list[float | None],
        pitch_allowed: list[bool],
    ) -> tuple[int | None, float]:
        # Start from the ordinary voiceprint thresholds.
        order = np.argsort(np.asarray(similarities, dtype=np.float32))[::-1]
        thresholds = {
            "Conservative": (0.12, 0.19),
            "Balanced": (0.16, 0.23),
            "Aggressive": (0.20, 0.28),
        }
        floor, full = thresholds.get(self._strictness, thresholds["Balanced"])

        for raw_index in order:
            index = int(raw_index)
            score = float(similarities[index])
            if score < floor:
                continue

            if self._pitch_enabled:
                has_pitch = pitches[index] is not None
                if has_pitch and not pitch_allowed[index]:
                    continue
                if self._fail_closed and not has_pitch:
                    continue

            if score < full:
                gain = 0.30 + 0.55 * ((score - floor) / max(full - floor, 1e-6))
            else:
                gain = 1.0

            other = 1 - index if len(similarities) > 1 else index
            margin = score - float(similarities[other])
            if margin < 0.035:
                if self._fail_closed:
                    return None, 0.0
                gain *= 0.65
            return index, float(np.clip(gain, 0.0, 1.0))

        return None, 0.0
