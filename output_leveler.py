from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class LevelingPreset:
    enabled: bool
    threshold_db: float
    ratio: float
    max_reduction_db: float
    attack_ms: float
    release_ms: float
    limiter_ceiling_db: float
    limiter_release_ms: float


OUTPUT_LEVEL_PRESETS: dict[str, LevelingPreset] = {
    "Off": LevelingPreset(False, -12.0, 1.0, 0.0, 20.0, 400.0, -0.5, 100.0),
    "Gentle": LevelingPreset(True, -16.0, 2.0, 6.0, 28.0, 650.0, -1.0, 130.0),
    "Stable": LevelingPreset(True, -19.0, 3.5, 10.0, 14.0, 520.0, -1.5, 110.0),
    "Broadcast": LevelingPreset(True, -22.0, 5.0, 14.0, 8.0, 360.0, -2.0, 90.0),
}


@dataclass(frozen=True)
class OutputLevelMetrics:
    preset: str = "Stable"
    compressor_reduction_db: float = 0.0
    limiter_reduction_db: float = 0.0
    total_reduction_db: float = 0.0
    input_peak_db: float = -60.0
    output_peak_db: float = -60.0
    input_rms_db: float = -60.0
    output_rms_db: float = -60.0


class OutputLeveler:
    """Downward-only speech stabilizer followed by a fast peak limiter.

    It never raises quiet audio or room noise. Loud passages are reduced with a
    slow-release compressor, then any remaining transient is caught by the
    limiter. State is continuous across callback blocks.
    """

    def __init__(self, preset: str = "Stable", sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self._lock = threading.RLock()
        self._preset_name = "Stable"
        self._preset = OUTPUT_LEVEL_PRESETS["Stable"]
        self._compressor_gain = 1.0
        self._limiter_gain = 1.0
        self._envelope = 0.0
        self._last_metrics = OutputLevelMetrics()
        self.configure(preset)

    @property
    def preset(self) -> str:
        with self._lock:
            return self._preset_name

    @property
    def metrics(self) -> OutputLevelMetrics:
        with self._lock:
            return self._last_metrics

    def configure(self, preset: str) -> None:
        selected = preset if preset in OUTPUT_LEVEL_PRESETS else "Stable"
        with self._lock:
            self._preset_name = selected
            self._preset = OUTPUT_LEVEL_PRESETS[selected]
            if not self._preset.enabled:
                self._compressor_gain = 1.0
                self._limiter_gain = 1.0
                self._envelope = 0.0

    def reset(self) -> None:
        with self._lock:
            self._compressor_gain = 1.0
            self._limiter_gain = 1.0
            self._envelope = 0.0
            self._last_metrics = OutputLevelMetrics(preset=self._preset_name)

    @staticmethod
    def _db(value: float) -> float:
        return max(-60.0, 20.0 * math.log10(max(float(value), 1e-6)))

    @staticmethod
    def _coefficient(duration_s: float, time_ms: float) -> float:
        tau = max(0.001, float(time_ms) / 1000.0)
        return 1.0 - math.exp(-max(0.0, duration_s) / tau)

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, OutputLevelMetrics]:
        block = np.asarray(audio, np.float32).reshape(-1)
        if not block.size:
            return block, self.metrics

        with self._lock:
            preset_name = self._preset_name
            preset = self._preset
            duration = block.size / max(self.sample_rate, 1)
            input_peak = float(np.max(np.abs(block)))
            input_rms = float(np.sqrt(np.mean(block * block) + 1e-12))

            if not preset.enabled:
                output = np.clip(block, -1.0, 1.0).astype(np.float32, copy=False)
                metrics = OutputLevelMetrics(
                    preset=preset_name,
                    input_peak_db=self._db(input_peak),
                    output_peak_db=self._db(float(np.max(np.abs(output)))),
                    input_rms_db=self._db(input_rms),
                    output_rms_db=self._db(float(np.sqrt(np.mean(output * output) + 1e-12))),
                )
                self._last_metrics = metrics
                return output, metrics

            # A smoothed RMS envelope prevents the compressor from chasing each
            # syllable. It rises quickly and falls slowly.
            env_attack = self._coefficient(duration, max(10.0, preset.attack_ms * 2.0))
            env_release = self._coefficient(duration, max(120.0, preset.release_ms))
            env_alpha = env_attack if input_rms > self._envelope else env_release
            self._envelope += (input_rms - self._envelope) * env_alpha
            envelope_db = self._db(self._envelope)

            if envelope_db > preset.threshold_db:
                compressed_db = preset.threshold_db + (
                    envelope_db - preset.threshold_db
                ) / max(preset.ratio, 1.0)
                target_reduction_db = max(
                    -preset.max_reduction_db,
                    compressed_db - envelope_db,
                )
            else:
                target_reduction_db = 0.0
            target_compressor_gain = 10.0 ** (target_reduction_db / 20.0)

            compressor_time = (
                preset.attack_ms
                if target_compressor_gain < self._compressor_gain
                else preset.release_ms
            )
            compressor_alpha = self._coefficient(duration, compressor_time)
            old_compressor_gain = self._compressor_gain
            self._compressor_gain += (
                target_compressor_gain - self._compressor_gain
            ) * compressor_alpha
            comp_ramp = np.linspace(
                old_compressor_gain,
                self._compressor_gain,
                block.size,
                dtype=np.float32,
            )
            compressed = block * comp_ramp

            ceiling = 10.0 ** (preset.limiter_ceiling_db / 20.0)
            compressed_peak = float(np.max(np.abs(compressed)))
            required_limiter = min(1.0, ceiling / max(compressed_peak, 1e-9))
            if required_limiter < self._limiter_gain:
                # Immediate attack: the current block cannot exceed the ceiling.
                self._limiter_gain = required_limiter
            else:
                release_alpha = self._coefficient(duration, preset.limiter_release_ms)
                self._limiter_gain += (1.0 - self._limiter_gain) * release_alpha

            output = compressed * self._limiter_gain
            output = np.clip(output, -ceiling, ceiling).astype(np.float32, copy=False)

            output_peak = float(np.max(np.abs(output)))
            output_rms = float(np.sqrt(np.mean(output * output) + 1e-12))
            compressor_reduction = max(0.0, -20.0 * math.log10(max(self._compressor_gain, 1e-6)))
            limiter_reduction = max(0.0, -20.0 * math.log10(max(self._limiter_gain, 1e-6)))
            metrics = OutputLevelMetrics(
                preset=preset_name,
                compressor_reduction_db=compressor_reduction,
                limiter_reduction_db=limiter_reduction,
                total_reduction_db=compressor_reduction + limiter_reduction,
                input_peak_db=self._db(input_peak),
                output_peak_db=self._db(output_peak),
                input_rms_db=self._db(input_rms),
                output_rms_db=self._db(output_rms),
            )
            self._last_metrics = metrics
            return output, metrics
