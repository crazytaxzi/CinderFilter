from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output_leveler import OUTPUT_LEVEL_PRESETS, OutputLeveler


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def main() -> int:
    sample_rate = 48_000
    leveler = OutputLeveler("Stable", sample_rate)
    time_axis = np.arange(4_800, dtype=np.float32) / sample_rate

    scenarios = (
        ("quiet", 0.02),
        ("normal", 0.10),
        ("loud", 0.58),
        ("normal-after-loud", 0.11),
    )

    print("CinderFilter output-leveler verification")
    print("Presets:", ", ".join(OUTPUT_LEVEL_PRESETS))
    for name, amplitude in scenarios:
        source = (amplitude * np.sin(2.0 * np.pi * 180.0 * time_axis)).astype(np.float32)
        if name == "normal-after-loud":
            source[700] = 0.99
        output, metrics = leveler.process(source)
        assert output.dtype == np.float32
        assert np.all(np.isfinite(output))
        assert peak(output) <= peak(source) + 1e-6, "The leveler must never boost audio"
        assert peak(output) <= 10.0 ** (-1.5 / 20.0) + 1e-5
        print(
            f"{name:18s} in={metrics.input_peak_db:6.1f} dBFS "
            f"out={metrics.output_peak_db:6.1f} dBFS "
            f"control={metrics.total_reduction_db:5.1f} dB"
        )

    print("OUTPUT LEVELER VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
