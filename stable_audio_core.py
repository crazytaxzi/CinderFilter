from __future__ import annotations

import queue
import time

import numpy as np

from cinderfilter_core import CinderFilterEngine
from output_leveler import OutputLevelMetrics, OutputLeveler


class StabilizedCinderFilterEngine(CinderFilterEngine):
    """Unified audio engine with one final output-level owner."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_leveler = OutputLeveler("Stable")
        self._last_leveler_event = 0.0

    @property
    def output_leveling_preset(self) -> str:
        return self.output_leveler.preset

    def set_output_leveling(self, preset: str) -> None:
        self.output_leveler.configure(preset)
        self.on_event("leveler_preset", self.output_leveler.preset)

    def _reset_stream_state(self) -> None:
        super()._reset_stream_state()
        self.output_leveler.reset()
        self._last_leveler_event = 0.0

    def _output_callback(self, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            self.on_status(f"Output stream: {status}")
        while self._pending_output.size < frames:
            try:
                self._pending_output = np.concatenate(
                    (self._pending_output, self._output_queue.get_nowait())
                )
            except queue.Empty:
                break

        available = min(frames, self._pending_output.size)
        mono = np.zeros(frames, np.float32)
        if available:
            mono[:available] = self._pending_output[:available]
            self._pending_output = self._pending_output[available:]
        if available < frames:
            self._output_underruns += 1

        if self._bypass.is_set():
            stabilized = np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)
            metrics = OutputLevelMetrics(preset="Bypass")
        else:
            stabilized, metrics = self.output_leveler.process(mono)

        self._output_level = self._peak_db(stabilized)
        if self._output_channels == 1:
            outdata[:, 0] = stabilized
        else:
            outdata[:, :] = stabilized[:, None]

        now = time.monotonic()
        if now - self._last_leveler_event >= 0.10:
            self._last_leveler_event = now
            self.on_event("leveler_metrics", metrics)
