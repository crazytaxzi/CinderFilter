from __future__ import annotations

import queue
import threading
import time

import numpy as np
from deepfilternet_rs import DeepFilterNetRealtime

import cinderfilter_voice_lock_v2_pitch_saved_gpu_noise as legacy
from cuda_noise_bridge_stable import StableCudaNoiseBridge
from voice_lock import SAMPLE_RATE


class StableCudaMainNoiseAudioEngine(legacy.CudaMainNoiseAudioEngine):
    """CUDA-locked engine with supervised worker recovery and no false CPU fallback."""

    def __init__(self, voice_service, separator, cuda_noise, on_status, on_metrics) -> None:
        super().__init__(voice_service, separator, cuda_noise, on_status, on_metrics)
        self._recovery_thread: threading.Thread | None = None
        self._recovery_stop = threading.Event()
        self._recovery_lock = threading.Lock()

    def start(self, input_device: int, output_device: int, strength: str) -> None:
        self._recovery_stop.clear()
        super().start(input_device, output_device, strength)

    def stop(self) -> None:
        self._recovery_stop.set()
        try:
            super().stop()
        finally:
            thread = self._recovery_thread
            if thread and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
            self._recovery_thread = None

    def _begin_cuda_recovery(self) -> None:
        if self._noise_backend != "CUDA" or self._recovery_stop.is_set():
            return
        with self._recovery_lock:
            if self._recovery_thread and self._recovery_thread.is_alive():
                return
            self._recovery_thread = threading.Thread(
                target=self._cuda_recovery_loop,
                name="CinderFilter-CUDA-Recovery",
                daemon=True,
            )
            self._recovery_thread.start()

    def _cuda_recovery_loop(self) -> None:
        delays = (0.5, 1.0, 2.0, 5.0, 10.0)
        attempt = 0
        while (
            not self._recovery_stop.is_set()
            and self._running.is_set()
            and self._noise_backend == "CUDA"
        ):
            delay = delays[min(attempt, len(delays) - 1)]
            if self._recovery_stop.wait(delay):
                return
            try:
                self.on_status(
                    f"Restarting lookahead-safe CUDA denoiser (attempt {attempt + 1}) — output muted, CPU fallback disabled"
                )
                self.cuda_noise.start(self._atten_lim, self._noise_preset)
                if self._recovery_stop.is_set() or not self._running.is_set():
                    self.cuda_noise.stop()
                    return
                self._cuda_noise_active = True
                self.on_status(
                    f"Lookahead-safe CUDA denoiser recovered on {self.cuda_noise.device}"
                )
                return
            except BaseException as exc:
                self._cuda_noise_active = False
                self.on_status(f"CUDA recovery attempt {attempt + 1} failed: {exc}")
                attempt += 1

    def _new_cpu_processor(self) -> DeepFilterNetRealtime:
        # CPU is legal only in explicit CPU or Auto mode. This guard prevents a
        # future code path from quietly undoing CUDA-locked operation.
        if self._noise_backend == "CUDA":
            raise RuntimeError("CPU denoiser creation blocked while Noise Engine is CUDA")
        return super()._new_cpu_processor()

    def _process_loop(self) -> None:
        processor: DeepFilterNetRealtime | None = None
        last_metrics = 0.0
        try:
            if not self._cuda_noise_active:
                if self._noise_backend == "CUDA":
                    raise RuntimeError("CUDA-locked noise engine started without an active CUDA worker")
                processor = super()._new_cpu_processor()
                self.on_status("Main noise reducer: CPU Rust DeepFilterNet")
            else:
                self.on_status(
                    f"Main noise reducer: lookahead-safe CUDA DeepFilterNet3 on {self.cuda_noise.device}"
                )
            self._model_ready.set()

            while self._running.is_set():
                try:
                    raw_block = self._input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                using_v2 = False
                source = raw_block
                if not self._bypass.is_set():
                    if self._v2_active:
                        extracted = self.separator.process_block(raw_block)
                        if extracted is None:
                            self._v2_active = False
                            if self._strict_fail_closed:
                                self._v2_failed_closed = True
                                source = np.zeros_like(raw_block, dtype=np.float32)
                                using_v2 = True
                                self.on_status(
                                    "STRICT PITCH LOCK: v2 failed or overloaded — output muted"
                                )
                            else:
                                self.on_status("Pitch Lock v2 fell back to v1")
                        else:
                            source = extracted
                            using_v2 = True
                    elif self._v2_requested and self._strict_fail_closed and self._v2_failed_closed:
                        source = np.zeros_like(raw_block, dtype=np.float32)
                        using_v2 = True

                if self._bypass.is_set():
                    processed = raw_block
                elif self._cuda_noise_active:
                    cuda_output = self.cuda_noise.process_block(source)
                    if cuda_output is None:
                        self._cuda_noise_active = False
                        if self._noise_backend == "CUDA":
                            # Explicit CUDA means CUDA, not a surprise 50% CPU spike.
                            processed = np.zeros_like(source, dtype=np.float32)
                            self.on_status(
                                "CUDA worker stopped — output muted while it restarts; CPU fallback is disabled"
                            )
                            self._begin_cuda_recovery()
                        else:
                            self.on_status(
                                "CUDA main denoiser failed in Auto mode — using CPU fallback"
                            )
                            processor = super()._new_cpu_processor()
                            processed = np.asarray(
                                processor.process_chunk(source), dtype=np.float32
                            ).reshape(-1)
                    else:
                        processed = np.asarray(cuda_output, dtype=np.float32).reshape(-1)
                elif self._noise_backend == "CUDA":
                    processed = np.zeros_like(source, dtype=np.float32)
                    self._begin_cuda_recovery()
                else:
                    if processor is None:
                        processor = super()._new_cpu_processor()
                    processed = np.asarray(
                        processor.process_chunk(source), dtype=np.float32
                    ).reshape(-1)

                if processed.size == 0:
                    continue
                processed = np.clip(processed, -1.0, 1.0)

                if using_v2:
                    similarity = self.separator.last_similarity
                    with self._voice_guard:
                        self._voice_similarity = similarity
                        self._voice_target_gain = 1.0
                        self._voice_current_gain = 1.0
                    gain = 1.0
                else:
                    self._feed_voice_lock(processed)
                    processed = self._apply_voice_gain(processed)
                    with self._voice_guard:
                        gain = self._voice_current_gain
                        similarity = self._voice_similarity

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


class StableCudaMainNoiseApp(legacy.CudaMainNoiseApp):
    def __init__(self) -> None:
        super().__init__()
        self.title("CinderFilter — Lookahead-Safe CUDA Noise Reducer")

        # Dispose of the original offline/chunked CUDA bridge and replace it.
        try:
            self.cuda_noise.stop()
        except Exception:
            pass
        self.cuda_noise = StableCudaNoiseBridge(
            self._thread_noise_status,
            self._thread_noise_metrics,
        )
        self.engine = StableCudaMainNoiseAudioEngine(
            self.voice_service,
            self.separator,
            self.cuda_noise,
            self._thread_status,
            self._thread_metrics,
        )
        self.noise_status_var.set(
            "CUDA engine ready to load — trained lookahead preserved, supervised restart, no 3-strike fallback"
        )
        self._refresh_noise_install_state()

    def _refresh_noise_install_state(self) -> None:
        if self.cuda_noise.installed:
            self.noise_status_var.set(
                "Lookahead-safe CUDA sidecar installed — CUDA mode never creates the CPU denoiser"
            )
            self.install_noise_button.configure(text="Repair CUDA Noise Engine")
        else:
            self.noise_status_var.set(
                "Stable CUDA worker is missing — apply the stable CUDA patch"
            )
            self.install_noise_button.configure(text="Install CUDA Noise Engine")


def main() -> None:
    app = StableCudaMainNoiseApp()
    app.mainloop()


if __name__ == "__main__":
    main()
