from __future__ import annotations

"""Thread-affinity hotfix for the DeepFilterNet Rust/PyO3 realtime object.

DeepFilterNetRealtime is marked unsendable, so its constructor, process calls,
and close call must all happen on the same Python thread.
"""

import cinderfilter as cf


def _start(self, input_device: int, output_device: int, strength: str) -> None:
    if self.running:
        self.stop()

    self._clear_queues()
    self._dropped_input = 0
    self._output_underruns = 0
    self._pending_output = cf.np.empty(0, dtype=cf.np.float32)

    out_info = cf.sd.query_devices(output_device)
    max_out = int(out_info["max_output_channels"])
    if max_out < 1:
        raise RuntimeError("The selected output device has no playback channels.")
    self._output_channels = 2 if max_out >= 2 else 1

    self._atten_lim = self.STRENGTHS.get(strength, self.STRENGTHS["Balanced"])
    self._model_ready = cf.threading.Event()
    self._worker_error: BaseException | None = None
    self.on_status("Loading DeepFilterNet model on AI thread...")

    self._running.set()
    self._worker = cf.threading.Thread(
        target=self._process_loop,
        name="CinderFilter-AI",
        daemon=True,
    )
    self._worker.start()

    if not self._model_ready.wait(timeout=30.0):
        self.stop()
        raise RuntimeError("DeepFilterNet model loading timed out.")
    if self._worker_error is not None:
        error = self._worker_error
        self.stop()
        raise RuntimeError(f"DeepFilterNet failed to initialize: {error}") from error
    if not self._running.is_set():
        self.stop()
        raise RuntimeError("The AI worker stopped before audio streams could start.")

    try:
        self._output_stream = cf.sd.OutputStream(
            device=output_device,
            samplerate=cf.SAMPLE_RATE,
            blocksize=0,
            channels=self._output_channels,
            dtype="float32",
            latency="low",
            callback=self._output_callback,
        )
        self._input_stream = cf.sd.InputStream(
            device=input_device,
            samplerate=cf.SAMPLE_RATE,
            blocksize=0,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._input_callback,
        )
        self._input_stream.start()
        prime_deadline = cf.time.monotonic() + 0.15
        while self._output_queue.qsize() < 2 and cf.time.monotonic() < prime_deadline:
            cf.time.sleep(0.005)
        self._output_stream.start()
    except Exception:
        self.stop()
        raise

    self.on_status("LIVE — AI filtering active")


def _stop(self) -> None:
    self._running.clear()

    for stream in (self._input_stream, self._output_stream):
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    self._input_stream = None
    self._output_stream = None

    if self._worker and self._worker.is_alive():
        self._worker.join(timeout=1.5)
    self._worker = None

    # Never close DeepFilterNet here. The AI worker owns and closes it.
    self._processor = None
    self._clear_queues()
    self.on_status("Stopped")


def _process_loop(self) -> None:
    processor: cf.DeepFilterNetRealtime | None = None
    last_metrics = 0.0

    try:
        processor = cf.DeepFilterNetRealtime(
            model_path=None,
            atten_lim=self._atten_lim,
            log_level="warn",
            compensate_delay=False,
            post_filter_beta=0.0,
        )
        if int(processor.sample_rate) != cf.SAMPLE_RATE:
            raise RuntimeError(
                f"Model requires {processor.sample_rate} Hz, expected {cf.SAMPLE_RATE} Hz."
            )
        self._model_ready.set()

        while self._running.is_set():
            try:
                block = self._input_queue.get(timeout=0.1)
            except cf.queue.Empty:
                continue

            if self._bypass.is_set():
                processed = block
            else:
                processed = processor.process_chunk(block)
                processed = cf.np.asarray(processed, dtype=cf.np.float32).reshape(-1)

            if processed.size == 0:
                continue
            processed = cf.np.clip(processed, -1.0, 1.0)
            self._output_level = self._peak_db(processed)

            try:
                self._output_queue.put_nowait(processed)
            except cf.queue.Full:
                try:
                    self._output_queue.get_nowait()
                    self._output_queue.put_nowait(processed)
                except cf.queue.Empty:
                    pass

            now = cf.time.monotonic()
            if now - last_metrics >= 0.10:
                self.on_metrics(
                    self._input_level,
                    self._output_level,
                    self._dropped_input,
                    self._output_underruns,
                )
                last_metrics = now
    except BaseException as exc:
        self._worker_error = exc
        self._last_status = f"AI processing error: {exc}"
        self.on_status(self._last_status)
        self._running.clear()
    finally:
        if processor is not None:
            try:
                processor.close()
            except BaseException:
                pass
        self._model_ready.set()


cf.AudioEngine.start = _start
cf.AudioEngine.stop = _stop
cf.AudioEngine._process_loop = _process_loop


if __name__ == "__main__":
    cf.main()
