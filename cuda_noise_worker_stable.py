from __future__ import annotations

import argparse
import contextlib
import json
import os
import struct
import sys
import time
import traceback
from multiprocessing.connection import Client
from typing import Any, Iterator

import numpy as np

REQUEST_HEADER = struct.Struct("<II")
RESPONSE_HEADER = struct.Struct("<IIf")
AUDIO_RATE = 48_000
FRAME_SAMPLES = 480


def install_torchaudio_backend_compat() -> None:
    """Provide the legacy type-only module imported by DeepFilterNet 0.5.6."""
    try:
        from torchaudio.backend.common import AudioMetaData as _AudioMetaData  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass

    import types
    from typing import NamedTuple

    import torchaudio

    class AudioMetaData(NamedTuple):
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    backend_module = types.ModuleType("torchaudio.backend")
    backend_module.__path__ = []
    common_module = types.ModuleType("torchaudio.backend.common")
    common_module.AudioMetaData = AudioMetaData
    backend_module.common = common_module
    sys.modules["torchaudio.backend"] = backend_module
    sys.modules["torchaudio.backend.common"] = common_module
    setattr(torchaudio, "backend", backend_module)


@contextlib.contextmanager
def suppress_native_stderr() -> Iterator[None]:
    """Hide DeepFilterNet's harmless git-version probe during package import.

    The installed wheel may call ``git`` to describe a source checkout.  A normal
    wheel install has no ``.git`` directory, so git prints a fatal warning even
    though model loading succeeds.  Only that import/load section is silenced;
    worker exceptions still print full tracebacks normally.
    """

    saved_fd: int | None = None
    null_fd: int | None = None
    try:
        saved_fd = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
        yield
    finally:
        if saved_fd is not None:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
        if null_fd is not None:
            os.close(null_fd)


def build_df_state() -> Any:
    from df.model import ModelParams
    from libdf import DF

    p = ModelParams()
    return DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )


def load_model() -> tuple[Any, Any, dict[str, int]]:
    install_torchaudio_backend_compat()
    with suppress_native_stderr():
        import torch
        from df.enhance import init_df
        from df.model import ModelParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch cannot access an NVIDIA GPU")

    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    with suppress_native_stderr():
        model, _unused_state, _ = init_df(
            model_base_dir="DeepFilterNet3",
            post_filter=False,
            log_level="ERROR",
            log_file=None,
            config_allow_defaults=True,
        )
    model.eval().to("cuda:0")

    p = ModelParams()
    lookahead = {
        "df": int(getattr(p, "df_lookahead", 0)),
        "conv": int(getattr(p, "conv_lookahead", 0)),
    }
    return model, torch, lookahead


def enhance_window(
    model: Any,
    torch: Any,
    audio: np.ndarray,
    atten_lim_db: float,
) -> np.ndarray:
    """Run the official whole-window path and return exactly the input length."""

    from df.enhance import enhance

    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    state = build_df_state()
    waveform = torch.from_numpy(audio).reshape(1, -1)
    with torch.inference_mode():
        result = enhance(
            model,
            state,
            waveform,
            pad=True,
            atten_lim_db=float(atten_lim_db),
        )
    output = result.detach().float().cpu().numpy().reshape(-1)
    if output.size < audio.size:
        output = np.pad(output, (0, audio.size - output.size))
    elif output.size > audio.size:
        output = output[: audio.size]
    return np.ascontiguousarray(np.clip(output, -1.0, 1.0), dtype=np.float32)


class LookaheadContextDenoiser:
    """Lookahead-aware overlap-save adapter around the official model path.

    Each request contains one target chunk followed by the exact future context
    needed by the trained model.  Real past context is retained across requests.
    The worker enhances the full context window and emits only the center target.
    """

    def __init__(
        self,
        model: Any,
        torch: Any,
        atten_lim_db: float,
        chunk_samples: int,
        model_lookahead: dict[str, int],
    ) -> None:
        if chunk_samples <= 0 or chunk_samples % FRAME_SAMPLES != 0:
            raise RuntimeError("Chunk size must be a positive multiple of 480 samples")
        self.model = model
        self.torch = torch
        self.atten_lim_db = float(atten_lim_db)
        self.chunk_samples = int(chunk_samples)

        required = max(model_lookahead.values(), default=0)
        self.future_frames = max(4, required + 2)
        chunk_frames = self.chunk_samples // FRAME_SAMPLES
        self.past_frames = min(30, max(12, int(round(chunk_frames * 0.40))))
        self.future_samples = self.future_frames * FRAME_SAMPLES
        self.past_samples = self.past_frames * FRAME_SAMPLES
        self.request_samples = self.chunk_samples + self.future_samples

        self._past = np.zeros(self.past_samples, dtype=np.float32)

    @property
    def nominal_latency_samples(self) -> int:
        return self.request_samples

    def process(self, audio_with_future: np.ndarray) -> np.ndarray:
        request = np.ascontiguousarray(
            np.asarray(audio_with_future, dtype=np.float32).reshape(-1)
        )
        if request.size != self.request_samples:
            raise RuntimeError(
                f"Expected {self.request_samples} samples including future context, "
                f"received {request.size}"
            )

        target = request[: self.chunk_samples]
        future = request[self.chunk_samples :]
        window = np.concatenate((self._past, target, future)).astype(np.float32, copy=False)
        enhanced = enhance_window(self.model, self.torch, window, self.atten_lim_db)
        start = self.past_samples
        stop = start + self.chunk_samples
        output = np.ascontiguousarray(enhanced[start:stop], dtype=np.float32)
        if output.size != self.chunk_samples:
            raise RuntimeError(
                f"Lookahead crop returned {output.size} samples, expected {self.chunk_samples}"
            )

        if self.past_samples:
            history = np.concatenate((self._past, target))
            self._past = np.ascontiguousarray(
                history[-self.past_samples :], dtype=np.float32
            )
        return output

def load_runtime(atten_lim_db: float, chunk_samples: int):
    model, torch, lookahead = load_model()

    # Warm the exact official path on a disposable adapter.  The live adapter is
    # created afterward, so its pending chunk and context begin cleanly.
    warm = LookaheadContextDenoiser(model, torch, atten_lim_db, chunk_samples, lookahead)
    silence = np.zeros(warm.request_samples, dtype=np.float32)
    warm.process(silence)
    warm.process(silence)
    torch.cuda.synchronize()

    return (
        LookaheadContextDenoiser(model, torch, atten_lim_db, chunk_samples, lookahead),
        torch,
        lookahead,
    )


def runtime_info(torch: Any, processor: LookaheadContextDenoiser, lookahead: dict[str, int]) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "model": "DeepFilterNet3",
        "sample_rate": AUDIO_RATE,
        "streaming": True,
        "stateful": False,
        "streaming_mode": "lookahead-context-overlap-save",
        "df_lookahead_frames": int(lookahead.get("df", 0)),
        "conv_lookahead_frames": int(lookahead.get("conv", 0)),
        "past_context_ms": round(processor.past_samples * 1000.0 / AUDIO_RATE, 1),
        "future_context_ms": round(processor.future_samples * 1000.0 / AUDIO_RATE, 1),
        "latency_ms": round(processor.nominal_latency_samples * 1000.0 / AUDIO_RATE, 1),
        "future_context_samples": int(processor.future_samples),
        "request_samples": int(processor.request_samples),
    }


def self_test(atten_lim_db: float, chunk_samples: int) -> int:
    processor, torch, lookahead = load_runtime(atten_lim_db, chunk_samples)
    rng = np.random.default_rng(7)
    timings: list[float] = []
    valid = True

    total = 20 * chunk_samples + processor.future_samples
    t = np.arange(total, dtype=np.float32) / AUDIO_RATE
    stream = 0.004 * rng.standard_normal(total).astype(np.float32)
    stream += (0.015 * np.sin(2 * np.pi * 130.0 * t)).astype(np.float32)

    for index in range(20):
        offset = index * chunk_samples
        request = stream[offset : offset + processor.request_samples]
        started = time.perf_counter()
        output = processor.process(request)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        valid = valid and output.size == chunk_samples and bool(np.isfinite(output).all())

    info = runtime_info(torch, processor, lookahead)
    duration = chunk_samples / AUDIO_RATE
    info.update(
        {
            "ok": bool(valid),
            "chunks": len(timings),
            "chunk_ms": round(duration * 1000.0, 1),
            "median_rtf": round(float(np.median(timings)) / duration, 4),
            "p95_rtf": round(float(np.percentile(timings, 95)) / duration, 4),
        }
    )
    print(json.dumps(info), flush=True)
    return 0 if valid else 1

def serve(pipe_name: str, auth_hex: str, atten_lim_db: float, chunk_samples: int) -> int:
    processor, torch, lookahead = load_runtime(atten_lim_db, chunk_samples)
    conn = Client(pipe_name, family="AF_PIPE", authkey=bytes.fromhex(auth_hex))
    try:
        info = runtime_info(torch, processor, lookahead)
        info["status"] = "ready"
        conn.send_bytes(json.dumps(info).encode("utf-8"))

        while True:
            payload = conn.recv_bytes()
            if len(payload) < REQUEST_HEADER.size:
                continue
            sequence, count = REQUEST_HEADER.unpack_from(payload, 0)
            if count == 0:
                break
            expected = REQUEST_HEADER.size + count * 4
            if len(payload) != expected:
                raise RuntimeError(
                    f"Bad CUDA noise request length: got {len(payload)}, expected {expected}"
                )
            audio = np.frombuffer(payload, dtype="<f4", offset=REQUEST_HEADER.size, count=count)
            started = time.perf_counter()
            output = processor.process(np.ascontiguousarray(audio, dtype=np.float32))
            elapsed = time.perf_counter() - started
            response = RESPONSE_HEADER.pack(sequence, output.size, float(elapsed)) + output.tobytes()
            conn.send_bytes(response)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe")
    parser.add_argument("--auth")
    parser.add_argument("--atten", type=float, default=45.0)
    parser.add_argument("--chunk-samples", type=int, default=24_000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            return self_test(args.atten, args.chunk_samples)
        if not args.pipe or not args.auth:
            raise RuntimeError("--pipe and --auth are required")
        return serve(args.pipe, args.auth, args.atten, args.chunk_samples)
    except BaseException as exc:
        print(
            f"Lookahead-safe CUDA noise worker failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
