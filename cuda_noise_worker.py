from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from multiprocessing.connection import Client

import numpy as np

REQUEST_HEADER = struct.Struct("<II")
RESPONSE_HEADER = struct.Struct("<IIf")


def build_df_state():
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


def load_model():
    import torch
    from df.enhance import init_df

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch cannot access an NVIDIA GPU")

    torch.set_grad_enabled(False)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

    loaded = init_df(
        model_base_dir="DeepFilterNet3",
        post_filter=False,
        log_level="ERROR",
        log_file=None,
        config_allow_defaults=True,
    )
    model = loaded[0]
    model.eval()
    device = next(model.parameters()).device
    if device.type != "cuda":
        model = model.to("cuda:0")
        device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"DeepFilterNet3 loaded on {device}, expected CUDA")
    return model, torch


def enhance_chunk(model, torch, audio: np.ndarray, atten_lim_db: float) -> np.ndarray:
    from df.enhance import enhance

    state = build_df_state()
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
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


def self_test(atten_lim_db: float) -> int:
    model, torch = load_model()
    test = np.zeros(48_000 // 4, dtype=np.float32)
    started = time.perf_counter()
    output = enhance_chunk(model, torch, test, atten_lim_db)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    props = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "ok": bool(output.size == test.size),
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "warmup_seconds": round(elapsed, 4),
            }
        )
    )
    return 0


def serve(pipe_name: str, auth_hex: str, atten_lim_db: float) -> int:
    model, torch = load_model()

    warm = np.zeros(48_000 // 4, dtype=np.float32)
    enhance_chunk(model, torch, warm, atten_lim_db)
    torch.cuda.synchronize()

    conn = Client(pipe_name, family="AF_PIPE", authkey=bytes.fromhex(auth_hex))
    try:
        props = torch.cuda.get_device_properties(0)
        conn.send_bytes(
            json.dumps(
                {
                    "status": "ready",
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                    "model": "DeepFilterNet3",
                    "sample_rate": 48_000,
                }
            ).encode("utf-8")
        )

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
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            started = time.perf_counter()
            output = enhance_chunk(model, torch, audio, atten_lim_db)
            torch.cuda.synchronize()
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
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            return self_test(args.atten)
        if not args.pipe or not args.auth:
            raise RuntimeError("--pipe and --auth are required")
        return serve(args.pipe, args.auth, args.atten)
    except BaseException as exc:
        print(f"CUDA noise worker failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
