# CUDA Main Noise Reducer

This build adds a genuine backend selector for the primary noise-reduction stage:

- **CUDA**: DeepFilterNet3's PyTorch model runs on the NVIDIA GPU. The Rust CPU denoiser is not created.
- **Auto**: prefers the CUDA sidecar, then uses the CPU Rust engine if CUDA is unavailable.
- **CPU**: uses the existing low-latency `deepfilternet-rs` engine.

The CUDA model runs in an isolated Python 3.11 sidecar because the official DeepFilterNet Windows library provides CPython 3.11 binaries, while the desktop UI currently runs Python 3.12. Audio is exchanged through a local authenticated Windows named pipe; no network port is opened.

## Install

1. Copy the patch into the CinderFilter directory.
2. Close CinderFilter.
3. Run `INSTALL_CUDA_NOISE_ENGINE.bat` once, or click **Install / Repair CUDA Noise Engine** in the application.
4. Wait for the CUDA and DeepFilterNet3 self-test to finish.
5. Select **Noise Engine: CUDA**.

The first model load may download DeepFilterNet3. `Low Latency` uses 200 ms overlapping windows, `Balanced` 400 ms, and `Quality` 800 ms. The UI reports real-time factor; values below 1.0 mean the GPU is keeping up.
