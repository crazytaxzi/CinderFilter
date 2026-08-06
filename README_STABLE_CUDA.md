# Stable Stateful CUDA Main Noise Reducer

This replaces the first experimental CUDA denoiser loop rather than hiding its failures.

## Root fixes

- Persistent DeepFilter STFT and normalization state.
- Persistent GRU hidden state across chunks.
- Persistent causal-convolution and multi-frame deep-filter context.
- Non-overlapping chunks; the old bridge processed every sample twice.
- No redundant global CUDA synchronization per chunk.
- No three-slow-window kill switch.
- Stale queued audio is discarded instead of abandoning CUDA.
- Genuine hangs are detected by response timeout and full tracebacks are logged.
- Genuine worker exits are restarted automatically.
- Explicit `CUDA` mode blocks CPU denoiser creation and mutes during recovery.
- `Auto` is the only mode allowed to fall back to CPU.

## Install and verify

1. Close CinderFilter.
2. Extract this patch over the CinderFilter directory.
3. Run `REPAIR_AND_VERIFY_STABLE_CUDA.bat`.
4. The script repairs the small DeepFilter dependency stack without redownloading PyTorch.
5. It runs 20 consecutive 500 ms chunks on the installed GPU and requires p95 RTF below 1.0.
6. On success it saves `CUDA / Balanced` and relaunches CinderFilter.

The usual launcher remains `START_CINDERFILTER_V2_PITCH.bat`.

Worker crashes and full Python tracebacks are written to:

`%LOCALAPPDATA%\CinderFilter\cuda-noise-worker.log`
