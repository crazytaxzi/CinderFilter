# CinderFilter consolidation audit

## Source inspected

Audit base: `main` at commit `ff25bc91cf783345e8b2f19f94cdbbe4630b1a66`.

## Problems found

1. **Nine application entry layers.** The original Tkinter app was successively subclassed for thread safety, Voice Lock, v2 separation, Pitch Lock, saved settings, CUDA detection, CUDA noise, and stable CUDA.
2. **Three supported-looking launchers.** They started materially different applications.
3. **Patch tooling in the product root.** Installation, force-install, repair, verification, and conflict scripts were mixed with runtime code.
4. **Duplicate CUDA implementations.** Experimental and stable bridges/workers remained present.
5. **Disconnected visual shell.** `main.py` displayed hardcoded devices, GPU state, meters, and placeholder pages rather than controlling the audio engine.
6. **Broken clean-install manifest.** `requirements.txt` contained only PySide6 after the UI overlay.
7. **Legacy UI coupling inside services.** `voice_lock.py` imported the original Tkinter application even when only the embedding service was needed.
8. **README fragmentation.** Patch-specific documentation contradicted other patch-specific documentation.

## Consolidated structure

```text
main.py                  application entry
cinderfilter_window.py   integrated frameless PySide6 UI
ui_components.py         custom dark/neon controls and animations
cinderfilter_core.py     single audio/controller pipeline
voice_lock.py            speaker-profile service only
target_separator_v2.py   target-speaker separation
pitch_lock_v2.py         pitch profile and pitch-aware selection
cuda_noise_bridge.py     one supervised CUDA bridge
cuda_noise_worker.py     one lookahead-safe CUDA worker
gpu_detector.py          hardware/runtime diagnostics
settings_store.py        atomic persistent settings
setup-and-run.ps1        one bootstrap path
START_CINDERFILTER.bat   one supported launcher
tools/                   runtime installers
```

## Removed from the checked-out product

- all `cinderfilter_voice_lock*.py` application variants
- `cinderfilter.py` and `cinderfilter_threadsafe.py`
- duplicate CUDA bridge/worker variants
- v2 and pitch alternate launchers
- top-level force-install, repair, and verification scripts
- patch notes and patch-specific README files

The removed files remain available in Git history at the audit-base commit.

## Safety rules retained

- DeepFilterNet Rust objects stay on their creator worker thread.
- Explicit CUDA mode never instantiates the CPU denoiser.
- Fail-closed target extraction produces silence on uncertainty or failure.
- Bypass is rejected while fail-closed target extraction is enabled.
- Physical input and processed output are independently selected and saved by endpoint identity.
- Voice and pitch profiles remain local.

## Verification

- Python compilation is enforced by GitHub Actions.
- Ruff checks undefined names and syntax-critical errors.
- Tests enforce one top-level launcher, no Tkinter runtime imports, and no hardcoded demo hardware.
- The CUDA installer performs a 20-window RTX acceptance test and requires p95 RTF below 1.0.

A live Windows audio/CUDA test still has to run on the target RTX 4070 system; no container test can substitute for the actual PortAudio endpoints, NVIDIA driver, and VoiceMeeter route.
