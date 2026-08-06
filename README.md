# CinderFilter

CinderFilter is one Windows desktop application for microphone cleanup, target-speaker filtering, pitch distinction, and routing into VoiceMeeter or another virtual audio endpoint.

```text
Physical microphone
  -> optional target-speaker extraction + pitch guard
  -> main noise reducer (CUDA or CPU)
  -> Voice Lock verification gate
  -> VoiceMeeter / VB-CABLE / hardware output
```

## One supported launcher

Double-click `START_CINDERFILTER.bat`.

There are no v1/v2/pitch/GPU alternate launchers. The first launch creates `.venv`, installs the UI and base runtime, selects CUDA PyTorch when an NVIDIA driver is present, and opens the unified application. Later launches skip installation unless `requirements.txt` changes.

## Interface

The PySide6 interface is frameless and follows the Cinder stream theme: charcoal glass panels, electric purple and neon green accents, custom dark dropdowns, animated meters, an ember backdrop, and smooth sidebar transitions. Every selector, meter, status pill, and profile state is connected to the real engine.

Pages:

- **Overview:** routing, main reducer, Voice Lock, Pitch Lock, live meters, and runtime diagnostics.
- **Devices:** Windows input/output endpoints and host APIs.
- **Voice:** enrollment, pitch calibration, profile deletion, and target-extraction preload.
- **Tuning:** strength, strictness, pitch margin, and latency presets.
- **Advanced:** internal CUDA installers; no patch files are added to the product root.
- **Diagnostics:** actual backend, RTF, queues, dropouts, and worker logs.

## Recommended VoiceMeeter Potato route

```text
CinderFilter input:  physical microphone
CinderFilter output: VoiceMeeter AUX Input (or another virtual playback endpoint)
Potato input:         matching virtual input strip
Potato bus:           B1/B2/B3 as required
```

Do not route the physical microphone into Potato at the same time. That would mix raw audio beside CinderFilter's processed output.

## Profiles and settings

- Voice and pitch profiles remain local under `profiles/`.
- Models are cached under `models/`.
- Settings live under `%LOCALAPPDATA%\CinderFilter\settings.json`.
- Old patch-era setting names are migrated automatically.
- Raw enrollment audio is not saved.

## CUDA behavior

- **CUDA:** the main DeepFilterNet3 worker must use NVIDIA CUDA. A genuine worker failure mutes output and restarts CUDA; it does not secretly instantiate the CPU denoiser.
- **Auto:** prefers CUDA and may use the CPU reducer when CUDA is unavailable.
- **CPU:** uses `deepfilternet-rs` on the processor.

The CUDA main reducer uses an isolated Python 3.11 sidecar because DeepFilterNet 0.5.6 provides a Windows CPython 3.11 binary. Install or repair it from **Advanced -> Install / Repair CUDA Noise Engine**.

## Clean migration from the patch-stacked folder

The consolidation commit removes all old tracked variants. Manually extracted ZIP files may remain untracked, so a clean checkout is safer:

```powershell
cd C:\Projects
Rename-Item CinderFilter CinderFilter.patch-backup
git clone https://github.com/crazytaxzi/CinderFilter.git CinderFilter

$old = 'C:\Projects\CinderFilter.patch-backup'
$new = 'C:\Projects\CinderFilter'
foreach ($name in '.venv','.venv_cuda_noise','models','profiles') {
    if (Test-Path "$old\$name") { Move-Item "$old\$name" "$new\$name" -Force }
}
```

Then run `C:\Projects\CinderFilter\START_CINDERFILTER.bat`. Keep the backup until the unified application passes a live test.

## Logs

CUDA worker tracebacks are written to `%LOCALAPPDATA%\CinderFilter\cuda-noise-worker.log`.

## License

MIT. DeepFilterNet, PyTorch, SpeechBrain, SepFormer, and pretrained model files retain their own licenses.
