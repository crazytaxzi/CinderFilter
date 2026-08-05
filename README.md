# CinderFilter

CinderFilter is a Windows-first, real-time microphone preprocessor. It captures a physical microphone, runs it through a weighted DeepFilterNet speech-enhancement model, and sends the cleaned signal to any playback or virtual-audio endpoint that Windows exposes.

The first usable build is deliberately direct:

```text
Physical microphone
    -> CinderFilter weighted AI
    -> VoiceMeeter / VB-CABLE / MIXLINE playback endpoint
    -> OBS, Discord, Voicemod, or another application
```

## Run it now

1. Download or clone this repository on Windows.
2. Double-click `START_CINDERFILTER.bat`.
3. The launcher installs Python 3.12 when necessary, creates an isolated environment, installs the model runtime, and opens CinderFilter.
4. Select the actual microphone under **Input device**.
5. Select a playback or virtual endpoint under **Output device**.
6. Choose an AI strength and press **START FILTERING**.

The first launch downloads the Python packages and the bundled model runtime. Later launches skip installation unless `requirements.txt` changes.

## VoiceMeeter Potato route

A useful direct route is:

```text
Input device:  your physical microphone
Output device: VoiceMeeter AUX Input (VB-Audio VoiceMeeter AUX VAIO)
```

The cleaned audio then arrives on Potato's AUX virtual input strip. Route that strip to the required B bus and select the corresponding VoiceMeeter Output recording device in OBS, Discord, or the stream application.

A VB-CABLE route also works:

```text
CinderFilter output: CABLE Input
Next application's input: CABLE Output
```

Windows names the two ends from opposite directions. CinderFilter's output selector intentionally shows only devices that accept playback audio.

## Device swapping

The input and output dropdowns enumerate every compatible endpoint PortAudio reports, including duplicate endpoints exposed through different Windows host APIs. While filtering, select a new route and press **START FILTERING** again. The engine closes the old route, loads the requested strength, and starts the new route.

When one version of a device fails, select another version of the same device under a different host API such as WASAPI, WDM-KS, DirectSound, or MME.

## Controls

- **Natural**: lighter attenuation and more room tone.
- **Balanced**: default stream setting.
- **Strong**: heavier suppression for fans and steady background noise.
- **Maximum**: permits the model's strongest attenuation and may sound less natural.
- **Bypass AI**: keeps the route active but sends the microphone through unprocessed. Use it to prove the device route works before blaming the model.

## Technical baseline

- 48 kHz mono internal processing
- 480-sample / 10 ms model frames
- DeepFilterNet real-time weighted neural model
- Separate capture, inference, and playback paths
- Bounded queues so an overloaded model cannot grow memory forever
- Input/output peak meters and dropout counters
- Dual-mono output when the selected virtual endpoint expects stereo

## Current limits

This is the rapid working baseline, not the finished product. It does not yet contain a custom Windows virtual microphone driver, automatic noise classification, speaker identity locking, persistent route presets, or a packaged `.exe`. Those belong after the real route is tested against actual VoiceMeeter and microphone devices.

## License

MIT. DeepFilterNet and `deepfilternet-rs` retain their own licenses and attribution requirements.
