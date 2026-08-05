# CinderFilter

CinderFilter is a Windows-first real-time microphone preprocessor. It captures a physical microphone, removes background noise with DeepFilterNet, optionally identifies the enrolled speaker with Voice Lock, and sends the result to any playback or virtual-audio endpoint exposed by Windows.

```text
Physical microphone
    -> DeepFilterNet speech cleanup
    -> Voice Lock speaker verification
    -> VoiceMeeter / VB-CABLE / MIXLINE output
```

## Run it

1. Download or clone the repository on Windows.
2. Double-click `START_CINDERFILTER.bat`.
3. The launcher uses Python 3.12, creates `.venv`, and installs the required runtimes.
4. Select the physical microphone under **Input device**.
5. Select `VoiceMeeter AUX Input`, another VoiceMeeter virtual input, or a VB-CABLE playback endpoint under **Output device**.
6. Choose a DeepFilterNet strength and press **START FILTERING**.

The Voice Lock update adds PyTorch and SpeechBrain, so the first updated launch has a larger dependency installation. The ECAPA speaker model is downloaded only when Voice Lock is first used.

## VoiceMeeter Potato route

```text
CinderFilter input:  physical microphone
CinderFilter output: VoiceMeeter AUX Input
Potato input strip:  AUX virtual input
Potato output:       B1/B2/B3 as required
```

Do not also route the raw microphone into Potato unless you deliberately want an unprocessed comparison channel.

## Voice Lock

Voice Lock v1 identifies the dominant speaker and smoothly attenuates speech that does not match the enrolled voice profile.

1. Stop filtering and select the physical microphone.
2. Press **Enroll My Voice**.
3. Speak naturally and continuously for 12 seconds. Use normal stream volume and distance.
4. Wait for **Voice profile: READY**. The first enrollment downloads the speaker model.
5. Enable **Voice Lock** and start filtering.

Profiles are stored locally under `profiles/`. No enrollment audio is uploaded or retained; only the resulting numerical embedding is saved.

### Controls

- **Other voice reduction:** attenuation applied when the dominant voice does not match.
- **Conservative:** favors preserving speech and suppresses fewer uncertain voices.
- **Balanced:** default compromise.
- **Aggressive:** requires a stronger match and may suppress more aggressively.

Start with **24 dB / Balanced**. Use Conservative if your own voice is being reduced. Use Aggressive if distant people or television speech still pass through.

### What v1 can and cannot do

Voice Lock v1 works best when one person is speaking at a time. It can strongly reduce a different dominant speaker, but it does not yet perform true target-speaker extraction when two people talk simultaneously. Uncertain overlapping speech is intentionally protected rather than hard-muted so CinderFilter does not erase the enrolled user along with the intruder.

## Processing design

- 48 kHz mono live audio path
- DeepFilterNet real-time weighted denoising
- ECAPA-TDNN speaker embeddings through SpeechBrain
- 16 kHz side-channel verification windows
- Speaker model isolated from the real-time audio callback
- Smoothed gain attack/release instead of a clicking hard gate
- Saved local voice profile
- Input/output selection and live route restart
- Peak meters and dropout counters

## Troubleshooting

- If an endpoint fails, select another copy of the same device under a different Windows host API.
- If Voice Lock reduces you, re-enroll at your normal mic position and use Conservative strictness.
- If other voices still pass, increase strictness before raising reduction to 48 dB.
- If two people speak over each other, v1 may preserve the mixture. True overlap separation is planned for Voice Lock v2.

## License

MIT. DeepFilterNet, SpeechBrain, PyTorch, and the pretrained speaker model retain their own licenses and attribution requirements.
