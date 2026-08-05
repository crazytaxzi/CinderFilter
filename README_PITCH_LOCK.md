# CinderFilter Pitch-Locked Voice Lock v2

This optional build adds a second identity constraint to Voice Lock v2:

1. SepFormer separates two estimated speech streams.
2. ECAPA compares both streams with the enrolled voiceprint.
3. Pitch Lock rejects streams above the locally calibrated target range.
4. Strict fail-closed mode outputs silence when voice identity or pitch is uncertain.

## Setup

1. Keep the normal v1 voice profile enrolled.
2. Launch `START_CINDERFILTER_V2_PITCH.bat`.
3. Select the physical microphone.
4. Click **Calibrate My Pitch** and speak normally for 12 seconds.
5. Keep **Fail closed on uncertain pitch** enabled for maximum rejection.
6. Enable v2 target-speaker extraction and start filtering.

Pitch is an additional discriminator, not a mathematical guarantee. Voices can overlap in pitch, and a person's F0 changes with emphasis, illness, whispering, yelling, and microphone conditions. Strict mode deliberately prefers muting uncertain audio over leaking another speaker.
