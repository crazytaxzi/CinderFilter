# Persistent settings update

The Pitch-Locked Voice Lock v2 launcher now remembers its setup under:

`%LOCALAPPDATA%\CinderFilter\settings.json`

Saved values:

- input and output endpoints, matched by device name and host API rather than PortAudio index
- AI strength and bypass
- Voice Lock enable state, reduction, and strictness
- v2 enable state, latency preset, and compute device
- Pitch Lock enable state, fail-closed mode, and pitch margin
- window size and position

Settings save automatically after changes, before Start/Preload, and when the app closes.
Voice Lock and Pitch Lock are only restored as enabled when their required local profiles exist.
