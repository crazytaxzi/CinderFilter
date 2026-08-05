# CinderFilter Voice Lock v2 (Experimental)

Voice Lock v2 adds overlapping-speaker extraction on top of the working v1 build.

Pipeline:

1. Capture the physical microphone at 48 kHz.
2. Buffer an overlapping speech chunk.
3. Run SpeechBrain SepFormer WHAMR16k to separate two speech streams.
4. Compare both streams with the existing ECAPA voice profile.
5. Keep the best matching stream and suppress chunks where neither stream matches.
6. Resample to 48 kHz and run DeepFilterNet cleanup.
7. Send the target voice to the selected VoiceMeeter or virtual-cable output.

## Install

Extract these files over the existing CinderFilter directory. Do not delete `.venv`, `models`, or `profiles`.

Launch `START_CINDERFILTER_V2.bat`.

## First run

1. Confirm the v1 voice profile says READY.
2. Choose Fast mode and Auto compute.
3. Click **Preload v2 Models**. The SepFormer model downloads on first use.
4. Enable target-speaker extraction.
5. Start filtering.

## Modes

- Fast: 1.6-second chunk, 0.8-second hop.
- Balanced: 2.4-second chunk, 1.2-second hop.
- Quality: 3.2-second chunk, 1.6-second hop.

The actual delay is the chunk duration plus model processing and audio buffering. Watch RTF: below 1.0 is required for sustained processing at the selected hop. CUDA is strongly preferred. If v2 overloads or errors, CinderFilter automatically falls back to Voice Lock v1.

This build is deliberately isolated from v1. Use `START_CINDERFILTER.bat` for the stable v1 path and `START_CINDERFILTER_V2.bat` for the experimental extractor.
