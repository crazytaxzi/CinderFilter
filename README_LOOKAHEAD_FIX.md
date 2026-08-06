# CinderFilter Lookahead-Safe CUDA Fix

The failed stateful worker assumed DeepFilterNet3 used zero deep-filter lookahead. The shipped model actually reports DF lookahead 2, so the wrapper correctly refused to lie about compatibility.

This replacement does not alter the trained model internals. It uses lookahead-aware overlap-save:

1. Hold one target chunk.
2. Add real past context.
3. Wait for enough real future context to cover convolution and DF lookahead.
4. Run the official DeepFilterNet whole-window path on CUDA.
5. Crop and emit only the center target chunk.

Explicit CUDA mode remains CUDA-locked. It mutes and restarts on a genuine worker failure; it does not instantiate the CPU denoiser. Auto mode may still fall back.

The verifier measures 20 enhanced windows after priming and requires p95 RTF below 1.0.
