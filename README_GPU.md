# GPU-aware CinderFilter

This patch separates GPU detection into three independent checks:

1. Windows graphics hardware (`Win32_VideoController`)
2. NVIDIA driver capability (`nvidia-smi`)
3. PyTorch runtime capability (`torch.cuda.is_available()`)

The Pitch-Locked v2 interface now displays the detected adapters, driver CUDA
version, PyTorch build, active backend, GPU name, and VRAM.

When an NVIDIA GPU is present but the CinderFilter virtual environment contains
a CPU-only PyTorch build, **Install CUDA PyTorch** performs a one-time repair:

- closes CinderFilter so Windows releases the PyTorch DLLs;
- installs the official PyTorch 2.11.0 CUDA 12.8 or CUDA 13.0 wheel selected from
  the NVIDIA driver's reported capability;
- verifies that CUDA is usable;
- relaunches CinderFilter.

AMD and Intel adapters are detected and reported, but the current SepFormer
runtime remains CUDA-or-CPU. The application does not falsely claim DirectML
compatibility.
