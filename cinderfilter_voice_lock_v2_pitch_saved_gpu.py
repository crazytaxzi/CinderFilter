from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import cinderfilter_voice_lock_v2_pitch_saved_safe as safe
from gpu_detector import GpuRuntimeStatus, detect_gpu_runtime


_CREATE_NEW_CONSOLE = 0x00000010 if os.name == "nt" else 0


class GpuAwarePitchLockApp(safe.SafePersistentPitchLockApp):
    """Persistent Pitch Lock app with hardware and PyTorch GPU diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.title("CinderFilter — GPU-Aware Pitch-Locked Voice Lock v2")
        self.geometry("940x1100")
        self.minsize(860, 900)

        self._gpu_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._gpu_status: GpuRuntimeStatus | None = None
        self.gpu_summary_var = tk.StringVar(value="Detecting Windows graphics hardware...")
        self.gpu_detail_var = tk.StringVar(value="Checking driver and PyTorch runtime")
        self.gpu_action_var = tk.StringVar(value="")

        self._add_gpu_ui()
        self.after(100, self._drain_gpu_events)
        self.after(150, self.detect_gpu)

    def _add_gpu_ui(self) -> None:
        frame = ttk.LabelFrame(
            self,
            text="GPU Runtime — hardware, driver, and AI backend",
            padding=12,
        )
        frame.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 18))
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            frame,
            textvariable=self.gpu_summary_var,
            font=("Segoe UI", 10, "bold"),
            wraplength=850,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(
            frame,
            textvariable=self.gpu_detail_var,
            wraplength=850,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        ttk.Label(
            frame,
            textvariable=self.gpu_action_var,
            wraplength=850,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 8))

        ttk.Button(frame, text="Detect GPU Again", command=self.detect_gpu).grid(
            row=3, column=0, sticky="w"
        )
        self.gpu_install_button = ttk.Button(
            frame,
            text="Enable NVIDIA CUDA",
            command=self.install_cuda_runtime,
            state="disabled",
        )
        self.gpu_install_button.grid(row=3, column=1, sticky="w", padx=(8, 0))

        ttk.Label(
            frame,
            text=(
                "Auto uses CUDA when PyTorch can access it and CPU otherwise. "
                "GPU detection does not weaken strict fail-closed filtering."
            ),
            wraplength=850,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def detect_gpu(self) -> None:
        self.gpu_summary_var.set("Detecting Windows graphics hardware...")
        self.gpu_detail_var.set("Checking NVIDIA driver and PyTorch runtime")
        self.gpu_action_var.set("")
        self.gpu_install_button.configure(state="disabled")
        threading.Thread(target=self._detect_gpu_worker, daemon=True).start()

    def _detect_gpu_worker(self) -> None:
        try:
            status = detect_gpu_runtime()
            self._gpu_events.put(("detected", status))
        except BaseException as exc:
            self._gpu_events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _drain_gpu_events(self) -> None:
        try:
            while True:
                kind, payload = self._gpu_events.get_nowait()
                if kind == "detected":
                    self._apply_gpu_status(payload)
                else:
                    self.gpu_summary_var.set("GPU detection failed")
                    self.gpu_detail_var.set(str(payload))
                    self.gpu_action_var.set("CinderFilter will continue on CPU.")
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._drain_gpu_events)

    def _apply_gpu_status(self, status: object) -> None:
        if not isinstance(status, GpuRuntimeStatus):
            return
        self._gpu_status = status
        self.gpu_summary_var.set(status.summary())
        self.gpu_detail_var.set(status.detail())

        if status.torch_cuda_available:
            self.gpu_install_button.configure(text="CUDA Ready", state="disabled")
            self.gpu_action_var.set(
                "NVIDIA acceleration is ready. Auto will select CUDA for Voice Lock v2."
            )
            if self.v2_device_var.get() == "CUDA":
                self.v2_status_var.set(f"CUDA ready — {status.torch_device_name}")
            return

        if status.has_nvidia:
            wheel = status.recommended_wheel_tag
            if wheel is None:
                self.gpu_install_button.configure(
                    text="CUDA Driver Update Needed",
                    state="disabled",
                )
                if status.driver_cuda_version is None:
                    self.gpu_action_var.set(
                        "NVIDIA hardware was found, but nvidia-smi did not report a usable "
                        "driver CUDA version. Update the NVIDIA driver, then detect again."
                    )
                else:
                    self.gpu_action_var.set(
                        f"NVIDIA hardware was found, but the driver exposes CUDA "
                        f"{status.driver_cuda_version:g}. PyTorch 2.11 needs a newer "
                        "driver for the supported CUDA wheels."
                    )
            else:
                self.gpu_install_button.configure(
                    text=f"Install CUDA PyTorch ({wheel})",
                    state="normal",
                )
                self.gpu_action_var.set(
                    "The GPU exists, but this virtual environment has a CPU-only or "
                    "unusable PyTorch build. The button repairs that runtime and restarts CinderFilter."
                )
            return

        self.gpu_install_button.configure(text="NVIDIA CUDA Unavailable", state="disabled")
        if status.has_amd or status.has_intel:
            self.gpu_action_var.set(
                "A non-NVIDIA GPU was detected. This Voice Lock v2 path currently uses "
                "CUDA on NVIDIA or CPU on Windows; it will not pretend DirectML compatibility."
            )
        else:
            self.gpu_action_var.set("No supported discrete GPU was identified; using CPU.")

    def install_cuda_runtime(self) -> None:
        status = self._gpu_status
        if status is None:
            messagebox.showinfo("Detecting GPU", "Wait for GPU detection to finish.")
            return
        wheel = status.recommended_wheel_tag
        if not status.has_nvidia or wheel is None:
            messagebox.showerror(
                "CUDA runtime unavailable",
                "CinderFilter could not choose a compatible CUDA wheel for this driver.",
            )
            return

        confirmed = messagebox.askyesno(
            "Install CUDA-enabled PyTorch",
            (
                f"CinderFilter detected NVIDIA hardware and will replace the current "
                f"PyTorch runtime with the official PyTorch 2.11.0 {wheel} build.\n\n"
                "This is a large one-time download. CinderFilter will close, install it "
                "in the existing .venv, verify CUDA, and relaunch.\n\nContinue?"
            ),
        )
        if not confirmed:
            return

        script = Path(__file__).resolve().parent / "install-cuda-runtime.ps1"
        if not script.exists():
            messagebox.showerror("Installer missing", f"Missing file:\n{script}")
            return

        try:
            self._save_settings_now()
            subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-PythonPath",
                    sys.executable,
                    "-AppRoot",
                    str(Path(__file__).resolve().parent),
                    "-ParentPid",
                    str(os.getpid()),
                    "-WheelTag",
                    wheel,
                ],
                cwd=str(Path(__file__).resolve().parent),
                creationflags=_CREATE_NEW_CONSOLE,
            )
        except BaseException as exc:
            messagebox.showerror("Could not launch CUDA installer", str(exc))
            return

        self.status_var.set("Closing for CUDA runtime installation...")
        self.after(300, self._on_close)


def main() -> None:
    app = GpuAwarePitchLockApp()
    app.mainloop()


if __name__ == "__main__":
    main()
