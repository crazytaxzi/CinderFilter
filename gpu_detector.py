from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Iterable


_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass(frozen=True)
class GpuAdapter:
    name: str
    vendor: str
    memory_mb: int | None = None
    driver_version: str | None = None
    source: str = "Windows"


@dataclass(frozen=True)
class GpuRuntimeStatus:
    adapters: tuple[GpuAdapter, ...]
    nvidia_smi_available: bool
    driver_cuda_version: float | None
    torch_version: str | None
    torch_build_cuda: str | None
    torch_cuda_available: bool
    torch_device_name: str | None
    torch_device_memory_mb: int | None
    torch_error: str | None

    @property
    def has_nvidia(self) -> bool:
        return any(adapter.vendor == "NVIDIA" for adapter in self.adapters)

    @property
    def has_amd(self) -> bool:
        return any(adapter.vendor == "AMD" for adapter in self.adapters)

    @property
    def has_intel(self) -> bool:
        return any(adapter.vendor == "Intel" for adapter in self.adapters)

    @property
    def selected_backend(self) -> str:
        return "CUDA" if self.torch_cuda_available else "CPU"

    @property
    def recommended_wheel_tag(self) -> str | None:
        if not self.has_nvidia:
            return None
        if self.driver_cuda_version is None:
            return None
        if self.driver_cuda_version >= 13.0:
            return "cu130"
        if self.driver_cuda_version >= 12.8:
            return "cu128"
        return None

    @property
    def needs_cuda_runtime(self) -> bool:
        return self.has_nvidia and not self.torch_cuda_available

    def summary(self) -> str:
        if not self.adapters:
            return f"No discrete GPU identified — backend: {self.selected_backend}"
        names = ", ".join(adapter.name for adapter in self.adapters)
        return f"{names} — backend: {self.selected_backend}"

    def detail(self) -> str:
        torch_version = self.torch_version or "not importable"
        torch_cuda = self.torch_build_cuda or "CPU build"
        parts = [f"PyTorch {torch_version}", f"Torch CUDA: {torch_cuda}"]
        if self.driver_cuda_version is not None:
            parts.append(f"Driver CUDA: {self.driver_cuda_version:g}")
        if self.torch_device_name:
            parts.append(f"Active GPU: {self.torch_device_name}")
        if self.torch_device_memory_mb:
            parts.append(f"VRAM: {self.torch_device_memory_mb} MB")
        if self.torch_error:
            parts.append(f"Torch error: {self.torch_error}")
        return "    ".join(parts)


def detect_gpu_runtime() -> GpuRuntimeStatus:
    nvidia_adapters, driver_cuda = _detect_nvidia_smi()
    windows_adapters = _detect_windows_adapters()
    adapters = _deduplicate([*nvidia_adapters, *windows_adapters])

    (
        torch_version,
        torch_build_cuda,
        torch_cuda_available,
        torch_device_name,
        torch_device_memory_mb,
        torch_error,
    ) = _detect_torch()

    return GpuRuntimeStatus(
        adapters=tuple(adapters),
        nvidia_smi_available=bool(nvidia_adapters),
        driver_cuda_version=driver_cuda,
        torch_version=torch_version,
        torch_build_cuda=torch_build_cuda,
        torch_cuda_available=torch_cuda_available,
        torch_device_name=torch_device_name,
        torch_device_memory_mb=torch_device_memory_mb,
        torch_error=torch_error,
    )


def _run(command: list[str], timeout: float = 8.0) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=_CREATE_NO_WINDOW,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _detect_nvidia_smi() -> tuple[list[GpuAdapter], float | None]:
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return [], None

    adapters: list[GpuAdapter] = []
    for line in query.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if not columns or not columns[0]:
            continue
        memory = _safe_int(columns[1]) if len(columns) > 1 else None
        driver = columns[2] if len(columns) > 2 and columns[2] else None
        adapters.append(
            GpuAdapter(
                name=columns[0],
                vendor="NVIDIA",
                memory_mb=memory,
                driver_version=driver,
                source="nvidia-smi",
            )
        )

    full_output = _run(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", full_output)
    driver_cuda = float(match.group(1)) if match else None
    return adapters, driver_cuda


def _detect_windows_adapters() -> list[GpuAdapter]:
    if os.name != "nt":
        return []

    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
    )
    output = _run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        timeout=12.0,
    )
    if not output:
        return []

    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []

    records = payload if isinstance(payload, list) else [payload]
    adapters: list[GpuAdapter] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = str(record.get("Name") or "").strip()
        if not name:
            continue
        raw_memory = record.get("AdapterRAM")
        memory_mb = None
        try:
            if raw_memory is not None:
                memory_mb = max(0, int(raw_memory) // (1024 * 1024))
        except (TypeError, ValueError, OverflowError):
            pass
        adapters.append(
            GpuAdapter(
                name=name,
                vendor=_vendor_from_name(name),
                memory_mb=memory_mb or None,
                driver_version=str(record.get("DriverVersion") or "").strip() or None,
                source="Windows CIM",
            )
        )
    return adapters


def _detect_torch() -> tuple[
    str | None,
    str | None,
    bool,
    str | None,
    int | None,
    str | None,
]:
    try:
        import torch

        version = str(getattr(torch, "__version__", "unknown"))
        build_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        available = bool(torch.cuda.is_available())
        device_name = None
        memory_mb = None
        if available:
            device_name = str(torch.cuda.get_device_name(0))
            properties = torch.cuda.get_device_properties(0)
            memory_mb = int(properties.total_memory // (1024 * 1024))
        return version, str(build_cuda) if build_cuda else None, available, device_name, memory_mb, None
    except BaseException as exc:
        return None, None, False, None, None, f"{type(exc).__name__}: {exc}"


def _deduplicate(adapters: Iterable[GpuAdapter]) -> list[GpuAdapter]:
    result: list[GpuAdapter] = []
    positions: dict[str, int] = {}
    for adapter in adapters:
        key = re.sub(r"\s+", " ", adapter.name).strip().casefold()
        if key in positions:
            current = result[positions[key]]
            # Prefer nvidia-smi because it reports reliable VRAM and driver data.
            if current.source != "nvidia-smi" and adapter.source == "nvidia-smi":
                result[positions[key]] = adapter
            continue
        positions[key] = len(result)
        result.append(adapter)
    return result


def _vendor_from_name(name: str) -> str:
    folded = name.casefold()
    if "nvidia" in folded or "geforce" in folded or "quadro" in folded or "rtx" in folded:
        return "NVIDIA"
    if "amd" in folded or "radeon" in folded:
        return "AMD"
    if "intel" in folded or "arc" in folded:
        return "Intel"
    return "Unknown"


def _safe_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
