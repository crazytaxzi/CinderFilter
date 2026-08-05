from __future__ import annotations

import cinderfilter_voice_lock_v2_pitch_saved as saved


class SafePersistentPitchLockApp(saved.PersistentPitchLockApp):
    """Persistent Pitch Lock app with compute-device validation.

    A saved CUDA preference must never make strict mode unstartable when the
    installed PyTorch runtime cannot access CUDA. In that case we switch the UI
    and saved preference to Auto, which selects CPU now and can select CUDA
    automatically after a compatible runtime is installed later.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ensure_compatible_compute(save=True)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except BaseException:
            return False

    def _ensure_compatible_compute(self, save: bool = False) -> None:
        if self.v2_device_var.get() != "CUDA":
            return
        if self._cuda_available():
            return

        self.v2_device_var.set("Auto")
        self.v2_status_var.set(
            "CUDA is unavailable in this PyTorch runtime — switched to Auto (CPU)"
        )
        self.status_var.set("Compute setting corrected — ready")
        if save:
            self._save_settings_now()

    def start_filtering(self) -> None:
        self._ensure_compatible_compute(save=True)
        super().start_filtering()

    def preload_v2(self) -> None:
        self._ensure_compatible_compute(save=True)
        super().preload_v2()


def main() -> None:
    app = SafePersistentPitchLockApp()
    app.mainloop()


if __name__ == "__main__":
    main()
