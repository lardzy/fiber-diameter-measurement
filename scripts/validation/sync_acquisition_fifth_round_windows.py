"""Sync the fifth-round candidate with the established internal runtime profile."""
from sync_acquisition_second_round_windows import release_snapshot, sync
import run_acquisition_second_round_frozen as bounded
from fifth_round_remote import run_powershell


WINDOWS_ROOT = r"C:\tmp\fdm-cu-live-fifth-round-20261020"
bounded.WINDOWS_ROOT = WINDOWS_ROOT
_windows_path = bounded.windows_path
bounded.windows_path = lambda relative: _windows_path(relative.replace("build/second-round/", "build/fifth-round/"))


def powershell(vm, code, *, timeout=240):
    # prlctl's command-size failure is misleadingly reported as no guest
    # session. Stage long trusted scripts only in this candidate's directory.
    bounded.powershell(vm, "$ErrorActionPreference='Stop'; [void](New-Item -ItemType Directory -Force -Path "
                       + bounded.ps_quote(WINDOWS_ROOT + r"\build\fifth-round") + ")", timeout=60)
    return run_powershell(vm, code, timeout=timeout)


if __name__ == "__main__":
    sync.DEFAULT_DESTINATION = WINDOWS_ROOT
    sync.snapshot = release_snapshot
    sync.powershell = powershell
    sync.main()
