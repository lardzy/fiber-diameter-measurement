"""Sync the third-round candidate with the established internal runtime profile."""
from sync_acquisition_second_round_windows import release_snapshot, sync


if __name__ == "__main__":
    sync.DEFAULT_DESTINATION = r"C:\tmp\fdm-cu-live-third-round-20261019"
    sync.snapshot = release_snapshot
    sync.main()
