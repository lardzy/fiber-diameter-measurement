"""Clean this release's owned build copies only after both retained packages verify.

Run from the host after receive_results.py has completed. This script refuses
partial source/UI/installer validation, changed evidence, active guest processes,
and linked paths. It never removes the retained installers or another build root.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/validation/professional-precision-release-20261027"
STAGE = ROOT / "build/professional-precision-release-20261027"
RETAINED_ROOT = ROOT / "dist/internal/professional-precision-20261027"
GUEST_ROOT = PureWindowsPath(r"C:\tmp\fdm-professional-precision-release-20261027")
SOURCE_COMMIT = "75a9e57304f3d2d6008461233adf906bc5b872a4"
PREVIOUS_INSTALLER = "dist/internal/professional-measurement-20261027/FDM-0.4.13-professional-measurement-ea3a9c2a985b.exe"
PREVIOUS_SHA256 = "330c059e82e61c4faa4d413e7820e7a78ce0bdc3f32ef840953658ed747c5d0c"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_digest(value: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid SHA-256 in release identity")
    return value


def contained_file(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    require(resolved.is_relative_to(base.resolve()) and resolved != base.resolve(),
            f"File escapes its allowed directory: {path}")
    require(path.is_file() and not path.is_symlink(), f"Missing or linked file: {path}")
    ancestor = path.parent
    while ancestor != base.parent:
        require(not ancestor.is_symlink(), f"Linked directory: {ancestor}")
        if ancestor == base:
            break
        ancestor = ancestor.parent
    return resolved


def verify_retained(rows: list[dict]) -> list[dict]:
    checked = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        path = contained_file(path, ROOT / "dist/internal")
        digest = checked_digest(row["sha256"])
        require(path.stat().st_size == row["bytes"] and sha(path) == digest,
                f"Retained installer mismatch: {path}")
        checked.append(dict(path=str(path), bytes=path.stat().st_size,
                            sha256=digest, verified=True))
    return checked


def guest_relative(path: str) -> str:
    candidate = PureWindowsPath(path)
    require(candidate.is_relative_to(GUEST_ROOT / "build/frozen-gui"),
            "GUI result is outside this release's frozen evidence directory")
    return candidate.relative_to(GUEST_ROOT / "build").as_posix()


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    retained = read_json(OUT / "host-retained-installer.json")
    new_path = contained_file(Path(retained["path"]), RETAINED_ROOT)
    require(new_path.suffix.lower() == ".exe", "Expected the retained installer")
    previous = read_json(OUT / "previous-installers.json")
    require(isinstance(previous, list) and len(previous) == 1
            and previous[0]["path"] == PREVIOUS_INSTALLER
            and previous[0]["sha256"] == PREVIOUS_SHA256,
            "The retained 0.4.13 rollback identity changed")
    retained_rows = previous + [retained]
    before = verify_retained(retained_rows)

    transfer = read_json(OUT / "transfer.json")
    require(transfer["installer_sha256"] == retained["sha256"]
            and transfer["installer_bytes"] == retained["bytes"]
            and transfer["installer"] == new_path.name,
            "Transferred installer does not match the retained installer")
    evidence = {}
    for row in transfer["files"]:
        relative = PurePosixPath(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and "\\" not in row["path"] and row["path"] not in evidence,
                "Invalid or duplicate transferred evidence path")
        path = contained_file(OUT / relative, OUT)
        digest = checked_digest(row["sha256"])
        require(path.stat().st_size == row["bytes"] and sha(path) == digest,
                f"Transferred evidence mismatch: {relative}")
        evidence[row["path"]] = row

    identity = read_json(OUT / "package/installer-identity.json")
    require(identity["passed"] is True and identity["version"] == "0.4.14"
            and identity["source_git_baseline"] == SOURCE_COMMIT
            and identity["sha256"] == retained["sha256"]
            and identity["bytes"] == retained["bytes"], "Wrong installer/build identity")
    source_status = read_json(OUT / "source-validation/status.json")
    require(source_status["exit_code"] == 0
            and len(source_status.get("ui", [])) == 3
            and {row["scale"] for row in source_status["ui"]} == {1, 1.5, 2}
            and all(row["exit_code"] == 0 for row in source_status["ui"]),
            "Windows source regression or DPI validation did not finish successfully")
    gui = read_json(OUT / "gui-status.json")
    require(gui["exit_code"] == 0 and gui["independent_verification"] is True,
            "Frozen GUI validation did not finish successfully")
    result_relative = guest_relative(gui["result"])
    verified_relative = str(PurePosixPath(result_relative).with_name("verified.json"))
    probe = read_json(OUT / result_relative)
    verified = read_json(OUT / verified_relative)
    require(probe["passed"] is True and not probe.get("errors")
            and probe["graceful_exit"] is True and probe["forced_termination"] is False,
            "Frozen GUI did not finish normally")
    require(verified["passed"] is True and verified["reference_comparison"] is True
            and verified["probe_sha256"] == sha(OUT / result_relative)
            and bool(verified["checks"]) and all(row["passed"] is True for row in verified["checks"]),
            "Independent exported-pixel verification is missing or changed")

    critical = ["package/installer-identity.json", "source-validation/status.json",
                "gui-status.json", result_relative, verified_relative]
    require(all(name in evidence for name in critical), "Critical validation evidence was not transferred")
    checks = ",\n".join("@{path=" + ps_quote(name.replace("/", "\\"))
                         + ";sha=" + ps_quote(evidence[name]["sha256"]) + "}" for name in critical)
    code = r"""$ErrorActionPreference='Stop'
$r='C:\tmp\fdm-professional-precision-release-20261027'
$expected=@(__EVIDENCE__)
foreach($entry in $expected){
 $path=Join-Path ($r+'\build') $entry.path
 if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha){throw 'Guest evidence changed after transfer'}
}
$id=Get-Content ($r+'\build\package\installer-identity.json') -Raw -Encoding UTF8|ConvertFrom-Json
$source=Get-Content ($r+'\build\source-validation\status.json') -Raw -Encoding UTF8|ConvertFrom-Json
$gui=Get-Content ($r+'\build\gui-status.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $id.passed -or $id.version -ne '0.4.14' -or $id.source_git_baseline -ne '__COMMIT__' -or $id.sha256 -ne '__SHA__'){throw 'Owned release identity mismatch'}
if($source.exit_code -ne 0 -or @($source.ui).Count -ne 3 -or @($source.ui|Where-Object{$_.exit_code -ne 0}).Count -ne 0){throw 'Source validation incomplete'}
if($gui.exit_code -ne 0 -or -not $gui.independent_verification){throw 'GUI validation incomplete'}
$snapshot=Get-Content ($r+'\.fdm_snapshot.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($snapshot.head -ne '__COMMIT__' -or (Get-FileHash ($r+'\.fdm_snapshot.json')).Hash.ToLowerInvariant() -ne $id.source_snapshot_sha256){throw 'Owned source snapshot changed'}
$installer=[IO.Path]::GetFullPath($id.installer)
if(-not $installer.StartsWith($r+'\dist\installer\',[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $installer).Hash.ToLowerInvariant() -ne '__SHA__'){throw 'Guest installer mismatch'}
$active=@(Get-CimInstance Win32_Process|Where-Object{$_.ProcessId -ne $PID -and (($_.ExecutablePath -and $_.ExecutablePath.StartsWith($r+'\',[StringComparison]::OrdinalIgnoreCase)) -or ($_.CommandLine -and $_.CommandLine.IndexOf($r,[StringComparison]::OrdinalIgnoreCase) -ge 0))})
if($active.Count){throw 'A process still references this release directory; no deletion'}
$ancestor=$r
while($ancestor){
 if(((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Reparse ancestor; no deletion'}
 $ancestor=[IO.Path]::GetDirectoryName($ancestor)
}
$items=@(Get-ChildItem -LiteralPath $r -Recurse -Force)
if(@($items|Where-Object{($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0}).Count){throw 'Reparse child; no deletion'}
$files=@($items|Where-Object{-not $_.PSIsContainer})
$freeBefore=([IO.DriveInfo]::new('C')).AvailableFreeSpace
$bytes=($files|Measure-Object Length -Sum).Sum
Remove-Item -LiteralPath $r -Recurse -Force
if(Test-Path -LiteralPath $r){throw 'Owned release directory remains'}
@{root=$r;removed=$true;file_count=$files.Count;bytes=$bytes;free_before=$freeBefore;free_after=([IO.DriveInfo]::new('C')).AvailableFreeSpace;at=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json
""".replace("__EVIDENCE__", checks).replace("__COMMIT__", SOURCE_COMMIT).replace("__SHA__", checked_digest(retained["sha256"]))
    sys.path.insert(0, str(ROOT / "scripts/validation"))
    import fifth_round_remote as remote
    # Staging scripts use this existing transport directory; it is not deleted.
    remote.WINDOWS_ROOT = r"C:\tmp\fdm-cu-live-professional-20261027"
    response = remote.run_powershell("Windows 11", code, timeout=240)
    receipt = json.loads(response.stdout)
    require(receipt["removed"] is True and receipt["root"] == str(GUEST_ROOT), "Unexpected guest cleanup result")
    receipt["retained_installers_before"] = before
    receipt["host_temporary_files_removed"] = []
    # Persist the guest outcome before touching any host temporary copy.
    (OUT / "cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    source_transfer = read_json(OUT / "source-transfer.json")
    source_zip = next(row for row in source_transfer["files"] if row["path"] == "source.zip")
    expected_hashes = {"source.zip": source_zip["sha256"],
                       "transfer/installer.exe": retained["sha256"],
                       "transfer/evidence.zip": transfer["evidence_sha256"]}
    for relative in ("source.zip", "transfer/installer.exe", "transfer/evidence.zip", "transfer/transfer.json"):
        path = STAGE / relative
        if not path.exists():
            continue
        path = contained_file(path, STAGE)
        digest = sha(path)
        if relative in expected_hashes:
            require(digest == checked_digest(expected_hashes[relative]), f"Host temporary file changed: {path}")
        else:
            require(read_json(path) == transfer, "Host temporary transfer manifest changed")
        row = dict(path=str(path), bytes=path.stat().st_size, sha256=digest)
        path.unlink()
        receipt["host_temporary_files_removed"].append(row)
        (OUT / "cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # All remaining host files are this task's generated transport/diagnostic
    # helpers. The authoritative scripts and evidence are retained under OUT.
    if STAGE.exists():
        require(not STAGE.is_symlink(), "Linked host staging directory")
        leftovers = list(STAGE.rglob("*"))
        require(not any(p.is_symlink() for p in leftovers), "Linked staging child")
        remaining = [dict(path=str(p.relative_to(STAGE)), bytes=p.stat().st_size, sha256=sha(p))
                     for p in leftovers if p.is_file()]
        shutil.rmtree(STAGE)
        require(not STAGE.exists(), "Host staging directory remains")
        receipt["host_owned_directory_removed"] = dict(root=str(STAGE), files=remaining)
    receipt["retained_installers_after"] = verify_retained(retained_rows)
    receipt["verified_evidence_files"] = len(evidence)
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
    (OUT / "cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (OUT / "retained-installers-after-cleanup.json").write_text(json.dumps(
        dict(at=receipt["completed_at"], files=receipt["retained_installers_after"]), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
