"""Scope the established Windows delivery tools to the sixth-round checkout.

The immutable adapter list below is generated from repository-owned fifth-round
validation sources. It replaces the round name, adds round-six tests, and makes
reviewable PowerShell files use the file transport even when their text is short;
all source/generated hashes are recorded in the transferred snapshot. Product
code and previous installers are never transformed by this adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_ROOT = r'C:\tmp\fdm-cu-live-sixth-round-20261020'
ADAPTERS = (
    'fifth_round_remote.py',
    'sync_acquisition_fifth_round_windows.py',
    'build_acquisition_fifth_round.ps1',
    'verify_fifth_round_installer.py',
    'verify_fifth_round_installed_selfcheck.py',
    'manage_acquisition_fifth_round_installation.ps1',
    'install_acquisition_fifth_round.ps1',
    'probe_acquisition_fifth_round_environment.ps1',
    'probe_acquisition_fifth_round_explorer.ps1',
    'probe_acquisition_fifth_round_frozen.ps1',
    'verify_acquisition_fifth_round_excel.ps1',
    'run_acquisition_fifth_round_frozen.py',
    'receive_acquisition_fifth_round_artifact.py',
)


def adapters():
    result = {}
    for name in ADAPTERS:
        source = ROOT / 'scripts/validation' / name
        data = source.read_text(encoding='utf-8-sig')
        transformed = data.replace('fifth', 'sixth').replace('FIFTH', 'SIXTH').replace('Fifth', 'Sixth')
        if name == 'fifth_round_remote.py':
            short_branch = (
                '    if len(base64.b64encode(code.encode("utf-16-le"))) <= 2000:\n'
                '        return shared.powershell(vm, code, timeout=timeout)\n'
            )
            if transformed.count(short_branch) != 1:
                raise RuntimeError('Expected one reviewed PowerShell short-command branch')
            # --code-file is already reviewed as a file. Parallels can strip
            # embedded quotes from its -Command argument; never reinterpret it.
            transformed = transformed.replace(short_branch, '', 1)
        if name == 'build_acquisition_fifth_round.ps1':
            transformed = transformed.replace(
                "$tests = @(\n",
                "$tests = @(\n    'tests/test_measurement_feedback.py', 'tests/test_trial_round6_ui.py', 'tests/test_round6_manual_start.py', 'tests/test_live_record_publication.py',\n", 1)
        destination = 'scripts/validation/' + name.replace('fifth', 'sixth')
        blob = (('\ufeff' if source.suffix == '.ps1' else '') + transformed).encode('utf-8')
        result[destination] = (blob, dict(source=source.relative_to(ROOT).as_posix(),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            path=destination, sha256=hashlib.sha256(blob).hexdigest(), size=len(blob)))
    return result


def load_adapter(name):
    path = 'scripts/validation/' + name
    payload, _ = adapters()[path]
    module = types.ModuleType(name.removesuffix('.py'))
    module.__file__ = str(ROOT / path)
    # Dependent imports resolve only to this fixed repository-generated list.
    sys.modules[module.__name__] = module
    exec(compile(payload.decode('utf-8-sig'), module.__file__, 'exec'), module.__dict__)
    return module


def snapshot(path, manual_reference=None, automatic_reference=None):
    from sync_acquisition_second_round_windows import release_snapshot
    manifest = release_snapshot(path, manual_reference, automatic_reference)
    saved = path.with_suffix('.original.zip')
    path.rename(saved)
    generated = adapters()
    with zipfile.ZipFile(saved) as archive, zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as output:
        for row in archive.infolist():
            if row.filename != '.fdm_snapshot.json':
                output.writestr(row, archive.read(row))
        for name, (blob, detail) in generated.items():
            output.writestr(name, blob)
            manifest['files'].append(dict(path=name, size=len(blob), sha256=detail['sha256']))
        manifest['generated_validation_adapters'] = [detail for _, detail in generated.values()]
        output.writestr('.fdm_snapshot.json', json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    saved.unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sync', 'command', 'gui', 'receive', 'plan'))
    parser.add_argument('--vm', default='Windows 11')
    parser.add_argument('--code-file', type=Path)
    parser.add_argument('--timeout', type=int, default=900)
    args, remaining = parser.parse_known_args()
    sys.path.insert(0, str(ROOT / 'scripts'))
    if args.action == 'plan':
        print(json.dumps(dict(windows_root=WINDOWS_ROOT,
            adapters=[detail for _, detail in adapters().values()]), ensure_ascii=False, indent=2))
        return 0
    remote = load_adapter('sixth_round_remote.py')
    if args.action == 'sync':
        adapter = load_adapter('sync_acquisition_sixth_round_windows.py')
        adapter.sync.DEFAULT_DESTINATION = WINDOWS_ROOT
        adapter.sync.snapshot = snapshot
        adapter.sync.powershell = adapter.powershell
        sys.argv = [str(Path(__file__)), '--vm', args.vm, *remaining]
        adapter.sync.main()
    elif args.action == 'command':
        if args.code_file is None:
            parser.error('command requires --code-file with a reviewable PowerShell script')
        print(remote.run_powershell(args.vm, args.code_file.read_text(encoding='utf-8-sig'), timeout=args.timeout).stdout)
    elif args.action == 'gui':
        adapter = load_adapter('run_acquisition_sixth_round_frozen.py')
        # The GUI driver hashes its probe on the host before executing the
        # snapshot copy in Windows. Materialize only generated validation
        # sources under ignored build storage, preserving repository sources.
        staging = ROOT / '.tmp/acquisition-sixth-round-20261020/adapters'
        for name, (blob, _) in adapters().items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        adapter.ROOT = staging
        if '--output' not in remaining:
            remaining += ['--output', str(ROOT / '.tmp/acquisition-sixth-round-20261020/windows-frozen-gui')]
        sys.argv = [str(Path(__file__)), '--vm', args.vm, *remaining]
        return adapter.main()
    elif args.action == 'receive':
        adapter = load_adapter('receive_acquisition_sixth_round_artifact.py')
        sys.argv = [str(Path(__file__)), *remaining]
        adapter.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
