"""Archive task evidence, excluding Windows' unrelated per-profile font cache."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
root=Path(r'C:\tmp\fdm-cu-live-ui-redesign-20261016\build')
evidence=root/('ui-redesign-final' if '--final' in sys.argv else 'ui-redesign')
output=root/('ui-redesign-evidence-final-compact.zip' if '--final' in sys.argv else 'ui-redesign-evidence-complete.zip')
assert not output.exists()
excluded=[]
files=[]
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(evidence.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(evidence)
        if 'isolated-config/Microsoft/' in relative.as_posix() or '__pycache__' in relative.parts:
            excluded.append(relative.as_posix());continue
        archive.write(path,relative.as_posix())
        files.append(relative.as_posix())
    archive.writestr('evidence-archive-inventory.json',json.dumps({'files':files,'excluded_windows_font_cache':excluded},ensure_ascii=False,indent=2))
with output.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'sha256':digest,'file_count':len(files),'excluded_font_cache_count':len(excluded)}))
