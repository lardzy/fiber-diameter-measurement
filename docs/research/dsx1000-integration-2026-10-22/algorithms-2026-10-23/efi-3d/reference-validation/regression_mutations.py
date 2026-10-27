"""Inject the three peer-review regressions into temporary copies only."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
source = (ROOT/'reference_efi.py').read_text()
start = source.index('class GuidedFilter:')
end = source.index('def guided_weighted_median(')
old_grid = '''def axial_candidates(z, start, count, interp=8):
    z=np.asarray(z,np.float32)
    step=float((z[-1]-z[0])/np.float32((len(z)-1)*interp))
    grid=z[0]+np.arange((len(z)-1)*interp+1,dtype=np.float64)*step
    return grid[(grid>=float(z[start]))&(grid<=float(z[start+count-1]))],step


'''
variants = {
    'wrong_odd_right_halo': source.replace(
        'n - 1 if length % 2 == 0 else max(0, n - 2)', 'n - 1'),
    'mixed_float64_axial_grid': source[:source.index('def axial_candidates(')]
        + old_grid + source[source.index('def estimate_height('):],
    'float32_guided_moments': source[:start]
        + source[start:end].replace('np.float64', 'np.float32') + source[end:]
}
report = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
          'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'native_executed': False, 'results': []}
for name, code in variants.items():
    with tempfile.TemporaryDirectory(prefix='dsx-efi-regression-') as directory:
        path = Path(directory)
        (path/'reference_efi.py').write_text(code)
        shutil.copy2(ROOT/'validate_reference.py', path/'validate_reference.py')
        run = subprocess.run([sys.executable,str(path/'validate_reference.py')],
                             capture_output=True,text=True)
        item = {'mutation':name,'caught':False,'returncode':run.returncode}
        result = path/'reference-validation/results.json'
        if result.exists():
            data = json.loads(result.read_text())
            item['failed_checks'] = [x['name'] for x in data['results'] if not x['passed']]
            item['caught'] = bool(item['failed_checks'])
        else:
            item['error'] = run.stderr.splitlines()[-1] if run.stderr else run.stdout
            item['caught'] = 'Singular matrix' in item['error']
        report['results'].append(item)
(ROOT/'reference-validation/independent-review-regressions.json').write_text(
    json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
raise SystemExit(0 if all(x['caught'] for x in report['results']) else 1)
