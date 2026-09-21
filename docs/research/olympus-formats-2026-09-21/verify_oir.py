"""Independent native-pixel comparison against oirfile 2026.9.6.

uv run --no-project --with oirfile==2026.9.6 --with tifffile --with pillow \
  python docs/research/olympus-formats-2026-09-21/verify_oir.py \
  --input '.tmp/DSX1000、OLS5000样张' --output '.tmp/olympus-oir-verification.json'

Private oirfile GUID ordering is used only for this pinned research comparison;
its default RGB order is not used as an image interpretation contract.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import oirfile

from probe import OirProbe


def verify(source, label, records):
    with zipfile.ZipFile(source) as archive:
        for member in archive.namelist():
            if member.endswith('.poir'):
                verify(io.BytesIO(archive.read(member)), label + '!' + member, records)
            elif member.endswith('.oir'):
                data = archive.read(member)
                ours = OirProbe(io.BytesIO(data))
                with oirfile.OirFile(io.BytesIO(data)) as other:
                    pixels = other.asarray()
                    guids = other._get_ordered_channel_guids()
                    comparisons = []
                    for channel in ours.channels:
                        plane = ours.plane(channel)
                        equal = bool(np.array_equal(plane, pixels[guids.index(channel['guid'])]))
                        if not equal:
                            raise AssertionError(label + '!' + member + ': ' + channel['kind'])
                        comparisons.append({'kind': channel['kind'], 'guid': channel['guid'],
                                            'equal': equal, 'shape': list(plane.shape)})
                    rgb_order = [c['kind'] for g in guids for c in ours.channels if c['guid'] == g]
                    records.append({'source': label + '!' + member, 'planes': comparisons,
                                    'library_array_order': rgb_order,
                                    'library_coordinate_scales': other.coord_scales,
                                    'factory_corrected_um': ours.channels[0]['calibration']['factory_corrected_um']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for source in sorted(args.input.rglob('*')):
        if source.suffix in ('.poir', '.mpoir'):
            verify(source, str(source.relative_to(args.input)), records)
    result = {'reference_library': 'oirfile', 'version': oirfile.__version__,
              'numpy_version': np.__version__, 'oir_count': len(records),
              'native_planes_equal': sum(len(r['planes']) for r in records), 'records': records}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
