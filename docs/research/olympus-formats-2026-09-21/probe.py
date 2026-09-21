"""Read-only, sample-scoped Olympus format research probe; not an FDM importer.

Run with the existing project environment (no dependency or lockfile changes):
    uv run --no-sync python docs/research/olympus-formats-2026-09-21/probe.py \
        --input '.tmp/DSX1000、OLS5000样张' --output '.tmp/olympus-probe-output'

--metadata-only skips pixel decoding and image output. ZIP members still need
inflation to access their OIR index/metadata. --channels selects decoded planes.
This probe supports only the single-frame, uncompressed OIR planes observed in
the supplied samples. It does not claim general OIR/DSX version compatibility.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image, JpegImagePlugin
import tifffile


def local(tag):
    return tag.rsplit('}', 1)[-1]


def find(element, path):
    return element.find('/'.join('{*}' + part for part in path.split('/')))


def value(element, path, default=None):
    node = find(element, path)
    return default if node is None or node.text is None else node.text.strip()


def xyz(element, default=1.0):
    return {axis: float(value(element, axis, default)) if element is not None
            else default for axis in ('x', 'y', 'z')}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
                    + '\n', encoding='utf-8')


def metrics(actual, expected):
    if actual.shape != expected.shape:
        return {'shape_mismatch': [list(actual.shape), list(expected.shape)]}
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    return {'mae_8bit': float(np.mean(np.abs(delta))),
            'rmse_8bit': float(np.sqrt(mse)),
            'psnr_db': float(10 * np.log10(255 ** 2 / mse)) if mse else None,
            'max_abs_8bit': float(np.max(np.abs(delta))),
            'exact_equal': bool(np.array_equal(actual, expected))}


def publish_plane(folder, kind, array, scale, preview=None):
    """Export native samples without quantization, then verify the TIFF readback."""
    path = folder / f'{kind}.tif'
    tifffile.imwrite(path, array, photometric='rgb' if array.ndim == 3 else 'minisblack',
                     metadata={'axes': 'YXS' if array.ndim == 3 else 'YX',
                               'research_calibration': scale})
    if not np.array_equal(tifffile.imread(path), array):
        raise ValueError(f'TIFF readback failed: {path}')
    if preview is None:
        if array.dtype == np.uint8:
            preview = array
        else:
            low, high = float(array.min()), float(array.max())
            preview = np.clip((array.astype(float) - low) * 255 / max(1, high - low),
                              0, 255).astype(np.uint8)
    Image.fromarray(preview).save(folder / f'{kind}_preview.png')
    result = {'file': str(path), 'preview': str(folder / f'{kind}_preview.png'),
              'shape': list(array.shape), 'dtype': str(array.dtype),
              'min': int(array.min()), 'max': int(array.max()),
              'raw_sha256': sha(array.tobytes()), 'tiff_readback_exact': True}
    if kind == 'invalid':
        values, counts = np.unique(array, return_counts=True)
        result['value_counts'] = {str(int(k)): int(v) for k, v in zip(values, counts)}
    return result, preview


class OirProbe:
    """Index-based reader written for the supplied OLS samples.

    Binary layout was cross-checked with cgohlke/oirfile 2026.9.6 (BSD-3-Clause).
    Channel semantics, RGB order, calibration, and LUT association are read
    independently from sample XML and GUIDs, not inferred from array order.
    """

    def __init__(self, stream):
        self.stream = stream
        self.pixel_bytes_read = 0
        self.read_requested = 0
        stream.seek(0, 2)
        self.size = stream.tell()
        header = self.read(0, 96)
        if header[:16] != b'OLYMPUSRAWFORMAT':
            raise ValueError('OIR signature mismatch')
        size, index_offset = struct.unpack_from('<QQ', header, 32)
        if size != self.size:
            raise ValueError('OIR declared size mismatch')
        index = self.read(index_offset, size - index_offset)
        if index[:4] != b'\xff' * 4 or (len(index) - 4) % 8:
            raise ValueError('Unsupported OIR index')
        offsets = [v[0] for v in struct.iter_unpack('<Q', index[4:])]
        self.blocks = []
        self.xml = []
        self.pixels = defaultdict(list)
        for offset in offsets:
            length, kind = struct.unpack('<II', self.read(offset, 8))
            if offset + 8 + length > index_offset:
                raise ValueError('Block exceeds indexed data area')
            self.blocks.append({'offset': offset, 'length': length, 'type': kind})
        for i, block in enumerate(self.blocks):
            offset, length, kind = block['offset'], block['length'], block['type']
            if kind == 3:
                content = self.read(offset + 8, length)
                uid_length = struct.unpack_from('<I', content, 8)[0]
                uid = content[12:12 + uid_length].decode('ascii')
                block['uid'] = uid
                pixel = self.blocks[i + 1]
                if pixel['type'] != 4:
                    raise ValueError('UID not followed by PIXEL')
                self.pixels[uid].append((pixel['offset'] + 8, pixel['length']))
            elif kind in (0, 1):
                content = self.read(offset + 8, length)
                cursor = 0
                while True:
                    pos = content.find(b'<?xml', cursor)
                    if pos < 0:
                        break
                    xml_length = struct.unpack_from('<I', content, pos - 4)[0]
                    xml_bytes = content[pos:pos + xml_length]
                    if len(xml_bytes) != xml_length:
                        raise ValueError('Truncated XML')
                    root = ET.fromstring(xml_bytes)
                    guid = None
                    if local(root.tag) == 'LUT' and pos >= 44:
                        guid_bytes = content[pos - 40:pos - 4]
                        if struct.unpack_from('<I', content, pos - 44)[0] == 36:
                            guid = guid_bytes.decode('ascii')
                    self.xml.append({'offset': offset + 8 + pos,
                                     'length': xml_length, 'root': root,
                                     'bytes': xml_bytes, 'channel_guid': guid})
                    cursor = pos + xml_length
        props = [item for item in self.xml if local(item['root'].tag) == 'imageProperties'
                 and find(item['root'], 'imageInfo') is not None]
        if not props:
            raise ValueError('Missing imageInfo')
        self.properties = props[-1]['root']
        info = find(self.properties, 'imageInfo')
        self.width, self.height = int(value(info, 'width')), int(value(info, 'height'))
        self.device = value(info, 'acquireDevice')
        config = find(self.properties, 'acquisition/microscopeConfiguration')
        factory = xyz(find(config, 'pixelCalibration'))
        user = xyz(find(config, 'userPixelCalibration'))
        self.channels = []
        for channel in info.findall('./{*}phase/{*}group/{*}channel'):
            nominal = xyz(find(channel, 'length'), default=0.0)
            units = {axis: value(channel, f'pixelUnit/{axis}') for axis in ('x', 'y', 'z')}
            if units['x'] != 'MICRO_METER' or units['y'] != 'MICRO_METER':
                raise ValueError('Probe only verified for MICRO_METER')
            calibration = {'nominal_length': nominal, 'units': units,
                           'pixelCalibration': factory, 'userPixelCalibration': user,
                           'factory_corrected_um': {a: nominal[a] * factory[a] for a in nominal},
                           'nonidentity_user_calibration_unverified': any(v != 1 for v in user.values())}
            elements = channel.findall('./{*}elementChannel')
            if elements:
                for elem in elements:
                    self.channels.append({'guid': elem.attrib['id'],
                                          'kind': value(elem, 'elementType').lower(),
                                          'depth': int(value(elem, 'depth')),
                                          'calibration': calibration})
            else:
                self.channels.append({'guid': channel.attrib['id'],
                                      'kind': value(channel, 'imageDefinition/imageType').lower(),
                                      'depth': int(value(channel, 'imageDefinition/depth')),
                                      'bits': int(value(channel, 'imageDefinition/bitCounts')),
                                      'calibration': calibration})
        self.scale_ranges = {}
        for elem in self.properties.findall('.//{*}productData/{*}scale'):
            self.scale_ranges[elem.attrib['ChannelId']] = [float(value(elem, 'min')),
                                                         float(value(elem, 'max'))]
        self.luts = {item['channel_guid']: item['root'] for item in self.xml
                     if local(item['root'].tag) == 'LUT' and item['channel_guid']}
        self.index_offset = index_offset

    def read(self, offset, length):
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError('Read outside OIR bounds')
        self.stream.seek(offset)
        result = self.stream.read(length)
        self.read_requested += length
        if len(result) != length:
            raise ValueError('Short read')
        return result

    def plane(self, channel):
        selected = []
        frame_keys = set()
        for uid, chunks in self.pixels.items():
            if uid.startswith('REF_'):
                continue
            prefix, guid, part = uid.rsplit('_', 2)
            if guid == channel['guid']:
                selected.extend((int(part), off, length) for off, length in chunks)
                frame_keys.add(prefix)
        if len(frame_keys) != 1:
            raise ValueError(f'Expected one frame for {channel["guid"]}: {frame_keys}')
        selected.sort()
        expected = self.width * self.height * channel['depth']
        if sum(length for _, _, length in selected) != expected:
            raise ValueError('Pixel length does not match XML dimensions/depth')
        chunks = [self.read(off, length) for _, off, length in selected]
        self.pixel_bytes_read += expected
        dtype = {1: np.dtype('u1'), 2: np.dtype('<u2')}[channel['depth']]
        return np.frombuffer(b''.join(chunks), dtype=dtype).reshape(self.height, self.width)

    def preview(self, channel, array):
        guid = channel['guid']
        if guid not in self.luts:
            return None
        lut = self.luts[guid]
        table = np.frombuffer(bytes.fromhex(value(lut, 'data')), dtype=np.uint8).reshape(-1, 4)
        low, high = self.scale_ranges.get(guid, [0, len(table) - 1])
        indices = np.clip((array.astype(float) - low) * (len(table) - 1) / (high - low),
                          0, len(table) - 1).astype(np.int64)
        return table[indices, :3]

    def summary(self):
        z_scales = []
        for item in self.xml:
            if local(item['root'].tag) == 'frameProperties':
                for group in item['root'].findall('./{*}additionalData'):
                    values = [n.text for n in group.findall('./{*}scales')]
                    if values:
                        z_scales.append({'channel_guid': group.attrib.get('channelId'),
                                         'count': len(values), 'values': dict(Counter(values))})
        return {'size': self.size, 'index_offset': self.index_offset,
                'dimensions_wh': [self.width, self.height], 'device': self.device,
                'blocks': self.blocks, 'channels': self.channels,
                'metadata': [{'offset': x['offset'], 'length': x['length'],
                              'root': x['root'].tag, 'channel_guid': x['channel_guid']}
                             for x in self.xml],
                'lut_names': {k: value(v, 'name') for k, v in self.luts.items()},
                'display_scale_ranges': self.scale_ranges, 'z_scales': z_scales,
                'pixel_bytes_read': self.pixel_bytes_read,
                'requested_oir_bytes': self.read_requested}


def read_oir(stream, folder, channels, metadata_only):
    reader = OirProbe(stream)
    folder.mkdir(parents=True, exist_ok=True)
    planes = {}
    files = {}
    for i, item in enumerate(reader.xml):
        if local(item['root'].tag) not in ('LUT', 'contents', 'annotationStore', 'eventList'):
            (folder / f'metadata_{i:02d}_{local(item["root"].tag)}.xml').write_bytes(item['bytes'])
    if not metadata_only:
        if reader.device.lower() == 'camera' and 'color' in channels:
            components = {c['kind']: reader.plane(c) for c in reader.channels}
            array = np.stack([components[c] for c in ('red', 'green', 'blue')], axis=-1)
            files['color'], planes['color'] = publish_plane(folder, 'color', array,
                                                           reader.channels[0]['calibration'])
        elif reader.device.upper() == 'LSM':
            for channel in reader.channels:
                kind = channel['kind']
                if kind in channels:
                    array = reader.plane(channel)
                    files[kind], planes[kind] = publish_plane(folder, kind, array,
                        channel['calibration'], reader.preview(channel, array))
    result = reader.summary()
    result['exports'] = files
    write_json(folder / 'oir_structure.json', result)
    return result, planes


def compare_exports(path, previews, is_dsx=False):
    result = {}
    for kind, suffix in [('color', 'C'), ('intensity', 'I'), ('height', 'H')]:
        if kind not in previews:
            continue
        candidates = [path.with_name(f'{path.stem}_{suffix}{ext}') for ext in ('.jpg', '.jpeg')]
        target = next((p for p in candidates if p.exists()), None)
        if target is None:
            continue
        with Image.open(target) as reference:
            expected = np.asarray(reference.convert('RGB'))
            sampling = JpegImagePlugin.get_sampling(reference)
            quantization = reference.quantization
        actual = previews[kind]
        if actual.ndim == 2:
            actual = np.repeat(actual[..., None], 3, axis=2)
        comparison = {'reference': str(target), 'full_image': metrics(actual, expected)}
        # Distinguish decoder/channel errors from the reference JPEG's chroma
        # subsampling. This is an in-memory codec control, not the raw export.
        buffer = io.BytesIO()
        Image.fromarray(actual).save(buffer, format='JPEG', qtables=quantization,
                                     subsampling=sampling)
        codec_control = np.asarray(Image.open(buffer).convert('RGB'))
        comparison['jpeg_sampling'] = sampling
        comparison['jpeg_codec_control'] = metrics(codec_control, expected)
        if is_dsx:
            limit = int(expected.shape[0] * 0.9)
            comparison['upper_90_percent_excluding_scale_annotation'] = metrics(actual[:limit], expected[:limit])
        result[kind] = comparison
    return result


def read_dsx(path, folder, channels, metadata_only):
    folder.mkdir(parents=True, exist_ok=True)
    result = {'path': str(path), 'kind': 'DSX', 'pages': [], 'exports': {}}
    previews = {}
    with tifffile.TiffFile(path) as tiff:
        main = tiff.pages[0]
        xml = ET.fromstring(main.description)
        (folder / 'ImageDescription.xml').write_bytes(main.description.encode('utf-8'))
        maker = main.tags['ExifTag'].value.get('MakerNote')
        if maker:
            (folder / 'MakerNote.xml').write_bytes(maker.encode('utf-8'))
        for i, page in enumerate(tiff.pages):
            description = page.description
            kind = 'color' if i == 0 and value(xml, 'emImageData') == 'Color' else description.lower()
            record = {'index': i, 'ifd_offset': page.offset, 'shape': list(page.shape),
                      'dtype': str(page.dtype), 'kind': kind,
                      'strip_offsets': list(page.dataoffsets), 'strip_lengths': list(page.databytecounts),
                      'description_offset': page.tags[270].valueoffset,
                      'compression': int(page.compression)}
            result['pages'].append(record)
            if kind in ('color', 'height'):
                prefix = kind.title()
                scale = {axis: float(value(xml, f'{prefix}ImageData/{prefix}DataPerPixel{axis.upper()}')) / 1e6
                         for axis in ('x', 'y', 'z')}
                record['pixel_scale_um'] = scale
                if not metadata_only and kind in channels:
                    result['exports'][kind], previews[kind] = publish_plane(folder, kind, page.asarray(), scale)
        result['image_data_per_pixel_raw'] = {a: value(xml, f'ImageDataPerPixel{a.upper()}') for a in 'xyz'}
        result['standard_dpi'] = {'x': main.tags['XResolution'].value,
                                  'y': main.tags['YResolution'].value}
    result['comparison'] = compare_exports(path, previews, is_dsx=True)
    write_json(folder / 'dsx_structure.json', result)
    return result


def layout_info(raw):
    root = ET.fromstring(raw)
    groups = []
    for group in root.findall('./{*}group'):
        groups.append({'id': group.attrib.get('objectId'),
                       'stitching': value(group, 'stitching'),
                       'stitch_image': value(group, 'stitchImage'),
                       'region': find(group, 'regionInfo/coordinates').attrib,
                       'area_info': {local(c.tag): c.text for c in find(group, 'areaInfo')},
                       'areas': [{local(c.tag): c.text for c in area} for area in group.findall('./{*}area')]})
    return {'map_image': value(root, 'map/image'), 'groups': groups}


def read_archive(source, folder, channels, metadata_only, compare_path=None):
    folder.mkdir(parents=True, exist_ok=True)
    result = {'members': [], 'oir': [], 'poir': []}
    previews = {}
    with zipfile.ZipFile(source) as archive:
        for i, info in enumerate(archive.infolist()):
            result['members'].append({'name': info.filename, 'size': info.file_size,
                                      'compressed_size': info.compress_size,
                                      'method': info.compress_type, 'crc32': info.CRC})
            suffix = Path(info.filename).suffix.lower()
            if suffix == '.oir':
                with archive.open(info) as stream:
                    record, images = read_oir(stream, folder / f'oir_{i:02d}', channels, metadata_only)
                record['member'] = info.filename
                result['oir'].append(record)
                previews.update(images)
            elif suffix == '.poir':
                nested = read_archive(io.BytesIO(archive.read(info)), folder / f'point_{i:02d}', channels, metadata_only)
                nested['member'] = info.filename
                result['poir'].append(nested)
            elif info.filename == 'matl.omp2info':
                raw = archive.read(info)
                (folder / info.filename).write_bytes(raw)
                result['layout'] = layout_info(raw)
        if 'layout' in result:
            result['layout']['map_member_present'] = result['layout']['map_image'] in archive.namelist()
    if compare_path is not None:
        result['comparison'] = compare_exports(compare_path, previews)
    write_json(folder / 'archive_structure.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--channels', default='color,intensity,height,invalid')
    args = parser.parse_args()
    channels = set(args.channels.split(','))
    if not channels <= {'color', 'intensity', 'height', 'invalid'}:
        parser.error('unknown channel')
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'input_root': str(args.input.resolve()), 'metadata_only': args.metadata_only,
              'requested_channels': sorted(channels), 'inventory': [], 'samples': []}
    for path in sorted(args.input.rglob('*')):
        if not path.is_file():
            continue
        with path.open('rb') as source:
            raw_hash = hashlib.file_digest(source, 'sha256').hexdigest()
        result['inventory'].append({'path': str(path.relative_to(args.input)),
                                     'bytes': path.stat().st_size, 'sha256': raw_hash})
        if path.suffix.lower() not in ('.dsx', '.poir', '.mpoir'):
            continue
        started = time.perf_counter()
        folder = args.output / path.relative_to(args.input).with_suffix('')
        if path.suffix.lower() == '.dsx':
            record = read_dsx(path, folder, channels, args.metadata_only)
        else:
            record = read_archive(path, folder, channels, args.metadata_only, compare_path=path)
        record.update({'source': str(path.relative_to(args.input)), 'sha256': raw_hash,
                       'elapsed_seconds': time.perf_counter() - started})
        result['samples'].append(record)
        print(record['source'], f'{record["elapsed_seconds"]:.3f}s', flush=True)
    write_json(args.output / 'analysis.json', result)


if __name__ == '__main__':
    main()
