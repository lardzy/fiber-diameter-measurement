"""Single-frame OIR index reader, derived from the local sample investigation.

See docs/research/olympus-formats-2026-09-21 for format evidence. This reader
never guesses channel order and does not support tiled or time/Z stacks.
"""
from __future__ import annotations
from collections import defaultdict
import struct
import math
import xml.etree.ElementTree as ET
import numpy as np

def local(tag):
    return tag.rsplit('}', 1)[-1]


def find(element, path):
    if element is None:
        return None
    return element.find('/'.join('{*}' + part for part in path.split('/')))


def value(element, path, default=None):
    node = find(element, path)
    return default if node is None or node.text is None else node.text.strip()


def number(raw):
    try:
        result = float(raw)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def xyz(element, default=1.0):
    return {axis: number(value(element, axis, default)) if element is not None
            else default for axis in ('x', 'y', 'z')}


class OirReader:
    """Index-based reader for the validated single-frame OLS layout.

    Binary layout was cross-checked with cgohlke/oirfile 2026.9.6 (BSD-3-Clause).
    Channel semantics, RGB order, calibration, and LUT association are read
    independently from sample XML and GUIDs, not inferred from array order.
    """

    def __init__(self, stream, check_cancel=lambda: None):
        self.check_cancel = check_cancel
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
                if i + 1 >= len(self.blocks):
                    raise ValueError("Truncated OIR UID")
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
                    if pos < 4:
                        raise ValueError('Invalid XML length prefix')
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
        factory = xyz(find(config, 'pixelCalibration'), default=None)
        user = xyz(find(config, 'userPixelCalibration'))
        self.channels = []
        for channel in info.findall('./{*}phase/{*}group/{*}channel'):
            nominal = xyz(find(channel, 'length'), default=0.0)
            units = {axis: value(channel, f'pixelUnit/{axis}') for axis in ('x', 'y', 'z')}
            calibration = {'nominal_length': nominal, 'units': units,
                           'pixelCalibration': factory, 'userPixelCalibration': user,
                           'missing_factory_calibration': any(factory[a] is None for a in ('x', 'y')),
                           'factory_corrected_um': {a: nominal[a] * factory[a]
                               if nominal[a] is not None and factory[a] is not None else None for a in nominal},
                           'raw_length': {a: value(channel, f'length/{a}') for a in nominal},
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
        self.check_cancel()
        self.stream.seek(offset)
        result = self.stream.read(length)
        self.read_requested += length
        if len(result) != length:
            raise ValueError('Short read')
        return result

    def pixel_layout(self, channel):
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
        if len(selected) != 1 or selected[0][0] != 0:
            raise ValueError("尚不支持分块 OIR 像素")
        expected = self.width * self.height * channel['depth']
        if sum(length for _, _, length in selected) != expected:
            raise ValueError('Pixel length does not match XML dimensions/depth')
        if channel['depth'] not in (1, 2):
            raise ValueError('Unsupported OIR sample depth')
        return selected, expected

    def plane(self, channel):
        selected, expected = self.pixel_layout(channel)
        chunks = [self.read(off, length) for _, off, length in selected]
        self.pixel_bytes_read += expected
        dtype = {1: np.dtype('u1'), 2: np.dtype('<u2')}[channel['depth']]
        return np.frombuffer(b''.join(chunks), dtype=dtype).reshape(self.height, self.width)
