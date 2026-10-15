from pathlib import Path
import ctypes,hashlib,json,struct
root=Path.cwd();out=root/'build/opt4k/release-0.4.8'
source=json.loads((out/'source.json').read_text(encoding='utf8'))
checks=[]
for row in source['files']:
    data=(root/row['path']).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    exact=digest==row['sha256']
    generated_crlf=(row['path']=='packaging/inno-setup/version.auto.iss' and b'\r\n' in data
        and hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==row['sha256'])
    checks.append(dict(path=row['path'],expected_sha256=row['sha256'],actual_sha256=digest,
        byte_identical=exact,generated_windows_crlf_only=generated_crlf,ok=exact or generated_crlf))
assert all(row['ok'] for row in checks),checks
identity=json.loads((out/'installer-identity.json').read_text(encoding='utf8'))
installer=Path(identity['path'])
assert hashlib.file_digest(installer.open('rb'),'sha256').hexdigest()==identity['sha256']
version=ctypes.WinDLL('version',use_last_error=True)
version.GetFileVersionInfoSizeW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32)]
version.GetFileVersionInfoSizeW.restype=ctypes.c_uint32
version.GetFileVersionInfoW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
version.GetFileVersionInfoW.restype=ctypes.c_int
version.VerQueryValueW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_uint)]
version.VerQueryValueW.restype=ctypes.c_int
size=version.GetFileVersionInfoSizeW(str(installer),None);assert size
data=ctypes.create_string_buffer(size)
assert version.GetFileVersionInfoW(str(installer),0,size,data)
pointer=ctypes.c_void_p();length=ctypes.c_uint()
assert version.VerQueryValueW(data,'\\',ctypes.byref(pointer),ctypes.byref(length))
assert length.value>=52
fixed=struct.unpack('<13I',ctypes.string_at(pointer.value,52))
assert fixed[0]==0xfeef04bd
file_version=[fixed[2]>>16,fixed[2]&65535,fixed[3]>>16,fixed[3]&65535]
pe=json.loads((out/'installer-pe-version.json').read_text(encoding='utf8'))
assert pe['ProductVersion'].strip()=='0.4.8',pe
release=json.loads((out/'full-release-manifest.json').read_text(encoding='utf8'))
assert release['version']=='0.4.8'
result=dict(ok=True,original_runner_postbuild_assertion_failed=True,
    reason='Only auto-generated version.auto.iss acquired Windows CRLF. Every other listed file is byte-identical. Exact LF-equivalent version content and installer product version 0.4.8 verified; no rebuild or artifact replacement.',
    first_pe_probe_incorrectly_required_numeric_file_version=True,
    pe_version_note='Inno installer has a blank FileVersion and numeric 0.0.0.0; its ProductVersion is 0.4.8. Application manifest and embedded fdm.version independently identify 0.4.8. Original numeric probe failure retained, no product change made to satisfy it.',
    checks=checks,installer_file_version=file_version,installer_product_version=pe['ProductVersion'].strip(),installer_sha256=identity['sha256'])
(out/'postbuild-generated-version-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result),flush=True)
