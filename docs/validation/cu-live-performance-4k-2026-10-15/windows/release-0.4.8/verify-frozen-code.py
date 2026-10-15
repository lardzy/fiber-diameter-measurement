"""Verify actual embedded PYZ code against the 0.4.8 source snapshot; Windows only."""
from pathlib import Path
import argparse,hashlib,json,types
from PyInstaller.archive.readers import CArchiveReader
ROOT=Path.cwd()
parser=argparse.ArgumentParser();parser.add_argument('app',type=Path);parser.add_argument('--label',required=True);args=parser.parse_args()
exe=args.app/'FiberDiameterMeasurement.exe';archive=CArchiveReader(str(exe));pyz=archive.open_embedded_archive(next(name for name,row in archive.toc.items() if row[-1]=='z'))
def normalize(value):
 if isinstance(value,types.CodeType):
  fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_code','co_consts','co_names','co_varnames','co_freevars','co_cellvars','co_name','co_qualname','co_firstlineno','co_linetable','co_exceptiontable')
  return tuple((key,normalize(getattr(value,key))) for key in fields)
 if isinstance(value,tuple):return tuple(normalize(item) for item in value)
 if isinstance(value,frozenset):return frozenset(normalize(item) for item in value)
 return value
modules=('fdm.version','fdm.live_experiment.models','fdm.ui.live_experiment_workspace','fdm.live_experiment.motion','fdm.live_experiment.controller','fdm.live_experiment.frames','fdm.live_experiment.sources','fdm.services.cu_live_self_check','fdm.live_experiment.png_stream','fdm.live_experiment.evidence','fdm.live_experiment.project_assets','fdm.ui.project_session_controller','fdm.services.cu_automatic_worker')
checks=[]
for name in modules:
 path=ROOT/'src'/Path(*name.split('.')).with_suffix('.py');data=path.read_bytes();source=compile(data,str(path),'exec',optimize=0);embedded=pyz.extract(name)
 checks.append({'module':name,'source_sha256':hashlib.sha256(data).hexdigest(),'actual_pyz_equals_compiled_source_ignoring_filename':normalize(source)==normalize(embedded)})
report={'ok':all(row['actual_pyz_equals_compiled_source_ignoring_filename'] for row in checks),'exe':str(exe),'exe_sha256':hashlib.file_digest(exe.open('rb'),'sha256').hexdigest(),'checks':checks,'normalization':'Only co_filename ignored; bytecode, constants, nested code, names, flags, arguments, exception/line tables and first lines all compared.'}
out=ROOT/'build/opt4k'/f'{args.label}-frozen-code.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(report,allow_nan=False));raise SystemExit(0 if report['ok'] else 1)
