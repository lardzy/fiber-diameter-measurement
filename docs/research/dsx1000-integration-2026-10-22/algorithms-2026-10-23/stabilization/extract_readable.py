import json,gzip
from pathlib import Path
root=Path(__file__).resolve().parent
wanted={
 'OlybImageProcessDomain.dll':lambda m: any(x in m['declaring_type'] for x in ['AUseableOlyNmvFilter','BlurringCorrectionFilter','CutInvalidRegionFilter','PreviewResizeImageFilter','LiveFilterRule','SnapFilterRule','ImageProcessService']) or m['name'] in ['_SetParameter','_SetEvaluationParameter'],
 'OlyiBlurringCorrectionSharp64.dll':lambda m:True,
 'OlybMemoryDatabase.dll':lambda m: any(x in m['declaring_type'] for x in ['NmvState','AImageEntityBase','ExtendFocusImageParam']),
 'OlyuImageDisplay.dll':lambda m:any(x in m['name'] for x in ['Blur','LiveImage','BitmapImage']),
 'OlybCameraDomain.dll':lambda m:m['name'] in ['_ReceiveInputFrameBuffer'],
 'OlyuUiCalculator.dll':lambda m:m['name'] in ['ConvertBlurShiftXForDisplay','ConvertBlurShiftYForDisplay','GetImageDisplayPosition'],
 'OlyuUiData.dll':lambda m:m['name'] in ['get_BlurCorrectionBufferSizeX','get_BlurCorrectionBufferSizeY','set_BlurCorrectionBufferSize'] or (m['declaring_type'].endswith('ObservationCommonData') and m['name']=='.ctor'),
 'OlyuMicroscope.dll':lambda m:m['token'] in ['0x060009e3','0x06000834','0x060007f1','0x06005376','0x060053a7'],
 'OlybProcessJoinDomain.dll':lambda m:m['name'] in ['_GetAnnotationAcquitionOffset','_GetExtendBlurCorrection','_GetExtendBlurCorrectionValue','_GetExtendImageResolution'],
}
seen=set()
outputs={}
for p in sorted([*root.glob('*-evidence.json'),*root.glob('*-evidence.json.gz')]):
 d=json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
 for e in d['records']:
  if e['evidence_id'] in seen:continue
  seen.add(e['evidence_id']); n=e.get('normalized_result',{}); name=e['subject']['name']
  if not isinstance(n,dict):
   if e['operation']=='procedure_pseudo_code':
    (root/(name+'.'+e['parameters']['procedure']+'.pseudo.c')).write_text('// '+e['evidence_id']+'\n// '+e['subject']['digest']['sha256']+'\n'+(n or ''))
   continue
  if not n.get('methods') or name not in wanted:continue
  refs={x['token']:(x.get('declaring_type') or x.get('parent_name') or '')+'::'+x.get('name','') for c in ['methods','fields','member_refs'] for x in n[c]}
  for x in n.get('user_strings',{}).get('literals',[]): refs[x['token']]=repr(x.get('value',x.get('text',x)))
  out=['// '+e['evidence_id'],'// '+e['subject']['digest']['sha256']]
  for m in n['methods']:
   if not wanted[name](m):continue
   out.append('\nMETHOD '+m['declaring_type']+'::'+m['name']+' '+m['token'])
   out += [f"{a['il_offset']:04d} {a['opcode']} {a['operand'] if a['operand'] is not None else ''} {refs.get(a['operand'],'')}" for a in m['body']['anchors']]
  outputs.setdefault(name,[]).extend(out)
for name,out in outputs.items(): (root/(name+'.il.txt')).write_text('\n'.join(out))
