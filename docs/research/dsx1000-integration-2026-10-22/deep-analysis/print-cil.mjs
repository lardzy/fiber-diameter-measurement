// Read-only CIL decoding with the installed REA 6.7.0 decoder; no CLR is loaded.
import fs from 'node:fs';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import {decodeInstructions} from '/Users/lishuyang/PycharmProjects/rea/dist/dotnet/ManagedMemberInstructionDecoder.js';
const [input,...names]=process.argv.slice(2);
const value=JSON.parse(zlib.gunzipSync(fs.readFileSync(input)).toString('utf8'));
const selectedId=Array.isArray(value.records)?names.shift():null;
const record=selectedId?value.records.find(r=>r.evidence_id===selectedId):value;
if(!record)throw new Error('Evidence ID not found');
const n=record.normalized_result;
const bytes=fs.readFileSync(record.subject.local_path);
if(crypto.createHash('sha256').update(bytes).digest('hex')!==record.subject.digest.sha256)throw new Error('Artifact SHA mismatch');
const lookup=new Map();
for(const col of ['methods','fields','member_refs','types'])for(const x of n[col])lookup.set(x.token,x.full_name||`${x.declaring_type||x.parent_name||''}.${x.name||''}`);
for(const x of n.user_strings.literals)lookup.set(x.token,JSON.stringify(x.value));
for(const x of n.call_edges)if(!lookup.has(x.target_token))lookup.set(x.target_token,`${x.target_declaring_type}.${x.target_name}`);
console.log('EVIDENCE',record.evidence_id,'SHA256',record.subject.digest.sha256);
for(const m of n.methods.filter(m=>!names.length||names.includes(m.name)||names.includes(m.token))){
 const b=m.body;if(b.status!=='present')continue;
 const o=b.file_offset,hs=b.header_format==='tiny'?1:(bytes.readUInt16LE(o)>>>12)*4;
 const il=bytes.subarray(o+hs,o+hs+b.il_size);
 if(crypto.createHash('sha256').update(il).digest('hex')!==b.il_sha256)throw new Error('IL SHA mismatch');
 const d=decodeInstructions(il);
 console.log('\nMETHOD',m.token,m.declaring_type,m.name,'IL_SHA256',b.il_sha256,'DECODE_ISSUE',d.issue);
 for(const x of d.parsed){
  if(x.opcode==='switch'){
   const count=il.readUInt32LE(x.offset+1),end=x.offset+5+count*4;
   x.switchTargets=Array.from({length:count},(_,i)=>end+il.readInt32LE(x.offset+5+i*4));
  }
  console.log(x.offset.toString(16).padStart(4,'0')+': '+x.opcode.padEnd(13)+' '+(x.operand??'')+' '+(lookup.get(x.operand)||'')+(x.switchTargets?' targets='+x.switchTargets.join(','):''));
 }
}
