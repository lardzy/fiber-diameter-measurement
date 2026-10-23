import base64,subprocess,sys
args=sys.argv[1:]; user=False
if args and args[0]=='--user': user=True;args.pop(0)
code=open(args[0],encoding='utf-8-sig').read() if args else sys.stdin.read()
cmd=['/Applications/Parallels Desktop.app/Contents/MacOS/prlctl','exec','Windows 11']
if user:cmd+=['--current-user']
cmd+=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand',base64.b64encode(code.encode('utf-16-le')).decode()]
r=subprocess.run(cmd,capture_output=True,timeout=300)
for stream in (r.stdout,r.stderr):
 try:s=stream.decode('utf-8')
 except UnicodeDecodeError:s=stream.decode('gb18030',errors='replace')
 print(s,flush=True)
sys.exit(r.returncode)
