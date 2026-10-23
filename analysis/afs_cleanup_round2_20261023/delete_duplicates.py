import pathlib,json,hashlib,stat,time,base64
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct');m=json.loads((out/'duplicate_manifest.json').read_text());root=pathlib.Path(m['root']);assert 0<m['allocated_bytes']<64*1024**2

def verify(item):
 p=pathlib.Path(item['path']);s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode)
 assert all(getattr(s,'st_'+a)==item[b] for a,b in [('size','size'),('mtime_ns','mtime_ns'),('ino','inode'),('dev','device'),('nlink','nlink')])
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 assert h.hexdigest()==item['sha256'];assert p.lstat()==s
for pair in m['pairs']:
 assert pathlib.Path(pair['remove']['path']).is_relative_to(root/'batches')
 assert pathlib.Path(pair['retain']['path']).is_relative_to(root) and not pathlib.Path(pair['retain']['path']).is_relative_to(root/'batches')
 verify(pair['remove']);verify(pair['retain'])
for pair in m['pairs']:pathlib.Path(pair['remove']['path']).unlink()
r={'files_removed':len(m['pairs']),'allocated_bytes_removed':m['allocated_bytes'],'retained_copies':[p['retain']['path'] for p in m['pairs']],'time':time.time()};(out/'duplicate_release.json').write_text(json.dumps(r,indent=2))
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
