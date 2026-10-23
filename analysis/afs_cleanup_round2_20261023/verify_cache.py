import pathlib,json,base64,os,time
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct');m=json.loads((out/'cache_manifest.json').read_text());r=json.loads((out/'cache_release.json').read_text())
remaining=[str(pathlib.Path(m['roots'][f['root']])/f['path']) for f in m['files'] if os.path.lexists(pathlib.Path(m['roots'][f['root']])/f['path'])]
probe=out/'.write-probe-final'
with probe.open('xb') as f:f.write(b'x'*4096);f.flush();os.fsync(f.fileno())
probe.unlink()
x={'deleted_manifest_files_absent':not remaining,'unexpected_remaining_count':len(remaining),'write_probe_succeeded':True,'released_bytes':r['allocated_bytes_removed'],'time':time.time()};(out/'cache_verification.json').write_text(json.dumps(x))
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(x).encode()).decode()+'B1KRUNTIMEEND',flush=True)
