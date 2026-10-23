import pathlib,json,hashlib,time,base64,os
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct')
cache=json.loads((out/'cache_release.json').read_text());dups=json.loads((out/'duplicate_release.json').read_text());tar=json.loads((out/'tar_release.json').read_text())
assert tar['state']=='completed'
missing_kept=[]
for item in tar['results']:
 p=pathlib.Path(item['source'])
 if item['state']=='deleted_duplicate_tar':
  assert not os.path.lexists(p) and pathlib.Path(item['proof']).is_file() and (p.parent/'bundle').is_dir()
 else:assert p.is_file()
m=json.loads((out/'duplicate_manifest.json').read_text())
for pair in m['pairs']:
 assert not pathlib.Path(pair['remove']['path']).exists()
 retained=pathlib.Path(pair['retain']['path']);assert retained.is_file()
 h=hashlib.sha256()
 with retained.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==pair['retain']['sha256']
probe=out/('.probe-'+str(time.time_ns()))
with probe.open('xb') as f:f.write(b'x'*4096);f.flush();os.fsync(f.fileno())
probe.unlink()
result={'state':'verified','cache_files':cache['files_removed'],'cache_bytes':cache['allocated_bytes_removed'],'duplicate_files':dups['files_removed'],'duplicate_bytes':dups['allocated_bytes_removed'],'tar_files':tar['files_removed'],'tar_bytes':tar['allocated_bytes_removed'],'tar_retained_count':len(tar['retained']),'total_files_removed':cache['files_removed']+dups['files_removed']+tar['files_removed'],'total_bytes_removed':cache['allocated_bytes_removed']+dups['allocated_bytes_removed']+tar['allocated_bytes_removed'],'write_probe_succeeded':True,'retained_duplicate_copies_sha256_verified':True,'deleted_tars_absent_and_bundles_present':True,'time':time.time()}
(out/'FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2))
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
