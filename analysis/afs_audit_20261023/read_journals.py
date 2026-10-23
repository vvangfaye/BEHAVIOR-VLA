import pathlib,json,base64,time
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-checkpoints-1792736315')
rows=[]
for name in ['installer_cacher4', 'baseline_aoss_copyr4', 'checkpoint01r4', 'checkpoint02r2r4', 'checkpoint03r2r4', 'checkpoint04r2r4', 'checkpoint05r2r4', 'checkpoint06r2r4', 'checkpoint07r2r4', 'checkpoint08r2r4']:
 p=root/name;f=p/'MANIFEST.partial.json';m=json.loads(f.read_text()) if f.exists() else {}
 rows.append({'id':name,'files_verified':len(m.get('files',[])),'bytes_verified':sum(x['size'] for x in m.get('files',[])),'complete':(p/'COMPLETE.json').exists(),'released':(p/'RELEASED.json').exists()})
result={'time':time.time(),'units':rows}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
