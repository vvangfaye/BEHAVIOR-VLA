import pathlib,json,base64,time
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-checkpoints-1792736315');rows=[]
for name in ['installer_cacher4r5', 'baseline_aoss_copyr4r5', 'checkpoint01r4r5', 'checkpoint02r2r4r5', 'checkpoint03r2r4r5', 'checkpoint04r2r4r5', 'checkpoint05r2r4r5', 'checkpoint06r2r4r5', 'checkpoint07r2r4r5', 'checkpoint08r2r4r5']:
 p=root/name;row={'id':name}
 for fn in ['TRANSFER.json','ERROR.json']:
  f=p/fn
  if f.exists():row[fn]=json.loads(f.read_text())
 row['complete']=(p/'COMPLETE.json').exists();rows.append(row)
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'time':time.time(),'rows':rows}).encode()).decode()+'B1KRUNTIMEEND',flush=True)
