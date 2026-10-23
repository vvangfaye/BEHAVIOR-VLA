import pathlib,json,base64,zipfile
base=pathlib.Path('/mnt/afs/260010165/data/urban_city_v2');r=[]
for p in (base/'archives').iterdir():
 x={'path':str(p),'size':p.stat().st_size}
 if zipfile.is_zipfile(p):
  with zipfile.ZipFile(p) as z:
   m=z.infolist();x.update(members=len(m),uncompressed=sum(v.file_size for v in m),sample=[v.filename for v in m[:5]])
 r.append(x)
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'archives':r,'dirs':[v.name for v in base.iterdir()]}).encode()).decode()+'B1KRUNTIMEEND',flush=True)
