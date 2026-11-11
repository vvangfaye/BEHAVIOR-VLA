import pathlib,tarfile,json,base64
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/eval_results/b1k-video1688977-20261013');files=list((root/'episodes').glob('*/artifact.tar.gz'))+list((root/'artifact_repairs').glob('*/artifact.tar.gz'));r=[]
for p in files[:2]:
 with tarfile.open(p) as t:r.append({'path':str(p),'size':p.stat().st_size,'nlink':p.stat().st_nlink,'sample':[{'name':x.name,'size':x.size,'type':x.type.decode()} for x in list(t)[:8]]})
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'count':len(files),'bytes':sum(p.stat().st_blocks*512 for p in files),'sample':r}).encode()).decode()+'B1KRUNTIMEEND',flush=True)
