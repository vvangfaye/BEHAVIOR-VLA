"""CPU-only classification of a stopped writer's preserved episode."""

def run(p):
 import hashlib,json,os,pathlib,sqlite3,time,urllib.request
 for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(k,None)
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 root=pathlib.Path(p['assignment_root'])
 assert root.parent==pathlib.Path(p['afs_output_root'])/'episodes'
 assert p['old_writer_stopped'] and p['old_state'] in ('FAILED','STOPPED','DELETED','CANCELLED','CANCELED')
 out=dict(old_uid=p['old_uid'],old_name=p['old_name'],old_state=p['old_state'],old_writer_stopped=True,checkpoint_step=2000000,task=p['task'],instance_id=p['instance_id'],observed_epoch=time.time(),gpu_count=0,no_model_execution=True,assignment_root=str(root))
 stem=p['task']+'_'+str(p['instance_id'])+'_0.json'
 entries=[];protected=[];attempts=[]
 if root.exists():
  assert not root.is_symlink()
  saved=root/'payload.json'
  if saved.exists():
   raw=saved.read_bytes();old=json.loads(raw)
   for k in ('task','instance_id','checkpoint_step','source_manifest_sha256','checkpoint_ready_sha256'):assert old[k]==p[k]
   out['old_payload_sha256']=hashlib.sha256(raw).hexdigest()
  batch=old.get('batch_context') if saved.exists() else None
  batch_root=None
  if batch:
   batch_root=pathlib.Path(batch['root'])
   assert batch_root.parent==pathlib.Path(p['afs_output_root'])/'batches'
   assert batch_root.name==batch['batch_name'] and not batch_root.is_symlink()
   # Search only this exact logical episode in the shared original attempt tree.
   for f in batch_root.rglob(stem):
    if f.is_file():
     raw=f.read_bytes();entries.append(dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
     try:
      v=json.loads(raw)
      if v.get('terminated') or v.get('truncated'):protected.append(str(f))
     except Exception:protected.append(str(f))
  for f in root.rglob(stem):
   if f.is_file():
    raw=f.read_bytes();entries.append(dict(path=str(f.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    try:
     v=json.loads(raw)
     if v.get('terminated') or v.get('truncated'):protected.append(str(f.relative_to(root)))
    except Exception:protected.append(str(f.relative_to(root)))
  receipt=root/'receipt.json'
  if receipt.exists():
   v=json.loads(receipt.read_bytes());out['old_receipt_state']=v.get('state');out['old_error_type']=v.get('error_type')
   if v.get('state') in ('artifacts_verified','complete','delivery_failed'):protected.append('receipt.json:complete_model_output')
  if (root/'bundle').exists() or (root/'_SUCCESS').exists():protected.append('existing_bundle_or_success')
  progress=root/'native_progress.json'
  if progress.exists():out['last_native_progress']=json.loads(progress.read_bytes())
  db=root/'queue.sqlite3'
  if not db.exists() and batch_root is not None:db=batch_root/'queue.sqlite3'
  if db.exists():
   connection=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
   if batch_root is not None:
    attempts=[dict(x) for x in connection.execute('SELECT attempts.* FROM attempts JOIN jobs ON jobs.id=attempts.job_id WHERE jobs.task=? AND jobs.instance_id=?',(p['task'],p['instance_id']))]
   else:attempts=[dict(x) for x in connection.execute('SELECT * FROM attempts')]
   connection.close()
  out['root_entries']=sorted(f.name for f in root.iterdir())
 else:out['root_absent']=True
 out.update(original_result_files=entries,protected_completed_outputs=protected,attempts=attempts,attempt_count=len(attempts),eligible=not protected,reason='complete_outputs_need_artifact_audit' if protected else 'incomplete_attempt_preserved')
 raw=(json.dumps(out,sort_keys=True,indent=2)+'\n').encode()
 request=urllib.request.Request(p['report_put'],data=raw,method='PUT')
 with opener.open(request,timeout=60) as response:response.read()
 print('STATUS_JSON='+json.dumps({k:v for k,v in out.items() if k not in ('attempts',)}),flush=True)
 return out
