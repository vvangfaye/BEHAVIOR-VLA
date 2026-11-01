import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import worker


class ArchiveContract(unittest.TestCase):
    def run_case(self, *, existing=False, corrupt=False, mutate=False, protect=False, sharded=False):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);source=base/'old'/'model.pt';source.parent.mkdir();source.write_bytes(b'weights'*1000)
            checkpoint=source.stat();objects={};puts=[]
            def identity(p):
                s=p.lstat();assert stat.S_ISREG(s.st_mode)
                return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'nlink':s.st_nlink,'blocks':s.st_blocks,
                        'inode':s.st_ino,'device':s.st_dev,'mode':stat.S_IMODE(s.st_mode)}
            def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
            def encode(x):return (json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
            def save(p,x):p.write_bytes(encode(x))
            def request(key,method,*,payload=None,path=None,expected=None,**kwargs):
                if method=='PUT':
                    puts.append(key);objects[key]=path.read_bytes() if path is not None else payload
                raw=objects[key]
                if method=='GET' and key=='data':
                    if corrupt:raw=raw+b'changed'
                    if mutate:source.write_bytes(b'new source')
                result={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                if expected is not None:assert result==expected
                return result
            plan_hash='plan'
            objects['lease']=encode({'plan_sha256':plan_hash,'expires_at':worker.time.time()+60,'observed_at':worker.time.time(),'allowed_indices':[0]})
            if existing:objects['data']=source.read_bytes()
            item={'index':0,'path':str(source),'size':checkpoint.st_size,'mtime_ns':checkpoint.st_mtime_ns,'nlink':1,
                  'key':'data','put':'data','get':'data','manifest_put':'manifest','manifest_get':'manifest',
                  'complete_put':'complete','complete_get':'complete','existing_object':existing}
            if sharded:
                item.update(sharded=True,parts=[{'number':i,'offset':offset,'size':min(2000,checkpoint.st_size-offset),
                    'key':f'part-{i}','put':f'part-{i}','get':f'part-{i}'}
                    for i,offset in enumerate(range(0,checkpoint.st_size,2000))])
            config={'output':str(base/'output'),'account_root':str(base),'helpers_source':'','files':[item],
                    'allowed_roots':[str(source.parent)],'protected_roots':[str(source)] if protect else [],
                    'plan_sha256':plan_hash,'deadline_seconds':30,'metadata_roots':[str(source.parent)],
                    'metadata_key':'metadata','metadata_put':'metadata','metadata_get':'metadata',
                    'metadata_complete_put':'metadata_complete','metadata_complete_get':'metadata_complete',
                    'status_put':'status','lease_get':'lease'}
            def read_small(url,limit=1<<24):
                raw=objects[url]
                if sharded and url=='part-1':
                    if corrupt:raw=raw+b'changed'
                    if mutate:source.write_bytes(b'new source')
                return raw
            replacements={'identity':identity,'sha_file':digest,'encode':encode,'save':save,'request':request,
                          'read_small':read_small}
            with mock.patch.multiple(worker,create=True,**replacements):
                if corrupt or mutate or protect:
                    with self.assertRaises(RuntimeError):worker.run(config)
                    self.assertTrue(source.exists())
                    self.assertNotIn('complete',objects)
                else:
                    worker.run(config)
                    self.assertFalse(source.exists())
                    self.assertTrue(source.with_name('model.pt.aoss.json').is_file())
                    self.assertTrue(json.loads(objects['complete'])['verified'])
                    self.assertEqual(json.loads(objects['status'])['stage'],'complete')
                    if existing:self.assertNotIn('data',puts)

    def test_upload_verify_then_release(self):self.run_case()
    def test_existing_object_is_read_only(self):self.run_case(existing=True)
    def test_corrupt_readback_preserves_source(self):self.run_case(corrupt=True)
    def test_source_change_preserves_source(self):self.run_case(mutate=True)
    def test_protected_file_is_not_released(self):self.run_case(protect=True)


if __name__=='__main__':unittest.main()
