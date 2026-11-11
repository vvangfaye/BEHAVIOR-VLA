import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from checkpoint_inventory import verified_checkpoints


class Client:
    def __init__(self,objects):self.objects=objects
    def list_objects_v2(self,**kw):return {'Contents':[{'Key':k,'Size':len(v)} for k,v in self.objects.items()]}
    def head_object(self,Bucket,Key):return {'Metadata':{'sha256':hashlib.sha256(self.objects[Key]).hexdigest()}}
    def get_object(self,Bucket,Key):return {'Body':io.BytesIO(self.objects[Key])}


class InventoryTest(unittest.TestCase):
    def test_task_becomes_ready_only_when_both_checkpoints_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);items=[];objects={}
            for tid in [21,26]:
                for model in ['image','serf']:
                    path=f'checkpoints/{model}--task-{tid:04d}/params';content=f'{tid}-{model}'.encode()
                    items.append({'destination':path,'size':len(content),'sha256':hashlib.sha256(content).hexdigest()})
                    if tid==21:objects['behavior1k/serf-multitask-20261029/assets/'+path]=content
            (root/'checkpoint_manifest.json').write_text(json.dumps({'items':items}))
            result=verified_checkpoints(Client(objects),root)
            self.assertEqual(result['tasks_ready'],[21]);self.assertFalse(result['complete'])
            objects.pop(next(k for k in objects if '/serf--' in k))
            (root/'checkpoint_inventory.json').unlink()
            result=verified_checkpoints(Client(objects),root)
            self.assertEqual(result['tasks_ready'],[])


if __name__=='__main__':unittest.main()
