import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import archive_worker as a

class Response(io.BytesIO):
 status=200
 def getheader(self,key):return str(len(self.getvalue())) if key=='Content-Length' else None
class Conn:
 def __init__(self,body):self.body=body
 def request(self,*args,**kwargs):pass
 def getresponse(self):return Response(self.body)
 def close(self):pass

class ArchiveTests(unittest.TestCase):
 def test_pack_full_readback_and_corruption(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'log.txt';p.write_bytes(b'important historical log'*100)
   entry={'relative_path':'log.txt',**a.identity(p)};dest=root/'bundle.gz'
   manifest=a.pack(root,[entry],dest,{'unit':'test'})
   body=dest.read_bytes()
   with patch.object(a,'connection',return_value=(Conn(body),'/')):
    report=a.verify_archive('unused',a.digest(body),len(body),manifest)
   self.assertEqual(report['files'],1)
   with patch.object(a,'connection',return_value=(Conn(body),'/')):
    with self.assertRaises(AssertionError):a.verify_archive('unused','0'*64,len(body),manifest)
 def test_protected_training_and_expert_paths(self):
  root=Path('/mnt/afs/260010165/behavior_1k/old')
  for name in ['checkpoints/round_01/rollout.pt','data/old.mp4','expert/a.bin','episode/model.pt','log.hdf5']:
   self.assertFalse(a.eligible(root/name,root,[]),name)
  self.assertTrue(a.eligible(root/'episode/rollout.pt',root,[]))
  self.assertFalse(a.eligible(root/'episode/rollout.pt',root,[root/'episode']))
 def test_source_change_blocks_pack(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'log.txt';p.write_bytes(b'a');item={'relative_path':'log.txt',**a.identity(p)};p.write_bytes(b'different')
   with self.assertRaises(AssertionError):a.pack(root,[item],root/'out.gz',{})
if __name__=='__main__':unittest.main()
