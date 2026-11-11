import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import urllib.request

from gpu_worker import restore_completed_archives


class ResumeTest(unittest.TestCase):
    def archive(self, root, name, entries):
        path=root/name
        with tarfile.open(path,"w:gz") as out:
            for key,value in entries.items():
                item=tarfile.TarInfo(key);item.size=len(value)
                out.addfile(item,io.BytesIO(value))
        return {"url":path.as_uri()}

    def test_preserves_completed_cases_and_identical_initial_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/"out";out.mkdir()
            archives=[self.archive(root,condition+".tgz",{
                "73_"+condition+"/result.json":json.dumps({"complete":True,"condition":condition}).encode(),
                "shared_initial_0/initial_state.npz":b"same-initial-state",
            }) for condition in ["image","serf"]]
            result=restore_completed_archives(archives,out,urllib.request.build_opener())
            self.assertEqual(set(result),{"73_image","73_serf"})
            self.assertEqual((out/"shared_initial_0/initial_state.npz").read_bytes(),b"same-initial-state")

    def test_rejects_conflicting_initial_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/"out";out.mkdir()
            archives=[self.archive(root,str(i)+".tgz",{"shared_initial_0/initial_state.npz":v})
                      for i,v in enumerate([b"a",b"b"])]
            with self.assertRaisesRegex(AssertionError,"initial state differs"):
                restore_completed_archives(archives,out,urllib.request.build_opener())
            self.assertEqual((out/"shared_initial_0/initial_state.npz").read_bytes(),b"a")

    def test_rejects_archive_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/"out";out.mkdir()
            archive=self.archive(root,"bad.tgz",{"../escape":b"bad"})
            with self.assertRaises(AssertionError):
                restore_completed_archives([archive],out,urllib.request.build_opener())
            self.assertFalse((root/"escape").exists())


if __name__=="__main__":unittest.main()
