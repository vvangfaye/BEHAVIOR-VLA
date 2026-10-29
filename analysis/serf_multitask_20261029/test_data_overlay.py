import hashlib
import io
import tarfile
import tempfile
import unittest
import urllib.request
from pathlib import Path
from gpu_worker import materialize_data


class OverlayTests(unittest.TestCase):
    def build(self,root,files):
        archive=root/'overlay.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for name,data in files.items():
                member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
        return archive

    def test_new_task_files_do_not_modify_shared_data(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'shared';base.mkdir();(base/'old').write_bytes(b'old');local=root/'job';local.mkdir()
            archive=self.build(root,{'old':b'old','instances/new':b'new'})
            out=materialize_data(base,local,archive.as_uri(),hashlib.sha256(archive.read_bytes()).hexdigest(),urllib.request.build_opener())
            self.assertEqual((out/'old').read_bytes(),b'old');self.assertTrue((out/'old').is_symlink())
            self.assertEqual((out/'instances/new').read_bytes(),b'new');self.assertFalse((base/'instances').exists())

    def test_shared_conflicting_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'shared';base.mkdir();(base/'old').write_bytes(b'old');local=root/'job';local.mkdir()
            archive=self.build(root,{'old':b'changed'})
            with self.assertRaises(AssertionError):materialize_data(base,local,archive.as_uri(),hashlib.sha256(archive.read_bytes()).hexdigest(),urllib.request.build_opener())
            self.assertEqual((base/'old').read_bytes(),b'old')

    def test_archive_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'shared';base.mkdir();local=root/'job';local.mkdir();archive=self.build(root,{'../escape':b'bad'})
            with self.assertRaises(AssertionError):materialize_data(base,local,archive.as_uri(),hashlib.sha256(archive.read_bytes()).hexdigest(),urllib.request.build_opener())
            self.assertFalse((local/'escape').exists())


if __name__=='__main__':unittest.main()
