import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import test_worker

spec=importlib.util.spec_from_file_location('parts_worker',Path(__file__).parent/'retry_a5/worker.py')
parts_worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parts_worker)


class PartsContract(unittest.TestCase):
    def check_case(self, **kwargs):
        case=test_worker.ArchiveContract()
        with mock.patch.object(test_worker,'worker',parts_worker):
            case.run_case(sharded=True,**kwargs)

    def test_concatenated_readback_matches_source(self):self.check_case()
    def test_corrupt_part_preserves_source(self):self.check_case(corrupt=True)
    def test_changed_source_is_retained(self):self.check_case(mutate=True)
    def test_protected_source_is_retained(self):self.check_case(protect=True)


if __name__=='__main__':unittest.main()
