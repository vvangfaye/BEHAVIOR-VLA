"""Local tests of queue scope and per-member result preservation; no cloud calls."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('batch_runner',ROOT/'frozen/batch_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
spec=importlib.util.spec_from_file_location('batch_controller',ROOT/'controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


class Response:
    def __enter__(self):return self
    def __exit__(self,*args):pass


class Opener:
    def __init__(self):self.sent=[]
    def open(self,request,timeout):
        self.sent.append((request.full_url,json.loads(request.data)))
        return Response()


class BatchTests(unittest.TestCase):
    def exercise(self, failure=None, timeout=None):
        old_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
                root=Path(temporary);members=[]
                for i in range(4):
                    id=f'candidate-{i:02d}';path=root/(id+'.json')
                    report=dict(id=id,state='failed' if i==failure else 'accepted')
                    source=('import json,time\nfrom pathlib import Path\n'+
                            ('time.sleep(2)\n' if i==timeout else '')+
                            f'Path({str(path)!r}).write_text(json.dumps({report!r}))\n')
                    members.append(dict(id=id,source=base64.b64encode(zlib.compress(source.encode())).decode(),
                                        sha256=hashlib.sha256(source.encode()).hexdigest(),report_path=str(path),
                                        report_url='https://example.invalid/'+id))
                config=dict(id='batch-00',root=str(root/'batch'),python=sys.executable,members=members,
                            member_walltime_seconds=0.5 if timeout is not None else 5,
                            protocol_sha256='p',worker_bundle_sha256='w',report_url='https://example.invalid/batch')
                opener=Opener()
                with patch.object(runner.urllib.request,'build_opener',return_value=opener):
                    result=runner.run(config)
                state=json.loads((root/'batch/status.json').read_text())
                self.assertEqual(len(state['members']),4)
                self.assertEqual(len({m['id'] for m in state['members']}),4)
                self.assertEqual(state['members'][0]['state'],'accepted')
                self.assertEqual(state['members'][3]['state'],'accepted')
                self.assertEqual(result,0 if failure is None and timeout is None else 1)
                member_uploads=[x for url,x in opener.sent if url.endswith('candidate-00')]
                self.assertEqual(member_uploads,[{'id':'candidate-00','state':'accepted'}])
                self.assertGreaterEqual(len(opener.sent),9)
        finally:
            for s,handler in old_handlers.items():signal.signal(s,handler)

    def test_all_four_sequential_results(self):self.exercise()
    def test_middle_failure_preserves_prior_and_runs_remaining(self):self.exercise(failure=1)
    def test_timeout_preserves_prior_and_runs_remaining(self):self.exercise(timeout=1)
    def test_fixed_scope_and_bootstrap(self):
        plan=controller.load(ROOT/'plan.json')
        indices=[i for g in plan['batch_groups'] for i in g['indices']]
        self.assertEqual(sorted(indices),list(range(20)))
        self.assertTrue(all(len(g['indices'])==4 for g in plan['batch_groups']))
        class Client:
            def generate_presigned_url(self,*args,**kwargs):
                return 'https://example.invalid/'+kwargs['Params']['Key']+'?signature='+'x'*200
        for group in plan['batch_groups']:
            startup=controller.bootstrap(Client(),plan,group,'local-test-'+group['id'])
            compile(startup,'<test>','exec')
            loader=controller.bootstrap_loader('https://example.invalid/startup?signature='+'x'*200,
                         hashlib.sha256(startup.encode()).hexdigest(),'/tmp/own/bootstrap.py')
            compile(loader,'<loader>','exec')
            self.assertLess(len(loader),2000)
        broken=dict(plan,max_active_gpu_jobs=6)
        with self.assertRaises(ValueError):controller.validate_plan(broken)


if __name__=='__main__':unittest.main()
