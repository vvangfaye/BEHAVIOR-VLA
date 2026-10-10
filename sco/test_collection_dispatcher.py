"""A failed pilot must never authorize Spot submission or lose its receipts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import continue_subtask_collection as dispatcher


class DispatcherTests(unittest.TestCase):
    def test_failed_standard_pilot_stops_before_any_resource_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pilot=dict(name='owned-pilot',uid='pilot-uid',source_manifest_sha256='a'*64)
            wave=dict(state='complete',source_manifest_sha256='a'*64,
                plan=dict(fingerprint='f'*64,expected_trials=30))
            (root/'pilot.json').write_text(json.dumps(pilot))
            (root/'wave.json').write_text(json.dumps(wave))
            (root/'client.py').write_text('def client():return object()\n')
            job=dict(name='owned-pilot',uid='pilot-uid',state='FAILED',ownership={'user_name':'260010165'})
            argv=['dispatcher','--pilot-receipt',str(root/'pilot.json'),'--wave-prepared',str(root/'wave.json'),
                '--output-dir',str(root/'receipts'),'--aoss-client-module',str(root/'client.py')]
            with patch.object(sys,'argv',argv),patch.object(dispatcher.subprocess,'check_output',return_value=json.dumps(job)),\
                    patch.object(dispatcher.subprocess,'run') as mutate:
                with self.assertRaisesRegex(AssertionError,'Standard pilot'):
                    dispatcher.main()
            mutate.assert_not_called()
            result=json.loads((root/'receipts/status.json').read_text())
            self.assertEqual(result['state'],'needs_attention')
            self.assertEqual(result['pilot_uid'],'pilot-uid')
            self.assertEqual(result['expected_trials'],30)


if __name__=='__main__':unittest.main()
