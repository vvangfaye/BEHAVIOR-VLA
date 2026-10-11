"""Use observed ACP worker phases to distinguish deletion from active writers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parent))
from submit_full_collection_shard import terminal_worker_evidence

UID='9a7f1475403e49ceabeddca207e5a9d6'
OBSERVED='| pt-'+UID+'-worker-0 | n12lp(加速芯片:1, | 10.115.73.114 | 10.120.1.55 | Deleted |'


def test_deleted_stopped_worker_is_valid_recovery_evidence():
    assert terminal_worker_evidence(OBSERVED,UID)==OBSERVED


@pytest.mark.parametrize('phase',['Failed','Succeeded'])
def test_other_terminal_worker_phases_remain_supported(phase):
    row=OBSERVED.replace('Deleted',phase)
    assert terminal_worker_evidence(row,UID)==row


@pytest.mark.parametrize('phase',['Running','Pending','Terminating','Unknown','Deleting'])
def test_active_or_uncertain_worker_cannot_be_recovered(phase):
    with pytest.raises(ValueError,match='terminal'):
        terminal_worker_evidence(OBSERVED.replace('Deleted',phase),UID)


def test_wrong_uid_absent_and_duplicate_rows_cannot_prove_stop():
    for table in ['',OBSERVED.replace(UID,'0'*32),OBSERVED+'\n'+OBSERVED]:
        with pytest.raises(ValueError,match='terminal'):
            terminal_worker_evidence(table,UID)
