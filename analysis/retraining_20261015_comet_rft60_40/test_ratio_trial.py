import importlib.util,copy
from pathlib import Path
import pytest
P=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('ratio',P/'prepare_ratio_trial.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_ratio_only_change_preserves_all_selected_data_and_clock():
    manifest,plan=m.build(.8,.2);base=m.load(m.BASE/'inputs/compiled/mixture.json')
    for a,b in zip(manifest['phases'][0]['sources'],base['phases'][0]['sources']):
        assert {k:v for k,v in a.items() if k!='weight'}=={k:v for k,v in b.items() if k!='weight'}
    assert manifest['seed']==base['seed']==42 and manifest['global_batch_size']==192
    assert plan.fingerprint != m.MixturePlan.load(m.BASE/'inputs/compiled/mixture.json').fingerprint
    assert set(plan.roots)==set(m.MixturePlan.load(m.BASE/'inputs/compiled/mixture.json').roots)

@pytest.mark.parametrize('a,b',[(1,0),(-.1,1.1),(.9,.2),(float('nan'),.2),(float('inf'),.2)])
def test_invalid_weights_rejected(a,b):
    with pytest.raises(ValueError):m.build(a,b)

def test_assembly_preserves_data_and_excludes_old_authorization(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'HERE',tmp_path)
    m.prepare(.8,.2,'test');r=m.assemble('test');bundle=tmp_path/'ratio_trials/test/bundle'
    assert r['full_bundle_assembled'] and not r['cpu_real_batch_verified'] and not r['gpu_smoke_verified']
    assert not m.load(bundle/'inputs/formal_authorization.json')['formal_training_allowed']
    assert not (bundle/'inputs/gpu_smoke_success.json').exists()
    original=m.load(m.BASE/'bundle_manifest.json');actual=m.load(bundle/'bundle_manifest.json')
    assert {k for k in actual if actual[k]!=original[k]}==set(r['changed_files'])
    assert all(actual[k]==original[k] for k in actual if '/indexes/' in k)
    with pytest.raises(FileExistsError):m.assemble('test')

def test_assembly_rejects_mixture_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'HERE',tmp_path)
    m.prepare(.8,.2,'test');p=tmp_path/'ratio_trials/test/mixture.json'
    v=m.load(p);v['seed']=99;m.save(p,v)
    with pytest.raises(ValueError,match='declared experiment'):m.assemble('test')
