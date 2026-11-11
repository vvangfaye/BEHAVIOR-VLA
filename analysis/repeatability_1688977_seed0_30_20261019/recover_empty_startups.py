"""Re-submit only jobs proven to have executed zero experiment episodes."""
import json
from pathlib import Path
import subprocess
import time
import controller as c

root=Path(__file__).resolve().parent
plan=c.load(root/'plan.json')
proof=c.load(root/'empty_startup_proof.json')
assert time.time()-proof['observed_at']<1800
client=c.aoss_client()
for identity,check in proof['groups'].items():
    assert check['all_execution_paths_absent'] and not any(check['paths'].values())
    path=root/'intents'/f'{identity}.json'
    old=c.load(path)
    assert not old.get('generation',0),'Only one bounded startup recovery per group'
    state=c.observe(old,plan);assert state['state']=='FAILED'
    for m in old['members']:
        episode=f"baseline-{m['index']:02d}"
        assert c.fetch(client,plan,episode) is None
        assert c.fetch(client,plan,episode+'/heartbeat',folder='telemetry') is None
    job=c.cli(['acp','jobs','describe',old['name'],'--workspace-name='+plan['workspace'],'-o','json'])
    command=job['roles'][0]['startup_script']
    assert c.digest(command.encode())==old['command_sha256']
    new_name=old['name']+'-g1'
    directory=root/'recoveries'/identity;directory.mkdir(parents=True,exist_ok=True)
    c.save(directory/'original_intent.json',old,exclusive=True)
    c.save(directory/'verification.json',dict(old_terminal_state=state,empty_startup_proof=str(root/'empty_startup_proof.json'),
           verified_no_reports=True,verified_no_heartbeats=True,model_episodes_executed=0,
           worker_source_changed=False,protocol_changed=False,at=time.time()),exclusive=True)
    new=dict(old,name=new_name,uid=None,state='creation_unknown',generation=1,
             previous_name=old['name'],previous_uid=old['uid'],created_at=time.time())
    c.save(directory/'intent.json',new,exclusive=True)
    args=['acp','jobs','copy','--workspace-name='+plan['workspace'],'--copy-job-name='+old['name'],
          '--name='+new_name,'--job-name='+new_name,'--quota-type=spot','--retry-times=0',
          '--command='+command]
    result=subprocess.run([c.CLI]+args,env=c.environment(),capture_output=True,text=True,timeout=90)
    new['cli_returncode']=result.returncode;c.save(directory/'intent.json',new)
    state=c.observe(new,plan);new.update(state)
    c.save(directory/'intent.json',new)
    c.save(path,new)
    print(json.dumps(state),flush=True)
