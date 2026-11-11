"""Read only decoded numerical quota fields; never inspect auth or environment."""
import json,subprocess,time
from pathlib import Path
CLI='/home/faye/.local/bin/sco'
ROOT=Path(__file__).resolve().parent

def query_quota():
    def command(args):
        return subprocess.run([CLI,*args],capture_output=True,text=True,timeout=40,check=True)
    previous=command(['config','get','core.zone']).stdout.strip()
    try:
        command(['zones','use','cn-sh-01g'])
        result=subprocess.run(['gdb','-q','-batch','-x',str(ROOT/'read_quota_response.gdb'),
            '--args',CLI,'afs','dir-quota','list','--volumeID=01a04263-91e5-7603-bc01-c67e503da6b5',
            '--dirPath=/260010165'],capture_output=True,text=True,timeout=45)
        rows=[json.loads(s.split('=',1)[1]) for s in result.stdout.splitlines() if s.startswith('B1K_QUOTA_JSON=')]
        if not rows or len(rows[-1])!=1:raise RuntimeError('No exact quota response; keep admission closed')
        row=rows[-1][0]
        assert row['dir_path']=='/260010165' and row['quota_bytes']>0 and row['used_bytes']>=0
        return {'observed_at':time.time(),'method':'normal SCO auth; decoded numerical quota response only',
                'volume_id':'01a04263-91e5-7603-bc01-c67e503da6b5','rows':[row]}
    finally:
        command(['zones','use',previous])

if __name__=='__main__':print(json.dumps(query_quota()))
