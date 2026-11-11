"""Bounded read-only observation of the five exact early1000 evaluation jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT/'paired_fixed20_step1000_5x4'
spec=importlib.util.spec_from_file_location('batch_controller_readonly',CAMPAIGN/'controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def snapshot():
    from botocore.exceptions import ClientError
    plan=controller.load(CAMPAIGN/'plan.json')
    intents=[controller.load(p) for p in sorted((CAMPAIGN/'intents').glob('*.json'))]
    if (len(intents)!=5 or {i['id'] for i in intents}!={g['id'] for g in plan['batch_groups']}
            or any(not i.get('uid') for i in intents)):
        raise ValueError('Expected five already-created, exact batch UIDs; observer never creates')
    client=controller.aoss_client()
    def read(relative):
        try:
            obj=client.get_object(Bucket='260010165',Key=plan['object_prefix']+'/'+relative)
            try:return json.loads(obj['Body'].read(8*1024**2))
            finally:obj['Body'].close()
        except ClientError as exc:
            if exc.response['Error']['Code'] in ('NoSuchKey','404'):return None
            raise
    def one(intent):
        native=controller.observe(intent)
        report=read('reports/'+intent['id']+'.json')
        current=(report or {}).get('current_member')
        heartbeat=read('telemetry/'+current+'/heartbeat.json') if current else None
        files=(heartbeat or {}).get('episode_files',{})
        videos=[v for k,v in files.items() if k.startswith('results/videos/')]
        return dict(native=native,batch_report=report,current_member_heartbeat=heartbeat,
                    batch_id=intent['id'],current_member=current,
                    seed_ack_file_present=files.get('seed_ack.json',{}).get('exists',False),
                    seed_ack_contents_read=False,
                    seed_ack_written_at='episode_end',
                    video_started=any(v.get('exists') and (v.get('bytes') or 0)>0 for v in videos),
                    acceptance_file_present=files.get('acceptance.json',{}).get('exists',False))
    with ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(one,intents))
    out=dict(observed_at=time.time(),read_only=True,job_count=5,rows=rows,
             training_modified=False,model_improvement_proven=False)
    folder=CAMPAIGN/'runtime_observations';folder.mkdir(exist_ok=True)
    controller.save(folder/(str(time.time_ns())+'.json'),out)
    controller.save(CAMPAIGN/'latest_runtime_progress.json',out)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=0)
    parser.add_argument('--until',choices=('video','completed'),default='video')
    args=parser.parse_args()
    if not 0<=args.seconds<=1800:raise ValueError('Bounded observation only')
    end=time.monotonic()+args.seconds
    previous=None
    while True:
        result=snapshot()
        event=dict(observed_at=result['observed_at'],
                   states={r['batch_id']:r['native']['state'] for r in result['rows']},
                   started_batches=sum(r['batch_report'] is not None for r in result['rows']),
                   current_members={r['batch_id']:r['current_member'] for r in result['rows']},
                   seed_ack_files=sum(r['seed_ack_file_present'] for r in result['rows']),
                   seed_ack_written_at='episode_end',
                   videos_started=sum(r['video_started'] for r in result['rows']))
        changed={k:v for k,v in event.items() if k!='observed_at'}
        if changed!=previous:
            print(json.dumps(event),flush=True);previous=changed
        reached=event['videos_started'] if args.until=='video' else event['seed_ack_files']
        if (reached or time.monotonic()>=end or
                any(s in controller.TERMINAL for s in event['states'].values())):
            print(json.dumps({'event':'bounded_observation_ended','read_only':True,
                              'observed_at':time.time()}),flush=True)
            return
        time.sleep(min(45,max(0,end-time.monotonic())))


if __name__=='__main__':main()
