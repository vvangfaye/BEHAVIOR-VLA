import concurrent.futures
import hashlib
import json
from pathlib import Path
from legacy_zip import decode_entry,fetch_range,index

HERE=Path(__file__).resolve().parent


def extract(selection,all_entries,target):
    selected=sorted(selection['entries'],key=lambda x:x['offset'])
    offsets=sorted(set(x['offset'] for x in all_entries))
    following=dict(zip(offsets,offsets[1:]))
    batches=[]
    for item in selected:
        start=item['offset'];end=following.get(start,start+30+131070+item['compressed_size'])-1
        end=min(end,selection['size']-1)
        if batches and start-batches[-1]['end']<=65536 and end-batches[-1]['start']<=16*1024**2:
            batches[-1]['items'].append(item);batches[-1]['end']=end
        else:batches.append({'start':start,'end':end,'items':[item]})
    target.mkdir(parents=True,exist_ok=True)
    records=[]
    def batch(job):
        blob=fetch_range(selection['url'],job['start'],job['end']);done=[]
        for item in job['items']:
            rel=Path(item['name']);assert not rel.is_absolute() and '..' not in rel.parts
            data=decode_entry(item,blob,job['start']);path=target/rel;path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_name(path.name+'.part');tmp.write_bytes(data);tmp.replace(path)
            done.append({'path':str(rel),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'zip_crc':item['crc']})
        return done
    print('batches',len(batches),'transfer bytes',sum(x['end']-x['start']+1 for x in batches),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(batch,b) for b in batches]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            records.extend(f.result())
            (target/'_EXTRACTION_PROGRESS.json').write_text(json.dumps({'completed_batches':n,'total_batches':len(batches),'files':len(records)}))
            print('completed batches',n,'/',len(batches),flush=True)
    (target/'_VERIFIED_FILES.json').write_text(json.dumps({'archive_url':selection['url'],'files':records},indent=2)+'\n')
    return records


if __name__=='__main__':
    selection=json.loads((HERE/'legacy_asset_selection.json').read_text())
    source=json.loads((HERE/'legacy_asset_zip_index.json').read_text())
    result=extract(selection,source['entries'],HERE/'legacy_assets/behavior-1k-assets')
    print('verified files',len(result),flush=True)
