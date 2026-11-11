"""Read bounded existing demonstration samples; no model/simulator calls."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare_probe_cache import (DEFAULT_ARCHIVE, DEFAULT_ARCHIVE_SHA256,
    metadata_plan, open_source_bucket, TransferBudget, read_selected_parquet,
    decode_rgb, make_raw_sample)

EPISODES = [18002, 18003, 18000, 18001]
MAX_BYTES = 512 * 1024**2
MAX_SECONDS = 1200

def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def plan():
    import pandas as pd
    meta = HERE.parent / 'pilot_three_tasks_cache_v1/dataset/meta'
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((meta/'episodes').glob('chunk-*/*.parquet'))]).set_index('episode_index')
    segments = pd.read_parquet(meta/'original_subtask_segments.parquet')
    samples = []
    for episode in EPISODES:
        rows = segments[segments.episode_index.eq(episode)].sort_values('start')
        placement = rows[rows.subtask.str.startswith('place in ')].iloc[0]
        end = int(placement.end)
        for offset in [-90, -1, 30, 150]:
            frame = end + offset
            label = rows[(rows.start <= frame) & (rows.end > frame)].iloc[0].subtask
            samples.append({
                'sample_id': f'ep{episode}_f{frame}', 'episode_index': episode,
                'frame_index': frame, 'task_name': 'composting_waste',
                'task_instance_id': int(episodes.loc[episode].task_instance_id),
                'subtask': label, 'boundary_id': f'ep{episode}_first_placement',
                'relative_to_first_placement_end': offset,
                'first_placement_annotation': placement.subtask,
            })
    result, _ = metadata_plan(DEFAULT_ARCHIVE, {'sample_keys': samples}, DEFAULT_ARCHIVE_SHA256)
    result['kind'] = 'natural_order_physical_audit'
    result['transfer_limits'] = {'maximum_bytes': MAX_BYTES, 'maximum_seconds': MAX_SECONDS}
    result['scope'] = 'Existing data only; annotation boundary is an anchor, not an achievement label.'
    return result

def prepare(p):
    import numpy as np
    from PIL import Image, ImageDraw
    out = HERE/'natural_order_frames'
    out.mkdir(exist_ok=True)
    receipt_path = out/'receipt.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get('complete'):
            print(json.dumps({'already_complete': True, 'samples': len(old['records'])}))
            return
        raise ValueError('Existing incomplete receipt requires explicit review before retry')
    budget = TransferBudget(MAX_BYTES, MAX_SECONDS)
    receipt = {'complete': False, 'records': [], 'transfer_bytes': 0, 'transfer_requests': 0,
               'model_calls': 0, 'simulator_calls': 0, 'new_expert_actions': 0}
    phase = 'open_source'
    try:
        bucket = open_source_bucket(Path.home()/'.ossutilconfig', p['source_bucket'])
        objects = {r['relative_path']: r for r in p['objects']}
        groups = {}
        for sample in p['samples']:
            groups.setdefault(sample['data_relative_path'], []).append(sample)
        for relative, samples in groups.items():
            phase = 'read_selected_actions'
            columns = read_selected_parquet(bucket, objects[relative], budget, {s['episode_index'] for s in samples})
            episodes, frames = np.asarray(columns['episode_index']), np.asarray(columns['frame_index'])
            for episode in sorted({s['episode_index'] for s in samples}):
                selected = [s for s in samples if s['episode_index'] == episode]
                selected.sort(key=lambda s: s['frame_index'])
                imgs = {s['sample_id']: {} for s in selected}
                for camera in p['cameras']:
                    phase = 'decode_rgb'
                    v = selected[0]['videos'][camera]
                    queries = []
                    for s in selected:
                        match = np.flatnonzero((episodes == episode) & (frames == s['frame_index']))
                        if len(match) != 1:
                            raise ValueError('Non-unique action row')
                        queries.append(float(columns['timestamp'][match[0]]) + v['from_timestamp'])
                    decoded, actual = decode_rgb(bucket, objects[v['relative_path']], budget, queries, v['shape'])
                    for s, pixels, query, pts in zip(selected, decoded, queries, actual, strict=True):
                        imgs[s['sample_id']][camera] = pixels
                sheet = Image.new('RGB', (3*512, len(selected)*418), 'white')
                draw = ImageDraw.Draw(sheet)
                for row, sample in enumerate(selected):
                    phase = 'save_sample'
                    sample_imgs = imgs[sample['sample_id']]
                    raw = make_raw_sample(sample, columns, sample_imgs)
                    raw_numeric = {k: v for k, v in raw.items() if k not in sample_imgs and not isinstance(v, str)}
                    numeric_path = out/(sample['sample_id']+'.npz')
                    np.savez_compressed(numeric_path, **raw_numeric)
                    image_records = []
                    for col, camera in enumerate(p['cameras']):
                        pixels = (sample_imgs[camera].transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8)
                        image_path = out/(sample['sample_id']+'_'+camera.rsplit('.',1)[-1]+'.png')
                        im = Image.fromarray(pixels)
                        im.save(image_path)
                        preview = im.copy()
                        preview.thumbnail((512,384))
                        sheet.paste(preview, (col*512, row*418+32))
                        draw.text((col*512+4,row*418+2), f"ep{episode} frame{sample['frame_index']} {camera.rsplit('.',1)[-1]}", fill='black')
                        draw.text((col*512+4,row*418+17), f"offset {sample['relative_to_first_placement_end']}", fill='black')
                        image_records.append({'camera': camera, 'path': str(image_path.relative_to(HERE)), 'sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()})
                    receipt['records'].append({'sample_id':sample['sample_id'], 'episode_index':episode,
                        'frame_index':sample['frame_index'], 'first_placement_annotation':sample['first_placement_annotation'],
                        'images':image_records, 'numeric_sha256': hashlib.sha256(numeric_path.read_bytes()).hexdigest()})
                sheet.save(out/f'ep{episode}_contact_sheet.png')
                receipt.update(transfer_bytes=budget.used_bytes, transfer_requests=budget.requests)
                write_json(receipt_path, receipt)
                print(json.dumps({'episode':episode,'saved_samples':len(receipt['records']),'transfer_bytes':budget.used_bytes}), flush=True)
        receipt['complete'] = len(receipt['records']) == len(p['samples'])
    except BaseException as error:
        receipt['failure'] = {'phase':phase, 'type':type(error).__name__}
        # No raw network exceptions or credential-bearing endpoints in output.
    receipt.update(transfer_bytes=budget.used_bytes, transfer_requests=budget.requests)
    write_json(receipt_path, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    p = plan()
    write_json(HERE/'natural_order_source_plan.json',p)
    print(json.dumps({'samples':len(p['samples']),'source_objects':len(p['objects']),'maximum_bytes':MAX_BYTES}), flush=True)
    if args.prepare:
        prepare(p)
