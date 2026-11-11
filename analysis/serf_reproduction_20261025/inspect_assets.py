"""Read public release manifests, without downloading checkpoint data."""
import json
from pathlib import Path
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
QUERIES = {
    'model': 'https://huggingface.co/api/models/byeonghyunpak/SERF-VLA?blobs=true',
    'maps_root': 'https://huggingface.co/api/datasets/suk063/SERF/tree/main',
    'maps_task22': 'https://huggingface.co/api/datasets/suk063/SERF/tree/main/map_models/task-0022?recursive=true',
    'points_task22': 'https://huggingface.co/api/datasets/suk063/SERF/tree/main/exported_neural_points/task-0022?recursive=true',
}

result = {}
for name, url in QUERIES.items():
    try:
        with urllib.request.urlopen(url, timeout=90) as response:
            raw = response.read(16 * 1024**2 + 1)
            assert len(raw) <= 16 * 1024**2
            data = json.loads(raw)
            result[name] = {'url': url, 'data': data, 'link': response.headers.get('Link')}
        if name == 'model':
            files = [x for x in data.get('siblings', []) if 'task-0022/' in x['rfilename']]
            print(name, 'revision', data.get('sha'), 'files', len(files),
                  'bytes', sum(x.get('size', 0) for x in files))
            print(json.dumps(files[:8], ensure_ascii=False))
        else:
            print(name, json.dumps(data, ensure_ascii=False)[:13000])
    except Exception as error:
        result[name] = {'url': url, 'error_type': type(error).__name__, 'error': str(error)}
        print(name, type(error).__name__, str(error))
(HERE / 'public_asset_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
